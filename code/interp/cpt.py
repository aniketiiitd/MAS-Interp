"""Causal Provenance Tracing (CPT): counterfactual receiver worlds, scoring, activation capture, patching (HF).

World factors for one receiver episode (item i, wrapper template t):
  ADV  in {o1, o2}                advocated option in the incoming message (o1 = target when the item has one)
  QUAL in {strong, weak, neutral} evidence quality (neutral = length-matched message with no verdict and no reasons)
  PROV in {expert, anon, none}    provenance descriptor in the message header (none = source-label ablation)
  PRIOR in {o1, o2}               the receiver's own earlier public verdict (social relation = PRIOR == ADV)
The receiver's reply is prefilled with "VERDICT:" and the next-token distribution over option letters is the endpoint.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from dataclasses import dataclass

import numpy as np
import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
from common import *  # noqa
from prompts import FORMAT_BLOCK, AGENT_NAMES

WRAPPERS = json.load(open(ROOT / "configs/wrappers.json"))  # list of {id, header, none_header, expert, anon}
NEUTRAL_TEXT = ("I went over the task and the discussion again, but I do not have a specific reason to offer for any of "
                "the options at this point, so I am not taking a position in this round and will leave the decision to the "
                "other participants.")
PRIOR_CONF, MSG_CONF = 70, 80


def load_stimuli():
    stim = {}
    for a in ("G31", "Q27", "G31fb"):  # G31fb (author fallback for items Q27 refused) overrides Q27 entries
        p = ROOT / f"data/mad_interp/stimuli_{a}.jsonl"
        if p.exists():
            for s in read_jsonl(p):
                stim[s["item_id"]] = s
    return stim


def receiver_system(labels):
    head = (f"You are {AGENT_NAMES[0]}, one of 3 AI agents working on the same task over several rounds. "
            "In later rounds you may be shown messages from other participants.")
    return head + "\n\n" + FORMAT_BLOCK.format(labels="exactly one option letter: " + ", ".join(labels))


def world_list(cv, templates, include_neutral=True):
    ws = []
    for t in templates:
        for adv, qual, prov, prior in itertools.product((cv["o1"], cv["o2"]), ("strong", "weak"), ("expert", "anon", "none"),
                                                        (cv["o1"], cv["o2"])):
            ws.append(dict(template=t, adv=adv, qual=qual, prov=prov, prior=prior))
        if include_neutral:
            for prov, prior in itertools.product(("expert", "anon"), (cv["o1"], cv["o2"])):
                ws.append(dict(template=t, adv=None, qual="neutral", prov=prov, prior=prior))
    return ws


def render_world(cv, stim, w):
    """Return (user_text, spans) where spans are char ranges inside user_text."""
    W = WRAPPERS[w["template"]]
    prior_reason = stim["args"][w["prior"]]["prior"]
    user = f"Task:\n{cv['task']}\n\nRound 2. Your previous response:\n"
    prior_block = f"VERDICT: {w['prior']}\nCONFIDENCE: {PRIOR_CONF}\nEVIDENCE: {prior_reason}"
    s_prior = (len(user), len(user) + len(prior_block))
    user += prior_block + "\n\nA response from another participant in the previous round:\n\n"
    if w["prov"] == "none":
        header = W["none_header"]
        s_src = None
        user += header
    else:
        pre, post = W["header"].split("{prov}")
        desc = W[w["prov"]]
        user += pre
        s_src = (len(user), len(user) + len(desc))
        user += desc + post
    user += "\n"
    msg_start = len(user)
    if w["qual"] == "neutral":
        arg = NEUTRAL_TEXT
        msg = f"VERDICT: undecided\nCONFIDENCE: 50\nEVIDENCE: "
    else:
        arg = stim["args"][w["adv"]][w["qual"]]
        msg = f"VERDICT: {w['adv']}\nCONFIDENCE: {MSG_CONF}\nEVIDENCE: "
    user += msg
    s_arg = (len(user), len(user) + len(arg))
    user += arg
    s_msg = (msg_start, len(user))
    user += "\n\nConsidering this response, give your updated response in the required format."
    return user, dict(prior=s_prior, src=s_src, arg=s_arg, msg=s_msg)


def build_ids(tok, cv, stim, w, merge_system=False):
    user, spans = render_world(cv, stim, w)
    sysm = receiver_system(cv["labels"])
    msgs = ([{"role": "user", "content": sysm + "\n\n" + user}] if merge_system else
            [{"role": "system", "content": sysm}, {"role": "user", "content": user}])
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    text = prefix + "VERDICT:"
    base = prefix.index(user)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]

    def tokspan(cs):
        if cs is None:
            return None
        a, b = cs[0] + base, cs[1] + base
        idx = [i for i, (s, e) in enumerate(offs) if e > a and s < b]
        return (idx[0], idx[-1] + 1)

    pos = dict(prior=tokspan(spans["prior"]), src=tokspan(spans["src"]), arg=tokspan(spans["arg"]), msg=tokspan(spans["msg"]))
    pos["prior_end"] = pos["prior"][1] - 1
    pos["msg_end"] = pos["msg"][1] - 1
    pos["arg_mid"] = (pos["arg"][0] + pos["arg"][1]) // 2
    pos["answer"] = len(ids) - 1
    if pos["src"]:
        pos["src_end"] = pos["src"][1] - 1
    return ids, pos


def letter_token_ids(tok):
    out = {}
    for L in LETTERS:
        ids = set()
        for s in (" " + L, L):
            t = tok(s, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                ids.add(t[0])
        out[L] = sorted(ids)
    return out


# ---------------------------------------------------------------------------
class Receiver:
    def __init__(self, key, dtype=torch.bfloat16, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.key = key
        self.path, self.sha = snapshot(key)
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, dtype=dtype, device_map=device)
        self.model.eval()
        m = self.model.model
        self.layers = m.language_model.layers if hasattr(m, "language_model") else m.layers
        self.n_layers = len(self.layers)
        self.softcap = getattr(self.model.config.get_text_config(), "final_logit_softcapping", None)
        # Gemma-4 E-series re-injects token identity at every layer via per-layer embeddings (PLE); patches must move them too
        self.has_ple = bool(getattr(self.model.config.get_text_config(), "hidden_size_per_layer_input", 0))
        # float32 unembedding: bf16 logits are rounded to ~1/128 relative precision, which quantizes small margins
        self.W32 = self.model.lm_head.weight.detach().float()
        assert getattr(self.model.lm_head, "bias", None) is None
        self.device = device
        self._letter_ids = {}
        for L in LETTERS:
            ids = set()
            for s in (" " + L, L):
                t = self.tok(s, add_special_tokens=False)["input_ids"]
                if len(t) == 1:
                    ids.add(t[0])
            self._letter_ids[L] = sorted(ids)
        self.hooks = []

    # -- rendering ----------------------------------------------------------
    def build(self, cv, stim, w):
        return build_ids(self.tok, cv, stim, w)

    # -- hooks ----------------------------------------------------------------
    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def add_capture(self, layer_idx, positions_per_row, store):
        """Capture residual output of decoder layer `layer_idx` at given positions (list per row) into store[layer_idx]."""
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            store[layer_idx] = [h[r, p].detach().to(torch.float32).cpu() for r, p in enumerate(positions_per_row)]
        self.hooks.append(self.layers[layer_idx].register_forward_hook(hook))

    def add_capture_mean(self, layer_idx, spans_per_row, store):
        """Capture the MEAN residual over a span of positions per row (Anthropic-style story pooling).

        add_capture takes one position per row; pooling a whole story needs the mean over its tokens, and doing that
        inside the hook avoids materialising [rows, tokens, d] on the GPU.
        """
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            store[layer_idx] = [h[r, p].mean(0).detach().to(torch.float32).cpu() for r, p in enumerate(spans_per_row)]
        self.hooks.append(self.layers[layer_idx].register_forward_hook(hook))

    def add_patch(self, layer_idx, row_patches):
        """row_patches: list of (row, positions LongTensor, values Tensor[len, d]) or add-mode (row, pos, delta, 'add')."""
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            for rp in row_patches:
                r, p, val = rp[0], rp[1], rp[2]
                if len(rp) > 3 and rp[3] == "add":
                    h[r, p] = h[r, p] + val.to(h.dtype)
                else:
                    h[r, p] = val.to(h.dtype)
            return out
        self.hooks.append(self.layers[layer_idx].register_forward_hook(hook))

    def add_ple_capture(self, layer_idx, positions_per_row, store):
        """Capture the per-layer input (PLE) consumed by layer `layer_idx` at given positions."""
        def pre(mod, args):
            ple = args[1]
            store[layer_idx] = [ple[r, p].detach().to(torch.float32).cpu() for r, p in enumerate(positions_per_row)]
        self.hooks.append(self.layers[layer_idx].register_forward_pre_hook(pre))

    def add_ple_patch(self, layer_idx, row_patches):
        """Overwrite the per-layer input consumed by layer `layer_idx`: row_patches = [(row, positions, values)]."""
        def pre(mod, args):
            ple = args[1]
            for r, p, val in row_patches:
                ple[r, p] = val.to(ple.dtype)
            return None
        self.hooks.append(self.layers[layer_idx].register_forward_pre_hook(pre))

    @torch.inference_mode()
    def forward(self, batch_ids):
        """Right-padded forward pass; returns log-probs at each row's last real token (float32, cpu)."""
        n = len(batch_ids)
        T = max(len(x) for x in batch_ids)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        inp = torch.full((n, T), pad, dtype=torch.long)
        att = torch.zeros((n, T), dtype=torch.long)
        for i, x in enumerate(batch_ids):
            inp[i, : len(x)] = torch.tensor(x)
            att[i, : len(x)] = 1
        inp, att = inp.to(self.device), att.to(self.device)
        last = torch.tensor([len(x) - 1 for x in batch_ids], device=self.device)
        out = self.model.model(input_ids=inp, attention_mask=att if n > 1 else None, use_cache=False)
        hs = out.last_hidden_state[torch.arange(n, device=self.device), last]
        logits = torch.nn.functional.linear(hs.float(), self.W32)
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return torch.log_softmax(logits.float(), -1).cpu()

    def letter_lp(self, lp_row, labels):
        return {L: float(torch.logsumexp(lp_row[self._letter_ids[L]], 0)) for L in labels}


def endpoints(lp_letters, cv, w, lp_row=None, letter_ids=None):
    labels = cv["labels"]
    vals = np.array([lp_letters[L] for L in labels])
    p = np.exp(vals - np.logaddexp.reduce(vals))
    P = dict(zip(labels, p))
    o1, o2 = cv["o1"], cv["o2"]
    rec = dict(lp_o1=lp_letters[o1], lp_o2=lp_letters[o2], m12=lp_letters[o1] - lp_letters[o2], p_o1=P[o1], p_o2=P[o2],
               p_label_mass=float(np.exp(np.logaddexp.reduce(vals))), argmax_label=labels[int(vals.argmax())])
    if w["adv"] is not None:
        other = o2 if w["adv"] == o1 else o1
        rec["m_adv"] = lp_letters[w["adv"]] - lp_letters[other]
        rec["p_adv"] = P[w["adv"]]
    rec["m_prior"] = lp_letters[w["prior"]] - lp_letters[o2 if w["prior"] == o1 else o1]
    return rec
