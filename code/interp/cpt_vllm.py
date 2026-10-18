"""Behavioral CPT scoring with vLLM (FP8 scale models and cross-family stress models).
Same worlds and token ids as the HF path; endpoint = top-k log-probs of the first token after "VERDICT:".
Letters outside the top-k get the k-th log-prob as an upper bound (flagged). World order is shuffled so batch-dependent
numerics are independent of the factors; 3% of worlds are scored twice to estimate the test-retest noise floor."""
import argparse, json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interp"))
from common import *  # noqa
from cpt import build_ids, endpoints, letter_token_ids, world_list
from run_cpt import usable_items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--split", default="release_600")
    ap.add_argument("--gpu_util", type=float, default=0.90); ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--all_templates", action="store_true")
    ap.add_argument("--quant", default=None, choices=[None, "fp8", "bitsandbytes"], help="E17 online quantization of a BF16 checkpoint")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    info = MODELS[args.model]; path, sha = snapshot(args.model)
    tok = AutoTokenizer.from_pretrained(path)
    merge = info["family"] in ("mistral",)
    items = usable_items(args.split)
    worlds = []
    for r, cv, s in items:
        tpls = [0, 1, 2] if args.all_templates else [h64("tpl", r["id"]) % 3]
        for w in world_list(cv, tpls):
            ids, pos = build_ids(tok, cv, s, w, merge_system=merge)
            worlds.append((dict(item_id=r["id"], family=cv["family"], domain=cv["domain"], stratum=cv["stratum"],
                                has_target=cv["has_target"], o1=cv["o1"], o2=cv["o2"], author=s["author"], **w,
                                agree=(w["adv"] == w["prior"]) if w["adv"] else None, n_tok=len(ids)), ids, cv))
    rng = random.Random(240091)
    order = list(range(len(worlds))); rng.shuffle(order)
    retest = rng.sample(range(len(worlds)), max(50, len(worlds) // 33))
    kw = {}
    if mm_limits(args.model):
        kw["limit_mm_per_prompt"] = mm_limits(args.model)
    if args.quant:
        kw["quantization"] = args.quant
    llm = LLM(model=path, tokenizer=path, dtype="auto", gpu_memory_utilization=args.gpu_util, max_model_len=4096,
              enable_prefix_caching=True, seed=0, max_logprobs=args.k, max_num_seqs=max_seqs(args.model), **kw)
    lids = letter_token_ids(tok)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=args.k)
    t = time.time()
    seq = order + retest
    outs = llm.generate([{"prompt_token_ids": worlds[i][1]} for i in seq], sp, use_tqdm=False)
    recs, rt = [None] * len(worlds), []
    for n, (i, o) in enumerate(zip(seq, outs)):
        meta, ids, cv = worlds[i]
        lp = o.outputs[0].logprobs[0]
        floor = min(v.logprob for v in lp.values())
        letters, missing = {}, []
        for L in cv["labels"]:
            vals = [lp[t_].logprob for t_ in lids[L] if t_ in lp]
            if vals:
                import math
                letters[L] = math.log(sum(math.exp(v) for v in vals))
            else:
                letters[L] = floor; missing.append(L)
        top = o.outputs[0].token_ids[0]
        rec = dict(meta, **endpoints(letters, cv, meta), lp_letters=letters, missing_letters=missing,
                   top_is_label=any(top in lids[L] for L in cv["labels"]), model=args.model, model_sha=sha)
        if n < len(order):
            recs[i] = rec
        else:
            rt.append(dict(world=i, m12_a=recs[i]["m12"], m12_b=rec["m12"]))
    out_dir = ACTS / args.model / f"cpt_vllm_{args.split}{'_alltpl' if args.all_templates else ''}{'_' + args.quant if args.quant else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "worlds.jsonl", recs); write_jsonl(out_dir / "retest.jsonl", rt)
    import statistics
    sd = statistics.pstdev([x["m12_a"] - x["m12_b"] for x in rt])
    print(f"[done] cpt_vllm {args.model} {args.split}: {len(recs)} worlds in {(time.time()-t)/60:.1f} min; retest SD(dm12)={sd:.4f}; "
          f"missing-letter worlds={sum(bool(r['missing_letters']) for r in recs)}", flush=True)

if __name__ == "__main__":
    main()
