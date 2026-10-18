"""Assemble every current figure into one PDF, contributions first, each figure with a full caption.

Every number quoted in the captions below is read from the tables at build time where a table exists, so the document
cannot drift from the data the way a hand-written summary does. Status labels are deliberately part of the layout:
a retracted claim and a solid one must not look the same on the page.

Run: python figures/build_pdf.py
Out: outputs/figures/MAD_Interp_figures.pdf
"""
from __future__ import annotations

import datetime as dt
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
import style  # noqa: E402

FIGDIR = Path(__file__).resolve().parents[2] / "outputs/figures/paper"
OUT = Path(__file__).resolve().parents[2] / "outputs/figures/MAD_Interp_figures.pdf"
L = 0.07   # left margin, in figure fraction; every page uses the same text column

CONTRIBUTIONS = [
    ("MAIN", "Debate settles the answer before it produces the argument",
     "A response written during a debate reads like reasoning that arrives at a conclusion. We checked whether that is "
     "what happens. An agent is stopped part-way through writing and made to answer at once, and we ask whether that "
     "forced answer already matches the one it will end up writing. If the argument were doing the work, stopping it "
     "early would catch the agent mid-thought and the two would often differ. They do not. Among the cases where an "
     "agent changes its mind after reading its peers, most of those changes are already fixed before it has written a "
     "single word of its own reasoning, and this holds on every architecture we tried: 65.4% against 22.3% on "
     "Qwen3.5-9B, 52.4% against 35.4% on Llama-3.1-8B and 48.5% against 12.6% on Gemma-4-E4B, each across the whole "
     "benchmark against the same agent on the same question with nobody to read. Two checks say the effect comes from "
     "having read the peers rather than from having text in front of it. Neutral filler of the same length before the "
     "forced answer leaves it intact, moving the rate by 2.5 points [-5.4, +0.4], an interval covering zero; "
     "another item's reasoning in that slot collapses the gap between the arms from 43 points to 4.6. Comparing "
     "each question against itself across the two conditions, the gap "
     "is 40.9 points on Qwen, 35.9 on Gemma and 15.6 on Llama, and a sign test over those items separates the models "
     "in a way the means hide: 87% and 86% of paired items favour debate in Qwen and Gemma against 64% in Llama, so "
     "the effect is large and consistent in two architectures and smaller and patchier in the third. The argument the "
     "agent appears to be reasoning through is produced after the decision it explains, which means reading it tells "
     "you less about that decision than it seems to."),
    ("MAIN", "What looks like respect for authority is about reliability",
     "Agents defer to sources described as authoritative, and the obvious reading is that credentials impress the "
     "model. We describe the same peer message in different ways, crossing the speaker's standing (a tenured professor "
     "against an ordinary member of the public) with the speaker's track record (strong, unknown or poor), and measure "
     "how far the reader's decision moves. The credential is plainly legible inside the model, readable at 0.85 AUROC. "
     "How much it is used is not a property of language models but of the particular one: swapping the standing is "
     "worth 0.011 logits on Llama-3.1-8B, 0.077 on Qwen3.5-9B and 0.793 on Gemma-4-E4B, a seventy-fold spread, while "
     "swapping the stated track record is worth 0.876, 0.560 and 1.905. Gemma plainly does attend to the credential, "
     "and any claim that models ignore credentials would be false of it. What is invariant is the ordering and what "
     "overrules what. The record leads the standing in every model, and it still leads after we amplify the standing "
     "inside the network until the two signals match in strength: by 14.3 times [7.3, 69.6] on Llama, 3.7 [3.0, 4.7] "
     "and 4.4 [3.4, 6.1] on Qwen at the two matched magnitudes, and 1.44 [1.33, 1.56] and 1.39 [1.31, 1.50] on Gemma, "
     "where the contest is closest. The sharpest cell is the same in all three: a professor with a poor record is "
     "trusted below a source carrying no description at all, by 0.203 logits on Qwen, 0.985 on Llama and 0.966 on "
     "Gemma. Given a poor record the credential is worth +0.010 on Qwen, whose interval covers zero, and turns "
     "negative on the other two, -0.383 on Llama and -0.097 on Gemma. So a credential buys a hearing in some models "
     "and almost none in others, but in none of them does it survive contact with a bad record. What looks like "
     "deference to authority is the model using a stated record as evidence about reliability, and a credential is "
     "only a stand-in for that, worth whatever that model has learned the stand-in is worth."),
    ("MAIN", "Influence does not arrive through attention to the peer's message",
     "The natural way to ask where a peer's influence travels is to cut every attention link to its message and see "
     "what is left. On a model built entirely from attention this works, and it works twice. In Llama-3.1-8B, whose 32 "
     "layers are all attention, blocking the message everywhere removes 132% [122, 144] of its effect, more than "
     "deleting the message outright. In Gemma-4-E4B it removes 117% [111, 123]. Gemma is worth describing exactly, "
     "because it is not a plain attention stack: its 42 layers are all attention, but only 7 are global and the other "
     "35 are sliding-window over 512 tokens, against a median debate prompt here of 1,668 tokens; 96.8% of those "
     "prompts exceed the window. For nearly every item the peer's message therefore sits outside the window of the "
     "layers, so the influence has to travel through the seven global "
     "ones. What matters for the argument is that every one of the 42 takes a mask, which is exactly what Qwen's 24 "
     "recurrent layers do not. On a hybrid it does not. Qwen3.5-9B is attention at only 8 of its 32 layers; the other "
     "24 are Gated-DeltaNet recurrent layers, where there is no attention mask to apply. The same complete block "
     "removes 80% [75, 84] across the whole benchmark, so about a fifth of the influence survives an intervention that "
     "takes all of it in both all-attention models, and the three intervals do not overlap. Gemma was a registered "
     "falsification arm: it is a different vendor, size and depth from Llama, and the prediction that it would behave "
     "like Llama rather than like Qwen was written down before its data existed, with retraction rather than caveat as "
     "the stated consequence of the other outcome. Two controls say the mask is doing what we think. Blocking an equal "
     "length of the agent's own earlier response instead of the peer's message leaves it more persuaded rather than "
     "less, by 14% on Qwen and 20% on Gemma, so the effect is not a generic consequence of removing context; and "
     "blocking the peer everywhere except at the deciding token removes only 9% on Qwen and 0.6% [-0.4, 1.6] on Gemma, "
     "so what matters is the message being readable while the agent writes, not at the moment it commits. A second "
     "measurement, which uses no attention mask at all, agrees. Interchanging the residual stream over the peer's "
     "message and asking how much of its effect that recovers, the influence survives markedly deeper in the hybrid: "
     "at layer 12 of 32 the recovery is 0.24 [0.22, 0.27] in Qwen against 0.03 [0.02, 0.04] in Llama, and it falls "
     "below a tenth four layers later in the hybrid than in the all-attention model. Gemma agrees at matched relative "
     "depth, 0.06 where the hybrid holds 0.24 at the same fraction of the way through the network. It also answers the "
     "one objection its own architecture invites. Per-layer embeddings are the single component Gemma has that an "
     "attention mask does not touch, so we patched them alone: they recover at most 0.018 of the peer's effect at any "
     "layer, and adding them to a residual patch moves the recovery by at most 0.011. They are a parameter-efficiency "
     "device rather than a channel between token positions, which is why masking attention takes everything in that "
     "model and not in the hybrid. Two methods with no machinery in common point the same way, which is the reason to "
     "believe the gap is in the architecture rather than in either measurement. The consequence for practice is that "
     "attention-based interpretability under-reports influence on hybrid models, and the share it misses is not small."),
    ("MAIN", "A peer moves the reader through its reasoning, not its answer",
     "A peer sends two things at once: what it concluded, and why. We separate them on the same question by giving the "
     "reader the message three ways, as written, with the reasoning stripped out so only the bare answer remains, and "
     "with the answer stripped out so only the reasoning remains. Across the whole benchmark the bare answer moves the "
     "reader 0.59 logits and the reasoning moves it 1.31, so 69% [67, 72] of a peer's influence is carried by its "
     "argument and 31% by the bare fact of its conclusion. That share barely moves across three unrelated kinds of "
     "task: 66% on knowledge, 72% on safety, 70% on values, and it survives a change of "
     "architecture. Gemma-4-E4B is almost three times as movable in absolute terms, shifting 5.23 logits where Qwen "
     "shifts 1.95, and still takes 60% [57, 63] of that from the reasoning; on values questions the three read "
     "70.0% [67.2, 73.2] in Qwen, 68.4% [64.5, 73.3] in Gemma and 65.8% [60.5, 73.0] in Llama, four points apart "
     "across a hybrid, a dense transformer and a small model with per-layer embeddings. Strict additivity is not "
     "stable in the same way, and how it fails "
     "is the evidence. In Qwen the two parts measured apart sum to what the whole message does, pooled and on "
     "knowledge and on values separately. In Llama they fall short by 0.69 and in Gemma they overshoot by 0.43: the "
     "same test failing in opposite directions in two architectures, which is what a shared interaction between the "
     "two channels cannot look like, and leaves additivity an architectural detail rather than a property of peer "
     "messages. The share the reasoning carries is what generalises. Beside the first claim this is a single sentence: "
     "the reasoning a reader takes in is what moves it, and the reasoning it writes afterwards is not."),
    ("MAIN", "Debate collapses a group into one agent, which is why voting over it stops paying",
     "The fair comparison for a debate is not a single agent but the same agents answering independently and voting, "
     "which costs the same compute. Debate beats that baseline by about three points, but most of that is an artifact "
     "of measurement: the vote is built from the agents' first answers, which are the hardest to parse, and an "
     "unparseable answer counts against the arm that produced it. On the questions where every agent actually "
     "answered, the gain is 1.2 points on Qwen, 1.2 on Gemma, and half a point on Llama whose interval covers zero. "
     "What debate reliably produces is not accuracy but agreement. Ending unanimous on a wrong answer rises 3.1 points "
     "on Qwen, 3.6 on Gemma and 7.4 on Llama against the same agents thinking for the same number of rounds without "
     "communicating, and that cost survives the same restriction in all three. Two measurements say why. The group "
     "stops moving early and then stays stopped, in all three models over twelve turns. The share of groups that "
     "have already made their last answer change after a single turn runs 90% [87, 92] debating against 80% "
     "[76, 83] alone in Qwen3.5-9B, 88% [85, 90] against 67% [64, 71] in Gemma-4-E4B, and 70% [67, 74] against "
     "14% [12, 17] in Llama-3.1-8B. The gap is 10.2, 20.5 and 56.2 points: present in every model and five times "
     "wider in one than in another, so the effect is general and its size is not. There is no model here in "
     "which thinking alone freezes faster. And between "
     "turns 6 and 11 the debating arm moves half a point while generating eight million tokens, and those tokens are "
     "not idling: paired by agent and question, a debating agent's response grows by 224 characters [167, 274] "
     "between turn 1 and turn 11 and 74% of agents write more, while the same agents alone shrink by 69 [32, 109], "
     "a gap of 293 [236, 347]. Llama-3.1-8B agrees about the gap and not about its parts: there debate grows by 573 "
     "[518, 625] and thinking alone grows as well, by 62 [20, 104], for a gap of 511 [444, 578]. What carries across "
     "is that debate outgrows thinking alone, not what thinking alone does by itself. Nor is the extra text more "
     "engagement with the peers in the same way: the rate at which a debating agent refers to them is flat in Qwen, "
     "21.7% to 20.3%, and climbs in Llama, 39.3% to 60.3%. Agents working alone keep producing changes for as long "
     "as you let them while debate does not, which is why the gap needs a long run to be seen at its full size: on "
     "Llama the same contrast measured at three turns is 76% against 38%, about two thirds of the twelve-turn gap. And the group "
     "stops being a group: scoring how often the three agents are right or wrong together and converting that to "
     "effective independent voters, three agents that have debated are worth 1.07 of them after three turns on the "
     "whole benchmark, and all three models land on that same 1.07 from different starting points: 1.31 alone in Qwen, "
     "1.27 in Gemma and 1.62 in Llama. On the questions where the three disagreed before anyone spoke, which are the "
     "ones a vote could have helped, agents that stay silent keep 2.34, 2.25 and 1.80 independent voters and debate "
     "leaves 1.18, 1.33 and 1.15. Over twelve turns the collapse goes further, and alike in all three: on the "
     "contested questions the debating group falls to 1.02 [1.00, 1.07] in Qwen, 1.02 [1.00, 1.06] in Gemma and "
     "1.07 [1.01, 1.14] in Llama, from 1.87, 1.51 and 1.51 at the first turn, while the same agents without "
     "communication still hold 2.10, 1.77 and 1.70. Three architectures, three different starting points and one "
     "ending: a debating group of three is worth one agent and a silent one is not. The same three-against-none "
     "pattern holds for agreement itself: after twelve turns on the contested questions, 98.9% [96.6, 100.0], "
     "97.5% [93.5, 100.0] and 93.8% [89.0, 98.1] of debating groups are unanimous, against 36.8% [26.4, 47.1], "
     "35.4% [23.8, 47.2] and 35.9% [26.3, 45.7] of the same agents thinking alone. The three debating intervals "
     "and the three silent ones do not overlap at all. A third measurement says where the "
     "agreed answer comes from. On the questions a "
     "group began split, debate reaches unanimity 2.1 to 2.7 times as often as the same agents thinking alone, and the "
     "answer it settles on is the one the plurality already held 76%, 73% and 83% of the time. That share is the same "
     "whether the group ends up right or wrong, 76% against 76% on Qwen and 82% against 84% on Gemma, with Llama the "
     "one exception at 64% against 80%, so overturning the starting plurality is not what makes a debate wrong. Debate "
     "changes how often a group agrees, not what it agrees on. A vote over one effective agent cannot correct errors, "
     "which is why the accuracy gain is small and why confident agreement on a wrong answer grows instead."),
    ("SUPPORTING", "Agents favour text labelled as their own",
     "Agents treat the same words differently depending on whose name is on them. Taking a peer's message and "
     "relabelling it as the agent's own earlier text is worth +0.749 logits on Qwen3.5-9B, +1.086 on Llama-3.1-8B and "
     "+1.302 on Gemma-4-E4B across the whole benchmark, and the comparison is between identical bytes, so nothing "
     "about the content differs. As a control, moving that text to a different position in the prompt without "
     "relabelling it shifts the decision by -0.019, +0.024 and -0.032 logits, which is a fortieth of the labelling "
     "effect in every model, 39, 45 and 41 times smaller respectively. What matters is whose the text is said to be "
     "rather than where it sits. The effect is already reported in the literature; the position control is ours, and "
     "it is the same size relative to the effect on three architectures whose labelling effects differ by a factor of "
     "nearly two."),
    ("SUPPORTING", "Concept probes on debate states need a permuted-label null, not a chance baseline",
     "We fitted probes for 171 emotion concepts and 22 moral concepts on a separate story corpus that never contained "
     "a debate. As instruments they are sound, reading held-out concepts at 0.82 and 0.955 out of fold and replicating "
     "almost exactly across two architectures. As a way to learn something about debate they produced six coherent and "
     "false stories in a row, because the states are so strongly organised by what the question is about that almost "
     "any direction looks like a good probe. A direction refit on deliberately shuffled concept labels still separates "
     "moral from knowledge questions, at a median AUROC of 0.685 in Qwen3.5-9B, 0.648 in Gemma-4-E4B and 0.602 in "
     "Llama-3.1-8B, and as high as 0.993. The separation is strongest exactly where the probes themselves are "
     "strongest: at Qwen's best layer the shuffled directions reach a 90th percentile of 0.932, so a probe reported "
     "at that depth at 0.85 against a 0.5 chance baseline sits below the 90th percentile of what carrying no concept "
     "information achieves. Effects that clear a chance baseline are not stable enough to carry a direction, let "
     "alone a mechanism. Take the two moral concepts that move most in Qwen over twelve turns, fix them in advance, "
     "and measure both models on the three-round runs they both have. While the agent reads its peers the two models "
     "reverse: self-righteous drifts +0.20 [+0.18, +0.22] in Qwen against -0.29 [-0.32, -0.26] in Llama, and "
     "conflicted-duty -0.29 [-0.31, -0.27] against +0.27 [+0.25, +0.30]. At the deciding token in the same episodes "
     "they do not reverse at all, both models drifting the same way on both concepts. The sign is a property of where "
     "in the network the probe is read and which layer is picked, not of the model: across ten layers the two models "
     "disagree about the sign at six and seven of them while reading and at two and five while deciding. The correct "
     "null is the same directions refit with the labels permuted inside each "
     "topic, and against that null none of our six analyses survives."),
    ("SUPPORTING", "Debate has no single emotional signature; it depends on what is being argued about",
     "A concept atlas read at every turn of a twelve-turn debate, compared against the same agents thinking for the "
     "same number of turns without communicating, shows that what debate does to a reader's affect depends on the "
     "kind of question. The three models validate different numbers of concepts, 158 for Qwen3.5-9B, 139 for "
     "Llama-3.1-8B and 105 for Gemma-4-E4B, so everything below is measured on the 99 all three validate, because "
     "a smaller and differently selected concept set is a different instrument rather than the same one applied "
     "three times. Concepts are chosen on one half of the items and re-tested on the other, against a null in "
     "which the domain labels are shuffled between item clusters; a spread is a maximum minus a minimum and so is "
     "positive even when nothing is happening, which is why every figure here is quoted against that shuffled "
     "null rather than against zero. The dependence itself is not subtle and it holds in all three: at the "
     "reading site 99 of 99 concepts shift differently by domain in Qwen, 88 of 99 in Llama and 98 of 99 in "
     "Gemma, and at the decision site 84, 94 and 98 of 99, at spreads 1.5 to 3.0 times the shuffled null. The "
     "sizes differ by a factor of five: the mean absolute drift at the last turn runs 0.35, 0.89 and 1.77 on "
     "knowledge questions and 0.42, 0.76 and 2.38 on values questions. What does not carry across is the "
     "*shape*, and the third model is what settles it. Over twelve turns at the reading site the domains "
     "converge in Qwen, 0.46 to 0.37, diverge in Gemma, 0.30 to 0.56, and do neither in Llama, 0.21 to 0.22. At "
     "the deciding token only Qwen moves at all, 0.45 to 0.57, against 0.91 to 0.89 in Gemma and 0.60 to 0.60 "
     "in Llama. So Qwen moves at both sites, Gemma only while reading, Llama at neither, and the one site where "
     "two models both move is the site where they move opposite ways. Nor do the three agree on how often "
     "domains "
     "are pushed opposite ways at the deciding token: 50% of "
     "confirmed concept-turn cells in Qwen, 21% in Llama and 3% in Gemma. We report the dependence as the "
     "finding and the dynamics as model-specific. Two models could only show that one shape failed to "
     "replicate; three show that there is no shared shape to find, which is a claim about these architectures "
     "rather than about a pair of them."),
]

STATUS_COLOR = {"MAIN": "#1B7F5A", "SUPPORTING": "#3E6B8A"}
PAGE = (8.27, 11.69)


def text_page(pdf, title, subtitle, blocks):
    """The claims, flowing onto as many pages as they need. A fixed single page silently truncated the last one."""
    BOTTOM = 0.06

    def new_page(first):
        fig = plt.figure(figsize=PAGE)
        fig.patch.set_facecolor("white")
        y = 0.955
        if first:
            fig.text(L, y, title, fontsize=19, weight="bold", color="#1A1A1A")
            y -= 0.030
            for line in textwrap.wrap(subtitle, 104):
                fig.text(L, y, line, fontsize=9.3, color="#666666", va="top")
                y -= 0.016
            y -= 0.030
        return fig, y

    fig, y = new_page(True)
    pages = 1
    for status, head, body in blocks:
        lines = textwrap.wrap(body, 100)
        # a claim is never split across pages: if it does not fit whole, it starts the next one
        need = 0.024 + 0.0158 * len(lines) + 0.018
        if y - need < BOTTOM and pages < 6:
            pdf.savefig(fig)
            plt.close(fig)
            fig, y = new_page(False)
            pages += 1
        # Past the page cap the old code simply kept drawing, so the remaining lines of a claim were rendered below
        # the bottom margin and did not appear. That is the same silent disappearance as a deleted claim or a figure
        # whose image was missing: the build succeeds and the document is short. It is now an error.
        if y - need < BOTTOM:
            raise SystemExit(f"[build_pdf] claim does not fit and the page cap ({pages} pages) is reached, so its "
                             f"text would be drawn off the page: {head!r} needs {need:.3f} of figure height, "
                             f"{y - BOTTOM:.3f} remains. Raise the cap or shorten the claim.")
        col = STATUS_COLOR.get(status, "#444444")
        fig.patches.append(plt.Rectangle((L, y - 0.003), 0.007, 0.018, transform=fig.transFigure,
                                         facecolor=col, edgecolor="none"))
        fig.text(L + 0.016, y, status, fontsize=6.6, weight="bold", color=col, va="bottom")
        fig.text(L + 0.115, y, head, fontsize=11.5, weight="bold", color="#1A1A1A", va="bottom")
        y -= 0.024
        for line in lines:
            fig.text(L + 0.115, y, line, fontsize=8.8, color="#333333", va="top")
            y -= 0.0158
        y -= 0.018
    pdf.savefig(fig)
    plt.close(fig)


def figure_page(pdf, path, title, caption, index):
    """One figure per page: the image sits at text width under the title, and the caption starts wherever it ends."""
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    fig.text(L, 0.957, f"Figure {index}", fontsize=8, weight="bold", color="#999999")
    # Titles wrap: a one-line title ran off the right edge once the captions started supplying their own headlines.
    ty = 0.934
    tlines = textwrap.wrap(title, 66)
    for line in tlines:
        fig.text(L, ty, line, fontsize=14, weight="bold", color="#1A1A1A")
        ty -= 0.024
    try:
        img = mpimg.imread(path)
    except FileNotFoundError:
        plt.close(fig)
        return False
    h, w = img.shape[0], img.shape[1]
    fw = 1 - 2 * L
    fh = fw * (h / w) * (PAGE[0] / PAGE[1])
    top = 0.917 - 0.024 * len(tlines)
    ax = fig.add_axes([L, top - fh, fw, fh])
    ax.imshow(img)
    ax.axis("off")
    y = top - fh - 0.030
    for line in textwrap.wrap(caption, 100):
        fig.text(L, y, line, fontsize=8.8, color="#333333", va="top")
        y -= 0.0158
    pdf.savefig(fig)
    plt.close(fig)
    return True


def load_hero():
    """Hero figures and captions, read from HERO_CAPTIONS.md so the document cannot drift from the figures.

    A heading names the file, which may be written as `fig_hero1_answer.{pdf,png}`. The first bold line under it is the
    caption's own one-line title; everything after that is the caption body.
    """
    import os
    import re
    path = f"{FIGDIR}/HERO_CAPTIONS.md"
    if not os.path.exists(path):
        return []
    head = re.compile(r"^#{1,3}\s*(?:Figure\s*\d+\s*[:.]?\s*)?`?([\w./-]+?)\.(?:png|pdf|\{pdf,png\}|\{png,pdf\})`?")
    out, fn, buf = [], None, []

    def flush():
        if not fn:
            return
        body = " ".join(buf).strip()
        t = re.match(r"\*\*(.+?)\*\*\s*(.*)$", body, re.S)
        title, rest = (t.group(1).strip(), t.group(2).strip()) if t else (fn, body)
        rest = re.split(r"Numbers on the figure:", rest)[0].strip()
        out.append((fn + ".png", title, rest))

    for line in open(path):
        m = head.match(line.strip())
        if m:
            flush()
            fn, buf = os.path.basename(m.group(1)), []
        elif fn and line.strip() and not line.startswith("---"):
            buf.append(line.strip())
    flush()
    # A caption whose image is absent used to be dropped here without a word, so a figure added to the manifest but
    # rendered to the wrong directory simply never appeared and the count in the log looked unremarkable. Same shape
    # as the claim that vanished from CONTRIBUTIONS: silence about a thing that is gone.
    keep, missing = [], []
    for f, t, c in out:
        (keep if os.path.exists(f"{FIGDIR}/{f}") else missing).append((f, t, c))
    if missing:
        print(f"[build_pdf] WARNING: {len(missing)} caption(s) have no image in {FIGDIR} and are being skipped:")
        for f, t, _ in missing:
            print(f"             {f}  ({t[:60]})")
    return keep


# A guard, not decoration. An index-slice edit to CONTRIBUTIONS silently deleted a MAIN claim once: the audit cannot
# notice, because a claim that is gone quotes no numbers for it to check.
EXPECTED_CLAIMS = (5, 3)   # (MAIN, SUPPORTING) -- raised deliberately 2026-09-19 for the affect claim


def check_claims():
    got = (sum(1 for c in CONTRIBUTIONS if c[0] == "MAIN"), sum(1 for c in CONTRIBUTIONS if c[0] == "SUPPORTING"))
    if got != EXPECTED_CLAIMS:
        raise SystemExit(f"[build_pdf] contribution count changed: expected {EXPECTED_CLAIMS} MAIN/SUPPORTING, "
                         f"found {got}. If this is intended, update EXPECTED_CLAIMS deliberately.")
    heads = [c[1] for c in CONTRIBUTIONS]
    if len(set(heads)) != len(heads):
        raise SystemExit("[build_pdf] duplicate claim headings")
    # Claim 1 sat in the paper ending "The argument it appears to." for an unknown number of builds: an index-slice
    # edit had cut its last sentence in half. The count guard cannot see that, because the claim is still present and
    # still quotes its numbers, so the audit passes too. A body is therefore checked for being a body.
    for kind, head, body in CONTRIBUTIONS:
        tail = body.rstrip()
        if not tail.endswith((".", "!", "?")):
            raise SystemExit(f"[build_pdf] claim body does not end in a full stop: {head!r} ends {tail[-60:]!r}")
        last = [w for w in tail.rstrip(".").split(".")[-1].split() if w]
        # A word count alone does not find it: the real corruption, "The argument it appears to.", is five words and
        # ends in a full stop, so a length test waves it through. What marks it is a SHORT final sentence left
        # hanging on a function word. A long sentence may legitimately end on one ("less than it seems to").
        DANGLING = {"to", "of", "the", "a", "an", "and", "or", "but", "in", "on", "at", "for", "with", "that",
                    "which", "is", "are", "was", "were", "by", "from", "as", "than", "into", "its", "their"}
        if len(last) < 5 or (len(last) < 8 and last[-1].lower().strip(".,;:") in DANGLING):
            raise SystemExit(f"[build_pdf] claim body ends in a fragment ({len(last)} words, last word "
                             f"{last[-1]!r}): {head!r} ends {tail[-70:]!r}")
        if len(body) < 300:
            raise SystemExit(f"[build_pdf] claim body is suspiciously short ({len(body)} chars): {head!r}")


def main():
    check_claims()
    style.apply()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with PdfPages(OUT) as pdf:
        text_page(pdf, "MAD-Interp: figures and claims",
                  f"Mechanisms of influence in multi-agent LLM debate. Generated {stamp}. "
                  "Status reflects the tightest check each claim has survived.",
                  # MAIN claims first, SUPPORTING after, each group in the order it is written above.
                  sorted(CONTRIBUTIONS, key=lambda c: 0 if c[0] == "MAIN" else 1))
        hero = load_hero()
        if not hero:
            # Never silently substitute: the built-in fallback list was retired to
            # figures/_retired_builtin_captions.py because it was unaudited and three of its captions carried
            # numbers that no longer appear anywhere. A build that cannot find its audited captions is broken.
            raise SystemExit(f"[build_pdf] no captions parsed from {FIGDIR}/HERO_CAPTIONS.md; refusing to build "
                             f"a document from unverified captions")
        figs = hero[:6]   # the paper carries at most six figures
        print(f"using {'HERO_CAPTIONS.md' if hero else 'the built-in list'}: {len(figs)} figures")
        i = 1
        for fn, title, cap in figs:
            if figure_page(pdf, f"{FIGDIR}/{fn}", title, cap, i):
                i += 1
    print(f"wrote {OUT} with {i - 1} figures")


if __name__ == "__main__":
    main()
