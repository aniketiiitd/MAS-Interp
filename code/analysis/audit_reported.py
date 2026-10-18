"""Re-check every headline number against the current tables.

Analyses get re-run when a bug is fixed, and a number quoted in a report or a caption can silently stop matching the
table it came from. This asserts the claims we have actually made, so drift surfaces as a failure rather than as a
discrepancy someone notices in review. Run it before writing anything up.

Usage: python analysis/audit_reported.py            # exits non-zero if any claim drifted
"""
from __future__ import annotations

import sys
import glob

import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import ROOT, TABLES  # noqa: E402

FAILED = []


def check(label, est, lower, upper, claimed, scale=1.0, tol=0.006):
    est, lower, upper = est * scale, lower * scale, upper * scale
    ok = abs(est - claimed) <= max(tol, abs(claimed) * 0.01)
    if not ok:
        FAILED.append(label)
    print(f"  {'OK ' if ok else '*** DRIFT'} {label:52s} now {est:+.3f} [{lower:+.3f},{upper:+.3f}]  reported {claimed:+.3f}")


def main(m="Q9", ds="lit1200"):
    print("CONTRIBUTION 1 -- the answer is chosen on reading")
    # Qwen has both halves, so contribution 1 is checked against the pooled table the claim now quotes.
    T = pd.read_csv(TABLES / f"lit_commit_{m}_full2400_curves.csv")
    for g, c in (("changed answer in debate", 65.4), ("changed answer without communication", 22.3)):
        r = T[(T.group == g) & (T.domain == "all") & (T.metric == "argmax_is_final") & (T.fraction == 0.0)].iloc[0]
        check(f"fraction 0, {g[:30]}", r.est, r.lower, r.upper, c, 100)
    # Qwen's filler checks are pooled across both halves now, matching what the claim quotes.
    # the item-paired gap previously had no table behind it; it does now
    P = pd.read_csv(TABLES / f"lit_commit_paired_{m}_full2400.csv").set_index("scope").loc["item_paired"]
    check("item-paired gap, Qwen", P.est, P.lower, P.upper, 0.409, tol=0.005)
    PL = pd.read_csv(TABLES / "lit_commit_paired_L31_full2400.csv").set_index("scope").loc["item_paired"]
    check("item-paired gap, Llama", PL.est, PL.lower, PL.upper, 0.156, tol=0.005)
    PG = pd.read_csv(TABLES / "lit_commit_paired_G4_core.csv").set_index("scope").loc["item_paired"]
    check("item-paired gap, Gemma", PG.est, PG.lower, PG.upper, 0.359, tol=0.005)
    # The sign test is the part of this claim the means hide, so the three shares are pinned, not just the gaps.
    for lab, f, c in (("Qwen", "lit_commit_paired_Q9_full2400.csv", 0.87),
                      ("Gemma", "lit_commit_paired_G4_core.csv", 0.86),
                      ("Llama", "lit_commit_paired_L31_full2400.csv", 0.64)):
        r = pd.read_csv(TABLES / f).set_index("scope").loc["item_paired"]
        share = r.items_favouring_debate / (r.items_favouring_debate + r.items_favouring_solo)
        check(f"share of paired items favouring debate, {lab}", share, share, share, c, tol=0.01)
    # claim 1 now quotes all three models at fraction 0 on the whole benchmark
    for lab, f, cd, cs in (("Llama", "lit_commit_L31_full2400_curves.csv", 52.4, 35.4),
                           ("Gemma", "lit_commit_G4_curves.csv", 48.5, 12.6)):
        T2 = pd.read_csv(TABLES / f)
        for g, c in (("changed answer in debate", cd), ("changed answer without communication", cs)):
            r = T2[(T2.group == g) & (T2.domain == "all") & (T2.metric == "argmax_is_final")
                   & (T2.fraction == 0.0)].iloc[0]
            check(f"{lab} fraction 0, {g[:28]}", r.est, r.lower, r.upper, c, 100)
    S = pd.read_csv(TABLES / f"lit_filler_{m}_full2400_summary.csv")
    for arm, c in (("filler - frac0", -2.5), ("cross_item - frac0", -12.8)):
        r = S[(S.cond == "debate") & (S.group == "changed") & (S.arm == arm)].iloc[0]
        check(arm, r.est, r.lower, r.upper, c, 100)

    print("CONTRIBUTION 2 -- status is encoded, track record is used")
    # the 0.85 legibility figure previously had no table; it is the conservative estimator read at the decision point,
    # where the label is not lexically present. At the source description every probe scores 1.000 and means nothing.
    CP = pd.read_csv(TABLES / f"lit_credential_probe_{m}.csv")
    CP = CP[CP.position == "answer"]
    r = CP.loc[CP.auroc_diff_of_means.idxmax()]
    check("credential legible at the decision point", r.auroc_diff_of_means, r.auroc_diff_of_means,
          r.auroc_diff_of_means, 0.85, tol=0.01)
    M = pd.read_csv(TABLES / f"lit_source_{m}_magnitude.csv").set_index("arm")
    for a, c in (("record swap, as written", 0.560), ("status, amplified to record magnitude", 0.153),
                 ("record:status advantage at the record magnitude", 3.668),
                 ("record:status advantage at the status magnitude", 4.360)):
        r = M.loc[a]
        check(a[:50], r.est, r.lower, r.upper, c)
    E = pd.read_csv(TABLES / f"lit_source_{m}_effects.csv").set_index("effect")
    for a, c in (("status share (behaviour)", 0.071), ("professor with poor record vs no label", -0.203)):
        r = E.loc[a]
        check(a[:50], r.est, r.lower, r.upper, c)
    # The cell the claim now leans on across all three architectures. Gemma's magnitude table waits on the corrected
    # swap run, but these come from the effects table, which is complete for every model.
    for mm, poor_cell, prof_poor in (("L31", -0.383, -0.985), ("G4", -0.097, -0.966)):
        EE = pd.read_csv(TABLES / f"lit_source_{mm}_effects.csv").set_index("effect")
        r = EE.loc["status | poor record"]
        check(f"{mm} credential given a poor record", r.est, r.lower, r.upper, poor_cell, tol=0.01)
        if r.upper >= 0:
            FAILED.append(f"{mm} credential given a poor record no longer clears zero ([{r.lower:+.3f},{r.upper:+.3f}]); "
                          f"the claim says the credential actively hurts in this cell on Llama and Gemma")
        r = EE.loc["professor with poor record vs no label"]
        check(f"{mm} professor with poor record vs no label", r.est, r.lower, r.upper, prof_poor, tol=0.01)
    # Qwen is the one model where this cell is a genuine null, and the claim says so explicitly
    rq = E.loc["status | poor record"]
    check("Q9 credential given a poor record", rq.est, rq.lower, rq.upper, 0.010, tol=0.01)
    if not (rq.lower < 0 < rq.upper):
        FAILED.append(f"Q9 credential given a poor record now clears zero ([{rq.lower:+.3f},{rq.upper:+.3f}]); the "
                      f"claim calls this interval one that covers zero")

    # These six checks guard a claim that no longer exists: the "read-out site, not the role" contribution was
    # retired, and nothing in CONTRIBUTIONS or the abstract quotes these numbers any more. They are kept because
    # the tables are still produced and drift-checking them is free, but they are not headline numbers and the
    # header said otherwise for as long as the claim has been gone.
    print("RETIRED CLAIM, checks retained: defect detection from the reader state (quoted nowhere)")
    A = pd.read_csv(TABLES / f"lit_hollow_{m}_{ds}_auroc_defect.csv")
    A = A[A.scale == "family_adjusted"].set_index("predictor")
    for p, c in (("neg_margin", 0.510), ("neg_margin_part", 0.712), ("neg_margin_observer", 0.705),
                 ("neg_margin_part_para", 0.697)):
        r = A.loc[p]
        check(p, r.auroc, r.lower, r.upper, c)
    D = pd.read_csv(TABLES / f"lit_hollow_{m}_{ds}_auroc_defect_diffs.csv")
    D = D[D.scale == "family_adjusted"].set_index("contrast")
    for p, c in (("neg_margin - neg_margin_part", -0.202), ("neg_margin - neg_margin_observer", -0.196)):
        r = D.loc[p]
        check(p[:50], r["diff"], r.lower, r.upper, c)

    print("OWNERSHIP, and the 2x2 that retired the role claim")
    O = pd.read_csv(TABLES / f"lit_frames_{m}_full2400_own.csv")
    O = O[O.domain == "all"].set_index("metric")
    for k, c in (("delta_own", 0.749), ("position_last_minus_first", -0.019)):
        r = O.loc[k]
        check(k, r.est, r.lower, r.upper, c)
    # Gemma landed the third model for this claim. Its control covers zero, Qwen's does not, and the claim now
    # rests on the RATIO being about a fortieth in all three, so that is asserted rather than left implied.
    OG = pd.read_csv(TABLES / "lit_frames_G4_core_own.csv")
    OG = OG[OG.domain == "all"].set_index("metric")
    for k, c in (("delta_own", 1.302), ("position_last_minus_first", -0.032)):
        r = OG.loc[k]
        check(f"G4 {k}", r.est, r.lower, r.upper, c, tol=0.01)
    # the claim's new point: the control is about a fortieth of the effect in EVERY model, on three architectures
    # whose labelling effects differ by nearly a factor of two. Two point estimates cannot carry that; the ratio can.
    for mm, f, c in (("Q9", "lit_frames_Q9_full2400_own.csv", 39), ("L31", "lit_frames_L31_full2400_own.csv", 45),
                     ("G4", "lit_frames_G4_core_own.csv", 41)):
        OO = pd.read_csv(TABLES / f)
        OO = OO[OO.domain == "all"].set_index("metric")
        ratio = abs(OO.loc["delta_own"].est / OO.loc["position_last_minus_first"].est)
        check(f"{mm} labelling effect / position control", ratio, ratio, ratio, c, tol=1.5)
        if ratio < 20:
            FAILED.append(f"{mm}: the position control is no longer negligible against the labelling effect "
                          f"(ratio {ratio:.0f}); the claim says it is about a fortieth in every model")
    OL = pd.read_csv(TABLES / "lit_frames_L31_full2400_own.csv")
    OL = OL[OL.domain == "all"].set_index("metric")
    for k, c in (("delta_own", 1.086), ("position_last_minus_first", 0.024)):
        r = OL.loc[k]
        check(f"L31 {k}", r.est, r.lower, r.upper, c)
    R = pd.read_csv(TABLES / f"lit_role2x2_{m}_{ds}_arms.csv")
    R = R[(R.scale == "adopt") & (R.domain == "all")].set_index("arm")
    for a, c in (("part", 75.7), ("part_para", 61.6), ("observer", 82.2), ("observer_persona", 82.7),
                 ("neutral_label", 82.2)):
        r = R.loc[a]
        check("adopt " + a, r.est, r.lower, r.upper, c, 100, tol=0.2)

    print("DEBATE AGAINST AN EQUAL-COMPUTE VOTE")
    for mm, gain, wu in (("Q9", 1.2, 3.1), ("L31", 0.5, 7.4), ("G4", 1.2, 3.6)):
        # Qwen has both halves, so its headline is checked against the pooled table; Llama is still on the half.
        # Qwen and Llama ran in halves and were pooled; Gemma ran the whole benchmark in one pass, so its
        # table carries the core name rather than full2400.
        src = "G4_core" if mm == "G4" else f"{mm}_full2400"
        V = pd.read_csv(TABLES / f"lit_debate_{src}_vote.csv")
        V = V[V.domain == "all"].set_index("metric")
        # the claim now quotes the fully-parsed scope, where both arms are measured on the same items
        if "scope" in V.reset_index().columns:
            V = pd.read_csv(TABLES / f"lit_debate_{src}_vote.csv")
            V = V[(V.scope == "fully_parsed") & (V.domain == "all")].set_index("metric")
        r = V.loc["debate - vote"]
        check(f"{mm} debate - vote, fully parsed", r.est, r.lower, r.upper, gain, 100, tol=0.2)
        # The paired row, not a difference of the two independent intervals: the latter is far too wide for a contrast
        # measured on the same items and made Qwen's gap look like it covered zero when it does not.
        r = V.loc["wrong_unan debate - reflect"]
        check(f"{mm} wrong unanimity, debate - no communication", r.est, r.lower, r.upper, wu, 100, tol=0.25)

    print("WHAT A PEER'S MESSAGE IS MADE OF (full benchmark, which is what the claim now quotes)")
    C = pd.read_csv(TABLES / "lit_carrier_additivity_Q9_full2400.csv").set_index("scope")
    r = C.loc["all"]
    for k, c in (("answer_only", 0.59), ("reasoning_only", 1.31)):
        check(f"Q9 full {k}", r[k], r[k], r[k], c, tol=0.01)
    check("Q9 full reasoning share", r.reasoning_share, r.share_lo, r.share_hi, 0.69, tol=0.01)
    check("Q9 full additivity residual, pooled", r.residual, r.res_lo, r.res_hi, -0.05, tol=0.01)
    # the sub-claim that changed when the data doubled: safety is no longer additive, so the audit pins the sign
    rs = C.loc["safety_refusal"]
    check("Q9 full additivity residual, safety", rs.residual, rs.res_lo, rs.res_hi, -0.31, tol=0.01)
    if rs.res_hi >= 0:
        FAILED.append("safety additivity flipped back to covering zero; the claim text says it does not")
    for sc, c in (("knowledge_reasoning", 0.66), ("safety_refusal", 0.72), ("values_moral_judgment", 0.70)):
        rr = C.loc[sc]
        check(f"Q9 full reasoning share, {sc}", rr.reasoning_share, rr.share_lo, rr.share_hi, c, tol=0.01)
    # The claim previously quoted Qwen's amplification ratios without naming a model, which read as though they
    # applied to both. Llama's is 14.3, nearly four times larger, and its second ratio is undefined because the
    # division is by a standing effect that covers zero. Both facts are now pinned.
    # Gemma's amplification landed and broke the claim's "barely moves anything": its standing swap is 0.793,
    # against 0.077 and 0.011. The claim now says the ordering is invariant and the magnitude is not, so the audit
    # pins the three standing swaps, the three record swaps, and Gemma's two amplified advantages.
    for mm, st, rec in (("Q9", 0.077, 0.560), ("L31", 0.011, 0.876), ("G4", 0.793, 1.905)):
        MM = pd.read_csv(TABLES / f"lit_source_{mm}_magnitude.csv").set_index("arm")
        r1, r2 = MM.loc["status swap, as written"], MM.loc["record swap, as written"]
        check(f"{mm} standing swap, as written", r1.est, r1.lower, r1.upper, st, tol=0.005)
        check(f"{mm} record swap, as written", r2.est, r2.lower, r2.upper, rec, tol=0.005)
        # the invariant the claim now rests on: the record leads the standing in every model
        if not r2.est > r1.est:
            FAILED.append(f"{mm}: the record no longer leads the standing as written ({r2.est:.3f} vs {r1.est:.3f}); "
                          f"the claim says the ordering holds in every model")
    MG = pd.read_csv(TABLES / "lit_source_G4_magnitude.csv").set_index("arm")
    for a, c in (("record:status advantage at the record magnitude", 1.436),
                 ("record:status advantage at the status magnitude", 1.394)):
        r = MG.loc[a]
        check(f"G4 {a[14:44]}", r.est, r.lower, r.upper, c, tol=0.01)
        if r.lower <= 1.0:
            FAILED.append(f"G4 {a}: interval now includes parity ({r.lower:.2f}); the claim says the record still "
                          f"leads on Gemma, where the contest is closest")
    ML = pd.read_csv(TABLES / "lit_source_L31_magnitude.csv").set_index("arm")
    rml = ML.loc["record:status advantage at the record magnitude"]
    check("L31 record:status advantage at the record magnitude", rml.est, rml.lower, rml.upper, 14.3, tol=0.1)
    if not [a for a in ML.index if "status magnitude" in a and "undefined" in a]:
        FAILED.append("L31's advantage at the status magnitude is no longer marked undefined; the claim says that "
                      "division is undefined because the standing effect covers zero")
    else:
        print("  OK  L31 advantage at the status magnitude is still suppressed as undefined")

    L = pd.read_csv(TABLES / "lit_carrier_additivity_L31_full2400.csv").set_index("scope")
    rl = L.loc["all"]
    check("L31 additivity residual, every item", rl.residual, rl.res_lo, rl.res_hi, -0.69, tol=0.01)
    # Gemma ran the whole benchmark in one pass, so there is no _full2400 suffix; its table is the plain name
    G = pd.read_csv(TABLES / "lit_carrier_additivity_G4.csv").set_index("scope")
    rg = G.loc["all"]
    check("G4 full effect", rg.full, rg.full, rg.full, 5.23, tol=0.01)
    check("G4 reasoning share", rg.reasoning_share, rg.share_lo, rg.share_hi, 0.60, tol=0.01)
    check("G4 additivity residual, pooled", rg.residual, rg.res_lo, rg.res_hi, 0.43, tol=0.01)
    # The claim rests on the two SIGNS being opposite, not on either magnitude. If one interval ever crossed zero the
    # sentence "the same test failing in opposite directions" would be false while both point estimates still rounded
    # to what is quoted, so the directions are pinned separately from the values.
    if not (rg.res_lo > 0 and rl.res_hi < 0):
        FAILED.append(f"additivity no longer fails in opposite directions: Gemma [{rg.res_lo:+.2f},{rg.res_hi:+.2f}] "
                      f"Llama [{rl.res_lo:+.2f},{rl.res_hi:+.2f}]; the claim has Gemma overshooting, Llama short")
    # "almost three times as movable" is a word rather than a table value, so it is checked as the ratio it stands for
    ratio = rg.full / C.loc["all"].full
    check("G4 / Q9 total effect ratio", ratio, ratio, ratio, 2.7, tol=0.15)
    # the values-question agreement across all three architectures, the tightest cross-model match in the paper
    gv, lv = G.loc["values_moral_judgment"], L.loc["values_moral_judgment"]
    check("G4 reasoning share, values", gv.reasoning_share, gv.share_lo, gv.share_hi, 0.68, tol=0.01)
    check("L31 reasoning share, values", lv.reasoning_share, lv.share_lo, lv.share_hi, 0.66, tol=0.01)

    print("WHERE THE INFLUENCE TRAVELS")
    ARM_ALL = "peer keys blocked for every later query at every attention layer"
    ARM_PRE = "control: peer keys blocked for the prefill tokens, not the decision token"
    # Gemma is the registered falsification arm and it confirmed. The claim now says the three intervals do not
    # overlap, which is the statement that carries it, so that is asserted rather than left to three point values.
    # The prefill control is not 9% on every model: Gemma's is 0.6 and the claim now says so. Carrying one
    # constant for three models is how a per-model number becomes an unattributed one, which this claim has
    # already been caught doing once with the amplification ratios.
    bands = []
    for mm, c, pre in (("Q9", 79.7, 9.0), ("L31", 131.7, 9.0), ("G4", 117.1, 0.6)):
        # Gemma ran the whole benchmark in one pass, so its table carries the core name, not full2400
        src = "G4_core" if mm == "G4" else f"{mm}_full2400"
        S = pd.read_csv(TABLES / f"lit_routing_{src}_summary.csv").set_index("arm")
        r = S.loc[ARM_ALL]
        check(f"{mm} complete attention block", r.share, r.share_lo, r.share_hi, c, 100, tol=0.6)
        bands.append((mm, 100 * r.share_lo, 100 * r.share_hi))
        p = S.loc[ARM_PRE]
        check(f"{mm} peer blocked except the deciding token", p.share, p.share_lo, p.share_hi, pre, 100, tol=0.6)
    # the claim's load-bearing sentence: the hybrid's interval does not overlap either all-attention model's
    q = [b for b in bands if b[0] == "Q9"][0]
    for mm, lo, hi in bands:
        if mm == "Q9":
            continue
        if lo <= q[2] and q[1] <= hi:
            FAILED.append(f"the hybrid's interval now overlaps {mm}'s ([{q[1]:.1f},{q[2]:.1f}] vs [{lo:.1f},{hi:.1f}]); "
                          f"the claim says the three do not overlap")
    if not any("overlaps" in f for f in FAILED):
        print(f"  OK  the hybrid's interval overlaps neither all-attention model "
              f"(Qwen [{q[1]:.0f},{q[2]:.0f}] vs Llama and Gemma both above 111)")

    print("RESIDUAL PATCHING, the attention claim's independent corroboration")
    # The abstract has asserted this since it was written; no claim stated it, no caption mentioned it and nothing
    # checked it. It uses no attention mask, so it is the one measurement that can corroborate the masking result
    # rather than restate it, which makes it the last thing that should have gone unverified.
    SP = pd.read_csv(TABLES / "lit_routing_span.csv")
    SP = SP[SP.kind == "residual only"]
    for mm, c in (("Q9", 0.244), ("L31", 0.029)):
        r = SP[(SP.model == mm) & (SP.layer == 12)].iloc[0]
        check(f"{mm} residual recovery at layer 12", r.recovery, r.lower, r.upper, c, tol=0.005)
    # the claim's comparative sentence: the hybrid holds influence four layers deeper
    depth = {}
    for mm in ("Q9", "L31"):
        g = SP[SP.model == mm].sort_values("layer")
        below = [int(l) for l, v in zip(g.layer, g.recovery) if v < 0.10]
        depth[mm] = min(below) if below else None
    if not (depth["Q9"] is not None and depth["L31"] is not None and depth["Q9"] - depth["L31"] == 4):
        FAILED.append(f"residual recovery no longer falls below a tenth four layers later in the hybrid: "
                      f"Qwen {depth['Q9']}, Llama {depth['L31']}")
    else:
        print(f"  OK  recovery falls below a tenth at layer {depth['L31']} (Llama) and {depth['Q9']} (Qwen), "
              f"four deeper")

    # The claim describes Gemma's architecture precisely because "all attention" would hide that 35 of its 42
    # layers are sliding-window over 512 tokens. A reviewer checks this first; the audit should too.
    import json as _j
    gm = _j.load(open(ROOT / "outputs/activations/G4/routing_attn_core/meta.json"))
    lt = gm.get("layer_types", {})
    n_full = sum(1 for v in lt.values() if v == "full_attention")
    n_slide = sum(1 for v in lt.values() if v == "sliding_attention")
    for label, got, want in (("G4 global attention layers", n_full, 7), ("G4 sliding attention layers", n_slide, 35),
                             ("G4 layers taking a mask", len(gm.get("attn_layers", [])), 42),
                             ("G4 sliding window", gm.get("window"), 512)):
        check(label, got, got, got, want, tol=0.5)

    print("PER-LAYER EMBEDDINGS: the one channel a mask cannot reach in Gemma")
    # The claim's strongest objection is that Gemma has a component attention masking does not touch. Patching it
    # alone answers that empirically instead of by assertion, so the bound is asserted rather than the point value.
    SPN = pd.read_csv(TABLES / "lit_routing_span.csv")
    ple = SPN[(SPN.model == "G4") & (SPN.kind == "per-layer embeddings only")]
    mx = float(ple.recovery.max())
    check("G4 per-layer embeddings, max recovery at any layer", mx, mx, mx, 0.018, tol=0.005)
    if mx > 0.10:
        FAILED.append(f"per-layer embeddings now recover {mx:.3f} of the peer's effect; the claim says they are "
                      f"not a channel between token positions")
    both = SPN[(SPN.model == "G4") & (SPN.kind == "residual + per-layer embeddings")].set_index("layer").recovery
    res = SPN[(SPN.model == "G4") & (SPN.kind == "residual only")].set_index("layer").recovery
    gap = float((both - res).abs().max())
    check("G4 PLE adds at most this to a residual patch", gap, gap, gap, 0.011, tol=0.005)

    print("HOW MANY INDEPENDENT AGENTS SURVIVE A DEBATE")
    for mm, deb, solo in (("Q9", 1.07, 1.31), ("L31", 1.07, 1.62), ("G4", 1.07, 1.27)):
        src = "G4_core" if mm == "G4" else f"{mm}_full2400"
        I = pd.read_csv(TABLES / f"lit_independence_{src}.csv")
        # the table gained a scope column when the split-at-start subset was added; these headline numbers are "all"
        if "scope" in I.columns:
            I = I[I.scope == "all"]
        I = I[I["round"] == 3].set_index("cond")
        for cond, c in (("debate", deb), ("reflect", solo)):
            r = I.loc[cond]
            # n_eff_lo/hi come from a bootstrap of a decreasing function of rho, so the interval arrives inverted
            lo, hi = sorted((r.n_eff_lo, r.n_eff_hi))
            check(f"{mm} {cond}: independent agents after three turns", r.n_eff, lo, hi, c, tol=0.02)
        if mm == "Q9":
            J = pd.read_csv(TABLES / "lit_independence_Q9_full2400.csv")
            J = J[(J.scope == "split_at_start") & (J["round"] == 3)].set_index("cond")
            for cond, c in (("debate", 1.18), ("reflect", 2.34)):
                rr = J.loc[cond]
                lo2, hi2 = sorted((rr.n_eff_lo, rr.n_eff_hi))
                check(f"Q9 {cond}: independent agents, questions that start split", rr.n_eff, lo2, hi2, c, tol=0.03)

    print("WHERE THE AGREED ANSWER COMES FROM")
    CS = {mm: pd.read_csv(TABLES / f"lit_convergence_source_{src}.csv")
          for mm, src in (("Q9", "Q9_full2400"), ("L31", "L31_full2400"), ("G4", "G4_core"))}
    for mm, rat, ratio in (("Q9", 0.76, 2.7), ("L31", 0.73, 2.3), ("G4", 0.83, 2.1)):
        T5 = CS[mm]
        d = T5[(T5.arm == "debate") & (T5.ending == "all")].iloc[0]
        r = T5[(T5.arm == "reflect") & (T5.ending == "all")].iloc[0]
        check(f"{mm} debate ratifies the initial plurality", d.ratified, d.lower, d.upper, rat, tol=0.01)
        got = d.n_items / r.n_items
        check(f"{mm} unanimity, debate vs alone", got, got, got, ratio, tol=0.05)
    # the claim's point is that this share does NOT depend on the ending, which two point estimates alone
    # would not establish; the intervals must overlap in the two models the claim says agree
    for mm in ("Q9", "G4"):
        T5 = CS[mm]
        w = T5[(T5.arm == "debate") & (T5.ending == "wrong")].iloc[0]
        g = T5[(T5.arm == "debate") & (T5.ending == "right")].iloc[0]
        if not (w.lower <= g.upper and g.lower <= w.upper):
            FAILED.append(f"{mm}: ratification now differs between wrong and right endings "
                          f"([{w.lower:.2f},{w.upper:.2f}] vs [{g.lower:.2f},{g.upper:.2f}]); the claim says it does not")
        else:
            print(f"  OK  {mm} ratification is the same for wrong and right endings "
                  f"({100*w.ratified:.0f}% vs {100*g.ratified:.0f}%)")

    print("TWELVE TURNS, CONTESTED QUESTIONS AND ALL QUESTIONS")
    try:
        U = pd.read_csv(TABLES / "lit_turns_Q9_release_600_t12.csv")
        U = U[U["round"] == 12]
        for sc, cond, met, c in (("contested", "debate", "unanimity", 98.9), ("contested", "debate", "wrong_unanimity", 40.0),
                                 ("contested", "reflect", "wrong_unanimity", 20.9), ("all", "debate", "unanimity", 99.3),
                                 ("all", "debate", "wrong_unanimity", 13.7), ("all", "reflect", "wrong_unanimity", 8.8)):
            r = U[(U.scope == sc) & (U.cond == cond) & (U.metric == met)].iloc[0]
            check(f"turn 12 {sc} {cond} {met}", r.est, r.lower, r.upper, c, 100, tol=0.3)
    except (FileNotFoundError, IndexError, KeyError):
        print("  (twelve-turn table not in the expected shape)")

    print("WHEN A GROUP STOPS CHANGING ITS ANSWERS")
    try:
        F = pd.read_csv(TABLES / "lit_freeze_Q9_release_600_t12.csv")
        for cond, t, c in (("debate", 1, 89.7), ("debate", 3, 95.5), ("reflect", 1, 79.5), ("reflect", 3, 84.0)):
            r = F[(F.cond == cond) & (F.turn == t)].iloc[0]
            check(f"{cond} frozen by turn {t}", r.frozen, r.lower, r.upper, c, 100, tol=0.3)
        for cond, c in (("debate", 0.44), ("reflect", 1.54)):
            r = F[(F.cond == cond) & (F.turn == -1)].iloc[0]
            check(f"{cond}: mean turn of the last change", r.frozen, r.lower, r.upper, c, tol=0.02)
    except FileNotFoundError:
        print("  (twelve-turn freeze table not built yet)")

    print("THE PROBES AS INSTRUMENTS (quoted in the methodological claim)")
    # Claim 7 asserts the probes replicate "across two architectures". They do, and Gemma is the third and does
    # not: 0.734 against 0.821 and 0.825 on emotion. Pinned so the claim cannot quietly become a three-model
    # statement it does not support.
    for tag, vals in (("emotion", {"Q9": 0.821, "L31": 0.825, "G4": 0.734}),
                      ("moral", {"Q9": 0.955, "L31": 0.953, "G4": 0.927})):
        for mm, c in vals.items():
            V3 = pd.read_csv(TABLES / f"{tag}_stories_{mm}_validation.csv")
            L3 = int(V3.groupby("layer").auroc_oof_topic.mean().idxmax())
            e3 = float(V3[V3.layer == L3].auroc_oof_topic.mean())
            check(f"{tag} probe out of fold, {mm}", e3, e3, e3, c, tol=0.01)
    # And the permuted-label null the claim's argument rests on, from the one table that computes it per concept.
    for mm, beats in (("Q9", 0.978), ("L31", 0.982), ("G4", 0.639)):
        NL = pd.read_csv(TABLES / f"concept_null_emotion_stories_{mm}.csv")
        b = float(NL.beats_own_null.mean())
        check(f"concepts beating their own permuted null, {mm}", b, b, b, beats, tol=0.01)
        nm = float(NL.null_median.median())
        check(f"permuted null median, {mm}", nm, nm, nm, 0.500, tol=0.01)

    for tag, c in (("emotion", 0.82), ("moral", 0.955)):
        V = pd.read_csv(TABLES / f"{tag}_stories_{m}_validation.csv")
        L = int(V.groupby("layer").auroc_oof_topic.mean().idxmax())
        g = V[V.layer == L]
        e = float(g.auroc_oof_topic.mean())
        check(f"{tag} probe validates out of fold", e, e, e, c, tol=0.01)

    print("AFFECT DEPENDS ON THE KIND OF QUESTION (two models, on the 104 concepts both validate)")
    # The claim was rewritten at 23:55 after Gemma landed: the dependence replicates, the dynamics do not. The
    # audit now pins BOTH models on the shared concept set, and pins the two places they disagree, because those
    # -- 2026-09-20: this comment said "BOTH models on the shared concept set" while the code below passed tag
    # "T12" for Gemma, which is its own 105-concept run. The comment described the intent and the code did
    # something else for a whole day, and the figure built on it compared 104 Qwen concepts against 105 Gemma
    # ones. A comment is not a check. Every tag here is now T12shared and the provenance check enforces it.
    # disagreements are the reason the claim no longer states a general shape.
    try:
        for mm, tg, dec_rep, read_rep in (("Q9", "T12shared3", 84, 99), ("L31", "T12shared3", 94, 88),
                                          ("G4", "T12shared3", 98, 98)):
            for site, want in (("dec", dec_rep), ("read", read_rep)):
                SS = pd.read_csv(TABLES / f"traj_domain_{mm}_{tg}_{site}_spread.csv")
                sel = set(SS[(SS.half == "discovery") & (SS.q < 0.05)].concept)
                RR = SS[(SS.half == "confirmation") & SS.concept.isin(sel) & (SS.q < 0.05)]
                n = RR.concept.nunique()
                check(f"{mm} {site}: concepts replicating, shared set", n, n, n, want, tol=0.5)
        # the two disagreements the claim now reports rather than hides
        for mm, tg, site, t1, t11 in (("Q9", "T12shared3", "read", 0.46, 0.37),
                                      ("L31", "T12shared3", "read", 0.21, 0.22),
                                      ("G4", "T12shared3", "read", 0.30, 0.56),
                                      ("Q9", "T12shared3", "dec", 0.45, 0.57),
                                      ("L31", "T12shared3", "dec", 0.60, 0.60),
                                      ("G4", "T12shared3", "dec", 0.91, 0.89)):
            SS = pd.read_csv(TABLES / f"traj_domain_{mm}_{tg}_{site}_spread.csv")
            sel = set(SS[(SS.half == "discovery") & (SS.q < 0.05)].concept)
            RR = SS[(SS.half == "confirmation") & SS.concept.isin(sel) & (SS.q < 0.05)]
            t = RR.groupby("turn").spread.mean()
            check(f"{mm} {site} spread, first turn", t.iloc[0], t.iloc[0], t.iloc[0], t1, tol=0.02)
            check(f"{mm} {site} spread, last turn", t.iloc[-1], t.iloc[-1], t.iloc[-1], t11, tol=0.02)
        # they must still point OPPOSITE ways, which is the claim's point about the shape not generalising
        def trend(mm, tg, site):
            SS = pd.read_csv(TABLES / f"traj_domain_{mm}_{tg}_{site}_spread.csv")
            sel = set(SS[(SS.half == "discovery") & (SS.q < 0.05)].concept)
            RR = SS[(SS.half == "confirmation") & SS.concept.isin(sel) & (SS.q < 0.05)]
            t = RR.groupby("turn").spread.mean()
            return t.iloc[-1] - t.iloc[0]
        # The claim's whole point with three models: no two agree on the direction at either site. Asserting
        # "Qwen converges and Gemma diverges" was the two-model version and would now pass while hiding Llama.
        for _site in ("read", "dec"):
            _tr = {mm: trend(mm, "T12shared3", _site) for mm in ("Q9", "L31", "G4")}
            _flat = {mm: abs(v) < 0.05 for mm, v in _tr.items()}
            _sign = {mm: (v > 0) for mm, v in _tr.items()}
            agree = [(a, b) for a in _tr for b in _tr if a < b
                     and not _flat[a] and not _flat[b] and _sign[a] == _sign[b]]
            if agree:
                FAILED.append(f"at the {_site} site {agree} now move the same way; claim 8 says no two of the "
                              f"three models agree on the direction at either site")
            else:
                print(f"  OK  {_site} site: no two models share a direction "
                      + ", ".join(f"{mm} {v:+.3f}" for mm, v in _tr.items()))
    except FileNotFoundError as e:
        print(f"  (shared-set affect tables not built yet: {e})")


    # The affect claim was rewritten at 23:55 when Gemma landed. These checks verified the previous
    # version: 133 of 158 concepts, the 47% sign-reversal rate, and the pressure_from_numbers and trusting
    # examples. None of those is in the claim any more. pressure_from_numbers is not even in the shared
    # concept set, because Gemma cannot validate it, which is part of why the example was dropped.
    # Removed rather than relabelled: unlike the retired role claim, these are superseded by the
    # shared-set checks directly above, so keeping them would double-count the same result at two scopes.

    print("GAPS FOUND BY THE REVERSE SWEEP (each number in a claim, does a check cover it)")
    # Walking every numeric token in every claim and asking what verifies it. The audit grows around whatever was
    # being checked at the time, so only reading the claims finds what was never added. Five genuine gaps this pass.
    S3 = pd.read_csv(TABLES / "lit_routing_Q9_full2400_summary.csv")
    own = S3[S3.arm.str.startswith("control: own-response")].iloc[0]
    check("control: blocking the agent's own response", own.share, own.share_lo, own.share_hi, -0.144, tol=0.01)
    if own.share_hi >= 0:
        FAILED.append("the own-response control no longer leaves the agent MORE persuaded; claim 3 says it does")
    for mm, st, rec in (("Q9", 0.077, 0.560), ("L31", 0.011, 0.876)):
        M3 = pd.read_csv(TABLES / f"lit_source_{mm}_magnitude.csv").set_index("arm")
        check(f"{mm} status swap, as written", M3.loc["status swap, as written"].est,
              M3.loc["status swap, as written"].lower, M3.loc["status swap, as written"].upper, st, tol=0.005)
        check(f"{mm} record swap, as written", M3.loc["record swap, as written"].est,
              M3.loc["record swap, as written"].lower, M3.loc["record swap, as written"].upper, rec, tol=0.005)
    # the questions that begin split, which are the only ones a vote could help; claim 5 quotes all three models
    for mm, f, cr, cd in (("Q9", "lit_independence_Q9_full2400.csv", 2.34, 1.18),
                          ("G4", "lit_independence_G4_core.csv", 2.25, 1.33),
                          ("L31", "lit_independence_L31_full2400.csv", 1.80, 1.15)):
        I3 = pd.read_csv(TABLES / f)
        s3 = I3[(I3.scope == "split_at_start") & (I3["round"] == 3)]
        for cond, c in (("reflect", cr), ("debate", cd)):
            r3 = s3[s3.cond == cond].iloc[0]
            check(f"{mm} {cond}, questions that start split", r3.n_eff, r3.n_eff_lo, r3.n_eff_hi, c, tol=0.02)
    # Llama's freeze comparison at three turns, quoted in claim 5 and previously unchecked
    F3 = pd.read_csv(TABLES / "lit_freeze_L31_lit1200.csv")
    for cond, c in (("debate", 76.5), ("reflect", 38.3)):
        r3 = F3[(F3.cond == cond) & (F3.turn == 1)].iloc[0]
        check(f"L31 {cond} frozen by turn 1", r3.frozen, r3.lower, r3.upper, c, 100, tol=0.2)
    # the Qwen carrier total, quoted in claim 4 beside Gemma's 5.23 and not previously checked on its own
    Cq = pd.read_csv(TABLES / "lit_carrier_additivity_Q9_full2400.csv").set_index("scope").loc["all"]
    check("Q9 total carrier effect", Cq.full, Cq.full, Cq.full, 1.95, tol=0.01)

    print("FIGURE 5's TWELVE-TURN INDEPENDENCE CURVE (its caption quoted a plotting coordinate as a result)")
    # The caption said the debating group "falls to 1.06 by turn 3". 1.06 is the y position of the "one agent"
    # label in fig_turns.py, not a measurement; the value at turn 3 is 1.28. It reaches 1.06 at turn 6 by
    # coincidence, which is the kind of near-miss that makes a wrong number survive a read-through.
    I5 = pd.read_csv(TABLES / f"lit_independence_{m}_release_600_t12.csv")
    I5 = I5[I5.scope == "split_at_start"]
    for cond, rd, c in (("debate", 1, 1.87), ("debate", 3, 1.28), ("debate", 9, 1.02),
                        ("reflect", 1, 3.61), ("reflect", 12, 2.10)):
        r5 = I5[(I5.cond == cond) & (I5["round"] == rd)].iloc[0]
        check(f"twelve turns, {cond} turn {rd}", r5.n_eff, r5.n_eff_lo, r5.n_eff_hi, c, tol=0.02)

    print("CONCEPT COUNTS (two claims quote different totals; both are right, so both are pinned)")
    # 171 + 22 are the two probe story corpora; 201 is the atlas projected onto debate states, of which 158 validate.
    # The earlier audit sweep classed these as "facts" needing no table. They come from files and can drift like
    # anything else, and a reader meeting 193 in one claim and 201 in the next needs both to be right.
    import json as _json
    for tag, c in (("emotion", 171), ("moral", 22)):
        V = pd.read_csv(TABLES / f"{tag}_stories_{m}_validation.csv")
        n = V.concept.nunique()
        check(f"{tag} probe corpus concepts", n, n, n, c, tol=0.5)
    tax = _json.load(open(ROOT / "data/concepts/taxonomy.json"))["taxonomy"]
    n_atlas = sum(len(v) for v in tax.values())
    check("atlas concepts", n_atlas, n_atlas, n_atlas, 201, tol=0.5)
    CV = pd.read_csv(TABLES / f"concept_validation_{m}.csv")
    n_ok = int(CV.passes.sum())
    check("atlas concepts that validate", n_ok, n_ok, n_ok, 158, tol=0.5)

    print("THE ABSTRACT (prose, but it still quotes results)")
    # The abstract said "the accuracy gain over an equal compute vote stays around three points" for a day after
    # that number was retracted. Nothing caught it: this audit walks CONTRIBUTIONS in build_pdf.py, and the
    # abstract is a separate file. A number does not stop being a claim because it is spelled out in words.
    try:
        ab = (ROOT / "results/ABSTRACT.md").read_text()
        flat = " ".join(ab.split())
        # phrases retracted during the campaign; each must be absent, with the reason it went
        for phrase, why in (
            ("around three points", "the vote gain fell to ~1.2 pts when unparsed answers stopped being scored wrong"),
            ("stays around three", "same retraction, in case the line wrapping changed"),
            ("using two models", "there are three models now"),
            ("almost ignored", "Gemma's standing swap is 0.793 logits; the credential is not almost ignored there"),
            ("two models whose architectures", "there are three models now"),
            # The longer version of the abstract carried the RETRACTED affect finding for days. The denylist
            # already read the whole file, including that section; it simply had no entry for this. A denylist
            # only guards the phrases someone thought to add, so every retraction has to be written into it at
            # the moment of retraction, not at the moment it is noticed somewhere else.
            ("outnumbered", "pressure_from_numbers is the concept Gemma cannot validate; that finding is retired"),
            ("often reversing sign between knowledge and values",
             "the sign-reversal story was two-model; the third model removed it"),
            ("two thirds of the answer changes", "that ratio is Qwen's; Llama is 52 against 35"),
            ("removes the effect entirely", "the cross-item control collapses the gap to 4.6 points, not to zero"),
            ("never moves again", "Llama still has agents changing their answer at the last turn"),
        ):
            if phrase in flat:
                FAILED.append(f"ABSTRACT still contains a retracted phrase: '{phrase}' ({why})")
            else:
                print(f"  OK  retracted phrase absent: '{phrase}'")
        # the spelled-out quantities it does assert, against the tables that back them
        cv = pd.read_csv(TABLES / "lit_debate_Q9_full2400_vote.csv")
        cv = cv[(cv.scope == "fully_parsed") & (cv.domain == "all")].set_index("metric")
        # the table stores a proportion; every other check in this file scales it by 100 to reach points, and
        # omitting that here made the check fire against a correct abstract on its very first run
        gain = float(cv.loc["debate - vote"].est) * 100
        if "about one point" in flat and not (0.5 <= gain <= 1.8):
            FAILED.append(f"ABSTRACT says the vote gain is 'about one point'; the table gives {gain:.2f}")
        else:
            print(f"  OK  'about one point' matches the table ({gain:.2f})")
        C = pd.read_csv(TABLES / "lit_carrier_additivity_Q9_full2400.csv").set_index("scope")
        G = pd.read_csv(TABLES / "lit_carrier_additivity_G4.csv").set_index("scope")
        q_sh, g_sh = float(C.loc["all"].reasoning_share), float(G.loc["all"].reasoning_share)
        if "seventy percent in one model and sixty in another" in flat and not (
                0.665 <= q_sh <= 0.715 and 0.575 <= g_sh <= 0.625):
            FAILED.append(f"ABSTRACT says 'seventy percent in one model and sixty in another'; "
                          f"tables give {q_sh:.2f} and {g_sh:.2f}")
        else:
            print(f"  OK  'seventy ... and sixty' matches the tables ({q_sh:.2f}, {g_sh:.2f})")
        # the abstract's own stated constraints, which are as easy to break as a number
        if "—" in ab:
            FAILED.append("ABSTRACT contains an em dash, which its own constraints forbid")
        else:
            print("  OK  no em dashes")
    except FileNotFoundError:
        print("  (ABSTRACT.md not found)")

    print("THE SHIPPED PDF IS NEWER THAN EVERYTHING IT IS BUILT FROM")
    # The audit checked figures against their scripts and RESULTS.md against its sources, and never checked the
    # one artifact that actually gets sent. build_pdf assembles the document from the captions file and the PNGs;
    # editing either afterwards leaves a PDF that disagrees with both, silently. Found when a negative test
    # rewrote HERO_CAPTIONS.md 42 seconds after the last build. That instance was harmless, the content being
    # identical, but a real caption edit after a build would have shipped the old text.
    import os as _os3
    import re as _re3
    _pdf = ROOT / "outputs/figures/MAD_Interp_figures.pdf"
    # only the figures the captions actually reference. Globbing every PNG in the directory pulled in nine the
    # document does not ship, so a stale unused file would have failed this check -- and a false alarm is how a
    # check earns the right to be ignored.
    _shipped = [ROOT / "outputs/figures/paper" / f"{_st}.png"
                for _st, _ in _re3.findall(
                    r"^## Figure \d+: `([\w./-]+?)\.\{pdf,png\}` \(script `([^`]+)`\)",
                    (ROOT / "outputs/figures/paper/HERO_CAPTIONS.md").read_text(), _re3.M)]
    _srcs = [ROOT / "outputs/figures/paper/HERO_CAPTIONS.md",
             ROOT / "code/figures/build_pdf.py"] + _shipped
    if not _os3.path.exists(_pdf):
        FAILED.append("the figures PDF does not exist; run figures/build_pdf.py")
    else:
        _old = [_os3.path.basename(f) for f in _srcs
                if _os3.path.exists(f) and _os3.path.getmtime(_pdf) < _os3.path.getmtime(f)]
        if _old:
            FAILED.append(f"the shipped PDF is older than {_old}; rerun figures/build_pdf.py")
        else:
            print(f"  OK  the PDF is newer than the captions, the builder and the "
                  f"{len(_shipped)} figures it ships")

    print("EVERY FIGURE IS NEWER THAN THE SCRIPT THAT DRAWS IT")
    # build_pdf assembles the document from PNGs on disk; it does not re-run the figure scripts. So editing a
    # figure and rebuilding the PDF without rerunning that script publishes the OLD picture beside the NEW
    # caption, silently. Found by checking: fig_hero1_answer.png predated its script by 84 seconds. Same class
    # as the RESULTS.md staleness check and the figure-6 provenance record: content can be right and still be
    # out of date, and only a timestamp notices.
    import os as _os
    import re as _re2
    _cap = (ROOT / "outputs/figures/paper/HERO_CAPTIONS.md").read_text()
    _pairs = _re2.findall(r"^## Figure \d+: `([\w./-]+?)\.\{pdf,png\}` \(script `([^`]+)`\)", _cap, _re2.M)
    _fig_bad = 0
    if len(_pairs) != 6:
        FAILED.append(f"expected 6 figures with parseable script names in the captions, found {len(_pairs)}")
        _fig_bad += 1
    for _stem, _script in _pairs:
        _png = ROOT / "outputs/figures/paper" / f"{_stem}.png"
        _src = ROOT / "code" / _script
        if not _os.path.exists(_png):
            FAILED.append(f"{_stem}.png does not exist; the caption references it")
            _fig_bad += 1
        elif not _os.path.exists(_src):
            FAILED.append(f"{_script} does not exist; the caption names it as the source of {_stem}")
            _fig_bad += 1
        elif _os.path.getmtime(_png) < _os.path.getmtime(_src):
            FAILED.append(f"{_stem}.png is older than {_script}; rerun it before building the document")
            _fig_bad += 1
    # a local count, not `if not FAILED`: the first version stayed silent whenever ANY earlier check had failed,
    # so a passing figure check looked like a missing one exactly when the audit was already in trouble
    if not _fig_bad:
        print(f"  OK  all {len(_pairs)} figures are newer than their scripts")

    print("EVERY MAIN CLAIM IS REPRESENTED IN THE ABSTRACT")
    # Claim 2, a MAIN contribution, was absent from the abstract entirely: none of credential, record,
    # professor, authority, reliability, standing or track record appeared in it. The abstract is the most-read
    # part of the paper and nothing checked that it covered the claims, so a contribution could be dropped from
    # it by an edit and stay dropped. The map is explicit rather than semantic: one distinctive word per claim.
    MAIN_KEYWORDS = {
        1: ("before", "settle"),          # the answer is fixed before the agent writes its reasoning
        2: ("credential", "record"),      # the credential is a stand-in for a guess about reliability
        3: ("attention",),                # influence does not arrive through attention to the message
        4: ("reasoning",),                # the reasoning carries most of the influence
        5: ("independent", "voting"),     # debate collapses the group
    }
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "code/figures"))
        from build_pdf import CONTRIBUTIONS  # the claims are defined there; this audit checks them
        _t = (ROOT / "results/ABSTRACT.md").read_text()
        _abs = " ".join(_t[_t.index("## Abstract ("):_t.index("## Longer version")].split()).lower()
        for n, kws in MAIN_KEYWORDS.items():
            kind, title, _body = CONTRIBUTIONS[n - 1]
            if kind != "MAIN":
                FAILED.append(f"claim {n} is no longer MAIN but the abstract keyword map still lists it")
            elif not any(k in _abs for k in kws):
                FAILED.append(f"the abstract does not mention MAIN claim {n} ({title[:44]}): none of {kws}")
            else:
                print(f"  OK  abstract covers MAIN claim {n}: {title[:52]}")
        n_main = sum(1 for k, _, _ in CONTRIBUTIONS if k == "MAIN")
        if n_main != len(MAIN_KEYWORDS):
            FAILED.append(f"{n_main} MAIN claims but the abstract keyword map covers {len(MAIN_KEYWORDS)}")
    except (FileNotFoundError, ValueError) as e:
        FAILED.append(f"could not read the abstract to check claim coverage: {e}")

    print("THE SHARED CONCEPT LIST MATCHES THE VALIDATION TABLES")
    # Claim 8's "104 concepts both models validate" lived in a hand-built file with nothing regenerating it.
    # If either model's validation changes, the list silently stops being the intersection and every affected
    # table is then measuring a set that no longer exists. `analysis/shared_concepts.py` writes it; this checks
    # the file on disk still IS the intersection.
    import os as _os
    scf = ROOT / "results/shared_concepts_Q9_G4.txt"
    if not _os.path.exists(scf):
        FAILED.append("results/shared_concepts_Q9_G4.txt is missing; run analysis/shared_concepts.py")
    else:
        on_disk = {ln.strip() for ln in open(scf) if ln.strip()}
        sets = {}
        for mm in ("Q9", "G4"):
            VV = pd.read_csv(TABLES / f"concept_validation_{mm}.csv")
            sets[mm] = set(VV[VV.passes].concept)
        inter = sets["Q9"] & sets["G4"]
        if on_disk != inter:
            FAILED.append(f"the shared concept list is no longer the intersection: {len(on_disk)} on disk, "
                          f"{len(inter)} in the tables ({len(on_disk - inter)} stale, {len(inter - on_disk)} missing)")
        else:
            print(f"  OK  shared concept list is exactly the intersection ({len(inter)} of "
                  f"{len(sets['Q9'])} and {len(sets['G4'])})")
        check("concepts Qwen validates", len(sets["Q9"]), len(sets["Q9"]), len(sets["Q9"]), 158, tol=0.5)
        check("concepts Gemma validates", len(sets["G4"]), len(sets["G4"]), len(sets["G4"]), 105, tol=0.5)
        check("concepts both validate", len(inter), len(inter), len(inter), 104, tol=0.5)

    print("THE CROSS-MODEL CONCEPT REVERSAL (claim 7's fourth untabled number, and it moved)")
    # The claim said "the two moral concepts that move most in Qwen over twelve turns both flip sign in Llama",
    # with no table and no mention that Llama has no twelve-turn run: the comparison was on the three-round data
    # both models have. Reproducing it found the reversal at the READING site and NO reversal at the deciding
    # token, so the sign turns on where you read. The claim now says that, which is the stronger statement.
    CX = pd.read_csv(TABLES / "concept_drift_crossmodel_moral_stories.csv")
    for site, c, mm, want in (("reading", "self-righteous", "Q9", 0.20), ("reading", "self-righteous", "L31", -0.29),
                              ("reading", "conflicted-duty", "Q9", -0.29), ("reading", "conflicted-duty", "L31", 0.27)):
        r = CX[(CX.site == site) & (CX.concept == c) & (CX.model == mm) & CX.is_reference_layer].iloc[0]
        check(f"{mm} {c} drift, {site} site", float(r.drift), float(r.lower), float(r.upper), want, tol=0.006)
    # the reversal at one site and its absence at the other are both load-bearing
    for site, expect_reversal in (("reading", True), ("decision", False)):
        for c in ("self-righteous", "conflicted-duty"):
            g = CX[(CX.site == site) & (CX.concept == c) & CX.is_reference_layer]
            q = float(g[g.model == "Q9"].iloc[0].drift)
            l = float(g[g.model == "L31"].iloc[0].drift)
            if ((q > 0) != (l > 0)) != expect_reversal:
                FAILED.append(f"claim 7 says {c} {'reverses' if expect_reversal else 'does not reverse'} at the "
                              f"{site} site between the two models; it now does the opposite")
    print("  OK  the reversal is present while reading and absent while deciding, as claim 7 states")
    SG = pd.read_csv(TABLES / "concept_drift_crossmodel_moral_stories_signs.csv")
    for site, c, want in (("reading", "self-righteous", 6), ("reading", "conflicted-duty", 7),
                          ("decision", "self-righteous", 2), ("decision", "conflicted-duty", 5)):
        r = SG[(SG.site == site) & (SG.concept == c)].iloc[0]
        check(f"layers where the models disagree on sign, {c} {site}",
              int(r.layers_sign_differs), int(r.layers_sign_differs), int(r.layers_sign_differs), want, tol=0.5)

    print("FIGURE 5's PER-PANEL COUNTS (one n was quoted for four panels)")
    # The caption said "91 of 600" once, for all four panels. Panels a and b use every contested group (87-91 by
    # turn); panels c and d ask whether the group is WRONG, which a values question cannot answer, and use 42-49.
    # Quoting the largest of three counts as though it were the only one overstates three of the four panels.
    # both models now, since figure 5 carries both and their contested subsets differ in size
    for mm, spec in (("Q9", (("unanimity", 87, 91), ("wrong_unanimity", 42, 46), ("switch_rate", 261, 273))),
                     ("L31", (("unanimity", 87, 114), ("wrong_unanimity", 49, 73), ("switch_rate", 261, 342))),
                     ("G4", (("unanimity", 78, 79), ("wrong_unanimity", 43, 44), ("switch_rate", 234, 237)))):
        TU2 = pd.read_csv(TABLES / f"lit_turns_{mm}_release_600_t12.csv")
        CT = TU2[TU2.scope == "contested"]
        for met, lo, hi in spec:
            g = CT[CT.metric == met]
            if not (lo <= int(g.n.min()) and int(g.n.max()) <= hi):
                FAILED.append(f"figure 5 caption, {mm}: {met} n is {int(g.n.min())}-{int(g.n.max())}, "
                              f"caption says {lo}-{hi}")
            else:
                print(f"  OK  figure 5 {mm} {met}: n = {int(g.n.min())}-{int(g.n.max())}, as the caption states")
            # These two compare the TABLE's bounds to the caption's. The first version passed `val` as both the
            # observed and the claimed value, which is a check that cannot fail on any input -- written only to
            # make the unsourced sweep go quiet. Satisfying a detector is not the same as verifying a number,
            # and it is the third vacuous check I have caught myself writing.
            check(f"fig5 {mm} {met} lowest n", int(g.n.min()), int(g.n.min()), int(g.n.min()), lo, tol=0.5)
            check(f"fig5 {mm} {met} highest n", int(g.n.max()), int(g.n.max()), int(g.n.max()), hi, tol=0.5)
    TU2 = pd.read_csv(TABLES / "lit_turns_Q9_release_600_t12.csv")
    CT = TU2[TU2.scope == "contested"]
    # the agent counts the caption prints, pinned individually so the reverse sweep can see them too
    sr = CT[CT.metric == "switch_rate"]
    for lab, got, want in (("lowest", int(sr.n.min()), 261), ("highest", int(sr.n.max()), 273)):
        check(f"fig5 panel a, {lab} agent count", got, got, got, want, tol=0.5)
    IN2 = pd.read_csv(TABLES / "lit_independence_Q9_release_600_t12.csv")
    sp = IN2[IN2.scope == "split_at_start"]
    if not (42 <= int(sp.n_items.min()) and int(sp.n_items.max()) <= 49):
        FAILED.append(f"figure 5 panel d: n_items is {int(sp.n_items.min())}-{int(sp.n_items.max())}, "
                      f"caption says 42 to 49")
    else:
        print(f"  OK  figure 5 panel d: n_items = {int(sp.n_items.min())}-{int(sp.n_items.max())}")
    for rnd, cond, want in ((1, "debate", 1.87), (3, "debate", 1.28), (9, "debate", 1.02),
                            (1, "reflect", 3.61), (12, "reflect", 2.10)):
        r = sp[(sp["round"] == rnd) & (sp.cond == cond)].iloc[0]
        check(f"fig5 effective voters, {cond} turn {rnd}", float(r.n_eff), float(r.n_eff_lo), float(r.n_eff_hi),
              want, tol=0.006)

    print("WHICH RATIO THE 69% ACTUALLY IS")
    # Figure 4's caption said "69% of the whole (1.31 of 1.95)", and 1.31/1.95 is 67%. Both numbers were right and
    # the sentence pairing them was not: the share is the reasoning against the TWO PARTS (1.31 of 1.90), not
    # against the whole message. A reader who divides the two numbers printed beside the claim gets a different
    # answer from the claim, which is the kind of thing a referee checks first.
    CA = pd.read_csv(TABLES / "lit_carrier_additivity_Q9_full2400.csv")
    ca = CA[CA.scope == "all"].iloc[0]
    parts = float(ca.answer_only) + float(ca.reasoning_only)
    check("reasoning share, against the two parts", float(ca.reasoning_only) / parts,
          float(ca.share_lo), float(ca.share_hi), 0.69, tol=0.006)
    if abs(float(ca.reasoning_share) - float(ca.reasoning_only) / parts) > 0.002:
        FAILED.append("the table's reasoning_share is no longer reasoning/(answer+reasoning); the claim and "
                      "caption both describe it that way")
    else:
        print("  OK  reasoning_share is the share of the two parts, as the claim and caption now say")
    # and the distinct number the caption used to conflate with it
    check("reasoning against the WHOLE message (the number the caption must not call 69%)",
          float(ca.reasoning_only) / float(ca.full), float(ca.reasoning_only) / float(ca.full),
          float(ca.reasoning_only) / float(ca.full), 0.672, tol=0.006)

    print("FIGURE 2's THREE-MODEL PANEL (normalised to each model's own record effect)")
    # Figure 2 was a two-model figure beside a claim that had been rewritten BECAUSE of the third model. Gemma is
    # the model that falsified "the credential barely moves anything", so a figure without it showed the evidence
    # for the claim and not the evidence that reshaped it. Panel b now normalises, because Gemma's effects are
    # two to seventy times the others' and raw logits would flatten Qwen and Llama into the axis.
    NORM = {}
    for mm, rec_w, sta_w, rec_n, sta_n in (("Q9", 0.56, 0.08, 0.60, 0.27),
                                           ("L31", 0.88, 0.01, 0.53, 0.07),
                                           ("G4", 1.91, 0.79, 0.58, 0.70)):
        MG2 = pd.read_csv(TABLES / f"lit_source_{mm}_magnitude.csv").set_index("arm")
        base = float(MG2.loc["record swap, as written"].est)
        NORM[mm] = base
        check(f"fig2 {mm} record swap as written", base, base, base, rec_w, tol=0.006)
        sw = float(MG2.loc["status swap, as written"].est)
        check(f"fig2 {mm} status swap as written", sw, sw, sw, sta_w, tol=0.006)
        rn = float(MG2.loc["record, shrunk to status magnitude"].est) / base
        check(f"fig2 {mm} record shrunk, share of its own effect", rn, rn, rn, rec_n, tol=0.006)
        sa = MG2.loc["status, amplified to record magnitude"]
        check(f"fig2 {mm} status amplified, share of the record's effect",
              float(sa.est) / base, float(sa.lower) / base, float(sa.upper) / base, sta_n, tol=0.006)
        # the claim's whole point: even amplified to equal size, the status does not overtake the record
        if float(sa.est) / base >= 1.0:
            FAILED.append(f"in {mm} the amplified status now matches or beats the record; claim 2 says it never does")
    if not (NORM["G4"] > NORM["L31"] > NORM["Q9"]):
        FAILED.append("the record-effect ordering across models changed; the caption states Qwen < Llama < Gemma")
    else:
        print("  OK  the amplified status never overtakes the record, in any of the three models")

    print("THE FREEZE RESULT, NOW ON BOTH TWELVE-TURN MODELS")
    # Claim 5's opening measurement was Qwen-only over twelve turns, with Llama represented by a three-turn run
    # and a hedge about needing a long run. Llama's twelve-turn data arrived, so the comparison is like for like
    # and much larger there: a 56-point gap against Qwen's 10.
    FZ = {}
    for mm, dbt, sol in (("Q9", 0.897, 0.795), ("L31", 0.703, 0.142), ("G4", 0.878, 0.673)):
        F4 = pd.read_csv(TABLES / f"lit_freeze_{mm}_release_600_t12.csv")
        F4 = F4[F4.turn == 1].set_index("cond")
        for cond, want in (("debate", dbt), ("reflect", sol)):
            r = F4.loc[cond]
            check(f"{mm} frozen by turn 1, {cond}", float(r.frozen), float(r.lower), float(r.upper), want, tol=0.002)
        FZ[mm] = float(F4.loc["debate"].frozen) - float(F4.loc["reflect"].frozen)
    # the claim quotes the three gaps in points and says the widest is five times the narrowest
    for mm, want in (("Q9", 10.2), ("G4", 20.5), ("L31", 56.2)):
        check(f"{mm} freeze gap, points", 100 * FZ[mm], 100 * FZ[mm], 100 * FZ[mm], want, tol=0.01)
    _ratio = max(FZ.values()) / min(FZ.values())
    if not 4.5 <= _ratio <= 6.5:
        FAILED.append(f"claim 5 says the freeze gap is five times wider in one model than another; it is "
                      f"{_ratio:.1f} times")
    else:
        print(f"  OK  the freeze gap spans {_ratio:.1f}x across the three models, as the claim states")
    # the claim's actual assertion: in no model does thinking alone freeze faster than debating
    for mm in ("Q9", "L31"):
        if FZ[mm] <= 0:
            FAILED.append(f"{mm}: thinking alone now freezes at least as fast as debate; claim 5 says there is "
                          f"no such model")
    else:
        print(f"  OK  debate freezes the group faster in both models (gaps {100*FZ['Q9']:.1f} and "
              f"{100*FZ['L31']:.1f} points)")
    # and the twelve-turn effective-voter collapse, now in two models
    for mm, t1, t12, solo12 in (("Q9", 1.87, 1.02, 2.10), ("L31", 1.51, 1.07, 1.70), ("G4", 1.51, 1.02, 1.77)):
        IN3 = pd.read_csv(TABLES / f"lit_independence_{mm}_release_600_t12.csv")
        sp3 = IN3[IN3.scope == "split_at_start"]
        for rnd, want in ((1, t1), (12, t12)):
            r = sp3[(sp3["round"] == rnd) & (sp3.cond == "debate")].iloc[0]
            check(f"{mm} effective voters, debate turn {rnd}", float(r.n_eff), float(r.n_eff_lo),
                  float(r.n_eff_hi), want, tol=0.006)
        r = sp3[(sp3["round"] == 12) & (sp3.cond == "reflect")].iloc[0]
        check(f"{mm} effective voters, no communication turn 12", float(r.n_eff), float(r.n_eff_lo),
              float(r.n_eff_hi), solo12, tol=0.006)
        # the claim's point is that debate collapses BELOW the silent arm in both, not that any one number holds
        d12 = float(sp3[(sp3["round"] == 12) & (sp3.cond == "debate")].iloc[0].n_eff)
        if not d12 < float(r.n_eff):
            FAILED.append(f"{mm}: debating agents are no longer less independent than silent ones at turn 12")

    print("FIGURE 5's TURN-12 ENDPOINTS IN ALL THREE MODELS")
    for mm, met, cond, want in (("Q9", "switch_rate", "debate", 0.0), ("G4", "switch_rate", "debate", 1.3),
                                ("L31", "switch_rate", "debate", 2.4), ("Q9", "switch_rate", "reflect", 1.5),
                                ("G4", "switch_rate", "reflect", 5.9), ("L31", "switch_rate", "reflect", 17.8),
                                ("Q9", "wrong_unanimity", "debate", 40.0), ("G4", "wrong_unanimity", "debate", 56.8),
                                ("L31", "wrong_unanimity", "debate", 40.3),
                                ("Q9", "wrong_unanimity", "reflect", 20.9), ("G4", "wrong_unanimity", "reflect", 9.1),
                                ("L31", "wrong_unanimity", "reflect", 18.9)):
        TU4 = pd.read_csv(TABLES / f"lit_turns_{mm}_release_600_t12.csv")
        r = TU4[(TU4.scope == "contested") & (TU4.metric == met) & (TU4.cond == cond)
                & (TU4["round"] == 12)].iloc[0]
        check(f"fig5 {mm} {met} {cond} at turn 12", float(r.est), float(r.lower), float(r.upper),
              want, scale=100.0, tol=0.06)
    # the caption says the ordering between models on wrong-unanimity should NOT be read; assert the intervals
    # really do overlap, so that caution stays honest rather than becoming false modesty
    _w = {}
    for mm in ("Q9", "L31", "G4"):
        TU4 = pd.read_csv(TABLES / f"lit_turns_{mm}_release_600_t12.csv")
        r = TU4[(TU4.scope == "contested") & (TU4.metric == "wrong_unanimity") & (TU4.cond == "debate")
                & (TU4["round"] == 12)].iloc[0]
        _w[mm] = (float(r.lower), float(r.upper))
    if max(lo for lo, _ in _w.values()) > min(hi for _, hi in _w.values()):
        FAILED.append("figure 5's caption says the wrong-unanimity ordering between models should not be read, "
                      "but the intervals no longer overlap; the caution has become false modesty")
    else:
        print("  OK  the wrong-unanimity intervals overlap, as the caption's caution says")

    print("CLAIM 4's VALUES-QUESTION SHARES, AND THE SUPERLATIVE THAT ROTTED")
    # The claim used to call this "the closest agreement anywhere in this paper". It was true when written. It
    # stopped being true when claim 5 gained a third model: the effective-voter endpoints are 1.02, 1.02 and 1.07,
    # a coefficient of variation of 2.7% against these shares' 3.1%. A superlative is a claim about every OTHER
    # result in the paper, so it silently becomes false whenever any of them improves. The claim now states the
    # three numbers and leaves the ranking alone.
    _vs = {}
    for mm, f, want, lo, hi in (("Q9", "lit_carrier_additivity_Q9_full2400.csv", 70.0, 67.2, 73.2),
                                ("G4", "lit_carrier_additivity_G4.csv", 68.4, 64.5, 73.3),
                                ("L31", "lit_carrier_additivity_L31_full2400.csv", 65.8, 60.5, 73.0)):
        C4 = pd.read_csv(TABLES / f)
        r = C4[C4.scope == "values_moral_judgment"].iloc[0]
        check(f"{mm} reasoning share on values questions", float(r.reasoning_share), float(r.share_lo),
              float(r.share_hi), want, scale=100.0, tol=0.06)
        _vs[mm] = float(r.reasoning_share)
    _spread = 100 * (max(_vs.values()) - min(_vs.values()))
    if not 3.0 <= _spread <= 5.0:
        FAILED.append(f"claim 4 says the three values-question shares are four points apart; they span "
                      f"{_spread:.1f} points")
    else:
        print(f"  OK  the three values-question shares span {_spread:.1f} points, as the claim states")
    # and the superlative must not come back
    _bodies = " ".join(" ".join(b.split()) for _, _, b in CONTRIBUTIONS).lower()
    # "the one quantity that survives a change of architecture" was the second of these, found by
    # analysis/cross_references.py rather than by reading. It was false against at least five other quantities
    # that also survive: the record-over-standing ordering, the voter collapse, the freeze gap, the position
    # control ratio, and the affect dependence. An exclusivity claim is a superlative wearing different words.
    for _sup in ("closest agreement anywhere", "the largest in this paper", "the only result in this paper",
                 "the one quantity that survives", "the only quantity that survives",
                 "the one thing that survives"):
        if _sup in _bodies:
            FAILED.append(f"a claim contains the superlative '{_sup}', which is an assertion about every other "
                          f"result and rots when any of them improves")
    print("  OK  no claim asserts a superlative over the rest of the paper")

    print("TWELVE-TURN UNANIMITY, ALL THREE MODELS")
    # Claim 5's newest sentence. The tempting version was "the silent arms land within 1.4 points of each other",
    # which reads precision into intervals twenty points wide -- the same trap the portability test fell into
    # this morning. What is actually well powered is that the two GROUPS of intervals do not overlap.
    _un = {}
    for mm, dbt, sol in (("Q9", 98.9, 36.8), ("G4", 97.5, 35.4), ("L31", 93.8, 35.9)):
        TU3 = pd.read_csv(TABLES / f"lit_turns_{mm}_release_600_t12.csv")
        C3 = TU3[(TU3.scope == "contested") & (TU3.metric == "unanimity") & (TU3["round"] == 12)].set_index("cond")
        for cond, want in (("debate", dbt), ("reflect", sol)):
            r = C3.loc[cond]
            check(f"{mm} unanimous at turn 12, {cond}", float(r.est), float(r.lower), float(r.upper),
                  want, scale=100.0, tol=0.06)
        _un[mm] = (float(C3.loc["debate"].lower), float(C3.loc["reflect"].upper))
    if not min(lo for lo, _ in _un.values()) > max(hi for _, hi in _un.values()):
        FAILED.append("claim 5 says the three debating unanimity intervals and the three silent ones do not "
                      "overlap; they now do")
    else:
        print(f"  OK  debating unanimity (lowest bound {100*min(lo for lo,_ in _un.values()):.1f}%) clears every "
              f"silent arm (highest bound {100*max(hi for _,hi in _un.values()):.1f}%)")

    print("RESPONSE LENGTH OVER TWELVE TURNS (claim 5's 'those tokens are not idling')")
    # Added after a qualitative impression from three transcripts failed its own base rate: peer-citation rate is
    # flat across turns (21.7% -> 20.3%), so "agents stop citing peers" was a sampling artifact. The arms must
    # also keep pointing OPPOSITE ways, which is the whole content of the sentence.
    RL = pd.read_csv(TABLES / "response_length_Q9_release_600_t12_t1to11.csv").set_index("cond")
    RL31 = pd.read_csv(TABLES / "response_length_L31_release_600_t12_t1to11.csv").set_index("cond")
    for cond, want in (("debate", 224.0), ("reflect", -69.0), ("debate_minus_solo", 293.0)):
        r = RL.loc[cond]
        check(f"Q9 {cond}: response length, turn 11 minus turn 1", float(r.delta), float(r.lower), float(r.upper),
              want, tol=0.01)
    for cond, want in (("debate", 573.0), ("reflect", 62.0), ("debate_minus_solo", 511.0)):
        r = RL31.loc[cond]
        check(f"L31 {cond}: response length, turn 11 minus turn 1", float(r.delta), float(r.lower), float(r.upper),
              want, tol=0.01)
    # the claim's portable statement: the ARM CONTRAST is large and positive in both, whatever each arm does
    for nm, tb in (("Qwen", RL), ("Llama", RL31)):
        if not float(tb.loc["debate_minus_solo"].lower) > 0:
            FAILED.append(f"{nm}: debate no longer outgrows thinking alone with an interval clear of zero, which "
                          f"is what claim 5 says carries across the two models")
    # and the parts really do disagree, which is why the claim reports the gap rather than the arms
    if not (float(RL.loc["reflect"].upper) < 0 < float(RL31.loc["reflect"].lower)):
        FAILED.append("claim 5 says the solo arms move in opposite directions in the two models; they no longer do")
    else:
        print("  OK  the solo arms move opposite ways (Qwen shrinks, Llama grows), as claim 5 states")
    for nm, tb, cf, cl in (("Qwen", RL, 0.217, 0.203), ("Llama", RL31, 0.393, 0.603)):
        r = tb.loc["debate"]
        check(f"{nm} peer-citation rate, turn 1", float(r.cite_rate_first), float(r.cite_rate_first),
              float(r.cite_rate_first), cf, tol=0.002)
        check(f"{nm} peer-citation rate, turn 11", float(r.cite_rate_last), float(r.cite_rate_last),
              float(r.cite_rate_last), cl, tol=0.002)
    check("debate: share of agents writing more by turn 11", float(RL.loc["debate"].frac_longer),
          float(RL.loc["debate"].frac_longer), float(RL.loc["debate"].frac_longer), 0.74, tol=0.01)
    if not (RL.loc["debate"].lower > 0 > RL.loc["reflect"].upper):
        FAILED.append("claim 5 says Qwen's debate arm grows while ITS solo arm shrinks (Llama's grows); "
                      "the intervals no longer separate in that direction")
    else:
        print("  OK  Qwen: debate grows and its solo arm shrinks, both intervals clear of zero")
    cf, cl = float(RL.loc["debate"].cite_rate_first), float(RL.loc["debate"].cite_rate_last)
    if abs(cf - cl) > 0.05:
        FAILED.append(f"Qwen's peer-citation rate is no longer flat ({100*cf:.1f}% -> {100*cl:.1f}%); the "
                      f"claim contrasts it against Llama's rise, so flatness here is load-bearing")
    else:
        print(f"  OK  Qwen's peer-citation rate is flat across turns ({100*cf:.1f}% -> {100*cl:.1f}%)")

    print("THE RESULTS WRITE-UP IS NOT STALE")
    # The agreed deliverable is tables, figures and a write-up; the LaTeX is the user's. The write-ups in
    # results/ silently went three days stale -- they described a paper from before the third model arrived and
    # before three claims were rewritten. RESULTS.md is generated from the claims so it cannot drift in content,
    # but it can still drift in TIME, which is the same defect the figure provenance check exists for.
    import os as _os
    rp = ROOT / "results/RESULTS.md"
    srcs = [ROOT / "code/figures/build_pdf.py",
            ROOT / "outputs/figures/paper/HERO_CAPTIONS.md",
            ROOT / "results/RESULTS_NARRATIVE.md"]
    if not _os.path.exists(rp):
        FAILED.append("results/RESULTS.md does not exist; run analysis/emit_results.py")
    else:
        stale = [f for f in srcs if _os.path.exists(f) and _os.path.getmtime(f) > _os.path.getmtime(rp)]
        if stale:
            FAILED.append("results/RESULTS.md is older than " + ", ".join(_os.path.basename(f) for f in stale)
                          + "; rerun analysis/emit_results.py")
        else:
            print("  OK  results/RESULTS.md is newer than the claims, captions and narrative it is built from")

    print("FIGURES 1, 4 AND 5's CAPTION NUMBERS")
    # The last 18 the unsourced sweep flagged. Nothing here had drifted, but two of them sent me to the wrong
    # table first: figure 5's caption says "agreement" where its table's metric is called "unanimity", and the
    # similarly named long_Q9_release_600_summary.csv gives different numbers for a differently defined
    # quantity. A caption number is only checkable if you can find the table it came from.
    for m, ds, group, want in (("Q9", "full2400", "changed answer in debate", 94.1),
                               ("Q9", "full2400", "changed answer without communication", 88.8),
                               ("L31", "full2400", "changed answer in debate", 86.8),
                               ("L31", "full2400", "changed answer without communication", 92.4),
                               ("G4", "", "changed answer in debate", 97.0),
                               ("G4", "", "changed answer without communication", 98.1)):
        f = f"lit_commit_{m}_curves.csv" if not ds else f"lit_commit_{m}_{ds}_curves.csv"
        C = pd.read_csv(TABLES / f)
        r = C[(C.domain == "all") & (C.metric == "argmax_is_final") & (C.fraction == 1.0) & (C.group == group)].iloc[0]
        check(f"fig1 {m} {group} at full reasoning", float(r.est), float(r.lower), float(r.upper), want,
              scale=100.0, tol=0.06)
    FI = pd.read_csv(TABLES / "lit_filler_Q9_full2400_summary.csv")
    # claim 1 used to say the filler "moves it under three points" beside a clause about the GAP, so a reader
    # naturally read both as the gap -- and the gap moves the other way, by +4.3. It now quotes the interval on
    # the level, which is what 2.5 was all along.
    # The -2.5 itself is already checked once, near the top, so only the NEW assertion lives here: the claim now
    # says in words that this interval covers zero, and a second check on the same value would be a second place
    # for the same number to drift from.
    fdiff = FI[(FI.cond == "debate") & (FI.group == "changed") & (FI.arm == "filler - frac0")].iloc[0]
    if not (float(fdiff.lower) < 0 < float(fdiff.upper)):
        FAILED.append("claim 1 says the filler interval covers zero; it no longer does")
    else:
        print("  OK  the filler interval covers zero, as claim 1 states")
    for cond, grp, arm, want in (("debate", "changed", "filler", 62.9), ("debate", "changed", "cross_item", 52.6),
                                 ("reflect", "changed", "cross_item", 48.0)):
        r = FI[(FI.cond == cond) & (FI.group == grp) & (FI.arm == arm)].iloc[0]
        check(f"fig1 filler {cond}/{arm}", float(r.est), float(r.lower), float(r.upper), want, scale=100.0, tol=0.06)

    # the caption also quotes the GAP between two filler arms, which is a number in its own right
    fd = float(FI[(FI.cond == "debate") & (FI.group == "changed") & (FI.arm == "filler")].iloc[0].est)
    fr = float(FI[(FI.cond == "reflect") & (FI.group == "changed") & (FI.arm == "filler")].iloc[0].est)
    check("fig1 filler gap, debate minus without communication", fd - fr, fd - fr, fd - fr, 47.0,
          scale=100.0, tol=0.5)

    for m, f, scope, want in (("L31", "lit_carrier_additivity_L31_full2400.csv", "values_moral_judgment", -1.06),
                              ("G4", "lit_carrier_additivity_G4.csv", "knowledge_reasoning", 1.77)):
        C = pd.read_csv(TABLES / f)
        r = C[C.scope == scope].iloc[0]
        check(f"fig4 {m} residual, {scope}", float(r.residual), float(r.res_lo), float(r.res_hi), want, tol=0.006)

    TU = pd.read_csv(TABLES / "lit_turns_Q9_release_600_t12.csv")
    for met, cond, want in (("unanimity", "debate", 99.3), ("unanimity", "reflect", 83.6),
                            ("wrong_unanimity", "debate", 13.7), ("wrong_unanimity", "reflect", 8.8)):
        r = TU[(TU.scope == "all") & (TU.metric == met) & (TU.cond == cond) & (TU["round"] == 12)].iloc[0]
        check(f"fig5 {met} {cond} at turn 12, all items", float(r.est), float(r.lower), float(r.upper), want,
              scale=100.0, tol=0.06)

    print("FIGURE 3's CAPTION, EVERY NUMBER (the vote slopegraphs)")
    # 33 of the 70 numbers the unsourced sweep flagged were in this one caption. They all matched the tables --
    # but nothing was reading them back, and this is the caption that carried the retracted +3.0 vote gain until
    # it was caught by hand. A caption is a results section with no audit attached unless one is attached.
    VOTE = [("Q9", "full2400", "all", "acc_vote", 88.0), ("Q9", "full2400", "all", "acc_debate_maj", 89.2),
            ("Q9", "full2400", "all", "debate - vote", 1.2),
            ("Q9", "full2400", "knowledge_reasoning", "acc_vote", 87.8),
            ("Q9", "full2400", "knowledge_reasoning", "acc_debate_maj", 88.9),
            ("Q9", "full2400", "safety_refusal", "acc_vote", 87.9),
            ("Q9", "full2400", "safety_refusal", "acc_debate_maj", 89.4),
            ("Q9", "full2400", "values_moral_judgment", "acc_vote", 88.9),
            ("Q9", "full2400", "values_moral_judgment", "acc_debate_maj", 89.6),
            ("Q9", "full2400", "all", "wrong_unan_reflect", 6.6),
            ("Q9", "full2400", "all", "wrong_unan_debate", 9.8),
            ("Q9", "full2400", "all", "wrong_unan debate - reflect", 3.1),
            ("Q9", "full2400", "knowledge_reasoning", "wrong_unan_reflect", 6.7),
            ("Q9", "full2400", "knowledge_reasoning", "wrong_unan_debate", 9.5),
            ("Q9", "full2400", "safety_refusal", "wrong_unan_reflect", 6.9),
            ("Q9", "full2400", "safety_refusal", "wrong_unan_debate", 10.1),
            ("Q9", "full2400", "values_moral_judgment", "wrong_unan_reflect", 6.1),
            ("Q9", "full2400", "values_moral_judgment", "wrong_unan_debate", 9.7),
            ("L31", "full2400", "all", "acc_vote", 77.3), ("L31", "full2400", "all", "acc_debate_maj", 77.8),
            ("L31", "full2400", "all", "debate - vote", 0.5),
            ("L31", "full2400", "knowledge_reasoning", "acc_vote", 75.7),
            ("L31", "full2400", "knowledge_reasoning", "acc_debate_maj", 71.6),
            ("L31", "full2400", "safety_refusal", "acc_vote", 73.9),
            ("L31", "full2400", "safety_refusal", "acc_debate_maj", 78.1),
            ("L31", "full2400", "values_moral_judgment", "acc_vote", 85.5),
            ("L31", "full2400", "values_moral_judgment", "acc_debate_maj", 86.0),
            ("L31", "full2400", "all", "wrong_unan_reflect", 11.8),
            ("L31", "full2400", "all", "wrong_unan_debate", 19.2),
            ("L31", "full2400", "all", "wrong_unan debate - reflect", 7.4),
            ("L31", "full2400", "knowledge_reasoning", "wrong_unan_reflect", 11.0),
            ("L31", "full2400", "knowledge_reasoning", "wrong_unan_debate", 22.9),
            ("L31", "full2400", "safety_refusal", "wrong_unan_reflect", 16.0),
            ("L31", "full2400", "safety_refusal", "wrong_unan_debate", 19.8),
            ("L31", "full2400", "values_moral_judgment", "wrong_unan_reflect", 5.4),
            ("L31", "full2400", "values_moral_judgment", "wrong_unan_debate", 12.8),
            # the conditional numbers the caption quotes as round figures, and Gemma's two
            ("Q9", "full2400", "all", "wrong_unan_debate | initial majority wrong", 76.0),
            ("Q9", "full2400", "all", "wrong_unan_reflect | initial majority wrong", 54.0),
            ("L31", "full2400", "all", "wrong_unan_debate | initial majority wrong", 76.0),
            ("L31", "full2400", "all", "wrong_unan_reflect | initial majority wrong", 43.0),
            ("G4", "core", "all", "debate - vote", 1.2),
            ("G4", "core", "all", "wrong_unan debate - reflect", 3.6)]
    for m, ds, dom, met, want in VOTE:
        V = pd.read_csv(TABLES / f"lit_debate_{m}_{ds}_vote.csv")
        r = V[(V.scope == "fully_parsed") & (V.domain == dom) & (V.metric == met)]
        if not len(r):
            FAILED.append(f"figure 3 caption: no {met} for {m} {dom} in the fully_parsed scope")
            continue
        r = r.iloc[0]
        # round figures in the caption ("76% of the time") get a whole point of slack; the rest get a tenth
        tol = 0.005 if float(want).is_integer() and want > 20 else 0.0006
        check(f"fig3 {m} {dom} {met}", float(r.est), float(r.lower), float(r.upper), want, scale=100.0, tol=tol * 100)

    print("THE PROMPT LENGTHS BEHIND THE SLIDING-WINDOW ARGUMENT")
    # Claim 3's "median prompt here of 597" was the third number found with no table behind it, and reproducing
    # it moved it by a factor of nearly three: the real median debate prompt is 1,668 tokens and 96.8% of them
    # exceed Gemma's 512-token window. The claim had understated its own architectural argument.
    PL = pd.read_csv(TABLES / "prompt_lengths_G4_core.csv").set_index("cond").loc["all_debate"]
    check("Gemma median debate prompt, tokens", float(PL["median"]), float(PL["median"]), float(PL["median"]),
          1668, tol=1.0)
    check("Gemma share of debate prompts over the 512-token window",
          float(PL.frac_over_window), float(PL.frac_over_window), float(PL.frac_over_window), 0.968, tol=0.001)
    if not float(PL["median"]) > 512:
        FAILED.append("claim 3 argues the peer sits outside the sliding window, but the median prompt fits in it")
    else:
        print(f"  OK  the median debate prompt ({PL['median']:.0f}) really does exceed the 512-token window")

    print("FIGURE 6's AFFECT NUMBERS (the caption states them; until now nothing read them back)")
    # Written into the caption during the two-model rebuild. A number I wrote an hour ago is exactly as capable
    # of drifting as one written a week ago, and the unsourced-number sweep listed every one of these.
    for m, tag, site, t1, t12 in (("Q9", "T12shared", "read", 0.46, 0.37), ("Q9", "T12shared", "dec", 0.45, 0.56),
                                  ("G4", "T12shared", "read", 0.30, 0.56), ("G4", "T12shared", "dec", 0.91, 0.89)):
        S = pd.read_csv(TABLES / f"traj_domain_{m}_{tag}_{site}_spread.csv")
        sel = set(S[(S.half == "discovery") & (S.q < 0.05)].concept)
        R = S[(S.half == "confirmation") & S.concept.isin(sel) & (S.q < 0.05)]
        g = R.groupby("turn").spread.mean()
        for turn, want in ((g.index[0], t1), (g.index[-1], t12)):
            check(f"{m} {site}-site domain spread, turn {turn}", float(g.loc[turn]), float(g.loc[turn]),
                  float(g.loc[turn]), want, tol=0.005)
    # figure 6 now plots `despondent` in three models; the caption states every domain at turn 11 with its CI
    for m, dom, want, lo, hi in (("Q9", "knowledge_reasoning", 0.53, 0.35, 0.71),
                                 ("Q9", "safety_refusal", 0.18, 0.03, 0.34),
                                 ("Q9", "values_moral_judgment", -0.37, -0.64, -0.10),
                                 ("L31", "safety_refusal", -0.12, -0.35, 0.12),
                                 ("L31", "knowledge_reasoning", -0.88, -1.02, -0.73),
                                 ("L31", "values_moral_judgment", -0.93, -1.11, -0.75),
                                 ("G4", "values_moral_judgment", 2.85, 2.66, 3.04),
                                 ("G4", "knowledge_reasoning", 2.16, 1.95, 2.35),
                                 ("G4", "safety_refusal", 1.98, 1.82, 2.15)):
        C = pd.read_csv(TABLES / f"traj_domain_{m}_T12shared3_dec_curves.csv")
        r = C[(C.half == "confirmation") & (C.concept == "despondent") & (C.turn == 11)
              & (C.domain == dom)].iloc[0]
        check(f"{m} 'despondent' at the decision token, turn 11, {dom}", float(r.drift), float(r.lower),
              float(r.upper), want, tol=0.006)
        for got, claimed, edge in ((float(r.lower), lo, "lower"), (float(r.upper), hi, "upper")):
            if abs(got - claimed) > 0.006:
                FAILED.append(f"{m} despondent {dom} {edge} bound: table {got:.3f}, caption {claimed}")
    # the sign-reversal rates the claim quotes, 50/21/3. Integers slip through the unsourced sweep 99% of the
    # time, so a clean sweep says nothing about these and a named check is the only guarantee.
    for m, want in (("Q9", 0.503), ("L31", 0.206), ("G4", 0.030)):
        SS = pd.read_csv(TABLES / f"traj_domain_{m}_T12shared3_dec_spread.csv")
        sel = set(SS[(SS.half == "discovery") & (SS.q < 0.05)].concept)
        keep = SS[(SS.half == "confirmation") & SS.concept.isin(sel) & (SS.q < 0.05)][["concept", "turn"]]
        CC = pd.read_csv(TABLES / f"traj_domain_{m}_T12shared3_dec_curves.csv")
        k = keep.merge(CC[CC.half == "confirmation"], on=["concept", "turn"])
        g = k.groupby(["concept", "turn"]).drift.agg(["min", "max"])
        rev = float(((g["min"] < 0) & (g["max"] > 0)).mean())
        check(f"{m} share of confirmed cells with opposite-signed domains", rev, rev, rev, want, tol=0.002)
    # and the ordering the claim rests on: Qwen reverses most often, Gemma least
    print("  OK  sign-reversal rates pinned for all three models")

    # the claim's magnitude sentence: mean |drift| at the last turn, per model and domain
    for m, dom, want in (("Q9", "knowledge_reasoning", 0.35), ("L31", "knowledge_reasoning", 0.89),
                         ("G4", "knowledge_reasoning", 1.77), ("Q9", "values_moral_judgment", 0.42),
                         ("L31", "values_moral_judgment", 0.76), ("G4", "values_moral_judgment", 2.38)):
        C = pd.read_csv(TABLES / f"traj_domain_{m}_T12shared3_dec_curves.csv")
        g = C[(C.half == "confirmation") & (C.turn == 11) & (C.domain == dom)].drift.abs().mean()
        check(f"{m} mean |drift| at turn 11, {dom}", float(g), float(g), float(g), want, tol=0.006)
    print("  OK  figure 6's caption numbers all read back from the tables")

    print("THE SHUFFLED-LABEL TOPIC AUROCs (claim 7's warning, quantified)")
    # This claim quoted 0.66 and 0.95 for a year of campaign days with NO table behind it: the numbers lived in a
    # ledger note, computed by hand, checkable by nobody including me. Reproducing them as a script moved them to
    # 0.685 and 0.993 -- the original had taken a single layer and reported it as the whole. The lesson is not that
    # the number was wrong (it understated its own point) but that a claim number without a table is not a result.
    for m, ds, med in (("Q9", "release_600_t12", 0.685), ("G4", "core", 0.648), ("L31", "lit1200", 0.602)):
        S = pd.read_csv(TABLES / f"shuffled_topic_auroc_{m}_{ds}.csv")
        g = S[S.contrast == "moral from knowledge"]
        check(f"{m} shuffled-label directions, moral vs knowledge, median across layers",
              float(g["median"].median()), float(g["median"].median()), float(g["median"].median()), med, tol=0.002)
    Sq = pd.read_csv(TABLES / "shuffled_topic_auroc_Q9_release_600_t12.csv")
    gq = Sq[Sq.contrast == "moral from knowledge"]
    check("Qwen shuffled-label maximum over all layers and directions",
          float(gq.maximum.max()), float(gq.maximum.max()), float(gq.maximum.max()), 0.993, tol=0.002)
    bq = gq.loc[gq["median"].idxmax()]
    check("Qwen shuffled-label 90th percentile at its strongest layer",
          float(bq.p90), float(bq.p90), float(bq.p90), 0.932, tol=0.002)
    # the rhetorical point is layer-dependent and the claim now says so: 0.85 must sit BELOW that 90th percentile
    if not (bq.p90 > 0.85):
        FAILED.append(f"claim 7 says a 0.85 probe is below the shuffled 90th percentile, but that is {bq.p90:.3f}")
    else:
        print(f"  OK  0.85 really is below the strongest layer's shuffled 90th percentile ({bq.p90:.3f})")
    # all three models must show the effect, or "almost any direction looks like a good probe" is one model's story
    if not all(pd.read_csv(TABLES / f"shuffled_topic_auroc_{m}_{ds}.csv").query("contrast == 'moral from knowledge'")["median"].median() > 0.55
               for m, ds in (("Q9", "release_600_t12"), ("G4", "core"), ("L31", "lit1200"))):
        FAILED.append("the shuffled-label separation is not present in all three models")
    else:
        print("  OK  all three models separate topic from shuffled-label directions")

    print("THE FIGURE CAPTIONS, AND WHAT THE FIGURE SCRIPTS ACTUALLY READ")
    # Third artifact that quotes results and was not audited, after CONTRIBUTIONS and the abstract. Figure 1 plotted
    # and captioned Qwen at 66.7/20.3 (one half of the benchmark) while the claim beside it said 65.4/22.3 (pooled),
    # because the figure script opened lit_commit_Q9_curves.csv and the audit opened the _full2400 table. Checking
    # the caption prose alone would not have found it: the drift lived in which table the SCRIPT read.
    import re as _re
    FIGDIR = Path(__file__).resolve().parents[2] / "outputs/figures/paper"
    try:
        cap = " ".join((FIGDIR / "HERO_CAPTIONS.md").read_text().split())
        for num, why in (("66.7%", "Qwen half-benchmark commit rate, superseded by 65.4%"),
                         ("50.8%", "Llama half-benchmark commit rate, superseded by 52.4%"),
                         ("63.8%", "Qwen half-benchmark filler arm, superseded by 62.9%"),
                         ("until a third model arbitrates", "Gemma has arbitrated"),
                         # the affect caption, retired 2026-09-20 when Gemma cost the claim two of three findings.
                         # Every number below was measured on Qwen's own 158 concepts, an instrument the second
                         # model does not have; the claim and figure are now on the 104 shared concepts.
                         ("+3.71", "pressure_from_numbers on Qwen's 158-concept set; Gemma cannot validate it"),
                         ("+2.16 on safety", "retired pressure_from_numbers safety arm; a bare "
                          "\"+2.16\" is too generic to denylist and collided with a legitimate value"),
                         ("+1.68", "same, knowledge arm"),
                         ("133 of 158", "Qwen-only replication count, superseded by 89 of 104"),
                         ("47%", "Qwen-only sign-reversal rate on 158 concepts, superseded by 49% on 104"),
                         ("0.48 to 0.39", "Qwen-only reading-site spread, superseded by 0.46 to 0.37"),
                         ("0.46 to 0.60", "Qwen-only decision-site spread, superseded by 0.45 to 0.56"),
                         ("201-concept", "the atlas is 158 validated in Qwen, 105 in Gemma, 104 shared"),
                         # 2026-09-20: Gemma's counts were quoted out of 105, its OWN set, inside a claim that
                         # says everything is measured on the 104 both models validate.
                         ("104 of 105", "Gemma counted against its own concept set, not the shared one"),
                         # Third instance of the exposed-comparison class and the first found in a caption.
                         # By spread of point estimates the SILENT unanimity arms (36.8, 35.9, 35.4) are
                         # tighter than the debating ones, so the superlative was at best unverifiable.
                         # The non-overlap stands on its own without a ranking.
                         ("most architecture-independent statement",
                          "a superlative over the rest of the figure"),
                         # figure 5's caption presented the artifact-inflated vote gain as the real one, beside a
                         # claim that spends two sentences explaining it is not
                         ("gain stops at about\nthree points", "the three-point vote gain is the measurement "
                          "artifact; the gain is 1.2 points on the fully-parsed scope"),
                         ("anyone spoke, 91 of 600", "the contested count is 87 to 91 by turn, and panels c "
                          "and d use 42 to 49; one n was presented for four panels")):
            if num in cap:
                FAILED.append(f"CAPTIONS still quote {num!r} ({why})")
            else:
                print(f"  OK  superseded value absent from captions: {num!r}")
        # the structural half: a figure script must not read a half-benchmark table when a pooled one exists
        # Figure 1 was found this way; sweeping the rest of the hero scripts then found the same drift in the vote
        # figure (which still plotted the retracted +3.0 gain) and the carrier figure. The class, not the instance.
        HALVES = {
            "fig_hero1_answer.py": ("lit_commit_Q9_curves.csv", "lit_commit_L31_curves.csv",
                                    "lit_filler_Q9_lit1200_summary.csv"),
            "fig_hero4_vote.py": ("lit_debate_Q9_lit1200_vote.csv", "lit_debate_L31_lit1200_vote.csv",
                                  'lit_debate_{m}_lit1200_vote.csv'),
            "fig_hero6_carrier.py": ('lit_carrier_additivity_{m}.csv',),
        }
        for script, halves in HALVES.items():
            src = (ROOT / "code/figures" / script).read_text()
            code = _re.sub(r'""".*?"""', "", src, flags=_re.S)
            code = "\n".join(l for l in code.splitlines() if not l.strip().startswith("#"))
            bad = [h for h in halves if h in code]
            if bad:
                FAILED.append(f"{script} reads half-benchmark tables {bad} while pooled ones exist")
            else:
                print(f"  OK  {script} reads the pooled tables")
        # The affect figure builds its table names by interpolation, so grepping its source for a forbidden
        # filename tests nothing: the first version of this check passed on a figure that could not have failed
        # it. It now writes down what it opened, and we read that instead -- but a record is only evidence if it
        # is newer than the script that wrote it, or it describes a figure nobody has rebuilt.
        import os as _os
        tscript = ROOT / "code/figures/fig_traj_domain.py"
        srcf = f"{FIGDIR}/V3_traj_domain.sources.txt"
        if not _os.path.exists(srcf):
            FAILED.append("fig_traj_domain.py has no source record; rerun it")
        elif _os.path.getmtime(srcf) < _os.path.getmtime(tscript):
            FAILED.append("fig_traj_domain.py's source record predates the script; rerun it")
        else:
            read = [l.strip() for l in open(srcf) if l.strip()]
            # Qwen's own twelve-turn run is tagged T12 and covers its 158 concepts; the shared-set rerun is
            # T12shared. Reading the former here silently compares 158 Qwen concepts against 105 Gemma ones.
            # The first version of this check tested the shared-set tag for Qwen only and accepted ANY Gemma
            # table. It passed for a month of nothing and then passed on the real defect: the figure was built
            # from Qwen's 104 shared concepts against Gemma's own 105, which is precisely the mismatch the
            # claim beside it warns against. A per-model rule that names only one model is not a per-model rule.
            bad = [r for r in read if _re.match(r"traj_domain_(Q9|L31|G4)_T12(shared)?_", r)]
            if bad:
                FAILED.append(f"fig_traj_domain.py read a narrower concept set {bad}; the three-model "
                              f"claim is measured on T12shared3")
            else:
                for mm in ("Q9", "L31", "G4"):
                    if not any(r.startswith(f"traj_domain_{mm}_T12shared3_") for r in read):
                        FAILED.append(f"fig_traj_domain.py read no shared-set table for {mm}")
                if not FAILED:
                    print(f"  OK  fig_traj_domain.py read the shared-set tables in BOTH models ({len(read)} files)")

        # the affect figure must plot a concept both models validate, and must plot both models
        ts = (ROOT / "code/figures/fig_traj_domain.py").read_text()
        tcode = _re.sub(r'""".*?"""', "", ts, flags=_re.S)
        if "pressure_from_numbers" in tcode:
            FAILED.append("fig_traj_domain.py still plots pressure_from_numbers, which Gemma cannot validate")
        elif '("G4"' not in tcode or '("Q9"' not in tcode:
            FAILED.append("fig_traj_domain.py is no longer a two-model figure")
        else:
            print("  OK  fig_traj_domain.py plots both models on the shared concept set")

        # the vote figure must also restrict to the scope the claim quotes
        vs = (ROOT / "code/figures/fig_hero4_vote.py").read_text()
        if 'scope == "fully_parsed"' not in vs:
            FAILED.append("fig_hero4_vote.py no longer restricts to the fully_parsed scope the claim quotes")
        else:
            print("  OK  fig_hero4_vote.py restricts to the fully-parsed scope")
    except FileNotFoundError as e:
        print(f"  (captions not found: {e})")

    print(f"\n{'ALL CLAIMS MATCH THE TABLES' if not FAILED else 'DRIFTED: ' + ', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
