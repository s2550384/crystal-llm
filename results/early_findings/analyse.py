"""Run from chris/results/early_findings:  python analyse.py

Early findings: humans vs LLM (direct) vs LLM + hypotheses vs LLM + Thompson sampling.

Reads the ef_* action CSVs and the human action_data.csv, writes per-game and
summary tables to OUT, and prints a summary.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[2]  # the chris folder
sys.path.insert(0, str(CODE))
import llm_play as L  # noqa: E402
from hypotheses import Hypothesis  # noqa: E402

RUNS = Path(__file__).resolve().parent
HUMAN = CODE.parent / "data" / "action_data.csv"
SETUP_NAMES = {"direct": "LLM", "hypo": "LLM + hypotheses", "ts": "LLM + Thompson"}
ORDER = ["Human", "LLM", "LLM + hypotheses", "LLM + Thompson"]

SH = L.SHAPES
lv = lambda n: int(n.split("_")[2])  # noqa: E731
shape = lambda n: n.split("_")[0]  # noqa: E731

# all kinds up to level 4 (level 5 cannot fuse), for scoring rules against the truth
KINDS = [{"shape": s, "texture": t, "level": l} for s in L.SHAPES for t in L.TEXTURES for l in range(5)]
PAIRS = [(a, b) for a in KINDS for b in KINDS]
TRUTH = {c: np.array([L.RULES[c](a, b) for a, b in PAIRS]) for c in ("low", "medium")}
_pred = {}


def predictions(rule):
    if rule not in _pred:
        h = Hypothesis.from_raw({"rule": rule})
        _pred[rule] = None if h.error else np.array([h.predict(a, b) for a, b in PAIRS])
    return _pred[rule]


def game_stats(g, cond):
    """Behavioural measures for one game (rows of one player's actions)."""
    f = g[g.action == "combine"]
    ok = f["yield"].notna() & (f["yield"].astype(str) != "")
    c = g[g.action == "consume"]
    seen, rep = set(), 0
    for (_, r), s in zip(f.iterrows(), ok):
        k = (r.held, r.target)
        rep += k in seen
        if not s:
            seen.add(k)
    pair = (f.held.map(lambda n: SH.index(shape(n))) + f.target.map(lambda n: SH.index(shape(n)))) == 3
    return {
        "points": g.points.sum() if "total_points" not in g else g.total_points.iloc[-1],
        "max_level": max([lv(y) for y in f["yield"][ok]] or [0]),
        "fusions": len(f),
        "success_rate": ok.mean() if len(f) else np.nan,
        "harvests": len(c),
        "first_harvest": c.action_id.min() if len(c) else np.nan,
        "same_level_share": (f.held.map(lv) == f.target.map(lv)).mean() if len(f) else np.nan,
        "shape_pair_share": pair.mean() if len(f) else np.nan,
        "repeated_failures": rep,
    }


def belief_stats(g, cond):
    """Measures of the stated hypotheses (LLM + hypotheses / Thompson only)."""
    truth = TRUTH[cond]
    acc_top, acc_w, none, refuted, n_distinct, top_conf, top_right = [], [], [], [], [], [], []
    found = None
    for _, r in g.iterrows():
        hs = json.loads(r.hypotheses)
        preds = [predictions(h["rule"]) for h in hs]
        ok = [(h, p) for h, p in zip(hs, preds) if p is not None]
        if not ok:
            continue
        w = np.array([h["weight"] for h, _ in ok])
        P = np.array([p for _, p in ok])
        acc_top.append((P[0] == truth).mean())
        # weighted prediction: P(fuses) from the rules (none_of_these left out, renormalised)
        pw = (w[:, None] * P).sum(0) / w.sum() if w.sum() > 0 else P.mean(0)
        acc_w.append(((pw > 0.5) == truth).mean())
        none.append(r.weight_none_of_these)
        refuted.append(r.weight_on_refuted)
        n_distinct.append(len({p.tobytes() for p in P}))
        top_conf.append(hs[0].get("stated_confidence"))
        top_right.append(bool((P[0] == truth).all()))
        if found is None and any((p == truth).all() for p in P):
            found = r.action_id
    return {
        "top_rule_accuracy_end": acc_top[-1] if acc_top else np.nan,
        "top_rule_accuracy_mean": np.mean(acc_top) if acc_top else np.nan,
        "weighted_accuracy_end": acc_w[-1] if acc_w else np.nan,
        "none_of_these_end": none[-1] if none else np.nan,
        "weight_on_refuted_mean": np.mean(refuted) if refuted else np.nan,
        "distinct_rules_mean": np.mean(n_distinct) if n_distinct else np.nan,
        "top_confidence_end": top_conf[-1] if top_conf else np.nan,
        "top_rule_exact_end": top_right[-1] if top_right else False,
        "exact_rule_first_step": found,
        "top_rule_end": json.loads(g.hypotheses.iloc[-1])[0]["explanation"] if len(g) else "",
    }


def main():
    rows = []
    # humans
    h = pd.read_csv(HUMAN)
    for cond in ("low", "medium"):
        for pid, g in h[h.condition == cond].groupby("id"):
            rows.append({"who": "Human", "condition": cond, "game": f"p{pid}", **game_stats(g, cond)})
    # LLM runs
    for f in sorted(glob.glob(str(RUNS / "ef_*_actions.csv"))):
        name = Path(f).name  # ef_<cond>_<setup>_s<seeds>_<cond>_<stamp>_actions.csv
        _, cond, setup = name.split("_")[:3]
        a = pd.read_csv(f)
        for seed, g in a.groupby("id"):
            g = g.sort_values("action_id")
            rec = {"who": SETUP_NAMES[setup], "condition": cond, "game": f"seed{seed}", **game_stats(g, cond)}
            if "hypotheses" in g:
                rec.update(belief_stats(g, cond))
            if "acted_on" in g:
                rec["thompson_none_draws"] = g.acted_on.fillna("").str.startswith("none").sum()
                rec["free_tests"] = int(g.free_test.fillna(False).sum())
            rec["api_calls"] = g.api_calls.sum()
            rec["reasoning_tokens"] = g.reasoning_tokens.sum()
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(RUNS / "per_game.csv", index=False)

    # human percentile of each LLM game
    hum = df[df.who == "Human"]
    df["human_percentile"] = [
        (hum[hum.condition == r.condition].points < r.points).mean() * 100 if r.who != "Human" else np.nan
        for r in df.itertuples()]

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for cond in ("low", "medium"):
        d = df[df.condition == cond]
        print(f"\n==================== {cond.upper()} ====================")
        s = d.groupby("who").agg(
            n=("points", "size"),
            points_median=("points", "median"),
            points_q25=("points", lambda x: x.quantile(.25)),
            points_q75=("points", lambda x: x.quantile(.75)),
            points_max=("points", "max"),
            max_level_median=("max_level", "median"),
            success_rate=("success_rate", "median"),
            fusions=("fusions", "median"),
            harvests=("harvests", "median"),
            first_harvest=("first_harvest", "median"),
            same_level=("same_level_share", "median"),
            shape_pair=("shape_pair_share", "median"),
            repeats=("repeated_failures", "median"),
            human_pct=("human_percentile", "median"),
        ).reindex([o for o in ORDER if o in set(d.who)])
        print(s.round(2).to_string())
        b = d[d.who.isin(["LLM + hypotheses", "LLM + Thompson"])]
        if len(b):
            print("\n  beliefs (median over games):")
            cols = ["top_rule_accuracy_end", "weighted_accuracy_end", "top_confidence_end", "none_of_these_end",
                    "weight_on_refuted_mean", "distinct_rules_mean", "top_rule_exact_end", "exact_rule_first_step"]
            print(b.groupby("who")[cols].agg(lambda x: x.median() if x.dtype != bool else x.mean()).round(2).to_string())
        print("\n  per game:")
        cols = ["who", "game", "points", "max_level", "success_rate", "fusions", "harvests", "first_harvest", "same_level_share"]
        extra = [c for c in ("top_rule_accuracy_end", "top_confidence_end", "top_rule_exact_end", "thompson_none_draws") if c in d]
        print(d[d.who != "Human"][cols + extra].round(2).to_string(index=False))
    df.to_csv(RUNS / "per_game.csv", index=False)
    print(f"\nsaved {RUNS / 'per_game.csv'}")


if __name__ == "__main__":
    main()
