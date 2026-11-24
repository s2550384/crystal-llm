"""Which model of choice best explains the fusions an LLM (or anyone who states
hypotheses each turn) actually made?

Takes saved games from `llm_play.py --hypotheses K` (actor llm), replays each
one, and at every turn where the player fused asks each model how probable the
chosen pair was, given the hypotheses and probabilities stated on that turn.

Models. All score a fusion of x onto y by its expected stop value (as the
policy players do: p * S(if it works) + (1 - p) * S(if it fails), see
llm_play.policy_scores) and differ only in where p, the chance it works,
comes from:

    random      every available pair equally likely (no scores)
    value_only  p = the success rate so far for every untried pair: no rules
                at all, only what a success would be worth and what is known
    greedy      p from the single most probable hypothesis
    thompson    one hypothesis drawn by its probability; a mixture over them
    expected    p averaged over all the hypotheses by probability
    information the pair whose outcome would say most about which hypothesis is
                right: expected information gain in bits, no value at all
                (Nelson, 2005; used on LLM-written rules by Piriyakulkij et
                al., 2024)
    value_info  expected, plus omega * information: an information bonus as in
                Wilson et al. (2014). Two parameters (beta, omega); omega = 0
                is the expected model

Options. By default (--options all) every pair is an option on every turn.
With --options held the choice is the one a participant faced
in the browser game, where you carry a crystal: after a fusion you hold its
result (or, if it failed, the crystal you held), so the options are the pairs
of that crystal with each kind on the ground. Turns where the player fused
something else (in the browser: dropped it and picked up another) are left
out and counted; after a harvest, or on the first move, the hand is empty and
every pair is an option.

Pairs already tried use their known result in every model, and "none of
these" contributes the success rate so far (Laplace), as in the players.

Choice rule. Within a turn the scores are rescaled to run from 0 (worst pair)
to 1 (best pair), and the pair is chosen by softmax: P(pair) is proportional
to exp(beta * rescaled score). beta = 0 is random choice; a large beta always
takes the best pair. One beta is fitted per model by maximum likelihood (and
omega for value_info). Information gain is already on a 0-1 scale (bits), so
it is used as it is: for a pair whose chance of working is q under all the
stated beliefs, it is H(q) - w_none * H(p0), where H is binary entropy, since
each stated rule predicts the outcome for certain and only "none of these"
(success rate p0) does not.
Harvest turns are left out: this is about which pair is fused, given a fusion.

Reported per model: fitted beta, log-likelihood, BIC, McFadden's pseudo-R2
(share of the gap between random and perfect prediction that it closes), the
average probability it gave the chosen pair, how often its best pair was the chosen one, a
leave-one-game-out score (beta fitted on the other games), and how many games
it explains best. With --recovery N, choices are re-drawn N times from each
fitted model at the same turns to check the models can be told apart.

Usage (from the chris folder):
    python fit_choice.py results/early_findings/ef_*_hypo_*_actions.csv
    python fit_choice.py results/*_actions.csv --recovery 200 --out results/choice_fit
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import llm_play as L
from hypotheses import parse_rule

MODELS = ["random", "value_only", "greedy", "thompson", "expected", "information", "value_info"]
N_PARAMS = {"random": 0, "value_info": 2}  # every other model: 1 (beta)
BETA_MAX = 200.0
OMEGA_GRID = np.concatenate([[0.0], np.logspace(-1.5, 1.5, 13)])

_rule_cache = {}


def rule_fn(rule):
    """Compiled rule (h, t) -> bool, or None if it does not parse."""
    if rule not in _rule_cache:
        try:
            _rule_cache[rule] = parse_rule(rule)[0]
        except ValueError:
            _rule_cache[rule] = None
    return _rule_cache[rule]


def rescale(scores):
    lo, hi = scores.min(), scores.max()
    return np.zeros_like(scores) if hi - lo < 1e-12 else (scores - lo) / (hi - lo)


def entropy(x):
    """Binary entropy in bits."""
    x = np.clip(np.asarray(x, float), 1e-12, 1 - 1e-12)
    return -(x * np.log2(x) + (1 - x) * np.log2(1 - x))


def find_crystal(game, kind, exclude=None):
    for c in game.crystals.values():
        if L.kind_name(c) == kind and c["id"] != exclude:
            return c
    raise ValueError(f"no {kind} crystal on the board")


def turn_record(game, row, held_kind=None):
    """Everything the models need for one fusion turn, before the move is applied.
    held_kind limits the options to pairs with that crystal held. Returns None if
    there was no real choice (fewer than two available pairs)."""
    crystals = list(game.crystals.values())
    n = game.actions_left
    known = {(h, t): bool(y) for h, t, y in game.fusion_log}
    s_fail = L.stop_value(game, crystals, [], n - 1)
    options, seen = [], set()
    for h in crystals:
        for t in crystals:
            key = (L.kind_name(h), L.kind_name(t))
            if h is t or key in seen or (held_kind and key[0] != held_kind):
                continue
            seen.add(key)  # crystals of the same kinds are interchangeable
            level = max(h["level"], t["level"]) + 1
            rest = [c for c in crystals if c is not h and c is not t]
            options.append({"key": key, "h": h, "t": t, "capped": level >= L.MAXLEVEL,
                            "known": known.get(key),
                            "s_ok": L.stop_value(game, rest, [10 ** level], n - 1)})
    if len(options) < 2:
        return None
    chosen = [i for i, o in enumerate(options) if o["key"] == (row.held, row.target)]
    if not chosen:
        raise ValueError(f"chosen pair {row.held} + {row.target} is not available")
    s_ok = np.array([o["s_ok"] for o in options], float)

    def full_p(p_untried):
        """Chance each pair works: known results and the cap override the beliefs."""
        return np.array([0.0 if o["capped"] else (float(o["known"]) if o["known"] is not None else pu)
                         for o, pu in zip(options, p_untried)])

    def scores(p_untried):
        """Rescaled expected stop values, given p for each untried pair."""
        p = full_p(p_untried)
        return rescale(p * s_ok + (1 - p) * s_fail)

    # the stated beliefs: parsable rules with their weights, and none_of_these
    hyps = [h for h in json.loads(row.hypotheses) if not h.get("error") and rule_fn(h["rule"])]
    w = np.array([h.get("weight") or 0.0 for h in hyps], float)
    w_none = float(row.weight_none_of_these) if pd.notna(row.weight_none_of_these) else 0.0
    total = w.sum() + w_none
    if total <= 0:
        w, w_none, total = np.ones(len(hyps)), 1.0, len(hyps) + 1.0
    w, w_none = w / total, w_none / total
    p0 = L.laplace_rate(game)
    none_p = np.full(len(options), p0)
    rule_p = [np.array([float(rule_fn(h["rule"])(o["h"], o["t"])) for o in options]) for h in hyps]

    comps = {"random": [(1.0, np.zeros(len(options)))],
             "value_only": [(1.0, scores(none_p))]}
    all_p, all_w = rule_p + [none_p], list(w) + [w_none]
    comps["greedy"] = [(1.0, scores(all_p[int(np.argmax(all_w))]))]
    comps["thompson"] = [(wi, scores(pi)) for wi, pi in zip(all_w, all_p) if wi > 0]
    mean_p = sum(wi * pi for wi, pi in zip(all_w, all_p))
    comps["expected"] = [(1.0, scores(mean_p))]
    # expected information gain (bits) about which hypothesis is right; 0 for pairs
    # whose outcome is already known (tried before, or over the level cap)
    q = full_p(mean_p)
    open_ = np.array([not o["capped"] and o["known"] is None for o in options])
    info = np.where(open_, np.maximum(entropy(q) - w_none * entropy(p0), 0.0), 0.0)
    comps["information"] = [(1.0, info)]
    o = options[chosen[0]]
    kind = ("capped" if o["capped"] else "untried" if o["known"] is None
            else "known to work" if o["known"] else "known to fail")
    return {"chosen": chosen[0], "n_options": len(options), "comps": comps, "pair_was": kind,
            "info_chosen": float(info[chosen[0]]), "info_max": float(info.max())}


SKIPPED = {"switched": 0}  # fusion turns left out under --options held


def replay(path, max_actions, options="all"):
    """Replay every game in an actions CSV; yield one record per fusion turn."""
    df = pd.read_csv(path)
    if "hypotheses" not in df:
        raise SystemExit(f"{path}: no hypotheses column (was it run with --hypotheses?)")
    if "actor" in df and df["actor"].isin(["thompson", "top", "expected"]).any():
        print(f"[note] {Path(path).name}: moves were chosen by a policy, not the model; skipped")
        return
    for seed, g in df.groupby("id"):
        g = g.sort_values("action_id").rename(columns={"yield": "yld"})
        cond = g.condition.iloc[0]
        human = "moves_from" in g and (g["moves_from"] == "human").any()
        game = L.CrystalGame(cond, seed=int(seed), max_actions=max_actions, **(L.HUMAN_REGEN if human else {}))
        game_id = f"{Path(path).name.split('_2026')[0]}|{cond}|seed{seed}"
        carried = None  # kind in hand: the last fusion's result, or its held crystal if it failed
        for row in g.itertuples():
            if row.action == "combine":
                held_kind = carried if options == "held" else None
                switched = held_kind is not None and row.held != held_kind
                SKIPPED["switched"] += switched
                rec = None if switched else turn_record(game, row, held_kind)
                h = find_crystal(game, row.held)
                t = find_crystal(game, row.target, exclude=h["id"])
                res = game.step(("fuse", h["id"], t["id"]))
                want = row.yld if isinstance(row.yld, str) else ""
                if (res["yield"] or "") != want:
                    raise ValueError(f"{game_id} action {row.action_id}: replay gave {res['yield']!r}, log has {want!r}")
                carried = res["yield"] or row.held
                if rec:
                    rec.update(game=game_id, condition=cond, action_id=row.action_id)
                    yield rec
            else:
                game.step(("harvest", find_crystal(game, row.held)["id"]))
                carried = None
        if game.points != g.total_points.iloc[-1]:
            raise ValueError(f"{game_id}: replay ended on {game.points} points, log has {g.total_points.iloc[-1]}")


def components(t, model, params):
    """[(weight, score vector)] for one turn: the softmax is taken over each vector."""
    if model == "value_info":
        return [(1.0, t["comps"]["expected"][0][1] + params[1] * t["comps"]["information"][0][1])]
    return t["comps"][model]


def dist(t, model, params):
    """P(each available pair) at one turn under a model. params = (beta,) or (beta, omega)."""
    p = np.zeros(t["n_options"])
    for w, u in components(t, model, params):
        z = params[0] * u
        e = np.exp(z - z.max())
        p += w * e / e.sum()
    return p


def turn_probs(turns, model, params):
    """P(chosen pair) at each turn under a model."""
    return np.array([dist(t, model, params)[t["chosen"]] for t in turns])


def loglik(turns, model, params):
    return float(np.log(np.maximum(turn_probs(turns, model, params), 1e-300)).sum())


def maximise(f, grid, refine=18):
    """Maximum of f over a 1-D grid, refined by golden-section search between the neighbours."""
    vals = [f(x) for x in grid]
    k = int(np.argmax(vals))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = f(a), f(b)
    for _ in range(refine):
        if fa < fb:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = f(b)
        else:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = f(a)
    x = (lo + hi) / 2
    fx = f(x)
    return (float(x), fx) if fx >= vals[k] else (float(grid[k]), vals[k])


BETA_GRID = np.concatenate([[0.0], np.logspace(-2, math.log10(BETA_MAX), 36)])


def fit(turns, model):
    """Maximum-likelihood parameters: (beta,), or (beta, omega) for value_info."""
    if model == "random" or not turns:
        return (0.0,)
    if model != "value_info":
        return (maximise(lambda b: loglik(turns, model, (b,)), BETA_GRID)[0],)

    def best_beta(omega):
        return maximise(lambda b: loglik(turns, model, (b, omega)), BETA_GRID, refine=10)
    omega, _ = maximise(lambda om: best_beta(om)[1], OMEGA_GRID, refine=8)
    return (best_beta(omega)[0], omega)


def agreement(turns, model, params):
    """How often the model's best pair is the chosen one (shared equally among ties)."""
    if model == "random":
        return float(np.mean([1 / t["n_options"] for t in turns]))
    tot = 0.0
    for t in turns:
        for w, u in components(t, model, params):
            best = np.flatnonzero(u >= u.max() - 1e-9)
            tot += w * (t["chosen"] in best) / len(best)
    return tot / len(turns)


def analyse(turns, label):
    games = sorted({t["game"] for t in turns})
    by_game = {gm: [t for t in turns if t["game"] == gm] for gm in games}
    ll_random = loglik(turns, "random", (0.0,))
    rows, per_turn, fitted = [], {}, {}
    for m in MODELS:
        params = fitted[m] = fit(turns, m)
        probs = per_turn[m] = turn_probs(turns, m, params)
        cv = 0.0  # leave one game out: parameters from the other games
        for gm in games:
            others = [t for t in turns if t["game"] != gm]
            cv += loglik(by_game[gm], m, fit(others, m) if len(games) > 1 else params)
        ll, k = float(np.log(probs).sum()), N_PARAMS.get(m, 1)
        rows.append({"data": label, "model": m, "beta": params[0], "omega": params[1] if len(params) > 1 else np.nan,
                     "n_params": k, "loglik": ll, "bic": k * math.log(len(turns)) - 2 * ll,
                     "pseudo_r2": 1 - ll / ll_random, "held_out_loglik": cv, "mean_prob": float(probs.mean()),
                     "agreement": agreement(turns, m, params)})
    res = pd.DataFrame(rows)
    game_ll = pd.DataFrame({m: [float(np.log(per_turn[m][[t["game"] == gm for t in turns]]).sum()) for gm in games]
                            for m in MODELS}, index=games)
    res["games_best"] = [int((game_ll.idxmax(axis=1) == m).sum()) for m in MODELS]
    res["n_fusions"], res["n_games"] = len(turns), len(games)
    return res, per_turn, game_ll, fitted


def recovery(turns, fitted, n_sims, rng):
    """Re-draw the choices from each fitted model at the same turns and refit all the
    models: rows = the model that made the choices, columns = how often each then had
    the lowest BIC."""
    wins = pd.DataFrame(0, index=MODELS, columns=MODELS)
    for true in MODELS:
        ps = [dist(t, true, fitted[true]) for t in turns]
        for _ in range(n_sims):
            sim = [{**t, "chosen": int(rng.choice(len(p), p=p / p.sum()))} for t, p in zip(turns, ps)]
            bic = {m: N_PARAMS.get(m, 1) * math.log(len(sim)) - 2 * loglik(sim, m, fit(sim, m)) for m in MODELS}
            wins.loc[true, min(bic, key=bic.get)] += 1
    return wins / n_sims


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", help="actions CSVs from llm_play.py --hypotheses K (globs allowed)")
    ap.add_argument("--actions", type=int, default=L.MAX_ACTIONS, help="actions per game in those runs")
    ap.add_argument("--recovery", type=int, default=0, metavar="N", help="model-recovery check with N simulations per model")
    ap.add_argument("--options", choices=["all", "held"], default="all",
                    help="all (default): every pair on every turn; held: only pairs with the crystal "
                         "in hand, as in the browser game")
    ap.add_argument("--out", default=None, help="prefix for the CSVs to save (default: choice_fit next to the first file)")
    args = ap.parse_args()

    files = sorted({f for pat in args.files for f in (glob.glob(pat) or [pat])})
    turns = [t for f in files for t in replay(f, args.actions, args.options)]
    if not turns:
        raise SystemExit("no fusion turns found")
    out = args.out or str(Path(files[0]).parent / "choice_fit")
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    kinds = pd.Series([t["pair_was"] for t in turns]).value_counts(normalize=True)
    print(f"{len(turns)} fusion choices in {len({t['game'] for t in turns})} games; "
          f"{np.mean([t['n_options'] for t in turns]):.0f} pairs to choose from on average")
    print("the chosen pair was: " + ", ".join(f"{k} {v:.0%}" for k, v in kinds.items()))
    if args.options == "held":
        n_sw = SKIPPED["switched"]
        print(f"options: pairs with the crystal in hand; {n_sw} fusion turns where another crystal was "
              f"fused left out ({n_sw / (n_sw + len(turns)):.0%} of fusions)")

    summaries, turn_rows = [], []
    datasets = [("all", turns)] + [(c, [t for t in turns if t["condition"] == c])
                                   for c in sorted({t["condition"] for t in turns})]
    datasets = datasets if len(datasets) > 2 else datasets[:1]
    for label, ts in datasets:
        res, per_turn, game_ll, fitted = analyse(ts, label)
        summaries.append(res)
        print(f"\n=== {label}: {len(ts)} fusions, {res.n_games.iloc[0]} games ===")
        print(res[["model", "beta", "omega", "loglik", "bic", "pseudo_r2", "held_out_loglik", "mean_prob",
                   "agreement", "games_best"]].to_string(index=False, na_rep=""))
        if label == "all":
            best = res.sort_values("bic").model.iloc[0]
            rng = np.random.default_rng(0)
            idx = np.array([list(game_ll.index).index(t["game"]) for t in ts])
            print(f"\nlog-likelihood of {best} minus each other model (95% interval, resampling games):")
            for m in MODELS:
                if m == best:
                    continue
                d = np.log(per_turn[best]) - np.log(per_turn[m])
                per_game = np.array([d[idx == k].sum() for k in range(len(game_ll))])
                boots = [per_game[rng.integers(0, len(per_game), len(per_game))].sum() for _ in range(5000)]
                print(f"  vs {m:10s} {d.sum():+8.1f}  [{np.percentile(boots, 2.5):+.1f}, {np.percentile(boots, 97.5):+.1f}]")
            for i, t in enumerate(ts):
                turn_rows.append({"game": t["game"], "condition": t["condition"], "action_id": t["action_id"],
                                  "n_options": t["n_options"], "pair_was": t["pair_was"],
                                  "info_chosen": t["info_chosen"], "info_max": t["info_max"],
                                  **{f"p_{m}": per_turn[m][i] for m in MODELS}})
            game_ll.to_csv(out + "_games.csv")
            if args.recovery:
                print(f"\nmodel recovery ({args.recovery} simulations per model; rows = model that made the choices, "
                      "columns = share of simulations in which each model then had the lowest BIC):")
                rec = recovery(ts, fitted, args.recovery, np.random.default_rng(1))
                print(rec.to_string())
                rec.to_csv(out + "_recovery.csv")
    pd.concat(summaries).to_csv(out + "_summary.csv", index=False)
    pd.DataFrame(turn_rows).to_csv(out + "_turns.csv", index=False)
    print(f"\nsaved {out}_summary.csv, {out}_turns.csv, {out}_games.csv")


if __name__ == "__main__":
    main()
