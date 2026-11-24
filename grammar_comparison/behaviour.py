"""Fit each grammar, as a Bayesian learner, to human combine choices.

Choice model: at each combine, given the crystal the participant holds and the crystal
kinds on the board (reconstructed from the action log), the participant picks the
target with probability  softmax(beta * P(success | history, grammar prior)).
P(success) is the posterior predictive under a prior over rules sampled from the
grammar, with a noisy likelihood (each outcome agrees with the rule w.p. 1-eps).
"""
import math, random, time, sys, json
from collections import Counter
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from domain import (NK, NP, kind_from_name, kind_index, CAP, TRUE_RULES, CONDITION_MAP, from_int, KL)
import grammars as G
from codes import CODES

rng_seed = 0
N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
U_MAX = 6000
EPS_GRID = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45]

# ---------------------------------------------------------------------------
# 1. reconstruct boards and build choice events
# ---------------------------------------------------------------------------
act = pd.read_csv("../../data/action_data.csv")
act = act.rename(columns={"yield": "yld"}).sort_values(["id", "action_id"])
participants = []
miss_t = miss_h = n_comb = 0
for pid, g in act.groupby("id"):
    cond = CONDITION_MAP[g.condition.iloc[0]]
    board = Counter(kind_index(s, t, 0) for s in range(4) for t in range(4))
    regen_done = set()
    events = []
    for r in g.itertuples():
        h = kind_from_name(r.held)
        if board[h] == 0:
            miss_h += r.action == "combine"; board[h] += 1
        if r.action == "combine":
            n_comb += 1
            t = kind_from_name(r.target)
            avail = board.copy(); avail[h] -= 1
            cand = sorted(k for k, c in avail.items() if c > 0)
            if t not in cand:
                miss_t += 1; cand = sorted(set(cand) | {t})
            ok = isinstance(r.yld, str) and len(r.yld) > 0
            events.append((h, np.array(cand), cand.index(t), h * NK + t, ok))
            if ok:
                board[h] -= 1; board[t] -= 1; board[kind_from_name(r.yld)] += 1
                for k in (h, t):
                    if KL[k] == 0 and k not in regen_done:
                        regen_done.add(k); board[k] += 1
        else:  # consume
            board[h] -= 1
            if KL[h] == 0 and h not in regen_done:
                regen_done.add(h); board[h] += 1
    participants.append((pid, cond, events))
print(f"{len(participants)} participants, {n_comb} combines; target not on reconstructed board: {miss_t}, held missing: {miss_h}")
mean_cand = np.mean([len(e[1]) for _, _, ev in participants for e in ev])
print(f"mean candidate targets per choice: {mean_cand:.1f}")

# ---------------------------------------------------------------------------
# 2. hypothesis spaces
# ---------------------------------------------------------------------------
def space(sampler, n=N_SAMPLES, seed=0):
    r = random.Random(seed); cnt = Counter(); deep = 0
    for _ in range(n):
        try:
            cnt[sampler(r)] += 1
        except G.TooDeep:
            deep += 1
    top = cnt.most_common(U_MAX)
    covered = sum(c for _, c in top) / (n - deep)
    M = np.stack([from_int(x) for x, _ in top]).astype(np.uint8)
    prior = np.array([c for _, c in top], float); prior /= prior.sum()
    return dict(M=M, logprior=np.log(prior), unique=len(cnt), covered=covered)


def theta_loco(held_out):
    tot = dict.fromkeys(G.THETA0, 0)
    for pid, (cat, mask, _) in CODES.items():
        pass
    msg = pd.read_csv("../../data/message_data.csv")
    for rr in msg.itertuples():
        cat, mask, _ = CODES[rr.id]
        if mask is None or CONDITION_MAP[rr.condition] == held_out:
            continue
        c = G.g2_derivation_counts(G.cells(mask))
        if c:
            for k in tot: tot[k] += c[k]
    return G.theta_from_counts(tot)


t0 = time.time()
SPACES = {
    "G1 feature-Boolean": space(G.sample_g1),
    "G2 Zhao-Arumugam (uniform)": space(G.sample_g2),
    "G3 relational DNF": space(lambda r: G.sample_g3(r, True)),
    "G3 relational DNF, no colour": space(lambda r: G.sample_g3(r, False)),
}
THETAS = {c: theta_loco(c) for c in ["high", "medium", "low"]}
def theta3_loco(held_out):
    tot = None
    msg = pd.read_csv("../../data/message_data.csv")
    for rr in msg.itertuples():
        cat, mask, _ = CODES[rr.id]
        if mask is None or CONDITION_MAP[rr.condition] == held_out:
            continue
        c = G.g3_derivation_counts(G.cells(mask))
        if c:
            tot = c if tot is None else {k: tot[k] + c[k] for k in tot}
    return G.g3_theta_from_counts(tot)


THETAS3 = {c: theta3_loco(c) for c in ["high", "medium", "low"]}
for c in THETAS3:
    SPACES[f"G3 fitted theta [{c}]"] = space(lambda r, th=THETAS3[c]: G.sample_g3w(r, th), seed=2)
for c in THETAS:
    SPACES[f"G2 fitted theta [{c}]"] = space(lambda r, th=THETAS[c]: G.sample_g2(r, th), seed=1)
print(f"spaces built in {time.time()-t0:.0f}s")
for k, v in SPACES.items():
    tr = {c: bool((v['M'] == TRUE_RULES[c].astype(np.uint8)).all(1).any()) for c in TRUE_RULES}
    print(f"  {k:32s} unique={v['unique']:7d} top-{U_MAX} mass={v['covered']:.3f} contains true rule: {tr}")

# ---------------------------------------------------------------------------
# 3. predictive success probabilities for each choice
# ---------------------------------------------------------------------------
def predictive(sp, eps, only=None):
    M, lp0 = sp["M"], sp["logprior"]
    Mf = M.astype(np.float32)
    lok, lbad = math.log(1 - eps), math.log(eps)
    out = []
    for pid, cond, ev in participants:
        if only and cond != only: continue
        ll = np.zeros(len(lp0))
        for h, cand, ci, pair, ok in ev:
            w = lp0 + ll; w = np.exp(w - w.max()); w /= w.sum()
            pred = w.astype(np.float32) @ Mf[:, h * NK + cand]
            out.append((pid, cond, pred, ci))
            if not CAP[pair]:
                agree = M[:, pair] == (1 if ok else 0)
                ll += np.where(agree, lok, lbad)
    return out


def pairwise_pred(p0, kappa=1.0):
    out = []
    for pid, cond, ev in participants:
        s = Counter(); n = Counter()
        for h, cand, ci, pair, ok in ev:
            key = lambda t: (h // 6, t // 6)  # shape-texture kinds, ignoring level
            pred = np.array([(p0 * kappa + s[key(t)]) / (kappa + n[key(t)]) for t in cand], np.float32)
            out.append((pid, cond, pred, ci))
            if not CAP[pair]:
                n[key(pair % NK)] += 1; s[key(pair % NK)] += ok
    return out


def oracle_pred():
    out = []
    for pid, cond, ev in participants:
        rule = TRUE_RULES[cond]
        for h, cand, ci, pair, ok in ev:
            out.append((pid, cond, rule[h * NK + cand].astype(np.float32), ci))
    return out


def fit_beta(preds):
    def nll(b):
        s = 0.0
        for _, _, p, ci in preds:
            z = b * p; z = z - z.max(); s -= z[ci] - math.log(np.exp(z).sum())
        return s
    res = minimize_scalar(nll, bounds=(0, 60), method="bounded")
    return res.x, -res.fun


def per_participant_ll(preds, b):
    ll = Counter()
    for pid, cond, p, ci in preds:
        z = b * p; z = z - z.max(); ll[pid] += z[ci] - math.log(np.exp(z).sum())
    return ll


results = {}
n_choices = sum(len(ev) for _, _, ev in participants)
rand_ll = -sum(math.log(len(e[1])) for _, _, ev in participants for e in ev)
results["Random"] = dict(LL=rand_ll, k=0, params="-")
pp_rand = Counter()
for pid, cond, ev in participants:
    pp_rand[pid] = -sum(math.log(len(e[1])) for e in ev)
PER = {"Random": pp_rand}

for name, sp in SPACES.items():
    if "fitted theta" in name:
        continue
    best = None
    for eps in EPS_GRID:
        pr = predictive(sp, eps)
        b, ll = fit_beta(pr)
        if best is None or ll > best[0]:
            best = (ll, eps, b, pr)
    results[name] = dict(LL=best[0], k=2, params=f"eps={best[1]}, beta={best[2]:.2f}")
    PER[name] = per_participant_ll(best[3], best[2])
    print(f"{name:32s} LL={best[0]:.1f}  {results[name]['params']}  ({time.time()-t0:.0f}s)")

# fitted-theta G2: each condition predicted with theta estimated from the OTHER conditions' messages
best = None
for eps in EPS_GRID:
    pr = []
    for c in ["high", "medium", "low"]:
        pr += predictive(SPACES[f"G2 fitted theta [{c}]"], eps, only=c)
    b, ll = fit_beta(pr)
    if best is None or ll > best[0]:
        best = (ll, eps, b, pr)
results["G2 Zhao-Arumugam (theta from messages)"] = dict(LL=best[0], k=2, params=f"eps={best[1]}, beta={best[2]:.2f}")
PER["G2 Zhao-Arumugam (theta from messages)"] = per_participant_ll(best[3], best[2])
best = None
for eps in EPS_GRID:
    pr = []
    for c in ["high", "medium", "low"]:
        pr += predictive(SPACES[f"G3 fitted theta [{c}]"], eps, only=c)
    b, ll = fit_beta(pr)
    if best is None or ll > best[0]:
        best = (ll, eps, b, pr)
results["G3 relational DNF (theta from messages)"] = dict(LL=best[0], k=2, params=f"eps={best[1]}, beta={best[2]:.2f}")
PER["G3 relational DNF (theta from messages)"] = per_participant_ll(best[3], best[2])

best = None
for p0 in [0.05, 0.1, 0.2, 0.3, 0.5]:
    pr = pairwise_pred(p0); b, ll = fit_beta(pr)
    if best is None or ll > best[0]: best = (ll, p0, b, pr)
results["Pairwise (no rules)"] = dict(LL=best[0], k=2, params=f"p0={best[1]}, beta={best[2]:.2f}")
PER["Pairwise (no rules)"] = per_participant_ll(best[3], best[2])
pr = oracle_pred(); b, ll = fit_beta(pr)
results["Oracle (knows true rule)"] = dict(LL=ll, k=1, params=f"beta={b:.2f}")
PER["Oracle (knows true rule)"] = per_participant_ll(pr, b)

# ---------------------------------------------------------------------------
# 4. summary
# ---------------------------------------------------------------------------
cond_of = {pid: cond for pid, cond, _ in participants}
rows = []
for name, r in results.items():
    bic = -2 * r["LL"] + r["k"] * math.log(n_choices)
    row = dict(model=name, LL=round(r["LL"], 1), BIC=round(bic, 1), params=r["params"],
               mean_p_choice=round(math.exp(r["LL"] / n_choices), 4))
    for c in ["high", "medium", "low"]:
        row[f"LL_{c}"] = round(sum(v for pid, v in PER[name].items() if cond_of[pid] == c), 1)
    rows.append(row)
summary = pd.DataFrame(rows).sort_values("BIC")
print(f"\nchoices={n_choices}")
print(summary.to_string(index=False))
summary.to_csv("behaviour_fit.csv", index=False)

# per-participant: which grammar fits best (grammar models only)
gnames = [n for n in PER if n.startswith("G")]
win = Counter()
wins_by_cond = Counter()
for pid in cond_of:
    w = max(gnames, key=lambda n: PER[n][pid]); win[w] += 1; wins_by_cond[(cond_of[pid], w)] += 1
print("\nbest-fitting grammar per participant:", dict(win))
print({f"{c}|{w}": v for (c, w), v in sorted(wins_by_cond.items())})
pd.DataFrame({n: pd.Series(PER[n]) for n in PER}).assign(condition=pd.Series(cond_of)).to_csv("per_participant_ll.csv")
json.dump({"G2": THETAS, "G3": THETAS3}, open("theta_from_messages.json", "w"), indent=1)

# ---------------------------------------------------------------------------
# 5. bootstrap over participants: difference in LL vs G2 uniform (params held fixed)
# ---------------------------------------------------------------------------
pids = list(cond_of)
ref = "G2 Zhao-Arumugam (uniform)"
rb = np.random.default_rng(0)
boot = {n: [] for n in PER if n != ref}
for _ in range(5000):
    samp = rb.choice(pids, len(pids))
    for n in boot:
        boot[n].append(sum(PER[n][p] - PER[ref][p] for p in samp))
print("\nLL difference vs", ref, "(positive = better), 95% bootstrap CI over participants:")
for n, v in boot.items():
    lo, hi = np.percentile(v, [2.5, 97.5]); print(f"  {n:40s} {np.mean(v):8.1f}  [{lo:7.1f}, {hi:7.1f}]")
print("\ntheta estimated from messages (held-out condition -> theta):")
for c, th in THETAS.items():
    print(" ", c, {k: round(v, 3) for k, v in th.items()})
for c, th in THETAS3.items():
    print(" G3", c, {k: round(v, 3) for k, v in th.items()})
