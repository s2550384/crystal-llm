"""Three candidate hypothesis grammars, as probabilistic context-free grammars (PCFGs).

G1 'feature-Boolean'  - the grammar in agent/hypotheses.py: and/or/not over single-object
                        feature tests (shape, texture, colour of held h or target t).
                        E -> and(E,E) | or(E,E) | not(E) | Test           (1/4 each)
                        Test -> is_shape(o,x) | is_texture(o,y) | is_colour(o,c), o in {h,t}  (1/6 each)
G2 'Zhao-Arumugam'    - the PCFG in Bonan's modelling notes (Table 1), uniform theta:
                        S -> A | B | and(A,B);  A -> sameShape | diffShape | C;  B likewise with D
                        C -> or(C,C) | isShape(o1,x) | isShape(o2,x) | and(isShape(o1,x),isShape(o2,y)) | pairShape(x,y)
G3 'relational DNF'   - proposed: Goodman et al. (2008)-style disjunctive normal form whose
                        literals include relations (same/different shape, texture, colour),
                        single-object tests and unordered pairs.
                        S -> C | or(S,S) (1/2 each);  C -> L | and(L,C) (1/2 each)
                        L -> 11 literal types (8 without colour)

Two things are computed per grammar:
  * the minimal description cost (negative log prior, nats) of a given rule;
  * Monte Carlo samples of rules (as 9216-bit sets over held/target kind pairs).
"""
import itertools, math, random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from domain import SH, ST, TH, TT, LH, LT, NS, NT, NL, to_int

L = math.log
# ---------------------------------------------------------------------------
# colour-free 256-cell view (held shape, held texture, target shape, target texture)
# ---------------------------------------------------------------------------
LEVEL0 = (LH == 0) & (LT == 0)
CELL_IDX = np.where(LEVEL0)[0]                 # 256 pair indices at level 0


def cells(mask):
    """9216 mask -> 256-bit int over level-0 cells (rules here ignore colour)."""
    return to_int_small(mask[CELL_IDX])


def to_int_small(b):
    return int("".join("1" if x else "0" for x in b[::-1]), 2)


ALL256 = (1 << 256) - 1
c_SH, c_ST, c_TH, c_TT = SH[CELL_IDX], ST[CELL_IDX], TH[CELL_IDX], TT[CELL_IDX]


def cmask(b):
    return to_int_small(b)


def mincover(target, pool, extra_per_clause=0.0):
    """Min-cost set of clauses (mask, cost) whose union is exactly `target`.
    Only clauses that are subsets of target are allowed. Returns (cost, chosen) or (inf, None)."""
    cand = [(m, c) for m, c in pool if m and (m & ~target) == 0]
    if not cand:
        return math.inf, None
    best = {}
    for m, c in cand:
        if m not in best or c < best[m]:
            best[m] = c
    cand = list(best.items())
    bits = [i for i in range(256) if target >> i & 1]
    A = np.zeros((len(bits), len(cand)))
    for j, (m, _) in enumerate(cand):
        for r, i in enumerate(bits):
            if m >> i & 1:
                A[r, j] = 1
    if (A.sum(1) == 0).any():
        return math.inf, None
    c = np.array([cc + extra_per_clause for _, cc in cand])
    res = milp(c, constraints=LinearConstraint(A, lb=1, ub=np.inf), integrality=np.ones(len(cand)),
               bounds=Bounds(0, 1))
    if not res.success:
        return math.inf, None
    chosen = [cand[j] for j in range(len(cand)) if res.x[j] > 0.5]
    return float(res.fun), chosen


def clauses(lits, kmax):
    out = []
    for k in range(1, kmax + 1):
        for combo in itertools.combinations(lits, k):
            m = ALL256
            for lm, _, _ in combo:
                m &= lm
            if m:
                out.append((m, k, sum(c for _, c, _ in combo), [n for _, _, n in combo]))
    return out


# ---------------------------------------------------------------------------
# G1: feature-Boolean (Chris's current grammar)
# ---------------------------------------------------------------------------
G1_NODE = L(4)                         # choosing and / or / not / Test
G1_TEST = L(4) + L(6) + L(4)           # E->Test, test type (6), value (4 shapes or textures)


def g1_literals():
    lits = []
    for o, (sv, tv) in {"h": (c_SH, c_TH), "t": (c_ST, c_TT)}.items():
        for x in range(NS):
            m = cmask(sv == x); lits += [(m, G1_TEST, f"is_shape({o},{x})"), (ALL256 & ~m, G1_TEST + G1_NODE, f"not is_shape({o},{x})")]
        for y in range(NT):
            m = cmask(tv == y); lits += [(m, G1_TEST, f"is_tex({o},{y})"), (ALL256 & ~m, G1_TEST + G1_NODE, f"not is_tex({o},{y})")]
    # relational macros spelled out in G1: same = or of 4 and(test,test)
    same = 3 * G1_NODE + 4 * (G1_NODE + 2 * G1_TEST)
    for name, m in [("sameShape", cmask(c_SH == c_ST)), ("sameTex", cmask(c_TH == c_TT))]:
        lits += [(m, same, name + "*"), (ALL256 & ~m, same + G1_NODE, "not " + name + "*")]
    return lits


def g1_cost_dnf(target, kmax=4):
    pool = [(m, (k - 1) * G1_NODE + c) for m, k, c, _ in clauses(G1_LITS, kmax)]
    cost, chosen = mincover(target, pool, extra_per_clause=G1_NODE)   # one 'or' per extra clause
    return cost - G1_NODE if chosen else math.inf


# ---------------------------------------------------------------------------
# G2: Zhao & Arumugam PCFG
# ---------------------------------------------------------------------------
def g2_dim_cost(f16, theta):
    """f16: 16-bit set over (held value, target value). Cost of A (or B) deriving it."""
    full = (1 << 16) - 1
    same = sum(1 << (a * 4 + a) for a in range(4))
    opts = []
    if f16 == same:
        opts.append(-L(theta["A_same"]))
    if f16 == full & ~same:
        opts.append(-L(theta["A_diff"]))
    # C: or-tree over atoms
    atoms = []
    for x in range(4):
        atoms.append((sum(1 << (x * 4 + b) for b in range(4)), -L(theta["C_o1"]) + L(4)))
        atoms.append((sum(1 << (a * 4 + x) for a in range(4)), -L(theta["C_o2"]) + L(4)))
        for y in range(4):
            atoms.append((1 << (x * 4 + y), -L(max(theta["C_pair"], theta["C_both"])) + 2 * L(4)))
    # small exact search: min cost cover of f16 with atoms that are subsets of f16
    cand = [(m, c) for m, c in atoms if m and (m & ~f16) == 0]
    best = {0: 0.0}
    frontier = [0]
    for _ in range(16):
        new = []
        for s in frontier:
            for m, c in cand:
                t = s | m
                v = best[s] + c - L(theta["C_or"])  # each extra leaf adds one 'or' node
                if t not in best or v < best[t] - 1e-12:
                    best[t] = v; new.append(t)
        frontier = new
        if not frontier:
            break
    if f16 in best and f16:
        opts.append(-L(theta["A_C"]) + best[f16] + L(theta["C_or"]))  # k leaves -> k-1 or nodes
    return min(opts) if opts else math.inf


THETA0 = dict(S_A=1 / 3, S_B=1 / 3, S_AB=1 / 3, A_same=1 / 3, A_diff=1 / 3, A_C=1 / 3,
              C_or=1 / 5, C_o1=1 / 5, C_o2=1 / 5, C_both=1 / 5, C_pair=1 / 5)


def g2_split(target):
    """If target (256 cells) = f(shapes) AND g(textures), return (f16, g16); else None."""
    f = g = 0
    for i in range(256):
        if target >> i & 1:
            f |= 1 << (int(c_SH[i]) * 4 + int(c_ST[i]))
            g |= 1 << (int(c_TH[i]) * 4 + int(c_TT[i]))
    rebuilt = 0
    for i in range(256):
        if (f >> (int(c_SH[i]) * 4 + int(c_ST[i])) & 1) and (g >> (int(c_TH[i]) * 4 + int(c_TT[i])) & 1):
            rebuilt |= 1 << i
    return (f, g) if rebuilt == target else None


def g2_cost(target, theta=THETA0):
    sp = g2_split(target)
    if sp is None or target == 0:
        return math.inf
    f, g = sp
    full = (1 << 16) - 1
    opts = []
    if g == full:
        opts.append(-L(theta["S_A"]) + g2_dim_cost(f, theta))
    if f == full:
        opts.append(-L(theta["S_B"]) + g2_dim_cost(g, theta))
    opts.append(-L(theta["S_AB"]) + g2_dim_cost(f, theta) + g2_dim_cost(g, theta))
    return min(opts)


def g2_derivation_counts(target):
    """Production counts of the minimal G2 derivation (for estimating theta from messages)."""
    sp = g2_split(target)
    if sp is None:
        return None
    f, g = sp
    full = (1 << 16) - 1
    cnt = dict.fromkeys(THETA0, 0)

    def dim(f16):
        same = sum(1 << (a * 4 + a) for a in range(4))
        if f16 == same:
            cnt["A_same"] += 1; return
        if f16 == full & ~same:
            cnt["A_diff"] += 1; return
        cnt["A_C"] += 1
        # greedy decomposition into atoms for counting purposes
        rem = f16
        rows = [sum(1 << (x * 4 + b) for b in range(4)) for x in range(4)]
        colsm = [sum(1 << (a * 4 + x) for a in range(4)) for x in range(4)]
        k = 0
        for r in rows:
            if r & f16 == r and r & rem:
                cnt["C_o1"] += 1; rem &= ~r; k += 1
        for c in colsm:
            if c & f16 == c and c & rem:
                cnt["C_o2"] += 1; rem &= ~c; k += 1
        npair = bin(rem).count("1")
        cnt["C_pair"] += npair; k += npair
        cnt["C_or"] += max(k - 1, 0)

    if g == full:
        cnt["S_A"] += 1; dim(f)
    elif f == full:
        cnt["S_B"] += 1; dim(g)
    else:
        cnt["S_AB"] += 1; dim(f); dim(g)
    return cnt


def theta_from_counts(cnt, alpha=1.0):
    th = {}
    for grp in (["S_A", "S_B", "S_AB"], ["A_same", "A_diff", "A_C"], ["C_or", "C_o1", "C_o2", "C_both", "C_pair"]):
        tot = sum(cnt[k] + alpha for k in grp)
        for k in grp:
            th[k] = (cnt[k] + alpha) / tot
    th["C_or"] = min(th["C_or"], 0.45)   # keep the grammar proper (finite expected size)
    return th


# ---------------------------------------------------------------------------
# G3: relational DNF (proposed)
# ---------------------------------------------------------------------------
def g3_literals(colour=True):
    ntypes = 11 if colour else 8
    base = L(ntypes)
    lits = [(cmask(c_SH == c_ST), base, "sameShape"), (cmask(c_SH != c_ST), base, "diffShape"),
            (cmask(c_TH == c_TT), base, "sameTex"), (cmask(c_TH != c_TT), base, "diffTex")]
    for o, (sv, tv) in {"h": (c_SH, c_TH), "t": (c_ST, c_TT)}.items():
        for x in range(4):
            lits.append((cmask(sv == x), base + L(2) + L(4), f"isShape({o},{x})"))
            lits.append((cmask(tv == x), base + L(2) + L(4), f"isTex({o},{x})"))
    for a, b in itertools.combinations(range(4), 2):
        lits.append((cmask(((c_SH == a) & (c_ST == b)) | ((c_SH == b) & (c_ST == a))), base + L(6), f"pairShape{{{a},{b}}}"))
        lits.append((cmask(((c_TH == a) & (c_TT == b)) | ((c_TH == b) & (c_TT == a))), base + L(6), f"pairTex{{{a},{b}}}"))
    # colour literals cannot change a colour-free rule; they only cost probability mass (via ntypes)
    return lits


def g3_cost(target, colour=True, kmax=4):
    lits = G3_LITS[colour]
    # clause with m literals: S->C (log2) + m C-decisions (log2 each) + literal costs
    pool = [(m, L(2) + k * L(2) + c) for m, k, c, _ in clauses(lits, kmax)]
    cost, chosen = mincover(target, pool, extra_per_clause=L(2))   # or-node per extra clause
    return cost - L(2) if chosen else math.inf


G1_LITS = g1_literals()
G3_LITS = {True: g3_literals(True), False: g3_literals(False)}

# ---------------------------------------------------------------------------
# Samplers over the full 9216-pair space (with colour/level), as python int bitsets
# ---------------------------------------------------------------------------
from domain import to_int as TI


def _m(b):
    return TI(b)


SH_IS = {(o, x): _m((SH if o == "h" else ST) == x) for o in "ht" for x in range(4)}
TX_IS = {(o, x): _m((TH if o == "h" else TT) == x) for o in "ht" for x in range(4)}
CO_IS = {(o, x): _m((LH if o == "h" else LT) == x) for o in "ht" for x in range(NL)}
SAME_S, SAME_T, SAME_C = _m(SH == ST), _m(TH == TT), _m(LH == LT)
from domain import FULL


class TooDeep(Exception):
    pass


def sample_g1(r, d=0):
    if d > 25: raise TooDeep
    c = r.randrange(4)
    if c == 0: return sample_g1(r, d + 1) & sample_g1(r, d + 1)
    if c == 1: return sample_g1(r, d + 1) | sample_g1(r, d + 1)
    if c == 2: return FULL & ~sample_g1(r, d + 1)
    ty = r.randrange(6); o = "h" if ty % 2 == 0 else "t"
    if ty < 2: return SH_IS[o, r.randrange(4)]
    if ty < 4: return TX_IS[o, r.randrange(4)]
    return CO_IS[o, r.randrange(NL)]


def _choice(r, probs):
    u = r.random(); acc = 0.0
    for k, p in probs:
        acc += p
        if u < acc: return k
    return probs[-1][0]


def sample_g2(r, th=THETA0):
    def dim(rel_same, isv, d):
        c = _choice(r, [("same", th["A_same"]), ("diff", th["A_diff"]), ("C", th["A_C"])])
        if c == "same": return rel_same
        if c == "diff": return FULL & ~rel_same
        return Cn(isv, d)

    def Cn(isv, d):
        if d > 25: raise TooDeep
        c = _choice(r, [("or", th["C_or"]), ("o1", th["C_o1"]), ("o2", th["C_o2"]), ("both", th["C_both"]), ("pair", th["C_pair"])])
        if c == "or": return Cn(isv, d + 1) | Cn(isv, d + 1)
        if c == "o1": return isv["h", r.randrange(4)]
        if c == "o2": return isv["t", r.randrange(4)]
        return isv["h", r.randrange(4)] & isv["t", r.randrange(4)]

    c = _choice(r, [("A", th["S_A"]), ("B", th["S_B"]), ("AB", th["S_AB"])])
    if c == "A": return dim(SAME_S, SH_IS, 0)
    if c == "B": return dim(SAME_T, TX_IS, 0)
    return dim(SAME_S, SH_IS, 0) & dim(SAME_T, TX_IS, 0)


def sample_g3(r, colour=True):
    nt = 11 if colour else 8

    def lit():
        k = r.randrange(nt)
        if k == 0: return SAME_S
        if k == 1: return FULL & ~SAME_S
        if k == 2: return SAME_T
        if k == 3: return FULL & ~SAME_T
        if k == 4: return SH_IS["ht"[r.randrange(2)], r.randrange(4)]
        if k == 5: return TX_IS["ht"[r.randrange(2)], r.randrange(4)]
        if k == 6:
            a, b = r.sample(range(4), 2); return (SH_IS["h", a] & SH_IS["t", b]) | (SH_IS["h", b] & SH_IS["t", a])
        if k == 7:
            a, b = r.sample(range(4), 2); return (TX_IS["h", a] & TX_IS["t", b]) | (TX_IS["h", b] & TX_IS["t", a])
        if k == 8: return SAME_C
        if k == 9: return FULL & ~SAME_C
        return CO_IS["ht"[r.randrange(2)], r.randrange(NL)]

    def C(d):
        if d > 25: raise TooDeep
        return lit() if r.random() < 0.5 else lit() & C(d + 1)

    def S(d):
        if d > 25: raise TooDeep
        return C(0) if r.random() < 0.5 else S(d + 1) | S(d + 1)

    return S(0)


# ---------------------------------------------------------------------------
# G3 with estimated probabilities (literal-type weights, or/and continuation)
# ---------------------------------------------------------------------------
G3_TYPES = ["sameShape", "diffShape", "sameTex", "diffTex", "isShape", "isTex", "pairShape", "pairTex"]


def _ltype(name):
    for t in G3_TYPES:
        if name.startswith(t):
            return t
    raise ValueError(name)


def g3_derivation_counts(target, kmax=4):
    """Counts of choices in the minimal G3 (no colour) DNF derivation of target."""
    lits = G3_LITS[False]
    cl = clauses(lits, kmax)
    pool = [(m, L(2) + k * L(2) + c) for m, k, c, _ in cl]
    names = {}
    for (m, k, c, nm), (_, cost) in zip(cl, pool):
        if (m & ~target) == 0 and (m not in names or cost < names[m][0]):
            names[m] = (cost, nm)
    cost, chosen = mincover(target, pool, extra_per_clause=L(2))
    if not chosen:
        return None
    cnt = {"S_or": 0, "S_stop": 0, "C_and": 0, "C_stop": 0, **dict.fromkeys(G3_TYPES, 0)}
    k = len(chosen)
    cnt["S_or"] += k - 1; cnt["S_stop"] += k
    for m, _ in chosen:
        nm = names[m][1]
        cnt["C_and"] += len(nm) - 1; cnt["C_stop"] += 1
        for n in nm:
            cnt[_ltype(n)] += 1
    return cnt


def g3_theta_from_counts(cnt, alpha=1.0):
    p_or = (cnt["S_or"] + alpha) / (cnt["S_or"] + cnt["S_stop"] + 2 * alpha)
    p_and = (cnt["C_and"] + alpha) / (cnt["C_and"] + cnt["C_stop"] + 2 * alpha)
    tot = sum(cnt[t] + alpha for t in G3_TYPES)
    return {"p_or": min(p_or, 0.45), "p_and": min(p_and, 0.7), **{t: (cnt[t] + alpha) / tot for t in G3_TYPES}}


def sample_g3w(r, th):
    types = [(t, th[t]) for t in G3_TYPES]

    def lit():
        k = _choice(r, types)
        if k == "sameShape": return SAME_S
        if k == "diffShape": return FULL & ~SAME_S
        if k == "sameTex": return SAME_T
        if k == "diffTex": return FULL & ~SAME_T
        if k == "isShape": return SH_IS["ht"[r.randrange(2)], r.randrange(4)]
        if k == "isTex": return TX_IS["ht"[r.randrange(2)], r.randrange(4)]
        if k == "pairShape":
            a, b = r.sample(range(4), 2); return (SH_IS["h", a] & SH_IS["t", b]) | (SH_IS["h", b] & SH_IS["t", a])
        a, b = r.sample(range(4), 2); return (TX_IS["h", a] & TX_IS["t", b]) | (TX_IS["h", b] & TX_IS["t", a])

    def C(d):
        if d > 25: raise TooDeep
        return lit() & C(d + 1) if r.random() < th["p_and"] else lit()

    def S(d):
        if d > 25: raise TooDeep
        return S(d + 1) | S(d + 1) if r.random() < th["p_or"] else C(0)

    return S(0)
