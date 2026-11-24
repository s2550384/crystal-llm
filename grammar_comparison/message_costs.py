import math, time, json
import pandas as pd
from codes import CODES
from domain import TRUE_RULES, CONDITION_MAP
from grammars import cells, g1_cost_dnf, g2_cost, g3_cost, g2_split, g2_derivation_counts
m = pd.read_csv('../../data/message_data.csv')
assert set(m.id) == set(CODES), set(m.id) ^ set(CODES)
rows = []
cache = {}
t0 = time.time()
for r in m.itertuples():
    cat, mask, gloss = CODES[r.id]
    cond = CONDITION_MAP[r.condition]
    row = dict(id=r.id, condition=cond, category=cat, gloss=gloss, total_points=r.total_points)
    if mask is not None:
        c = cells(mask)
        row["matches_true_rule"] = bool(c == cells(TRUE_RULES[cond]))
        if c not in cache:
            cache[c] = dict(G1=g1_cost_dnf(c), G2=g2_cost(c), G3=g3_cost(c, True), G3nc=g3_cost(c, False),
                            G2_separable=g2_split(c) is not None)
        row.update(cache[c])
    rows.append(row)
print('time', round(time.time()-t0,1), 'unique rules', len(cache))
df = pd.DataFrame(rows)
df.to_csv('message_coding.csv', index=False)
print(df.category.value_counts())
print(pd.crosstab(df.condition, df.category))
R = df[df.category.isin(['rule','history'])]
print('rule msgs', len(R), 'true-rule matches', R.matches_true_rule.sum())
for g in ['G1','G2','G3','G3nc']:
    fin = R[g].apply(math.isfinite)
    print(g, 'expressible', fin.sum(), '/', len(R), ' mean cost (nats, expressible)', round(R.loc[fin,g].mean(),2), ' median', round(R.loc[fin,g].median(),2))
common = R[R[['G1','G2','G3','G3nc']].map(math.isfinite).all(axis=1)]
print('common support n=', len(common)); print(common[['G1','G2','G3','G3nc']].sum().round(1))
print(R[['id','condition','gloss','matches_true_rule','G1','G2','G3','G3nc']].sort_values('condition').to_string())
