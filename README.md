# Chris's project code

Honours project: *Does AI act on hypotheses like people?* Everything in this
folder is mine; `agent/`, `analysis/`, `modeling/` and `experiment/` are
Bonan's (plus my old runner `agent/run_agent.py` / `view_results.py` for her
agent).

| Path | What it is |
|---|---|
| `llm_play.py` | The game (a port of the JS) and the LLM player. `--hypotheses K` makes it state K rules with probabilities alongside each action. |
| `hypotheses.py` | The rule grammar, parser and scoring; can also propose hypotheses on their own from a human or random game. |
| `grammar_comparison/` | Comparison of three hypothesis grammars against the human messages and choices (see its README). |
| `results/` | Outputs of the scripts above. |
| `elm-test.py` | Probe of what the ELM API proxy supports. |

`hypo_play.py` (a separate hypothesis-generator LLM feeding the player) was
removed; it is in the git history before this move if needed.

## Setup

The scripts read `chris\.env` (same contents as `agent\.env`, not committed):

```text
ELM_API_KEY=your-key
ELM_BASE_URL=https://elm.edina.ac.uk/api/v1
```

The existing environment in `agent\.venv` works; run from this folder:

```powershell
Set-Location "D:\University\Year-4\Project\discovery-rules\chris"
..\agent\.venv\Scripts\Activate.ps1
```

`grammar_comparison/` also needs `numpy` and `scipy`.

## Running

```powershell
# LLM plays directly
python llm_play.py --condition medium --seeds 0 1 2 --name med

# LLM states 5 hypotheses + "none of these" with every action
python llm_play.py --condition medium --seeds 0 1 2 --hypotheses 5 --name med_h5

# the same, but the model keeps the whole conversation, its own earlier replies included
python llm_play.py --condition medium --seeds 0 1 2 --hypotheses 5 --history full --name med_h5_full

# policy players: the LLM states hypotheses each turn, the policy picks the move
# thompson = sample a hypothesis by its weight; top = the most likely one; expected = all of them
python llm_play.py --condition low --seeds 0 1 2 --hypotheses 5 --actor thompson --name low_ts

# your ELM spend and limit
python llm_play.py --costs

# check the game rules reproduce every human fusion
python llm_play.py --check-human-data ..\data\action_data.csv
```

## Costs

Every run prints your ELM spend before and after, and a usage line per game: API calls,
re-asks, input tokens (and how many were served from the prompt cache, which is billed
cheaper) and output tokens (and how many were hidden reasoning). The same numbers are
saved per action in the actions CSV (`api_calls`, `prompt_tokens`, `cached_tokens`,
`completion_tokens`, `reasoning_tokens`), per game in each transcript (`usage`), and per
run in `*_usage.json`. ELM updates its spend figure only every so often, so the change
printed at the end of a run can lag; `--costs` later shows the settled figure.

## Policy players (`--actor thompson | top | expected`)

The LLM is asked for hypotheses only (no action in its prompt or JSON) and the policy
picks the move. All three use one rule: play the move with the highest expected stop
value, where the stop value S is points banked plus the n most valuable things still
harvestable (n = actions left). Harvesting the best crystal leaves S unchanged; fusing
x onto y scores p * S(if it works) + (1 - p) * S(if it fails). They differ only in p:

- `expected`: p = total weight of the hypotheses that say the pair fuses (all beliefs).
- `thompson`: p = 1 or 0 according to one hypothesis sampled by its weight.
- `top`: p = 1 or 0 according to the highest-weight hypothesis.

For "none of these", p is the success rate so far by Laplace's rule, (successes + 1) /
(fusions tried + 2). Pairs already tried are known; failures are never repeated. Ties go
to an untried fusion. Extra CSV columns: `actor`, `acted_on`, `acted_on_rule`,
`acted_on_weight`, `policy_score`, `policy_p` (p of the move played), `p0`, `free_test`.
