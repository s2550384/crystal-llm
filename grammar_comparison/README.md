# Grammar comparison for the discovery game

Compares three hypothesis grammars (PCFGs) for the crystal game:

* G1 feature-Boolean: the grammar in agent/hypotheses.py
* G2 Zhao & Arumugam: the PCFG in Bonan's modelling notes (Table 1)
* G3 relational DNF: proposed alternative (relations, unordered pairs, OR across dimensions)

Two tests:

1. `message_costs.py`: codes every participant's end-of-game message (`codes.py`) and computes,
   per grammar, whether the stated rule is expressible and its minimal description length
   (negative log prior, nats). Output: `message_coding.csv`.
2. `behaviour.py`: each grammar as a Bayesian learner (prior = 200k rules sampled from the
   grammar, noisy likelihood), predicting which target each participant picks for the crystal
   they hold, softmax over posterior-predictive success. Fitted theta versions estimate rule
   probabilities from the messages of the *other two* conditions. Outputs: `behaviour_fit.csv`,
   `per_participant_ll.csv`, `theta_from_messages.json`, `run_full2.log`.

Run from this folder: `python message_costs.py` then `python behaviour.py 200000` (about 4 min).
Data paths at the top of each script point to data/action_data.csv and data/message_data.csv.
