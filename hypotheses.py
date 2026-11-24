"""LLM hypothesis generation in a small formal grammar.

Given what a player has observed in the crystal game, ask an LLM for k
candidate rules for which fusions work. Each rule is written as

    rule(h, t) = <expr>        h = held crystal, t = crystal on the ground
    <expr> := <expr> and <expr> | <expr> or <expr> | not <expr> | ( <expr> )
            | same_<dim>(h, t) | different_<dim>(h, t)     dim = shape, texture, colour
            | is_<value>(h) | is_<value>(t)                value = a shape, texture or colour

The primitives are those of Zhao's PCFG (sameShape, diffShape, isShape(o, x)
and the texture versions), extended to colour. A pair test such as
pairShape(o1, o2, x, x') is written is_x(h) and is_x'(t). Unlike the PCFG,
and/or/not combine freely, so every PCFG rule can be written here (the
reverse does not hold, e.g. same_shape(h, t) or same_texture(h, t)).
"color" spellings are accepted. GRAMMAR_TEXT below is the description shown
to the LLM, here and in llm_play.py. Each rule is checked against this grammar, compiled,
and evaluated on every (held, target) pair of the 96 crystal kinds, so
hypotheses can be compared by what they predict rather than how they are
written. For each one we record its length, how many observed fusions it
predicts correctly, whether it is symmetric, and which real condition
rule(s) it is equivalent to.

Usage:
    # observations from a human participant's first 15 actions
    python hypotheses.py --human ..\\data\\action_data.csv --participant 2 --upto 15 --k 5

    # observations from a random player in a fresh game
    python hypotheses.py --condition low --seed 0 --random-actions 15 --k 5
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from llm_play import COLORS, MAXLEVEL, RULES, SHAPES, TEXTURES, CrystalGame, create_completion, load_env, make_client

FEATURES = {f"is_{s}": ("shape", s) for s in SHAPES}
FEATURES.update({f"is_{t}": ("texture", t) for t in TEXTURES})
FEATURES.update({f"is_{c}": ("level", i) for i, c in enumerate(COLORS)})
TESTS = {name: (lambda attr, val: lambda x: x[attr] == val)(*fv) for name, fv in FEATURES.items()}

# Relations between the two crystals: same_<dim>(h, t) / different_<dim>(h, t).
# A crystal's colour is its level, so same_colour compares levels.
DIMENSIONS = {"shape": "shape", "texture": "texture", "colour": "level", "color": "level"}
RELATIONS = {}
for _dim, _attr in DIMENSIONS.items():
    RELATIONS[f"same_{_dim}"] = (lambda a: lambda x, y: x[a] == y[a])(_attr)
    RELATIONS[f"different_{_dim}"] = (lambda a: lambda x, y: x[a] != y[a])(_attr)

GRAMMAR_TEXT = """Write each hypothesis in this formal language. h is the crystal being held and t is the crystal on the ground:
    rule(h, t) = <expression>

Tests that compare the two crystals:
    same_shape(h, t)    different_shape(h, t)
    same_texture(h, t)  different_texture(h, t)
    same_colour(h, t)   different_colour(h, t)

Tests that look at ONE crystal (x is h or t):
    shape:   is_triangle(x)  is_circle(x)  is_square(x)  is_diamond(x)
    texture: is_plain(x)  is_checkered(x)  is_stripes(x)  is_dots(x)
    colour:  is_red(x)  is_orange(x)  is_yellow(x)  is_green(x)  is_blue(x)  is_purple(x)

Combine tests with: and, or, not, parentheses. Nothing else is allowed.

Examples of the syntax (these are NOT hints about the answer):
    rule(h, t) = is_red(h) and not is_square(t)
    rule(h, t) = different_colour(h, t) and (is_plain(t) or is_triangle(h))"""

# Every crystal kind that can exist: 4 shapes x 4 textures x 6 levels = 96.
DOMAIN = [{"shape": s, "texture": t, "level": l}
          for s in SHAPES for t in TEXTURES for l in range(len(COLORS))]
PAIRS = [(a, b) for a in DOMAIN for b in DOMAIN]  # 9,216 ordered pairs
BASE_PAIRS = [(a, b) for a, b in PAIRS if a["level"] == 0 and b["level"] == 0 and a != b]  # 240

# ---------------------------------------------------------------------------
# Parsing: the grammar is a subset of Python boolean expressions, so we parse
# with Python's ast module and reject every node outside the grammar.
# ---------------------------------------------------------------------------

def parse_rule(text):
    """Return (fn(h, t) -> bool, length), or raise ValueError with a message for the LLM."""
    expr = re.sub(r"^\s*rule\s*\(\s*h\s*,\s*t\s*\)\s*:?=", "", text.strip(), flags=re.I)
    expr = expr.strip()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"could not parse ({e.msg}): {text}")
    length, call_parts = 0, set()
    for node in ast.walk(tree):  # breadth-first, so a call is seen before its parts
        if isinstance(node, (ast.Expression, ast.And, ast.Or, ast.Not, ast.Load)) or id(node) in call_parts:
            continue
        if isinstance(node, ast.BoolOp) or (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)):
            length += 1
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "id", None)
            args = [getattr(a, "id", None) for a in node.args]
            if name in FEATURES:
                if len(args) != 1 or node.keywords or args[0] not in ("h", "t"):
                    raise ValueError(f"{name} takes exactly one argument, h or t")
            elif name in RELATIONS:
                if len(args) != 2 or node.keywords or sorted(args) != ["h", "t"]:
                    raise ValueError(f"{name} takes exactly two arguments: {name}(h, t)")
            else:
                raise ValueError(f"unknown test {name!r} (allowed: same_<shape|texture|colour>(h, t), "
                                 f"different_<shape|texture|colour>(h, t), is_<shape|texture|colour>(x))")
            call_parts.update([id(node.func)] + [id(a) for a in node.args])
            length += 1
        else:
            raise ValueError(f"not allowed in a rule: {ast.unparse(node)!r} (use only tests, and, or, not)")
    code = compile(tree, "<rule>", "eval")
    return (lambda h, t: bool(eval(code, {"__builtins__": {}}, {**TESTS, **RELATIONS, "h": h, "t": t}))), length


def truth_table(fn):
    return bytes(1 if fn(a, b) else 0 for a, b in PAIRS)


TRUE_TABLES = {c: truth_table(r) for c, r in RULES.items()}


def parse_kind(name):
    """'circle_dots_2' -> crystal dict."""
    s, t, l = name.split("_")
    return {"shape": s, "texture": t, "level": int(l)}


def kind_text(name):
    c = parse_kind(name)
    return f"{COLORS[c['level']]} {c['shape']} {c['texture']}"


@dataclass
class Hypothesis:
    rule: str
    explanation: str = ""
    stated_confidence: float | None = None
    error: str | None = None  # grammar error, if it failed to parse
    length: int | None = None  # number of tests, connectives and constants
    n_true_base: int | None = None  # how many of the 240 base pairs it allows
    symmetric: bool | None = None  # rule(h,t) == rule(t,h) for all pairs
    matches: list = field(default_factory=list)  # real conditions it equals
    n_obs: int = 0
    n_correct: int = 0
    key: str | None = None  # id of the truth table (equal => same predictions)
    fn: object = field(default=None, repr=False)

    @classmethod
    def from_raw(cls, raw):
        conf = raw.get("confidence")
        h = cls(rule=str(raw.get("rule", "")).strip(), explanation=str(raw.get("explanation", "")).strip(),
                stated_confidence=float(conf) if isinstance(conf, (int, float)) else None)
        try:
            h.fn, h.length = parse_rule(h.rule)
        except ValueError as e:
            h.error = str(e)
            return h
        table = truth_table(h.fn)
        h.key = hashlib.md5(table).hexdigest()[:8]
        h.matches = [c for c, tt in TRUE_TABLES.items() if tt == table]
        h.n_true_base = sum(h.fn(a, b) for a, b in BASE_PAIRS)
        h.symmetric = all(h.fn(a, b) == h.fn(b, a) for a, b in PAIRS)
        return h

    def predict(self, held, target):
        """Predicted fusion success, including the game's known max-level cap."""
        return max(held["level"], target["level"]) + 1 < MAXLEVEL and self.fn(held, target)

    def to_dict(self):
        d = asdict(self)
        d.pop("fn")
        return d


def score(hyps, observations):
    """Fill n_obs / n_correct from observations [(held_name, target_name, yield_name or ''), ...]."""
    obs = [(parse_kind(h), parse_kind(t), bool(y)) for h, t, y in observations]
    for hyp in hyps:
        if hyp.error is None:
            hyp.n_obs = len(obs)
            hyp.n_correct = sum(hyp.predict(h, t) == ok for h, t, ok in obs)


def posterior(hyps, prior="uniform", eps=0.05, lam=0.5):
    """Weights over truth tables (call score() first).

    Likelihood: each observed fusion is predicted correctly with prob 1-eps.
    prior 'uniform': equal weight per proposal; proposals with the same truth
    table pool their weight, so the LLM's proposal frequency acts as the prior.
    prior 'length': exp(-lam * length) per proposal.
    """
    ok = [h for h in hyps if h.error is None]
    if not ok:
        return {}
    logw = [(-lam * h.length if prior == "length" else 0.0)
            + h.n_correct * math.log(1 - eps) + (h.n_obs - h.n_correct) * math.log(eps) for h in ok]
    w = [math.exp(x - max(logw)) for x in logw]
    out = Counter()
    for h, wi in zip(ok, w):
        out[h.key] += wi / sum(w)
    return dict(out)


# ---------------------------------------------------------------------------
# Prompt and generator
# ---------------------------------------------------------------------------

PROMPT = """You are helping a player in a crystal-fusion game. Crystals differ in shape (triangle, circle, square, diamond) and texture (plain, checkered, stripes, dots). Naturally occurring crystals are red, and a crystal's colour shows its value: red 1 point, orange 10, yellow 100, green 1,000, blue 10,000, purple 100,000. A player can try to fuse a crystal they are holding with a crystal on the ground. Not all crystals can be fused, but a successful fusion makes a new crystal worth 10 times more; for example, successfully fusing two red crystals produces an orange crystal. Keep in mind that crystals differ in their shape and texture, and not all kinds of fusions are likely to work out. A hidden rule decides which pairs fuse. Your job is to propose hypotheses about that hidden rule.

{grammar}

{observations}

Propose {k} different hypotheses about the hidden rule. Each must be consistent with every observation above if possible. Make them genuinely different from each other (they should disagree about at least some untried pairs), and order them from most to least plausible.

Reply with JSON only, in exactly this format:
{{"hypotheses": [
  {{"rule": "rule(h, t) = ...", "explanation": "...", "confidence": 0.0}}
]}}
{explanation}
"confidence" is your probability (0-1) that the hypothesis is exactly right."""

# Shared with llm_play.py, so both scripts ask for the same thing.
EXPLANATION_TEXT = ('"explanation" is the rule itself restated as one plain-English sentence, not the evidence for it; '
                    'for the first example above: "A held red crystal fuses with any crystal on the ground that is not a square."')


def format_observations(observations, rounds=None):
    """The fusion list. rounds: [(n_fusions_seen, [Hypothesis, ...]), ...] from earlier calls;
    each set is shown at the point in the list where it was proposed."""
    rounds = [(n, [h for h in hs if h.error is None]) for n, hs in rounds or []]
    if not observations and not rounds:
        return "No fusions have been attempted yet."
    lines = ["Fusions attempted so far (held + ground -> result)"
             + (", with the hypotheses you proposed along the way. You may keep, refine or replace them:"
                if rounds else ":")]
    shown_keys = None

    def add_rounds(n):
        nonlocal shown_keys
        for m, hs in rounds:
            if m != n or not hs:
                continue
            keys = [h.key for h in hs]
            where = "before any fusions" if n == 0 else f"after fusion {n}"
            if keys == shown_keys:
                lines.append(f"     [your hypotheses {where}: unchanged]")
                continue
            shown_keys = keys
            lines.append(f"     [your hypotheses {where}:]")
            for h in hs:
                conf = "" if h.stated_confidence is None else f" (your confidence {h.stated_confidence:g})"
                lines.append(f"       - {h.rule}  {h.explanation}{conf}")

    add_rounds(0)
    for i, (h, t, y) in enumerate(observations, 1):
        lines.append(f"  {i}. {kind_text(h)} + {kind_text(t)} -> "
                     + (f"SUCCESS, made a {kind_text(y)} crystal" if y else "FAILED"))
        add_rounds(i)
    return "\n".join(lines)


@dataclass
class HypothesisGenerator:
    client: object
    model: str = "gpt-4o-mini"
    k: int = 5
    temperature: float | None = None  # None = model default (some models only accept 1)
    reasoning_effort: str | None = None  # e.g. none/low/medium/high; None = model default
    max_retries: int = 2  # re-asks to fix JSON / grammar errors

    def propose(self, observations, rounds=None):
        """observations: [(held_name, target_name, yield_name or ''), ...] with names
        like 'circle_dots_0'. rounds: earlier [(n_fusions_seen, hypotheses), ...], shown
        in the fusion list where they were proposed. Returns (hypotheses, log)."""
        prompt = PROMPT.format(k=self.k, observations=format_observations(observations, rounds),
                               grammar=GRAMMAR_TEXT, explanation=EXPLANATION_TEXT)
        messages = [{"role": "user", "content": prompt}]
        log = {"prompt": prompt, "attempts": []}
        kwargs = {k: v for k, v in (("temperature", self.temperature),
                                    ("reasoning_effort", self.reasoning_effort)) if v is not None}
        hyps = []
        for _ in range(self.max_retries + 1):
            text = create_completion(self.client, model=self.model, messages=messages,
                                     **kwargs).choices[0].message.content or ""
            messages.append({"role": "assistant", "content": text})
            try:
                body = re.sub(r"```(?:json)?", "", text)
                raw = json.loads(body[body.find("{"):body.rfind("}") + 1])["hypotheses"]
                hyps = [Hypothesis.from_raw(r) for r in raw]
                problems = [f"Hypothesis {i}: {h.error}" for i, h in enumerate(hyps, 1) if h.error]
                if len(hyps) < self.k:
                    problems.append(f"You gave {len(hyps)} hypotheses; {self.k} are needed.")
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                problems = [f"Your reply could not be read as the requested JSON ({e})."]
            log["attempts"].append({"output": text, "problems": problems})
            if not problems:
                break
            messages.append({"role": "user", "content": "There were problems:\n" + "\n".join(problems)
                             + f"\nPlease reply again with the full JSON containing all {self.k} hypotheses, fixed."})
        score(hyps, observations)
        return hyps, log


# ---------------------------------------------------------------------------
# Observation sources
# ---------------------------------------------------------------------------

def observations_from_human(csv_path, participant, upto):
    """Fusions in the first `upto` actions of one participant in data/action_data.csv."""
    import pandas as pd

    p = pd.read_csv(csv_path)
    p = p[p.id == participant].sort_values("action_id").head(upto)
    if p.empty:
        raise SystemExit(f"participant {participant} not found in {csv_path}")
    obs = [(h, t, y if isinstance(y, str) else "")
           for a, h, t, y in zip(p.action, p.held, p.target, p["yield"]) if a == "combine"]
    return obs, str(p.condition.iloc[0])


def observations_from_random_play(condition, seed, n_actions):
    """Play n random actions (fusions 80% of the time) and return the fusions made."""
    game = CrystalGame(condition, seed=seed)
    rng = random.Random(seed)
    for _ in range(n_actions):
        ids = list(game.crystals)
        if len(ids) >= 2 and rng.random() < 0.8:
            game.step(("fuse", *rng.sample(ids, 2)))
        else:
            game.step(("harvest", rng.choice(ids)))
    return list(game.fusion_log)


def main():
    load_env()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--human", metavar="CSV", help="human action_data.csv to take observations from")
    src.add_argument("--condition", choices=list(RULES), default="low", help="condition for random play")
    p.add_argument("--participant", type=int, help="participant id (with --human)")
    p.add_argument("--upto", type=int, default=15, help="use the participant's first N actions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random-actions", type=int, default=15)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--model", default=os.getenv("ELM_MODEL", "gpt-4o-mini"))
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--reasoning-effort", default=os.getenv("ELM_REASONING_EFFORT"),
                   help="e.g. none, low, medium, high (model-dependent); default: ELM_REASONING_EFFORT or the model's default")
    p.add_argument("--output", default="results")
    args = p.parse_args()

    if args.human:
        if args.participant is None:
            raise SystemExit("--participant is required with --human")
        obs, condition = observations_from_human(args.human, args.participant, args.upto)
        source = f"human {args.participant}, first {args.upto} actions"
    else:
        condition = args.condition
        obs = observations_from_random_play(condition, args.seed, args.random_actions)
        source = f"random play, seed {args.seed}, {args.random_actions} actions"

    print(f"Condition {condition} | {source} | {len(obs)} fusions observed | model {args.model} | "
          f"reasoning effort {args.reasoning_effort or 'model default'}")
    gen = HypothesisGenerator(make_client(), model=args.model, k=args.k, temperature=args.temperature,
                              reasoning_effort=args.reasoning_effort)
    hyps, log = gen.propose(obs)
    w_uniform, w_length = posterior(hyps, "uniform"), posterior(hyps, "length")

    for i, h in enumerate(hyps, 1):
        print(f"\n[{i}] {h.explanation}\n     {h.rule}")
        if h.error:
            print(f"     PARSE ERROR: {h.error}")
            continue
        print(f"     fits {h.n_correct}/{h.n_obs} | length {h.length} | allows {h.n_true_base}/240 base pairs | "
              f"{'symmetric' if h.symmetric else 'asymmetric'} | table {h.key} | "
              f"posterior {w_uniform.get(h.key, 0):.2f} (length prior {w_length.get(h.key, 0):.2f}) | "
              f"stated conf {h.stated_confidence}"
              + (f" | EQUALS true '{', '.join(h.matches)}' rule" if h.matches else ""))
    parsed = [h for h in hyps if h.error is None]
    print(f"\n{len({h.key for h in parsed})} distinct predictions among {len(parsed)} parsed hypotheses")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"hypotheses_{condition}_{datetime.now():%Y-%m-%d_%H-%M-%S}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"condition": condition, "source": source, "model": args.model,
                   "reasoning_effort": args.reasoning_effort, "observations": obs,
                   "hypotheses": [h.to_dict() for h in hyps], "posterior_uniform": w_uniform,
                   "posterior_length": w_length, "log": log}, f, indent=2, ensure_ascii=False)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
