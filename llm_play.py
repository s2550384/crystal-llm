"""LLM plays the crystal-fusion discovery game.

An action-only port of experiment/js (config.js + grid.js). The grid,
movement, pick-up and drop are left out: in the browser game they are free,
so a player's choices reduce to two actions that each cost one of the
40 actions:

    fuse <held_id> <target_id>    (JS: combineItem(held, target))
    harvest <id>                  (JS: consumeItem(held))

Game logic follows the JS exactly:
  * 16 base crystals (4 shapes x 4 textures), all level 0 (red).
  * Points for harvesting a level-n crystal = 10 ** n.
  * A successful fusion makes a crystal with the HELD crystal's shape, the
    TARGET crystal's texture and level max(levels) + 1. Fusions that would
    reach level 6 fail (MAXLEVEL = 6, so the top crystal is level 5).
  * Failed fusions still cost an action and leave both crystals in place.
  * Base crystals (level 0) regenerate once each, by kind, after being
    harvested OR used in a successful fusion. In the browser this happens
    after ~5s (+2s for the second crystal of a fusion); here it happens straight
    after the action (REGEN_DELAY = 0), the only setting every participant's moves fit.
  * Shape/texture orderings are the ones in config.js. They matter because
    the medium/low rules use indices. Checked against data/action_data.csv:
    every human fusion outcome is reproduced (see check_rules_against_human_data).

Conditions use the labels in data/subject_data.csv:
    high   - same shape                                     (JS isSameShape)
    medium - shape indices sum to 3, textures differ        (JS diffObjs1)
    low    - shape indices sum to 3, texture(held) >= texture(target)
             (JS capSet; the rule in the CogSci 2025 paper)
    low-3  - shape indices sum to 3, texture(held) even or texture(target)
             odd (JS modRule3; data/pilot_newhardrule)

--hypotheses K makes the player state K hypotheses about the hidden rule in
the same reply as its action, every turn. Each hypothesis is a rule in the
formal grammar of hypotheses.py with a probability, plus a probability that
none of them is right. The rules are parsed and checked like in
hypotheses.py, and each action record logs them together with measures of
whether the action agrees with them (see HYPOTHESIS_PROMPT and play()).

Replies are JSON only (the models' reasoning is separate), e.g.
    {"action": "fuse", "held": 3, "ground": 7}   or   {"action": "harvest", "crystal": 5}
with "hypotheses" and "none_of_these" fields added under --hypotheses.

Usage:
    python llm_play.py --condition high --seed 0 --name test
    python llm_play.py --condition medium --seeds 0 1 2 --hypotheses 5 --name med_h5
    python llm_play.py --condition low --seeds 0 1 2 --hypotheses 5 --actor thompson --name low_ts
    (--actor: llm | thompson | top | expected)
    python llm_play.py --check-human-data ../data/action_data.csv
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Game definition (mirrors experiment/js/config.js)
# ---------------------------------------------------------------------------

MAX_ACTIONS = 40
MAXLEVEL = 6  # fusion fails if it would produce this level
SHAPES = ["triangle", "circle", "square", "diamond"]
TEXTURES = ["plain", "checkered", "stripes", "dots"]
COLORS = ["red", "orange", "yellow", "green", "blue", "purple"]  # by level
REGEN_DELAY = 0  # actions until a used base crystal reappears: 0 = straight after the action
# (as in the human data: the browser game brings it back after ~5 s of real time,
# and 26% of participants' reuses come sooner than a 2-action delay allows)


def _s(shape):
    return SHAPES.index(shape)


def _t(texture):
    return TEXTURES.index(texture)


RULES = {
    "high": lambda h, t: h["shape"] == t["shape"],
    "medium": lambda h, t: _s(h["shape"]) + _s(t["shape"]) == 3
    and h["texture"] != t["texture"],
    "low": lambda h, t: _s(h["shape"]) + _s(t["shape"]) == 3
    and _t(h["texture"]) >= _t(t["texture"]),
    "low-3": lambda h, t: _s(h["shape"]) + _s(t["shape"]) == 3
    and (_t(h["texture"]) % 2 == 0 or _t(t["texture"]) % 2 == 1),
}


def kind_name(c):
    """Crystal name in the JS/human-data format, e.g. 'circle_dots_0'."""
    return f"{c['shape']}_{c['texture']}_{c['level']}"


def _kind_text(name):
    """'circle_dots_1' -> 'orange circle dots'."""
    s, t, l = name.split("_")
    return f"{COLORS[int(l)]} {s} {t}"


def describe(c):
    return f"{COLORS[c['level']]} {c['shape']} {c['texture']} ({10 ** c['level']:,} pts)"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CrystalGame:
    def __init__(self, condition="high", seed=None, max_actions=MAX_ACTIONS,
                 regen_delay=REGEN_DELAY, allow_regeneration=True, regen_stagger=False):
        if condition not in RULES:
            raise ValueError(f"Unknown condition {condition!r}; choose from {list(RULES)}")
        self.condition = condition
        self.rule = RULES[condition]
        self.max_actions = max_actions
        self.regen_delay = regen_delay
        self.allow_regeneration = allow_regeneration
        self.regen_stagger = regen_stagger  # second crystal of a fusion one action later, as the JS's 2 s stagger
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        kinds = [(s, t) for s in SHAPES for t in TEXTURES]
        self.rng.shuffle(kinds)  # so ids carry no information about the rule
        self.crystals = {}
        self._next_id = 0
        for s, t in kinds:
            self._add(s, t, 0)
        self.points = 0
        self.actions_used = 0
        self.regenerated = set()  # base kinds already regenerated (once each)
        self.pending = []  # (due_action_count, shape, texture)
        self.fusion_log = []  # (held_name, target_name, yield_name or "")
        self.harvest_log = []  # (name, points)
        return self

    def _add(self, shape, texture, level):
        cid = self._next_id
        self._next_id += 1  # ids are never reused
        self.crystals[cid] = {"id": cid, "shape": shape, "texture": texture, "level": level}
        return cid

    @property
    def actions_left(self):
        return self.max_actions - self.actions_used

    @property
    def done(self):
        return self.actions_left <= 0 or not self.crystals and not self.pending

    def _schedule_regen(self, crystals):
        if not self.allow_regeneration:
            return
        for i, c in enumerate(crystals):
            key = (c["shape"], c["texture"])
            if c["level"] == 0 and key not in self.regenerated:
                self.regenerated.add(key)
                # JS staggers the second crystal of a fusion by 2s
                self.pending.append((self.actions_used + self.regen_delay + (i if self.regen_stagger else 0), *key))

    def _process_regen(self):
        due = [p for p in self.pending if p[0] <= self.actions_used]
        self.pending = [p for p in self.pending if p[0] > self.actions_used]
        return [self._add(s, t, 0) for _, s, t in due]

    def validate(self, action):
        """Return an error message, or None if the action is legal."""
        kind, *ids = action
        if kind == "harvest" and len(ids) == 1:
            if ids[0] not in self.crystals:
                return f"There is no crystal with id {ids[0]}."
        elif kind == "fuse" and len(ids) == 2:
            if ids[0] == ids[1]:
                return "You cannot fuse a crystal with itself."
            missing = [i for i in ids if i not in self.crystals]
            if missing:
                return f"There is no crystal with id {missing[0]}."
        else:
            return "Unrecognised action."
        return None

    def step(self, action):
        """Apply a legal action. Returns a dict describing what happened."""
        err = self.validate(action)
        if err:
            raise ValueError(err)
        kind, *ids = action
        self.actions_used += 1
        result = {"action": kind, "points": 0, "yield": ""}

        if kind == "harvest":
            c = self.crystals.pop(ids[0])
            pts = 10 ** c["level"]
            self.points += pts
            self.harvest_log.append((kind_name(c), pts))
            self._schedule_regen([c])
            result.update(held=kind_name(c), target="", points=pts,
                          text=f"You harvested crystal {c['id']} ({describe(c)}) and earned {pts:,} points.")
        else:
            h, t = self.crystals[ids[0]], self.crystals[ids[1]]
            new_level = max(h["level"], t["level"]) + 1
            ok = new_level < MAXLEVEL and self.rule(h, t)
            result.update(held=kind_name(h), target=kind_name(t))
            if ok:
                del self.crystals[h["id"]], self.crystals[t["id"]]
                nid = self._add(h["shape"], t["texture"], new_level)
                new = self.crystals[nid]
                result["yield"] = kind_name(new)
                result["text"] = (f"Fusion succeeded: crystal {h['id']} ({describe(h)}) + crystal {t['id']} "
                                  f"({describe(t)}) -> new crystal {nid} ({describe(new)}).")
                self._schedule_regen([h, t])
            else:
                result["text"] = (f"Fusion failed: crystal {h['id']} ({describe(h)}) and crystal {t['id']} "
                                  f"({describe(t)}) did not fuse. Both crystals are unchanged.")
            self.fusion_log.append((result["held"], result["target"], result["yield"]))

        new_ids = self._process_regen()
        if new_ids:
            result["text"] += " " + " ".join(
                f"A {describe(self.crystals[i])} crystal regenerated as crystal {i}." for i in new_ids)
        result["regenerated"] = [kind_name(self.crystals[i]) for i in new_ids]
        return result

    def observation(self):
        """What the player sees: the harvests and attempted fusions participants
        were shown, then the crystals, score and actions left that participants were shown."""
        # Ordered for prompt caching: the logs only grow at their ends, so they come
        # first and each turn's prompt starts with the previous turn's; the parts that
        # change every turn (crystals, score) come last.
        lines = ["Crystals harvested:"]
        lines += [f"  {_kind_text(n)}: {p:,} points" for n, p in self.harvest_log] or ["  (none)"]
        lines.append("Fusions attempted (held + ground -> result):")
        lines += [f"  {_kind_text(h)} + {_kind_text(t)} -> {_kind_text(y) if y else 'failed'}"
                  for h, t, y in self.fusion_log] or ["  (none)"]
        lines.append("Crystals on the planet:")
        for c in sorted(self.crystals.values(), key=lambda c: c["id"]):
            lines.append(f"  [{c['id']}] {describe(c)}")
        if not self.crystals:
            lines.append("  (none)")
        lines.append(f"Energy points: {self.points:,}. Actions remaining: {self.actions_left}.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM player
# ---------------------------------------------------------------------------

# The participant instructions from experiment/p/task.html, as close to
# verbatim as the text-only game allows. Changes:
#   * The move / pick-up / drop controls are left out (free in the browser
#     game); the fuse and harvest steps keep their wording, minus the keys.
#   * Colours are named (participants saw coloured squares). task.html has a
#     typo in the list (blue 100,000, purple 1,000,000, no 10,000); the values
#     below are the ones the game awards.
#   * The table of attempted fusions and harvests participants were shown is
#     the turn history here, so its sentence is kept.
#   * "After the game" (the message for a future player and the bonus) is
#     left out: the LLM writes no message.
TASK_PROMPT = """In this game, you are a space explorer. You have discovered a faraway planet with a curious property — the surface of the planet is dotted with valuable crystals that can be harvested for energy points. Some of these crystals can also be fused to make new and much more powerful crystals; however, scientists still do not understand why some pairs of crystals can fuse while others cannot. Your goal is to harvest the most energy points possible by collecting, combining, and harvesting crystals. Explore the planet to discover new, more rewarding crystals and maximize your score.

How to play:
To fuse crystals: fuse a crystal you are holding with a crystal on the ground. Not all crystals can be fused—but, if you do discover a successful fusion, it will make a new crystal that is worth 10 times more energy points.
To harvest a crystal: harvest the crystal you are holding. Harvesting crystals is the only way to earn energy points.

Fusion guide:
On this planet, naturally-occurring crystals are red. Each red crystal is worth 1 energy point, and they can regenerate once.
Each crystal's color shows how many energy points it's worth:
red: 1 point
orange: 10 points
yellow: 100 points
green: 1,000 points
blue: 10,000 points
purple: 100,000 points
When you discover a successful fusion, you'll make a new crystal that is worth 10 times as many energy points as the original crystals! For example, successfully fusing two red crystals (1 point) will produce a new orange crystal (10 points).

Your hover has fuel for up to {max_actions} actions—fusing crystals and harvesting them for energy points each consume one action. Use your actions wisely! Focus on finding and fusing crystals to unlock new, more valuable crystals! To help you keep track of how you've used your actions, we'll show you a table of all the fusions you've attempted and the energy points of all the crystals you have harvested.

Keep in mind that crystals differ in their shape and texture, and not all kinds of fusions are likely to work out."""

# How a turn works and the reply format (not from task.html).
TURN_PROMPT = """

Each turn you will see the table of fusions attempted and crystals harvested so far, and the crystals on the planet, each with an id in square brackets. You can pick up any crystal. Take one action per turn. Reply with only this JSON:
{"action": "fuse", "held": <id of the crystal you hold>, "ground": <id of the crystal on the ground>}
or
{"action": "harvest", "crystal": <id>}"""

SYSTEM_PROMPT = TASK_PROMPT + TURN_PROMPT

# Replaces TURN_PROMPT with --hypotheses K. {grammar} and {explanation} are
# filled from hypotheses.py (GRAMMAR_TEXT, EXPLANATION_TEXT), so both scripts
# describe the same language and fields.
HYPOTHESIS_PROMPT = """

A hidden rule decides which pairs of crystals fuse. Each turn, give your {k} best hypotheses about it, then your action.

{grammar}

The {k} hypotheses must disagree about at least some untried pairs. Order them from most to least plausible. "confidence" is the probability that a hypothesis is exactly the hidden rule and "none_of_these" the probability that none of them is; together they sum to 1. {explanation}

Each turn you will see the table of fusions attempted and crystals harvested so far, and the crystals on the planet, each with an id in square brackets. You can pick up any crystal. Take one action per turn. Reply with only this JSON:
{{"hypotheses": [{{"rule": "rule(h, t) = ...", "explanation": "...", "confidence": 0.0}}, ...], "none_of_these": 0.0, "action": "fuse", "held": <id of the crystal you hold>, "ground": <id of the crystal on the ground>}}
with "action": "harvest", "crystal": <id> in place of the last three fields to harvest."""

# Replaces HYPOTHESIS_PROMPT with --actor thompson | top: the moves are chosen by
# the policy, so the model is asked for hypotheses only, with no action.
HYPOTHESES_ONLY_PROMPT = """

A hidden rule decides which pairs of crystals fuse. Each turn, give your {k} best hypotheses about it.

{grammar}

The {k} hypotheses must disagree about at least some untried pairs. Order them from most to least plausible. "confidence" is the probability that a hypothesis is exactly the hidden rule and "none_of_these" the probability that none of them is; together they sum to 1. {explanation}

Each turn you will see the table of fusions attempted and crystals harvested so far, and the crystals on the planet. Reply with only this JSON:
{{"hypotheses": [{{"rule": "rule(h, t) = ...", "explanation": "...", "confidence": 0.0}}, ...], "none_of_these": 0.0}}"""

ACTION_RE = re.compile(r"action\s*:\s*(fuse|combine|harvest|consume)\s+(\d+)(?:\s*(?:,|\+|and|with)?\s*(\d+))?",
                       re.IGNORECASE)


def find_json(text, key):
    """The first JSON object in text that has `key` (code fences and stray text tolerated), or None."""
    body = re.sub(r"```(?:json)?", "", text or "")
    decoder = json.JSONDecoder()
    for i, ch in enumerate(body):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(body, i)
        except ValueError:
            continue
        if isinstance(obj, dict) and key in obj:
            return obj
    return None


def _id(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def parse_action(text):
    """Read the action from the reply JSON. Returns ('fuse', held, ground) / ('harvest', id) or None.
    An old-style 'Action: fuse A B' line is still accepted as a fallback."""
    obj = find_json(text, "action")
    if obj is not None:
        verb = str(obj.get("action", "")).strip().lower()
        if verb in ("fuse", "combine"):
            a, b = _id(obj.get("held")), _id(obj.get("ground", obj.get("target")))
            return ("fuse", a, b) if a is not None and b is not None else None
        if verb in ("harvest", "consume"):
            a = _id(obj.get("crystal", obj.get("id")))
            return ("harvest", a) if a is not None else None
        return None
    matches = list(ACTION_RE.finditer(text or ""))
    if not matches:
        return None
    verb, a, b = matches[-1].groups()
    if verb.lower() in ("fuse", "combine"):
        return ("fuse", int(a), int(b)) if b is not None else None
    return ("harvest", int(a))


def parse_hypotheses(text, k):
    """Read the hypotheses from the reply JSON.

    Returns (hypotheses, none_of_these, problems). hypotheses are
    hypotheses.Hypothesis objects (parse errors are kept on each one);
    problems is a list of messages for the model, empty if the block is fine."""
    from hypotheses import Hypothesis  # here, not at the top: hypotheses.py imports this module

    block = find_json(text, "hypotheses")
    if block is None or not isinstance(block.get("hypotheses"), list):
        return [], None, ['The JSON has no "hypotheses" list.']

    hyps = [Hypothesis.from_raw(r if isinstance(r, dict) else {}) for r in block["hypotheses"]]
    other = block.get("none_of_these")
    other = float(other) if isinstance(other, (int, float)) else None
    problems = [f"Hypothesis {i}: {h.error}" for i, h in enumerate(hyps, 1) if h.error]
    if len(hyps) != k:
        problems.append(f"You gave {len(hyps)} hypotheses; {k} are needed.")
    if any(h.stated_confidence is None for h in hyps) or other is None:
        problems.append('Every hypothesis needs a numeric "confidence", and "none_of_these" needs a number.')
    return hyps, other, problems


def hypothesis_weights(hyps, other):
    """Normalised weights: (one per parsed hypothesis, none_of_these).

    Stated probabilities are rescaled to sum to 1; rules that failed to parse
    are dropped (their probability is not moved to the others)."""
    ok = [h for h in hyps if h.error is None]
    raw = [max(h.stated_confidence or 0.0, 0.0) for h in ok]
    rest = max(other or 0.0, 0.0)
    total = sum(raw) + rest
    if total <= 0:  # no usable numbers: spread evenly
        return [1 / (len(ok) + 1)] * len(ok), 1 / (len(ok) + 1)
    return [r / total for r in raw], rest / total


def load_env():
    """Load agent/.env (then a .env in the current folder, if any)."""
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
    load_dotenv()


OPTIONAL_PARAMS = ("temperature", "prompt_cache_key")
_REJECTED = {}  # model -> optional params it (or the ELM proxy) rejected


def create_completion(client, prompt_cache_key=None, **kwargs):
    """chat.completions.create, dropping optional settings that are rejected and
    remembering that for the model: temperature (many reasoning models only
    allow the default) and prompt_cache_key (helps OpenAI route requests with
    the same prompt start to the same cache; the proxy may not accept it)."""
    model = kwargs.get("model")
    rejected = _REJECTED.setdefault(model, set())
    while True:
        call = dict(kwargs)
        if "temperature" in rejected:
            call.pop("temperature", None)
        if prompt_cache_key and "prompt_cache_key" not in rejected:
            call["extra_body"] = {**call.get("extra_body", {}), "prompt_cache_key": prompt_cache_key}
        try:
            return client.chat.completions.create(**call)
        except Exception as e:
            msg = str(e).lower()
            sent = set(call) | set(call.get("extra_body", {}))
            bad = [p for p in OPTIONAL_PARAMS if p in sent and p in msg]
            if getattr(e, "status_code", None) != 400 or not bad:
                raise
            for p in bad:
                print(f"  [note] {model} rejected {p}; continuing without it")
                rejected.add(p)


def usage_of(resp):
    """Token counts from a response. cached = prompt tokens served from the
    provider's prompt cache (billed cheaper); reasoning = hidden thinking tokens,
    which are billed as completion tokens. None where the API gave no number."""
    def get(obj, *path):
        for p in path:
            obj = obj.get(p) if isinstance(obj, dict) else getattr(obj, p, None)
            if obj is None:
                return None
        return obj

    u = getattr(resp, "usage", None)
    return {"prompt_tokens": get(u, "prompt_tokens"),
            "cached_tokens": get(u, "prompt_tokens_details", "cached_tokens"),
            "completion_tokens": get(u, "completion_tokens"),
            "reasoning_tokens": get(u, "completion_tokens_details", "reasoning_tokens")}


TOKEN_FIELDS = ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens")


def usage_summary(api_log):
    """Totals over a list of api_log entries."""
    tot = {f: sum(e.get(f) or 0 for e in api_log) for f in TOKEN_FIELDS}
    tot["calls"] = len(api_log)
    tot["retries"] = len(api_log) - len({(e.get("seed"), e.get("step")) for e in api_log})  # re-asks after a bad reply
    tot["cache_hit_rate"] = tot["cached_tokens"] / tot["prompt_tokens"] if tot["prompt_tokens"] else None
    return tot


def format_usage(u):
    hit = "n/a" if u["cache_hit_rate"] is None else f"{u['cache_hit_rate']:.0%}"
    return (f"{u['calls']} calls ({u['retries']} retries) | input {u['prompt_tokens']:,} tokens, "
            f"{u['cached_tokens']:,} cached ({hit}) | output {u['completion_tokens']:,} tokens, "
            f"of which reasoning {u['reasoning_tokens']:,}")


def elm_costs():
    """ELM's running spend for this API key: {'usage', 'limit', 'currency', 'asOf'}, or None.
    ELM only syncs this every so often, so a run that just finished may not show yet."""
    import urllib.request

    base = os.getenv("ELM_BASE_URL", "https://elm.edina.ac.uk/api/v1").strip().rstrip("/")
    req = urllib.request.Request(base + "/costs",
                                 headers={"Authorization": f"Bearer {os.getenv('ELM_API_KEY', '').strip()}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.load(r)
    except Exception as e:  # never stop a run over this
        print(f"  [note] could not read ELM /costs: {e}")
        return None


def format_costs(c):
    return f"{c.get('usage')} of {c.get('limit')} {c.get('currency', '')} used (as of {c.get('asOf')})"


def make_client():
    from openai import OpenAI

    load_env()
    return OpenAI(base_url=os.getenv("ELM_BASE_URL", "https://elm.edina.ac.uk/api/v1"),
                  api_key=os.environ["ELM_API_KEY"])


@dataclass
class LLMPlayer:
    client: object
    model: str = "gpt-4o-mini"
    temperature: float = 1.0
    reasoning_effort: str | None = None  # e.g. none/low/medium/high; None = model default
    max_retries: int = 3  # re-asks after an unparseable / illegal reply (costs no action)
    k_hypotheses: int = 0  # >0: state this many hypotheses in every reply (--hypotheses)
    cache_key: str | None = None  # prompt_cache_key; None = llm_play-h<k>, so runs with the same prompt share it
    # "full": the whole conversation stays in context, including the model's own earlier replies.
    # "table": each turn the model sees only the system prompt and the current screen (last result,
    # crystals, table of attempted fusions and harvests), like a participant; no earlier replies.
    history: str = "table"
    # False (policy actors): the model states hypotheses only; act() returns no action
    chooses_action: bool = True
    messages: list = field(default_factory=list)  # what the model is sent
    log: list = field(default_factory=list)  # every message of the game, for the transcript
    api_log: list = field(default_factory=list)
    # set by act() when k_hypotheses > 0: what the accepted reply said
    last_hypotheses: list = field(default_factory=list)
    last_none_of_these: float | None = None
    last_hypothesis_problems: list = field(default_factory=list)

    def start(self, game):
        prompt = TASK_PROMPT.format(max_actions=game.max_actions)
        if self.k_hypotheses:
            from hypotheses import EXPLANATION_TEXT, GRAMMAR_TEXT  # lazy: hypotheses.py imports this module
            template = HYPOTHESIS_PROMPT if self.chooses_action else HYPOTHESES_ONLY_PROMPT
            prompt += template.format(k=self.k_hypotheses, grammar=GRAMMAR_TEXT, explanation=EXPLANATION_TEXT)
        else:
            prompt += TURN_PROMPT
        self.messages = [{"role": "system", "content": prompt}]
        self.log = list(self.messages)

    def _chat(self, step):
        for attempt in range(5):
            try:
                extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
                t0 = time.time()
                resp = create_completion(self.client, model=self.model, messages=self.messages,
                                         temperature=self.temperature,
                                         prompt_cache_key=self.cache_key or (
                                             f"llm_play-h{self.k_hypotheses}" + ("" if self.chooses_action else "-policy")),
                                         **extra)
                text = resp.choices[0].message.content or ""
                rejected = _REJECTED.get(self.model, set())
                self.api_log.append({"step": step, "attempt": attempt, "output": text,
                                     "seconds": round(time.time() - t0, 2),
                                     "temperature_sent": "temperature" not in rejected,
                                     "cache_key_sent": "prompt_cache_key" not in rejected,
                                     **usage_of(resp)})
                return text
            except Exception as e:  # network / rate limit
                print(f"  [api error, retrying] {e}")
                time.sleep(2 ** attempt)
        raise RuntimeError("API failed repeatedly")

    def act(self, game, user_msg, step):
        """Send user_msg, get a legal action. With history "full" the whole exchange
        stays in context; with "table" the context is cut back to the system prompt
        first, so the model sees only this turn (re-asks within the turn included)."""
        if self.history == "table":
            self.messages = self.messages[:1]
        n0 = len(self.messages)
        try:
            return self._act(game, user_msg, step)
        finally:
            self.log += self.messages[n0:]

    def _act(self, game, user_msg, step):
        """With k_hypotheses > 0 the reply must also hold a valid hypothesis block;
        a bad block is re-asked like a bad action (no action is used). If the
        retries run out with a legal action but a bad block, the action is still
        played and whatever hypotheses parsed are kept (problems are logged).
        With chooses_action False only the hypotheses are checked, and the
        returned action is None (the policy picks the move)."""
        self.messages.append({"role": "user", "content": user_msg})
        best = None  # last reply with a legal action, in case the hypotheses never come right
        for _ in range(self.max_retries + 1):
            text = self._chat(step)
            self.messages.append({"role": "assistant", "content": text})
            if self.chooses_action:
                action = parse_action(text)
                err = "The JSON has no valid action." if action is None else game.validate(action)
            else:
                action, err = None, None
            problems = []
            if self.k_hypotheses:
                hyps, other, problems = parse_hypotheses(text, self.k_hypotheses)
                if err is None:
                    best = (action, text, hyps, other, problems)
            if err is None and not problems:
                if self.k_hypotheses:
                    self.last_hypotheses, self.last_none_of_these, self.last_hypothesis_problems = hyps, other, []
                return action, text
            parts = ([err] if err else []) + problems
            self.messages.append({"role": "user", "content": "\n".join(parts) + (
                "\nNo action was used." if self.chooses_action else "") + "\nReply again with the corrected JSON."})
        if best:
            action, text, hyps, other, problems = best
            self.last_hypotheses, self.last_none_of_these, self.last_hypothesis_problems = hyps, other, problems
            return action, text
        self.last_hypotheses, self.last_none_of_these, self.last_hypothesis_problems = [], None, ["no legal reply"]
        return None, text


def hypothesis_record(game, action, hyps, other, problems):
    """Per-action hypothesis columns (--hypotheses). Call BEFORE game.step(action).

    weight_* use the stated probabilities, rescaled to sum to 1 with none_of_these:
      weight_predicts_success  weight on rules saying the chosen fusion works
      weight_on_refuted        weight on rules already contradicted by a past fusion
    top_* refer to the first parsed hypothesis (the model's most plausible)."""
    from hypotheses import score

    ok = [h for h in hyps if h.error is None]
    w, w_other = hypothesis_weights(hyps, other)
    score(ok, list(game.fusion_log))  # fills n_obs / n_correct against fusions before this action
    rec = {
        "hypotheses": json.dumps([{**h.to_dict(), "weight": wi} for h, wi in zip(ok, w)]
                                 + [{"rule": h.rule, "error": h.error} for h in hyps if h.error],
                                 ensure_ascii=False),
        "n_hypotheses_parsed": len(ok),
        "none_of_these": other, "weight_none_of_these": w_other,
        "top_rule": ok[0].rule if ok else None,
        "top_matches_true_rule": bool(ok and game.condition in ok[0].matches),
        "any_matches_true_rule": any(game.condition in h.matches for h in ok),
        "weight_on_true_rule": sum(wi for h, wi in zip(ok, w) if game.condition in h.matches),
        "weight_on_refuted": sum(wi for h, wi in zip(ok, w) if h.n_correct < h.n_obs),
        "top_predicts_success": None, "weight_predicts_success": None,
        "hypothesis_problems": "; ".join(problems),
    }
    if action[0] == "fuse":
        h, t = game.crystals[action[1]], game.crystals[action[2]]
        preds = [hy.predict(h, t) for hy in ok]
        rec["top_predicts_success"] = preds[0] if ok else None
        rec["weight_predicts_success"] = sum(wi for p, wi in zip(preds, w) if p)
    return rec


# ---------------------------------------------------------------------------
# Policy players (--actor thompson | top | expected)
# ---------------------------------------------------------------------------
# The LLM states its hypotheses only (HYPOTHESES_ONLY_PROMPT: no action is
# asked for) and the policy picks the move. All three policies use ONE rule:
# one-step look-ahead on the expected stop value.
#
#   S(s) = points banked + values of the n most valuable crystals on the board
#          (n = actions left): what "harvest from now on" earns, whatever the
#          rule. Crystals that will regenerate later are not counted: the policy
#          uses only what is on the screen, like the LLM and the participants.
#   harvest the best crystal:  score = S (unchanged)
#   fuse x onto y:             score = p * S(if it works) + (1 - p) * S(if it fails)
#
# and the highest score is played. They differ only in p(x, y), the
# probability that the fusion works:
#   expected  p = sum of the weights of the hypotheses that say it fuses
#                 (+ weight of none_of_these * p0)       -- uses all the beliefs
#   thompson  p = r(x, y) in {0, 1} for ONE hypothesis r sampled by weight
#   top       p = r(x, y) for the highest-weight hypothesis
# If none_of_these is the one sampled / the top, p = p0 for every untried pair.
# p0 is Laplace's rule of succession on the game so far: (successes + 1) /
# (fusions tried + 2). Kind pairs already tried are known (p = 1 or 0), since
# the game is deterministic.
#
# Because "harvest from now on" is worth S under every rule, these scores are
# its exact Q-values in the belief MDP, so by policy improvement the policy's
# expected return (under the stated beliefs) is never below S. Zhao, Velez &
# Griffiths' (2024) stopping rule is the same comparison with a constant p.
# Ties are broken uniformly at random over all the tied moves, with no
# preference between known-working fusions, untried fusions and harvesting.
# (When more actions are left than there are crystals on the board, a failed
# fusion costs nothing, so untried fusions tie with harvesting early on.)

ACTORS = ("llm", "thompson", "top", "expected")


def _value(c):
    return 10 ** c["level"]


def _top_sum(values, n):
    return sum(sorted(values, reverse=True)[:max(n, 0)])


def laplace_rate(game):
    """P(an untried fusion works) by the rule of succession on this game's fusions."""
    return (sum(1 for *_, y in game.fusion_log if y) + 1) / (len(game.fusion_log) + 2)


def stop_value(game, crystals, extra, n):
    """S: points banked + the n most valuable crystals on the board (plus `extra`
    values, e.g. a fusion's result). Only what is on the screen counts: crystals
    that will regenerate later are left out, as the player cannot see them yet."""
    values = [_value(c) for c in crystals] + extra
    return game.points + _top_sum(values, n)


def policy_scores(game, p_fn):
    """Expected stop value of every move: [(score, action, p, untried)].
    p_fn(h, t) is the probability that fusing h onto t works, for an untried
    kind pair. One entry per kind pair (crystals of the same kinds are
    interchangeable), plus harvesting the best crystal, which leaves S unchanged.
    Uses only what is on the screen: no knowledge of regeneration."""
    n = game.actions_left
    crystals = list(game.crystals.values())
    known = {(h, t): bool(y) for h, t, y in game.fusion_log}
    out = []
    if crystals:
        best = max(crystals, key=_value)  # harvesting a smaller crystal never scores higher
        out.append((stop_value(game, crystals, [], n), ("harvest", best["id"]), None, False))
    s_fail = stop_value(game, crystals, [], n - 1)
    seen = set()
    for h in crystals:
        for t in crystals:
            key = (kind_name(h), kind_name(t))
            if h is t or key in seen:
                continue
            seen.add(key)
            level = max(h["level"], t["level"]) + 1
            if level >= MAXLEVEL:
                p = 0.0  # the cap: fails under every rule
            elif key in known:
                p = 1.0 if known[key] else 0.0
            else:
                p = float(p_fn(h, t))
            if key in known and not known[key]:
                continue  # never repeat a known failure
            rest = [c for c in crystals if c is not h and c is not t]
            s_ok = stop_value(game, rest, [10 ** level], n - 1)  # h and t replaced by the new crystal
            out.append((p * s_ok + (1 - p) * s_fail, ("fuse", h["id"], t["id"]), p,
                        key not in known and level < MAXLEVEL))
    return out


def policy_action(game, hyps, other, actor, rng):
    """Choose the move for a policy actor from the LLM's stated hypotheses.
    Returns (action, info); info says what was acted on and the move's score."""
    ok = [h for h in hyps if h.error is None]
    w, w_other = hypothesis_weights(hyps, other)
    p0 = laplace_rate(game)
    if actor == "expected":
        def p_fn(h, t):
            return sum(wi for hy, wi in zip(ok, w) if hy.predict(h, t)) + w_other * p0
        info = {"acted_on": "all hypotheses (expected value)", "acted_on_rule": None, "acted_on_weight": None}
    else:
        options, weights = ok + [None], w + [w_other]
        if actor == "thompson":
            choice = rng.choices(options, weights)[0] if sum(weights) > 0 else None
        else:  # top
            choice = options[max(range(len(options)), key=lambda i: weights[i])]
        weight = weights[options.index(choice)]
        if choice is None:
            p_fn = lambda h, t: p0  # noqa: E731
            info = {"acted_on": "none_of_these", "acted_on_rule": None, "acted_on_weight": weight}
        else:
            p_fn = lambda h, t: 1.0 if choice.predict(h, t) else 0.0  # noqa: E731
            info = {"acted_on": choice.explanation or choice.rule, "acted_on_rule": choice.rule,
                    "acted_on_weight": weight}

    scores = policy_scores(game, p_fn)
    top = max(s for s, *_ in scores)
    tied = [x for x in scores if x[0] >= top - 1e-9 * max(1.0, abs(top))]
    score, action, p, untried = rng.choice(sorted(tied, key=lambda x: x[1]))  # uniform over ties
    harvest = next((s for s, a, *_ in scores if a[0] == "harvest"), None)
    info.update(policy_score=score, policy_p=p, p0=p0,
                # a fusion that scores no better than harvesting: played because it costs nothing
                free_test=bool(action[0] == "fuse" and harvest is not None and score <= harvest + 1e-9))
    return action, info


def step_usage(api_log, step):
    """Per-action token columns: every API call made for this action, re-asks included."""
    calls = [e for e in api_log if e.get("step") == step]
    return {"api_calls": len(calls), **{f: sum(e.get(f) or 0 for e in calls) for f in TOKEN_FIELDS}}


# ---------------------------------------------------------------------------
# Replaying human games: the model is asked exactly as in its own games, but the
# participant's logged move is played
# ---------------------------------------------------------------------------

# In the browser game a used base crystal reappears after a few seconds of real
# time (JS: max(1 s, N(5 s, 2 s)), plus 2 s for the second crystal of a fusion),
# and the data has no timestamps. Reappearing straight away, once per kind, is the
# only setting under which every low and medium participant's moves are possible.
HUMAN_REGEN = {"regen_delay": 0, "regen_stagger": False}  # = the defaults; kept explicit for the replays


def crystal_of_kind(game, kind, exclude=None):
    """The lowest-id crystal of a kind ('circle_dots_0'), or None."""
    for c in sorted(game.crystals.values(), key=lambda c: c["id"]):
        if kind_name(c) == kind and c["id"] != exclude:
            return c
    return None


def load_human_games(path, condition, participants=None):
    """{participant id: [(action, held, target, yield), ...]} from a human action_data CSV.
    Only these columns are read (the file also holds participant tokens)."""
    import pandas as pd
    df = pd.read_csv(path, usecols=["id", "action_id", "action", "held", "target", "yield", "condition"])
    df = df[df.condition == condition]
    if participants:
        df = df[df.id.isin(participants)]
    return {int(pid): [(r["action"], r["held"], r["target"] if isinstance(r["target"], str) else "",
                        r["yield"] if isinstance(r["yield"], str) else "")
                       for _, r in g.sort_values("action_id").iterrows()]
            for pid, g in df.groupby("id")}


def human_action(game, move):
    """The engine action for a logged human move (by crystal kinds)."""
    kind, held, target, _ = move
    h = crystal_of_kind(game, held)
    t = crystal_of_kind(game, target, exclude=h["id"]) if h and kind == "combine" else None
    if h is None or (kind == "combine" and t is None):
        raise ValueError(f"{held if h is None else target} is not on the board")
    return ("fuse", h["id"], t["id"]) if kind == "combine" else ("harvest", h["id"])


def check_human_replay(condition, moves, seed):
    """Replay a human game without the model; raise if a move is impossible or a
    fusion gives a different result from the log."""
    game = CrystalGame(condition, seed=seed, max_actions=len(moves), **HUMAN_REGEN)
    for i, move in enumerate(moves, 1):
        r = game.step(human_action(game, move))
        if move[0] == "combine" and (r["yield"] or "") != move[3]:
            raise ValueError(f"action {i}: replay gave {r['yield']!r}, log has {move[3]!r}")
    return game


def play(game, player, player_id=0, verbose=True, actor="llm", policy_seed=None, moves=None):
    """Play one game. Returns per-action records in the human action_data format
    (plus hypothesis columns when the player states hypotheses).

    actor "llm" plays the model's own action; "thompson" / "top" play the policy's
    action on the model's stated hypotheses (the model's action is still logged).

    moves (a participant's logged moves) replays a human game: the model gets exactly
    the prompt it would get in its own game, its move is logged as llm_action and
    discarded, and the participant's move is played. On the participant's harvest
    turns the model is not called: with history "table" it keeps nothing between
    turns, so this changes no later prompt."""
    player.start(game)
    records = []
    rng = random.Random(policy_seed)
    # the closing line asks for what this player's reply must hold (no "Result:" line:
    # the result of the last action is the last line of the table above it)
    ask = ("State your hypotheses." if not getattr(player, "chooses_action", True)
           else "State your hypotheses and your action." if getattr(player, "k_hypotheses", 0)
           else "State your action.")
    msg = f"{game.observation()}\n\n{ask}"
    while not game.done:
        step = game.actions_used + 1
        asked = moves is None or moves[step - 1][0] == "combine"
        llm_action, raw = player.act(game, msg, step) if asked else (None, "")
        action, policy_cols = llm_action, {}
        if moves is not None:
            action = human_action(game, moves[step - 1])
            policy_cols = {"moves_from": "human", "llm_action": json.dumps(llm_action) if llm_action else ""}
        if actor != "llm":  # the model stated hypotheses only; the policy picks the move
            action, info = policy_action(game, player.last_hypotheses, player.last_none_of_these, actor, rng)
            policy_cols = {"actor": actor, **info}
        fallback = action is None
        if fallback:  # model never produced a legal action: harvest a random crystal
            action = ("harvest", game.rng.choice(list(game.crystals)))
        hyp_cols = (hypothesis_record(game, action, player.last_hypotheses, player.last_none_of_these,
                                      player.last_hypothesis_problems)
                    if getattr(player, "k_hypotheses", 0) and asked else {})
        r = game.step(action)
        if moves is not None and moves[step - 1][0] == "combine" and (r["yield"] or "") != moves[step - 1][3]:
            raise ValueError(f"participant {player_id} action {step}: replay gave {r['yield']!r}, "
                             f"log has {moves[step - 1][3]!r}")
        records.append({
            "id": player_id, "action_id": step, "action": "combine" if r["action"] == "fuse" else "consume",
            "held": r["held"], "target": r["target"], "yield": r["yield"], "points": r["points"],
            "condition": game.condition, "total_points": game.points, "fallback": fallback,
            "regenerated": ";".join(r["regenerated"]), "model_output": raw, **hyp_cols, **policy_cols,
            **step_usage(player.api_log, step),
        })
        if verbose:
            if hyp_cols:  # the stated hypotheses, in plain English, with the model's probabilities
                for hy in player.last_hypotheses:
                    p = "  ?  " if hy.stated_confidence is None else f"{hy.stated_confidence:5.2f}"
                    note = (" [unparsable rule]" if hy.error
                            else " [= true rule]" if game.condition in hy.matches else "")
                    print(f"      {p}  {hy.explanation or hy.rule}{note}")
                other = player.last_none_of_these
                print(f"      {'  ?  ' if other is None else f'{other:5.2f}'}  (none of these)")
            extra = (f"  [P(success) {hyp_cols['weight_predicts_success']:.2f}]"
                     if hyp_cols and hyp_cols["weight_predicts_success"] is not None else "")
            if policy_cols and moves is not None:
                print(f"      model would have played: {llm_action}")
            elif policy_cols:
                wt = policy_cols["acted_on_weight"]
                pp = policy_cols["policy_p"]
                print(f"      {actor} acted on: {policy_cols['acted_on']}"
                      f"{'' if wt is None else f' ({wt:.2f})'}"
                      f"{'' if pp is None else f' | P(works) {pp:.2f}'}"
                      f"{' | free test' if policy_cols['free_test'] else ''}")
            print(f"  {step:2d}. {action} -> {r['text']}{extra}")
        msg = f"{game.observation()}\n\n{ask}"
    return records


# ---------------------------------------------------------------------------
# Check the rules reproduce the human data
# ---------------------------------------------------------------------------

def check_rules_against_human_data(path):
    import pandas as pd

    def parse(name):
        s, t, lvl = name.split("_")
        return {"shape": s, "texture": t, "level": int(lvl)}

    df = pd.read_csv(path)
    df = df[df.action == "combine"]
    all_ok = True
    for cond, g in df.groupby("condition"):
        if cond not in RULES:
            print(f"{cond}: no rule defined, skipped")
            continue
        errors = 0
        for h, t, y in zip(g.held, g.target, g["yield"].fillna("")):
            h, t = parse(h), parse(t)
            new_level = max(h["level"], t["level"]) + 1
            ok = new_level < MAXLEVEL and RULES[cond](h, t)
            pred = f"{h['shape']}_{t['texture']}_{new_level}" if ok else ""
            errors += pred != y
        all_ok &= errors == 0
        print(f"{cond}: {len(g)} human fusions, {errors} mismatches")
    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    load_env()  # before argparse, so ELM_MODEL from .env sets the --model default
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", choices=list(RULES), default="high")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--actions", type=int, default=MAX_ACTIONS)
    p.add_argument("--model", default=os.getenv("ELM_MODEL", "gpt-5.6-sol"))
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--reasoning-effort", default=os.getenv("ELM_REASONING_EFFORT"),
                   help="e.g. none, low, medium, high (model-dependent); default: ELM_REASONING_EFFORT or the model's default")
    p.add_argument("--regen-delay", type=int, default=REGEN_DELAY)
    p.add_argument("--hypotheses", type=int, default=0, metavar="K",
                   help="state K hypotheses (with probabilities) in the same reply as each action; 0 = off")
    p.add_argument("--actor", choices=ACTORS, default="llm",
                   help="who picks the move: llm = the model's own action. The others play the move with the "
                        "highest expected stop value given the model's hypotheses (which is then all the model "
                        "is asked for; needs --hypotheses): thompson = as if one hypothesis, sampled by its "
                        "weight, were true; top = as if the highest-weight hypothesis were true; expected = "
                        "averaging over all the hypotheses by weight")
    p.add_argument("--history", choices=["full", "table"], default="table",
                   help="table (default): each turn the model sees only the current screen (last result, crystals, "
                        "table of attempted fusions and harvests), like a participant; "
                        "full: it keeps the whole conversation, its own earlier replies included")
    p.add_argument("--output", default="results")
    p.add_argument("--name", default="llm_play")
    p.add_argument("--check-human-data", metavar="CSV", help="verify the rules against a human action_data CSV and exit")
    p.add_argument("--costs", action="store_true", help="print your ELM spend and limit, and exit")
    p.add_argument("--replay-human", metavar="CSV",
                   help="replay participants' games from a human action_data CSV (with --condition): the model "
                        "states hypotheses on each of their fusion turns with exactly the prompt of its own games, "
                        "and their move is played instead of the model's. Needs --hypotheses and --history table")
    p.add_argument("--participants", type=int, nargs="+", help="with --replay-human: these participant ids only")
    p.add_argument("--limit", type=int, help="with --replay-human: the first N participants only")
    p.add_argument("--dry-run", action="store_true",
                   help="with --replay-human: check the replays and count the calls, without calling the model")
    args = p.parse_args()

    if args.check_human_data:
        raise SystemExit(0 if check_rules_against_human_data(args.check_human_data) else 1)
    if args.actor != "llm" and not args.hypotheses:
        p.error(f"--actor {args.actor} acts on the model's hypotheses, so it needs --hypotheses K")
    games = {}
    if args.replay_human:
        if not args.hypotheses or args.history != "table" or args.actor != "llm":
            p.error("--replay-human needs --hypotheses K, --history table and --actor llm")
        games = load_human_games(args.replay_human, args.condition, args.participants)
        if args.limit:
            games = dict(list(games.items())[:args.limit])
        if not games:
            p.error(f"no {args.condition} participants found in {args.replay_human}")
        for pid, mv in games.items():
            check_human_replay(args.condition, mv, pid)
        calls = sum(m[0] == "combine" for mv in games.values() for m in mv)
        print(f"{len(games)} {args.condition} participants replay correctly; {calls} fusion turns, "
              f"so {calls} model calls plus any re-asks")
        if args.dry_run:
            raise SystemExit(0)
        args.seeds = list(games)
    if args.costs:
        c = elm_costs()
        print(format_costs(c) if c else "ELM /costs unavailable")
        raise SystemExit(0 if c else 1)

    import pandas as pd

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    client = make_client()
    all_records, all_calls, per_game_usage = [], [], []
    costs_start = elm_costs()
    if costs_start:
        print(f"ELM spend before this run: {format_costs(costs_start)}")
    for seed in args.seeds:
        print(f"=== condition={args.condition} seed={seed} model={args.model} "
              f"reasoning_effort={args.reasoning_effort or 'model default'} "
              f"hypotheses={args.hypotheses or 'off'} history={args.history} actor={args.actor} base_url={client.base_url} ===")
        moves = games.get(seed)
        game = (CrystalGame(args.condition, seed=seed, max_actions=len(moves), **HUMAN_REGEN) if moves else
                CrystalGame(args.condition, seed=seed, max_actions=args.actions, regen_delay=args.regen_delay))
        player = LLMPlayer(client, model=args.model, temperature=args.temperature,
                           reasoning_effort=args.reasoning_effort, k_hypotheses=args.hypotheses,
                           history=args.history, chooses_action=args.actor == "llm")
        records = play(game, player, player_id=seed, actor=args.actor,
                       policy_seed=f"policy-{seed}", moves=moves)  # own stream, not the game's Random(seed)
        all_records += records
        all_calls += [{**e, "seed": seed} for e in player.api_log]
        usage = usage_summary(player.api_log)
        per_game_usage.append(usage)
        print(f"Final score: {game.points:,}")
        print(f"Usage, seed {seed}: {format_usage(usage)}")
        base = out / f"{args.name}_{args.condition}_{'p' if moves else 'seed'}{seed}_{stamp}"
        with open(f"{base}_transcript.json", "w", encoding="utf-8") as f:
            json.dump({"condition": args.condition, "seed": seed, "model": args.model,
                       "temperature": args.temperature, "reasoning_effort": args.reasoning_effort,
                       "hypotheses": args.hypotheses, "final_points": game.points, "usage": usage,
                       "history": args.history, "actor": args.actor,
                       "messages": player.log, "api_log": player.api_log}, f, indent=2, ensure_ascii=False)
    # the games in the file name, so runs started in the same second don't overwrite each other
    tag = ("p" if games else "s") + "-".join(map(str, args.seeds[:6])) + ("-etc" if len(args.seeds) > 6 else "")
    csv = out / f"{args.name}_{args.condition}_{tag}_{stamp}_actions.csv"
    pd.DataFrame(all_records).to_csv(csv, index=False)
    print(f"Saved {csv}")

    total = usage_summary(all_calls)
    total["retries"] = sum(u["retries"] for u in per_game_usage)  # steps repeat across seeds
    print(f"\nUsage, whole run: {format_usage(total)}")
    if total["calls"] and not total["cached_tokens"]:
        print("  (no cached tokens reported: either ELM does not pass prompt caching through, or it does not report it)")
    costs_end = elm_costs()
    if costs_start and costs_end:
        try:
            spent = float(costs_end["usage"]) - float(costs_start["usage"])
            print(f"ELM spend after this run: {format_costs(costs_end)}\n"
                  f"Change during the run: {spent:.4f} {costs_end.get('currency', '')} "
                  f"(ELM syncs spend periodically, so recent calls may not show yet; run --costs later to check)")
        except (KeyError, TypeError, ValueError):
            print(f"ELM spend after this run: {format_costs(costs_end)}")
    with open(out / f"{args.name}_{args.condition}_{tag}_{stamp}_usage.json", "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "seeds": args.seeds, "actor": args.actor, "usage": total,
                   "elm_costs_before": costs_start, "elm_costs_after": costs_end}, f, indent=2)


if __name__ == "__main__":
    main()
