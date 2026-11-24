"""Shared game domain: shapes, textures, levels, pair indexing, true rules.

Index conventions follow agent/llm_play.py (and the JS config), which the
medium/low rules depend on.
"""
import numpy as np

SHAPES = ["triangle", "circle", "square", "diamond"]
TEXTURES = ["plain", "checkered", "stripes", "dots"]
COLOURS = ["red", "orange", "yellow", "green", "blue", "purple"]  # = level 0..5
NS, NT, NL = 4, 4, 6
NK = NS * NT * NL            # 96 crystal kinds (shape, texture, level)
NP = NK * NK                 # 9216 ordered (held, target) kind pairs


def kind_index(s, t, l):
    return (s * NT + t) * NL + l


def kind_from_name(name):
    s, t, l = name.split("_")
    return kind_index(SHAPES.index(s), TEXTURES.index(t), int(l))


KS = np.array([k // (NT * NL) for k in range(NK)])
KT = np.array([(k // NL) % NT for k in range(NK)])
KL = np.array([k % NL for k in range(NK)])

# per-pair feature arrays over the 9216 ordered pairs (held = h, target = t)
H = np.repeat(np.arange(NK), NK)
T = np.tile(np.arange(NK), NK)
SH, TH, LH = KS[H], KT[H], KL[H]
ST, TT, LT = KS[T], KT[T], KL[T]
CAP = np.maximum(LH, LT) + 1 >= 6   # fusion impossible at max level (game cap)

si = SHAPES.index
ti = TEXTURES.index

TRUE_RULES = {
    "high": SH == ST,
    "medium": (SH + ST == 3) & (TH != TT),
    "low": (SH + ST == 3) & (TH >= TT),
}
CONDITION_MAP = {"easy": "high", "medium-1": "medium", "hard": "low",
                 "high": "high", "medium": "medium", "low": "low"}


def to_int(mask):
    """bool array (NP,) -> python int bitset"""
    return int.from_bytes(np.packbits(mask.astype(np.uint8), bitorder="little").tobytes(), "little")


def from_int(x):
    b = np.frombuffer(x.to_bytes(NP // 8, "little"), dtype=np.uint8)
    return np.unpackbits(b, bitorder="little").astype(bool)


FULL = (1 << NP) - 1
