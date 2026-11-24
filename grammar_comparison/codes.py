"""Hand coding of every participant's end-of-game messages (messageHow + messageRules).

Each participant gets a category:
  rule      - states a (possibly wrong) condition on which pairs fuse, with no colour
  colour    - the stated condition refers to colour / level (e.g. "same colour")
  history   - condition depends on earlier fusions (not a fixed pair rule)
  outcome   - only describes what a fusion produces (shape from held, texture from ground)
  none      - no rule: uncertainty, controls, scoring tips, or too vague to formalise

For 'rule' (and 'history', coded on its fixed-pair part) the condition is written in a
neutral notation, evaluated over all 9216 (held, target) kind pairs. Names people used
are normalised: solid/blank/plain -> plain; spotty/dotted -> dots; lines/diagonal ->
stripes; checked/crosshatch/squared -> checkered; rhombus/kite -> diamond;
rectangle -> square; cycle/round -> circle.
"""
from domain import SH, ST, TH, TT, si, ti

# ---- neutral notation (numpy masks over the 9216 ordered pairs) ----
SS = SH == ST                     # same shape
DS = SH != ST                     # different shape
STX = TH == TT                    # same texture
DTX = TH != TT                    # different texture


def P(a, b):      # unordered shape pair {a, b}
    return ((SH == si(a)) & (ST == si(b))) | ((SH == si(b)) & (ST == si(a)))


def PO(a, b):     # ordered: held a, ground b
    return (SH == si(a)) & (ST == si(b))


def PT(a, b):     # unordered texture pair
    return ((TH == ti(a)) & (TT == ti(b))) | ((TH == ti(b)) & (TT == ti(a)))


def Sh(o, x):
    return (SH if o == "h" else ST) == si(x)


def Tx(o, y):
    return (TH if o == "h" else TT) == ti(y)


ONE_PLAIN = (Tx("h", "plain") & ~Tx("t", "plain")) | (Tx("t", "plain") & ~Tx("h", "plain"))
SQCI = P("square", "circle")
TRDI = P("triangle", "diamond")

# id -> (category, mask or None, short gloss)
CODES = {
    # ---------------- high (easy) condition: true rule = same shape ----------------
    3: ("rule", SS, "same shape"),
    5: ("rule", SS, "same shape"),
    8: ("rule", SS, "same shape"),
    16: ("rule", SS | STX, "same shape OR same pattern"),
    18: ("none", None, "controls"),
    19: ("rule", SS, "same shape (patterns 'similar' too, ambiguous)"),
    30: ("rule", SS, "same shape"),
    32: ("none", None, "'long spacing'"),
    38: ("rule", SS, "same shape"),
    39: ("none", None, "pay attention"),
    40: ("colour", None, "blue circle + red -> purple"),
    43: ("rule", SS & ONE_PLAIN, "same shape, one plain and one patterned (+ level sequence)"),
    44: ("rule", SS, "same shape"),
    47: ("rule", SS, "same shape, colour/pattern irrelevant"),
    51: ("colour", None, "lower energy fuses with higher energy"),
    54: ("rule", SS, "mostly same shape"),
    59: ("rule", SS & DTX, "same shape, different patterns"),
    61: ("rule", STX, "same pattern"),
    62: ("rule", SS, "same shape"),
    67: ("colour", None, "colour arithmetic"),
    72: ("none", None, "interface"),
    76: ("rule", SS, "same shape, different colours"),
    78: ("colour", None, "red with red"),
    79: ("rule", Sh("h", "square") & Sh("t", "square"), "squares together"),
    84: ("rule", SS, "only same shape"),
    85: ("rule", SS, "same shape"),
    86: ("rule", SS, "same shape"),
    92: ("rule", SS, "same shape"),
    99: ("none", None, "vague: 'coordinated' patterns"),
    101: ("rule", SS, "alike shapes"),
    102: ("rule", SS, "only same shape"),
    103: ("none", None, "none found"),
    104: ("rule", SS, "same shape (some patterns won't)"),
    106: ("rule", SS, "only identical shapes"),
    108: ("rule", SS, "form (shape) decides"),
    110: ("rule", SS & ~(Tx("h", "plain") | Tx("t", "plain")), "same shape, plain ones never fuse"),
    111: ("none", None, "none"),
    114: ("history", SS, "same shape, but no repeated pattern in a chain"),
    119: ("rule", SS & PT("plain", "stripes"), "same shape, plain with striped"),
    # ---------------- low (hard) condition ----------------
    2: ("none", None, "tips only"),
    4: ("none", None, "no pattern"),
    7: ("rule", SQCI & (Tx("h", "plain") | Tx("t", "plain")), "circle-square, usually with a plain one"),
    9: ("rule", (SQCI | TRDI) & Tx("h", "plain") & Tx("t", "plain"), "plain square-circle, plain triangle-diamond"),
    10: ("none", None, "find the orange square"),
    13: ("rule", DS, "always two different shapes"),
    14: ("none", None, "random"),
    15: ("rule", STX, "same pattern, any shape"),
    20: ("none", None, "nothing"),
    23: ("none", None, "unclear"),
    25: ("none", None, "controls"),
    26: ("colour", None, "same colour"),
    27: ("none", None, "no pattern"),
    28: ("none", None, "not sure"),
    33: ("none", None, "missing"),
    34: ("outcome", None, "shape from first, texture from second"),
    36: ("rule", DS, "same shapes don't combine"),
    41: ("none", None, "none"),
    45: ("rule", SS & STX, "look alike"),
    46: ("rule", P("diamond", "circle") | SQCI, "diamond-circle, square-circle"),
    48: ("none", None, "vague"),
    49: ("rule", SQCI | TRDI, "square-circle, triangle-diamond"),
    50: ("rule", SQCI, "square-circle"),
    53: ("none", None, "none found"),
    56: ("colour", None, "nothing combines with green"),
    58: ("rule", DS & DTX, "different shape and pattern (triangle-diamond best)"),
    60: ("rule", (SQCI & STX) | (TRDI & DTX), "circle-square same pattern OR triangle-diamond different pattern"),
    65: ("rule", SQCI & STX, "circle-square with matching pattern"),
    69: ("colour", None, "same colour"),
    70: ("rule", TRDI | SQCI, "triangle-diamond, circle-square"),
    73: ("rule", SQCI | TRDI, "square-circle, triangle-diamond"),
    77: ("rule", TRDI, "diamond-triangle"),
    82: ("rule", SQCI | TRDI, "circle-square, diamond-triangle"),
    88: ("none", None, "vague"),
    89: ("none", None, "none"),
    94: ("none", None, "none"),
    96: ("rule", DS, "same shapes don't combine"),
    98: ("colour", None, "similar colours rarely"),
    # ---------------- medium condition ----------------
    6: ("none", None, "rejects same shape, no positive rule"),
    11: ("none", None, "points multiply"),
    12: ("none", None, "controls"),
    17: ("rule", SQCI | TRDI, "square-circle, triangle-diamond, any pattern"),
    21: ("none", None, "strategy only"),
    22: ("none", None, "n/a"),
    24: ("none", None, "random"),
    29: ("rule", PT("plain", "dots"), "plain with dots"),
    31: ("rule", SQCI, "square-circle"),
    35: ("rule", SQCI, "circle-square (not all)"),
    37: ("none", None, "vague"),
    42: ("none", None, "point values"),
    52: ("rule", (SQCI | TRDI) & DTX, "square-circle, triangle-diamond, different patterns"),
    55: ("rule", STX, "same pattern"),
    57: ("rule", SQCI, "square-circle, pattern irrelevant"),
    63: ("none", None, "random"),
    64: ("outcome", None, "shape from held, pattern from ground"),
    66: ("rule", PT("plain", "stripes"), "plain with striped"),
    68: ("rule", SQCI & DTX, "square-circle, different patterns"),
    71: ("none", None, "controls"),
    74: ("none", None, "difficult"),
    75: ("none", None, "none"),
    80: ("rule", DS & DTX, "different shapes and patterns"),
    81: ("none", None, "none worked"),
    83: ("rule", SQCI & DTX, "circle-square, different patterns"),
    87: ("none", None, "guess"),
    90: ("rule", DS & DTX, "not same shape, not same pattern (+ recipe list)"),
    91: ("rule", SQCI | TRDI, "square-circle, kite-triangle"),
    93: ("none", None, "none"),
    95: ("rule", DS & DTX, "avoid similar shape or pattern"),
    97: ("rule", (SQCI | TRDI) & DTX, "triangle-diamond, square-circle, different pattern"),
    100: ("none", None, "colour values"),
    105: ("none", None, "no clue"),
    107: ("none", None, "none"),
    109: ("rule", TRDI, "triangle-diamond (mostly diamonds)"),
    112: ("rule", SQCI | TRDI, "diamond-triangle, circle-square"),
    113: ("rule", (SQCI | (Sh("h", "triangle") & Sh("t", "triangle"))) & ONE_PLAIN,
          "circle-square or triangle-triangle, one plain one patterned"),
    115: ("none", None, "luck"),
    116: ("rule", DS, "similar shapes never fuse"),
    117: ("rule", SQCI, "circle-square"),
    118: ("outcome", None, "shape from first, pattern from second"),
    120: ("rule", PO("triangle", "diamond") | PO("square", "circle"), "triangle->diamond, square->circle, order matters"),
}
