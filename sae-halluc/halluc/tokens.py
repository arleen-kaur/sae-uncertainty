"""Coarse token-type classes, used to match / stratify hallucinated vs faithful tokens.

Entropy depends heavily on token type (function words are easy to predict; names and
numbers are not), and hallucinated spans are content-heavy, so token type must be
controlled before "uncertainty" can be interpreted.
"""

STOPWORDS = frozenset("""
a an the and or but nor so yet for of in on at to from by with without about into onto over
under between among through during before after above below up down out off again further
then once here there when where why how all any both each few more most other some such no
not only own same than too very can will just don should now is am are was were be been being
have has had having do does did doing i me my myself we our ours ourselves you your yours
yourself yourselves he him his himself she her hers herself it its itself they them their
theirs themselves what which who whom this that these those would could may might must shall
also if as because while until against per via
""".split())

SPACE = "space"
PUNCT = "punct"
NUMERIC = "numeric"
FUNCTION = "function"
CAPITALIZED = "capitalized"
CONTENT = "content"

ALL_TYPES = (SPACE, PUNCT, NUMERIC, FUNCTION, CAPITALIZED, CONTENT)


def token_type(text):
    """Classify a token's surface text (leading/trailing whitespace is ignored)."""
    t = text.strip()
    if not t:
        return SPACE
    if not any(c.isalnum() for c in t):
        return PUNCT
    if any(c.isdigit() for c in t):
        return NUMERIC
    if t.lower() in STOPWORDS:
        return FUNCTION
    if t[0].isupper():
        return CAPITALIZED
    return CONTENT
