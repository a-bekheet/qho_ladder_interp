# src/qho_ladder_interp/config.py

from typing import Final, Tuple


# Physics / task settings

N_MAX: Final[int] = 63

OPERATORS: Final[Tuple[str, ...]] = (
    "LOWER",
    "RAISE",
    "ID",
)

# Special Tokens

BOS_TOKEN: Final[str]= "<BOS>"
SEP_TOKEN: Final[str]= "<SEP>"
EOS_TOKEN: Final[str]= "<EOS>"
NULL_TOKEN: Final[str]= "NULL"

# Token Prefixes

OP_PREFIX: Final[str] = "OP="
STATE_PREFIX: Final[str] = "S"

DEFAULT_SEED: Final[int] = 42 # for randomness