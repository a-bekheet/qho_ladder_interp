"""
Responsible for defining vocabulary,
token <-> id mappings,
encode and decode utils
"""
import enum
import operator
from typing import Dict, List
from .config import (
    N_MAX,
    OPERATORS,
    BOS_TOKEN,
    SEP_TOKEN,
    EOS_TOKEN,
    NULL_TOKEN,
    OP_PREFIX,
    STATE_PREFIX,
)

CONTROL_TOKENS: List[str] = [
    BOS_TOKEN,
    SEP_TOKEN,
    EOS_TOKEN,
    NULL_TOKEN,
]

OPERATOR_TOKENS: List[str] = [
    f"{OP_PREFIX}{op}" for op in OPERATORS
]

STATE_TOKENS: List[str] = [
    f"{STATE_PREFIX}{n}" for n in range(N_MAX + 1) # encodes kets to set of states
]

VOCAB: List[str] = (
    CONTROL_TOKENS
    + OPERATOR_TOKENS
    + STATE_TOKENS
)

assert len(VOCAB) == 4 + len(OPERATORS) + (N_MAX+1) # makes sure no stray tokens
assert len(VOCAB) == len(set(VOCAB)) # no duplicates

TOKEN_TO_ID: Dict[str, int] = {
    token: idx for idx, token in enumerate(VOCAB)
}

ID_TO_TOKEN: Dict[int, str] = {
    idx: token for token, idx in TOKEN_TO_ID.items()
}

for token, idx in TOKEN_TO_ID.items():
    assert ID_TO_TOKEN[idx] == token

# encoding

def encode(tokens: List[str]) -> List[int]:
    return [TOKEN_TO_ID[token] for token in tokens]

# deconding

def decode(ids: List[int]) -> List[str]:
    return[ID_TO_TOKEN[idx] for idx in ids]


