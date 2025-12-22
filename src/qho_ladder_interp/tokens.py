"""
Responsible for defining vocabulary,
token <-> id mappings,
encode and decode utils
"""
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

    