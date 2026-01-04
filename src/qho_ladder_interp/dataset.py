from typing import Dict, List, Optional
import random

from .config import (
    N_MAX,
    OPERATORS,
    BOS_TOKEN,
    SEP_TOKEN,
    EOS_TOKEN,
    NULL_TOKEN,
    OP_PREFIX,
    STATE_PREFIX,
    DEFAULT_SEED,
)
from .tokens import encode
from .oracle import apply_ladder_operator

def sample_state(op: str, rng: random.Random) -> int:
    """
    Sample a valid quantum number n conditioned on the operator
    """
    if op == "RAISE":
        return rng.randint(0, N_MAX - 1)    
    return rng.randint(0, N_MAX)

def generate_example(rng: random.Random) -> Dict:
    """
    Generate a single supervised example with metadata
    """
    op = rng.choice(OPERATORS) # random operator
    n = sample_state(op, rng)

    result: Optional[int] = apply_ladder_operator(op, n)

    # construct token sequence
    input_tokens: List[str] = [
        BOS_TOKEN,
        f"{OP_PREFIX}{op}",
        f"{STATE_PREFIX}{n}",
        SEP_TOKEN
    ]

    if result is None:
        output_token = NULL_TOKEN
    else:
        output_token = f"{STATE_PREFIX}{result}"

    full_tokens: List[str] = (
        input_tokens
        + [output_token, EOS_TOKEN]
    )

    input_ids: List[int] = encode(full_tokens)

    # loss mask - supervise only first output token
    labels: List[int] = [-100] * len(input_ids)
    output_position = len(input_tokens)
    labels[output_position] = input_ids[output_position]

    metadata = {
        "operator": op,
        "n_in": n,
        "n_out": result,
        "is_boundary_case": (op == "LOWER" and n == 0),
    }

    return {
        "input_ids": input_ids,
        "labels": labels,
        "metadata": metadata,
    }

def generate_dataset(num_examples: int, seed: int = DEFAULT_SEED) -> List[Dict]:
    """
    Generate a list of supervised examples.
    """
    return [generate_example(random.Random(seed)) for _ in range(num_examples)] # deterministic given a seed