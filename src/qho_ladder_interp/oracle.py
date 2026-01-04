"""
Contains the ladder operator function
"""
from typing import Optional
from .config import N_MAX, OPERATORS

def _validate_inputs(op: str, n: int) -> None:
    if op not in OPERATORS:
        raise ValueError(f"Unknown operator: {op}")
    if not isinstance(n, int):
        raise TypeError(f"n must be int, got {type(n)}")
    if n < 0 or n > N_MAX:
        raise ValueError(f"n must be in [0, {N_MAX}], got {n}")
    
def apply_ladder_operator(op: str, n: int) -> Optional[int]:
    """
    Apply a ladder operator to number state |n>
    
    PARAMS:
    op : str
        {"LOWER", "RAISE", "ID"}.
    n : int
        Quantum Number, 0 <= n <= N_MAX.

    RETURNS:
    Optional[int]
        Resulting quantum number n', or None if null state (LOWER at n=0).
    """

    _validate_inputs(op, n)

    if op == "LOWER":
        if n == 0:
            return None
        return n - 1
    
    if op == "RAISE":
        return n + 1 # we have already validated input 
    
    if op == "ID":
        return n
    
    raise RuntimeError(f"Unhandled operator: {op}") # we can't really get here