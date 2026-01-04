import pytest
from qho_ladder_interp.oracle import apply_ladder_operator


def test_lower_boundary():
    assert apply_ladder_operator("LOWER", 0) is None

def test_lower_nonzero():
    assert apply_ladder_operator("LOWER", 2) == 1

def test_raise():
    assert apply_ladder_operator("RAISE", 5) == 6

def test_identity():
    assert apply_ladder_operator("ID", 42) == 42

def test_invalid_operator():
    with pytest.raises(ValueError):
        apply_ladder_operator("POOP", 3)

def test_invalid_neg_n():
    with pytest.raises(ValueError):
        apply_ladder_operator("LOWER", -2)
    
def test_invalid_large_n():
    with pytest.raises(ValueError):
        apply_ladder_operator("RAISE", 9000) # its over 9000