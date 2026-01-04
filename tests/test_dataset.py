from importlib import metadata
import pytest
from qho_ladder_interp.config import N_MAX
from qho_ladder_interp.dataset import generate_dataset
from qho_ladder_interp.tokens import VOCAB, encode, decode

def test_no_invalid_raise():
    data = generate_dataset(1000)
    for ex in data:
        meta = ex["metadata"]
        if meta["operator"] == "RAISE":
            assert meta["n_in"] < N_MAX

def test_boundary_cases_labeled():
    data = generate_dataset(1000)
    for ex in data:
        meta = ex["metadata"]
        if meta["operator"] == "LOWER" and meta["n_in"] == 0:
            assert meta["is_boundary_case"]

def test_single_supervision_point():
    data = generate_dataset(10)
    for ex in data:
        labels = ex["labels"]
        assert sum(l != -100 for l in labels) == 1