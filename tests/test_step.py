import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import hdc  # Triggers C++ code compilation
from hdc import _mmhdc_cpp


@pytest.fixture
def step_inputs():
    """Simple 3-class, 4-feature problem with known non-trivial layout."""
    torch.manual_seed(0)
    num_classes = 3
    out_channels = 4

    # One sample per class that is clearly separated
    x = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # class 0
        [0.0, 1.0, 0.0, 0.0],  # class 1
        [0.0, 0.0, 1.0, 0.0],  # class 2
    ], dtype=torch.float32)
    y = torch.tensor([0, 1, 2], dtype=torch.int64)

    # Initialise prototypes to zeros so any non-trivial update is visible
    prototypes = torch.zeros(num_classes, out_channels, dtype=torch.float32)

    return x, y, prototypes


def test_step_executes(step_inputs):
    """step() should run without raising an exception."""
    x, y, prototypes = step_inputs
    lr, C = 0.1, 1.0
    result = _mmhdc_cpp.step(x, y, prototypes, lr, C)
    assert result is not None


def test_step_returns_tensor(step_inputs):
    """step() should return a torch.Tensor."""
    x, y, prototypes = step_inputs
    lr, C = 0.1, 1.0
    result = _mmhdc_cpp.step(x, y, prototypes, lr, C)
    assert isinstance(result, torch.Tensor)


def test_step_updates_prototypes(step_inputs):
    """step() should change at least one prototype value."""
    x, y, prototypes = step_inputs
    lr, C = 0.1, 1.0
    prototypes_before = prototypes.clone()
    result = _mmhdc_cpp.step(x, y, prototypes, lr, C)
    assert not torch.equal(result, prototypes_before), (
        "step() returned prototypes identical to the input — no update occurred"
    )
