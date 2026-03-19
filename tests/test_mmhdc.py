import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import hdc  # Triggers C++ code compilation
from hdc.mmhdc import MultiMMHDC


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NUM_CLASSES = 3
OUT_CHANNELS = 8
NUM_SAMPLES = 30  # 10 per class


@pytest.fixture
def training_data():
    """Synthetically separable multi-class dataset (float32)."""
    torch.manual_seed(42)
    xs, ys = [], []
    for cls in range(NUM_CLASSES):
        # Each class lives in a different quadrant so initialisation is non-trivial
        center = torch.zeros(OUT_CHANNELS)
        center[cls] = 2.0
        x_cls = center + 0.1 * torch.randn(NUM_SAMPLES // NUM_CLASSES, OUT_CHANNELS)
        xs.append(x_cls)
        ys.append(torch.full((NUM_SAMPLES // NUM_CLASSES,), cls, dtype=torch.long))
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    # L2-normalise so the model sees unit-norm inputs (matching typical HDC usage)
    x = x / x.norm(dim=1, keepdim=True)
    return x, y


@pytest.fixture(scope="module")
def mnist_training_data():
    """MNIST-derived training subset for optimisation-consistency checks."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, repo_root)
    from data import load_mnist

    # data loader resolves paths relative to cwd.
    orig_cwd = os.getcwd()
    os.chdir(repo_root)
    try:
        X_raw, y_raw, _, _ = load_mnist("mnist")
    finally:
        os.chdir(orig_cwd)

    x = torch.tensor(X_raw, dtype=torch.float32).reshape(len(X_raw), -1)
    y = torch.tensor(y_raw, dtype=torch.long)

    # Keep runtime low while preserving all classes.
    per_class = 64
    idx_parts = []
    for cls in torch.unique(y):
        cls_idx = (y == cls).nonzero(as_tuple=False).flatten()[:per_class]
        idx_parts.append(cls_idx)
    idx = torch.cat(idx_parts)

    x = x[idx]
    y = y[idx]
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return x, y


@pytest.fixture(params=["python",])
def model(request):
    """Return a fresh MultiMMHDC model for each backend."""
    return MultiMMHDC(
        num_classes=NUM_CLASSES,
        out_channels=OUT_CHANNELS,
        lr=0.5,
        C=1.0,
        backend=request.param,
    )


# ---------------------------------------------------------------------------
# 1. Training actually runs
# ---------------------------------------------------------------------------

class TestTrainingRuns:
    """Verify that the full training loop executes without errors."""

    def test_initialize_does_not_raise(self, model, training_data):
        x, y = training_data
        model.initialize(x, y)  # should not raise

    def test_single_step_does_not_raise(self, model, training_data):
        x, y = training_data
        model.initialize(x, y)
        model.step(x, y)  # should not raise

    def test_multiple_steps_do_not_raise(self, model, training_data):
        x, y = training_data
        model.initialize(x, y)
        for _ in range(5):
            model.step(x, y)

    def test_step_returns_loss_tensor(self, model, training_data):
        x, y = training_data
        model.initialize(x, y)
        loss = model.step(x, y)
        assert isinstance(loss, torch.Tensor), (
            f"step() should return a loss Tensor, got {type(loss)}"
        )
        assert loss.ndim == 0, "Loss should be a scalar tensor"

    def test_forward_runs_after_training(self, model, training_data):
        x, y = training_data
        model.initialize(x, y)
        model.step(x, y)
        preds = model(x)
        assert preds.shape == (x.shape[0],), (
            f"forward() output shape {preds.shape} does not match input batch size"
        )

    def test_forward_output_in_valid_range(self, model, training_data):
        x, y = training_data
        model.initialize(x, y)
        model.step(x, y)
        preds = model(x)
        assert preds.min() >= 0 and preds.max() < NUM_CLASSES, (
            "Predictions contain class indices outside [0, num_classes)"
        )


# ---------------------------------------------------------------------------
# 2. Prototypes are changing
# ---------------------------------------------------------------------------

class TestPrototypesChange:
    """Verify that training updates the prototype vectors."""

    def test_initialize_sets_nonzero_prototypes(self, model, training_data):
        x, y = training_data
        assert torch.all(model.prototypes == 0), "Prototypes should start at zero"
        model.initialize(x, y)
        assert not torch.all(model.prototypes == 0), (
            "initialize() left all prototypes at zero"
        )

    def test_step_changes_prototypes(self, model, training_data):
        x, y = training_data
        model.initialize(x, y)
        prototypes_before = model.prototypes.data.clone()
        model.step(x, y)
        assert not torch.equal(model.prototypes.data, prototypes_before), (
            "step() did not change any prototype"
        )

    def test_prototypes_differ_across_steps(self, model, training_data):
        x, y = training_data
        model.initialize(x, y)
        snapshots = []
        for _ in range(3):
            model.step(x, y)
            snapshots.append(model.prototypes.data.clone())

        assert not torch.equal(snapshots[0], snapshots[1]) or \
               not torch.equal(snapshots[1], snapshots[2]), (
            "Prototypes did not change across any consecutive steps"
        )

    def test_each_class_prototype_changes(self, model, training_data):
        x, y = training_data
        model.initialize(x, y)
        prototypes_before = model.prototypes.data.clone()
        # Run enough steps so that all class prototypes get updated
        for _ in range(10):
            model.step(x, y)
        for cls in range(NUM_CLASSES):
            assert not torch.equal(
                model.prototypes.data[cls], prototypes_before[cls]
            ), f"Prototype for class {cls} was never updated"


# ---------------------------------------------------------------------------
# 3. Step update consistency with gradient descent
# ---------------------------------------------------------------------------

class TestStepMatchesGradientDescent:
    """Verify that manual step-based optimisation tracks autograd GD on the same loss."""

    def test_step_and_gradient_descent_losses_match(self, mnist_training_data):
        x, y = mnist_training_data
        lr = 0.05
        C = 1.0
        num_steps = 10

        num_classes = int(y.max().item()) + 1
        out_channels = x.shape[1]

        model_step = MultiMMHDC(
            num_classes=num_classes,
            out_channels=out_channels,
            lr=lr,
            C=C,
            backend="python",
        )
        model_step.initialize(x, y)

        model_gd = MultiMMHDC(
            num_classes=num_classes,
            out_channels=out_channels,
            lr=lr,
            C=C,
            backend="python",
        )
        model_gd.prototypes = torch.nn.Parameter(
            model_step.prototypes.detach().clone(),
            requires_grad=True,
        )

        # Start SGD from the same initial prototypes as the step-based model.
        optimizer = torch.optim.SGD([model_gd.prototypes], lr=lr)

        step_losses = []
        gd_losses = []

        for _ in range(num_steps):
            # Step-based optimisation path.
            model_step.step(x, y)
            step_losses.append(model_step.loss(x, y).detach())

            # SGD path on the same objective.
            optimizer.zero_grad()
            loss_gd = model_gd.loss(x, y)
            loss_gd.backward()
            optimizer.step()

            gd_losses.append(model_gd.loss(x, y).detach())

        step_losses_t = torch.stack(step_losses)
        gd_losses_t = torch.stack(gd_losses)
        abs_diff_t = (step_losses_t - gd_losses_t).abs()

        print("\n[step-vs-sgd] loss table")
        print("step | step_loss      | sgd_loss       | abs_diff")
        print("-----+----------------+----------------+----------------")
        for idx, (s_loss, g_loss, diff) in enumerate(zip(step_losses_t, gd_losses_t, abs_diff_t), start=1):
            print(f"{idx:>4} | {s_loss.item():>14.6f} | {g_loss.item():>14.6f} | {diff.item():>14.6f}")
        print(f"[step-vs-sgd] max abs diff: {abs_diff_t.max().item():.3e}")

        # The two optimisation trajectories should be very close.
        assert torch.allclose(step_losses_t, gd_losses_t, atol=1e-4, rtol=1e-4), (
            "Step-based and gradient-descent losses diverge. "
            f"max_abs_diff={abs_diff_t.max().item():.3e}"
        )
