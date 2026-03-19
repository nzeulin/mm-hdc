import time
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


# ── Real-data convergence: C++ step vs Python step on MNIST ─────────────────

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load all hyper-parameters from the project configs so that this test
# automatically stays in sync whenever configs/default_mnist_config.py or
# the referenced HDC config change.
sys.path.insert(0, _REPO_ROOT)
from importlib.util import spec_from_file_location, module_from_spec
from configs.default_mnist_config import get_config as _get_main_config

def _load_hdc_config(rel_path: str):
    abs_path = os.path.join(_REPO_ROOT, rel_path)
    spec = spec_from_file_location("_hdc_config", abs_path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()

_MAIN_CFG = _get_main_config()
_HDC_CFG  = _load_hdc_config(_MAIN_CFG.model_config_paths[0])


@pytest.fixture(scope="module")
def mnist_hd_data():
    """
    Fixture for loading MNIST dataset.
    """
    sys.path.insert(0, _REPO_ROOT)
    from data import load_mnist
    from hdc import HDTransform

    # data/__init__.py resolves MNIST file paths relative to os.getcwd()
    orig_cwd = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        X_raw, y_raw, _, _ = load_mnist("mnist")
    finally:
        os.chdir(orig_cwd)

    # N = 500
    X = torch.tensor(X_raw, dtype=torch.float32)
    y = torch.tensor(y_raw, dtype=torch.int64)

    # Flatten and apply HD transform – parameters come from default_mnist_config.py
    X_flat = X.reshape(X.shape[0], -1)                                       # (N, 784)
    transform = HDTransform(
        in_channels=X_flat.shape[1],                                # 784
        out_channels=_MAIN_CFG.dataset.model_dim,                   # dataset.model_dim
        seed=0,
        batch_size=1024,
        transform_type=_MAIN_CFG.dataset.mapping,                   # dataset.mapping
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    return transform(X_flat), y                                     # (N, model_dim), (N)


class TestCppVsPythonStepMNIST:
    """
    Verify that the C++ step function produces numerically identical results
    to the Python reference implementation when both are fed the same
    hyperdimensional MNIST features.

    Hyper-parameters are loaded at import time from:
      • configs/default_mnist_config.py              – num_classes, model_dim, mapping
      • configs/mnist/hdc/mmhdc_multi_config.py      – learning_rate, C
    """

    NUM_CLASSES = _MAIN_CFG.dataset.num_classes   # dataset.num_classes
    MODEL_DIM   = _MAIN_CFG.dataset.model_dim     # dataset.model_dim
    LR          = _HDC_CFG.learning_rate           # learning_rate
    C           = float(_HDC_CFG.C)               # C

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _make_py_model(self):
        from hdc.mmhdc import MultiMMHDC
        return MultiMMHDC(
            num_classes=self.NUM_CLASSES,
            out_channels=self.MODEL_DIM,
            lr=self.LR,
            C=self.C,
            backend="python",
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # tests
    # ------------------------------------------------------------------

    def test_single_step_matches_python(self, mnist_hd_data):
        """
        Test to verify that C++ and Python implementations of step() return identical prototypes.
        """
        X_hd, y = mnist_hd_data
        batch_size = _MAIN_CFG.training.batch_size   # training.batch_size

        py_model = self._make_py_model()

        # Shuffle with a fixed seed for reproducibility
        rng = torch.Generator()
        rng.manual_seed(0)
        perm = torch.randperm(X_hd.shape[0], generator=rng)
        X_batch = X_hd[perm[:batch_size]].to(py_model.device)
        y_batch = y[perm[:batch_size]].to(py_model.device)

        py_model.initialize(X_hd, y)
        proto_init = py_model.prototypes.data.clone()

        # Python reference step – modifies py_model.prototypes in-place
        py_model.step(X_batch, y_batch)
        py_protos = py_model.prototypes.data.clone()

        # C++ step – returns the updated prototypes
        cpp_protos = _mmhdc_cpp.step(X_batch, y_batch, proto_init.clone(), self.LR, self.C)

        max_diff = (cpp_protos - py_protos).abs().max().item()
        assert torch.allclose(cpp_protos, py_protos, atol=1e-5), (
            f"C++ and Python prototypes diverge after 1 step on MNIST data. "
            f"Max diff = {max_diff:.3e}"
        )

    def test_epoch_matches_python(self, mnist_hd_data):
        """
        Test to verify that C++ and Python step implementations produce identical prototypes after several training epochs.
        """
        X_hd, y = mnist_hd_data
        batch_size = _MAIN_CFG.training.batch_size

        py_model = self._make_py_model()
        py_model.initialize(X_hd, y)

        # C++ chain starts from the same initial prototypes
        cpp_protos = py_model.prototypes.data.clone()

        num_batches = (X_hd.shape[0] + batch_size - 1) // batch_size
        py_total_s  = 0.0
        cpp_total_s = 0.0

        epochs = 10
        for epoch in range(epochs):
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end   = start + batch_size
                X_batch = X_hd[start:end].to(py_model.device)
                y_batch = y[start:end].to(py_model.device)

                # Python step (updates py_model.prototypes in-place)
                t0 = time.perf_counter()
                py_model.step(X_batch, y_batch)
                py_total_s += time.perf_counter() - t0
                py_protos = py_model.prototypes.data.clone()

                # C++ step: carry the output of the previous batch forward
                t0 = time.perf_counter()
                cpp_protos = _mmhdc_cpp.step(X_batch, y_batch, cpp_protos.clone(), self.LR, self.C)
                cpp_total_s += time.perf_counter() - t0
                
                # Making sure that C++ and Python implementations result closely match
                max_diff = (cpp_protos - py_protos).abs().max().item()
                assert torch.allclose(cpp_protos, py_protos, atol=1e-4), (
                    f"C++ and Python prototypes diverge at batch "
                    f"{batch_idx + 1}/{num_batches} of the epoch on MNIST data. "
                    f"Max diff = {max_diff:.3e}"
                )

        # Checking speedup from using C++ implementation
        print(
            f"\n[test_epoch_matches_python] {num_batches} batches × {batch_size} samples"
            f"\n  Python  mean: {py_total_s*1e3 / (num_batches*epochs):.1f} ms"
            f"\n  C++     mean: {cpp_total_s*1e3 / (num_batches*epochs):.1f} ms"
            f"\n  Speedup:       {py_total_s / cpp_total_s:.2f}×"
        )


class TestStepMatchesGradientDescent:
    """
    Verify that step-based optimisation tracks autograd SGD descending on the same
    loss function, using the same HD-transformed MNIST batch.
    """

    NUM_CLASSES = _MAIN_CFG.dataset.num_classes
    MODEL_DIM   = _MAIN_CFG.dataset.model_dim
    LR          = _HDC_CFG.learning_rate
    C           = float(_HDC_CFG.C)

    def _make_py_model(self):
        from hdc.mmhdc import MultiMMHDC
        return MultiMMHDC(
            num_classes=self.NUM_CLASSES,
            out_channels=self.MODEL_DIM,
            lr=self.LR,
            C=self.C,
            backend="python",
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dtype=torch.float32,
        )

    def test_step_and_gradient_descent_losses_match(self, mnist_hd_data):
        """
        After each step-based update and each SGD update (both starting from the
        same initial prototypes), the losses evaluated on the same batch should be
        numerically identical — confirming that the closed-form step is the exact
        subgradient descent rule for the hinge loss objective.
        """
        X_hd, y = mnist_hd_data
        batch_size = _MAIN_CFG.training.batch_size
        X_batch = X_hd[:batch_size].to('cuda' if torch.cuda.is_available() else 'cpu')
        y_batch = y[:batch_size].to('cuda' if torch.cuda.is_available() else 'cpu')

        model_step = self._make_py_model()
        model_step.initialize(X_hd, y)

        model_gd = self._make_py_model()
        model_gd.prototypes = torch.nn.Parameter(
            model_step.prototypes.detach().clone(),
            requires_grad=True,
        )
        optimizer = torch.optim.SGD([model_gd.prototypes], lr=self.LR)

        num_steps = 50
        step_losses = []
        gd_losses   = []

        for _ in range(num_steps):
            # Step-based path.
            model_step.step(X_batch, y_batch)
            step_losses.append(model_step.loss(X_batch, y_batch).detach())

            # SGD path on the same objective.
            optimizer.zero_grad()
            model_gd.loss(X_batch, y_batch).backward()
            optimizer.step()
            gd_losses.append(model_gd.loss(X_batch, y_batch).detach())

        step_losses_t = torch.stack(step_losses)
        gd_losses_t   = torch.stack(gd_losses)
        abs_diff_t    = (step_losses_t - gd_losses_t).abs()

        print("\n[step-vs-sgd] loss table")
        print("step | step_loss      | sgd_loss       | abs_diff")
        print("-----+----------------+----------------+----------------")
        for idx, (s, g, d) in enumerate(zip(step_losses_t, gd_losses_t, abs_diff_t), start=1):
            print(f"{idx:>4} | {s.item():>14.6f} | {g.item():>14.6f} | {d.item():>14.6f}")
        print(f"[step-vs-sgd] max abs diff: {abs_diff_t.max().item():.3e}")

        assert torch.allclose(step_losses_t, gd_losses_t, atol=1e-4, rtol=1e-4), (
            "Step-based and gradient-descent losses diverge. "
            f"max_abs_diff={abs_diff_t.max().item():.3e}"
        )