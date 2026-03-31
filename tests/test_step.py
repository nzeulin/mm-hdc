import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.examples.mnist import get_config as _get_main_config
from data import load_mnist
from mmhdc import MultiMMHDC
from mmhdc.utils import HDTransform


_MAIN_CFG = _get_main_config()
_MODEL_CFG = _MAIN_CFG.model
_DEVICE = _MAIN_CFG.device
_TRANSFORM_DTYPE = _MODEL_CFG.get("transform_dtype", None) or torch.float32
_TRANSFORM_BATCH_SIZE = _MODEL_CFG.get("transform_batch_size", None)


def _get_batch():
    X_raw, y_raw, _, _ = load_mnist("mnist")
    X = X_raw.to(dtype=torch.float32)
    y = y_raw.to(dtype=torch.int64)

    X_flat = X.reshape(X.shape[0], -1)
    transform = HDTransform(
        in_channels=X_flat.shape[1],
        out_channels=_MAIN_CFG.dataset.model_dim,
        seed=0,
        batch_size=_TRANSFORM_BATCH_SIZE,
        normalize=bool(_MODEL_CFG.get("normalize", True)),
        device=_DEVICE,
        dtype=_TRANSFORM_DTYPE,
    )

    X_hd = transform(X_flat)
    batch_size = _MAIN_CFG.training.batch_size
    X_batch = X_hd[:batch_size].to(_DEVICE)
    y_batch = y[:batch_size].to(_DEVICE)
    return X_hd, y, X_batch, y_batch


def _make_model(backend: str, init_prototypes: torch.Tensor):
    model = MultiMMHDC(
        num_classes=_MAIN_CFG.dataset.num_classes,
        out_channels=_MAIN_CFG.dataset.model_dim,
        lr=float(_MODEL_CFG.learning_rate),
        C=float(_MODEL_CFG.C),
        backend=backend,
        device=_DEVICE,
        dtype=torch.float32,
    )
    model.prototypes.data = init_prototypes.detach().clone().to(_DEVICE)
    return model


def test_step_and_gradient_descent_losses_match(output_plot: bool = False):
    X_hd, y_all, X_batch, y_batch = _get_batch()

    # Build one common initialization so every method starts identically.
    init_model = MultiMMHDC(
        num_classes=_MAIN_CFG.dataset.num_classes,
        out_channels=_MAIN_CFG.dataset.model_dim,
        lr=float(_MODEL_CFG.learning_rate),
        C=float(_MODEL_CFG.C),
        backend="python",
        device=_DEVICE,
        dtype=torch.float32,
    )

    init_model.initialize(X_hd, y_all)
    init_prototypes = init_model.prototypes.detach().clone()

    model_cpp = _make_model("cpp", init_prototypes)
    model_py_opt = _make_model("python", init_prototypes)
    model_py_ref = _make_model("python", init_prototypes)

    model_gd = _make_model("python", init_prototypes)
    model_gd.prototypes = torch.nn.Parameter(
        init_prototypes.detach().clone().to(_DEVICE),
        requires_grad=True,
    )
    optimizer = torch.optim.SGD([model_gd.prototypes], lr=float(_MODEL_CFG.learning_rate))

    num_steps = 50
    losses_cpp = []
    losses_py_opt = []
    losses_py_ref = []
    losses_gd = []

    for _ in range(num_steps):
        model_cpp.step(X_batch, y_batch)
        losses_cpp.append(model_cpp.loss(X_batch, y_batch).detach())

        model_py_opt._py_step(X_batch, y_batch, optimized=True)
        losses_py_opt.append(model_py_opt.loss(X_batch, y_batch).detach())

        model_py_ref._py_step(X_batch, y_batch, optimized=False)
        losses_py_ref.append(model_py_ref.loss(X_batch, y_batch).detach())

        optimizer.zero_grad()
        model_gd.loss(X_batch, y_batch).backward()
        optimizer.step()
        losses_gd.append(model_gd.loss(X_batch, y_batch).detach())

    losses_cpp_t = torch.stack(losses_cpp)
    losses_py_opt_t = torch.stack(losses_py_opt)
    losses_py_ref_t = torch.stack(losses_py_ref)
    losses_gd_t = torch.stack(losses_gd)

    diff_cpp_t = (losses_cpp_t - losses_gd_t).abs()
    diff_py_opt_t = (losses_py_opt_t - losses_gd_t).abs()
    diff_py_ref_t = (losses_py_ref_t - losses_gd_t).abs()

    print("\n[step-loss-table]")
    print("step |          sgd |          cpp |       py_opt |       py_ref |   |cpp-sgd| | |py_opt-sgd| | |py_ref-sgd|")
    print("-----+--------------+--------------+--------------+--------------+-------------+---------------+--------------")
    for step_idx in range(num_steps):
        print(
            f"{step_idx + 1:>4} | "
            f"{losses_gd_t[step_idx].item():>12.6f} | "
            f"{losses_cpp_t[step_idx].item():>12.6f} | "
            f"{losses_py_opt_t[step_idx].item():>12.6f} | "
            f"{losses_py_ref_t[step_idx].item():>12.6f} | "
            f"{diff_cpp_t[step_idx].item():>11.6f} | "
            f"{diff_py_opt_t[step_idx].item():>13.6f} | "
            f"{diff_py_ref_t[step_idx].item():>12.6f}"
        )

    if output_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "step_vs_gd_losses.png")

        steps = range(1, num_steps + 1)
        plt.figure(figsize=(9, 5))
        plt.plot(steps, losses_gd_t.cpu().numpy(), label="SGD", marker="o")
        plt.plot(steps, losses_cpp_t.cpu().numpy(), label="C++", marker="s")
        plt.plot(steps, losses_py_opt_t.cpu().numpy(), label="Python optimized", marker="^")
        plt.plot(steps, losses_py_ref_t.cpu().numpy(), label="Python reference", marker="d")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Step Procedure Loss Comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[step-loss-plot] saved: {plot_path}")

    assert torch.allclose(losses_cpp_t, losses_gd_t, atol=1e-4, rtol=1e-4), (
        "C++ step losses diverge from SGD. "
        f"max_abs_diff={(losses_cpp_t - losses_gd_t).abs().max().item():.3e}"
    )
    assert torch.allclose(losses_py_opt_t, losses_gd_t, atol=1e-4, rtol=1e-4), (
        "Python optimized step losses diverge from SGD. "
        f"max_abs_diff={(losses_py_opt_t - losses_gd_t).abs().max().item():.3e}"
    )
    assert torch.allclose(losses_py_ref_t, losses_gd_t, atol=1e-4, rtol=1e-4), (
        "Python reference step losses diverge from SGD. "
        f"max_abs_diff={(losses_py_ref_t - losses_gd_t).abs().max().item():.3e}"
    )
