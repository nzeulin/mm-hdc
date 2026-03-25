#include <torch/extension.h>
#include <cmath>
#include <vector>

/*
Optimised step procedure — global GEMM reformulation:

  All per-class loops are eliminated. The entire batch is handled with
  exactly two GEMMs regardless of the number of classes K:

    1. scores   = x @ prototypes.T          (N, K)  — one GEMM
    2. W        built from the boolean margin-violation mask (N, K)
    3. update   = W.T @ x                   (K, D)  — one GEMM

  This maximises GPU utilisation (two large, dense kernels instead of K
  small ones) and removes all GPU→CPU syncs (.item() calls).
*/

torch::Tensor step(torch::Tensor &x, torch::Tensor &y, torch::Tensor &prototypes, float lr, float C) {
    // scores[i, k] = x_i · w_k  for every sample and every class.
    auto scores = torch::mm(x, prototypes.t());                    // (N, K)

    // correct_scores[i] = x_i · w_{y_i}
    auto correct_scores = scores.gather(1, y.unsqueeze(1));        // (N, 1)

    // margin_gap[i, k] = x_i · w_{y_i} - x_i · w_k
    //                  = correct_score_i  - score_{i,k}
    // Margin is violated when gap < 2  (and k ≠ y_i).
    auto violated = (correct_scores - scores) < 2;                 // (N, K) bool

    // Zero out the diagonal: a sample cannot violate a margin against its own class.
    violated.scatter_(1, y.unsqueeze(1),
                      torch::zeros_like(y.unsqueeze(1), torch::kBool));

    // Build weight matrix W  (N, K):
    //   W[i, k]   = -1  if sample i violates margin against class k
    //   W[i, y_i] = +m_i where m_i is the number of violated margins for sample i
    //               (matches Python: each sample contributes once per violated class).
    auto W = -violated.to(x.scalar_type());                        // (N, K)
    W.scatter_add_(1, y.unsqueeze(1),
                   violated.sum(1, /*keepdim=*/true).to(x.scalar_type()));

    // prototypes_update[k] = Σ_i  W[i,k] * x_i  — single GEMM.
    auto prototypes_update = torch::mm(W.t(), x);                  // (K, D)

    // Regularised update.
    prototypes = (1 - lr / C) * prototypes + lr * prototypes_update;
    return prototypes;
}

  torch::Tensor round_divide_int(const torch::Tensor &numer, int64_t denom) {
    auto abs_q = torch::floor_divide(numer.abs() + denom / 2, denom);
    auto sign = numer.sign().to(torch::kInt64);
    return abs_q.to(torch::kInt64) * sign;
  }

  std::vector<torch::Tensor> step_int(
    torch::Tensor &x,
    torch::Tensor &y,
    torch::Tensor &prototypes,
    torch::Tensor &prototypes_fp,
    double lr,
    double C,
    int64_t fixed_point_frac_bits,
    int64_t dtype_min,
    int64_t dtype_max) {

    auto x_acc = x.to(torch::kInt64);
    auto fp_scale = static_cast<int64_t>(1) << fixed_point_frac_bits;

    auto p_acc = round_divide_int(prototypes_fp, fp_scale);

    auto scores = torch::mm(x_acc, p_acc.t());
    auto correct_scores = scores.gather(1, y.unsqueeze(1));

    auto violated = (correct_scores - scores) < 2;
    violated.scatter_(1, y.unsqueeze(1), torch::zeros_like(y.unsqueeze(1), torch::kBool));

    auto W = -violated.to(torch::kInt64);
    W.scatter_add_(1, y.unsqueeze(1), violated.sum(1, /*keepdim=*/true).to(torch::kInt64));
    auto prototypes_update = torch::mm(W.t(), x_acc);

    auto decay_q = static_cast<int64_t>(std::llround((1.0 - lr / C) * static_cast<double>(fp_scale)));
    auto lr_q = static_cast<int64_t>(std::llround(lr * static_cast<double>(fp_scale)));

    auto updated_fp = round_divide_int(decay_q * prototypes_fp, fp_scale) + lr_q * prototypes_update;
    auto updated = round_divide_int(updated_fp, fp_scale);

    auto clamped = torch::clamp(updated, dtype_min, dtype_max).to(prototypes.scalar_type());
    return {clamped, updated_fp};
  }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("step", &step, "MM-HDC prototype update function");
    m.def("step_int", &step_int, "MM-HDC integer prototype update function");
}
