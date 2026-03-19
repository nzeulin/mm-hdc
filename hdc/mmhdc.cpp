#include <torch/extension.h>

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("step", &step, "MM-HDC prototype update function");
}
