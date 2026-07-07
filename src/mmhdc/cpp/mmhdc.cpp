#include <torch/extension.h>

torch::Tensor step(torch::Tensor &x, torch::Tensor &y, torch::Tensor &prototypes, float lr, float C, float margin_width) {
    auto scores = torch::mm(x, prototypes.t());
    auto correct_scores = scores.gather(1, y.unsqueeze(1));

    // Checking if margin is violated
    auto violated = (correct_scores - scores) < 2 * margin_width;

    // Zero out the diagonal: a sample cannot violate a margin against its own class.
    violated.scatter_(1, y.unsqueeze(1),
                      torch::zeros_like(y.unsqueeze(1), torch::kBool));

    // Build weight matrix W:
    //   W[i, k]   = -1  if sample i violates margin against class k
    //   W[i, y_i] = +m_i where m_i is the number of violated margins for sample i
    auto W = -violated.to(x.scalar_type());
    W.scatter_add_(1, y.unsqueeze(1),
                   violated.sum(1, true).to(x.scalar_type()));

    // Computing update for the prototypes
    auto prototypes_update = torch::mm(W.t(), x);

    // Update prototypes and add regularization
    prototypes = (1 - lr / C) * prototypes + lr * prototypes_update;
    return prototypes;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("step", &step, "MM-HDC prototype update function");
}
