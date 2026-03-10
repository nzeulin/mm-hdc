#include <torch/extension.h>

/*
Implementing step procedure in C++ to accelerate it a bit
*/

torch::Tensor step(torch::Tensor &x, torch::Tensor &y, torch::Tensor &prototypes, float lr, float C) {
    auto num_classes = prototypes.size(0);
    auto unique_labels = std::get<0>(torch::_unique(y));
    auto prototypes_update = torch::zeros_like(prototypes);

    for (auto i = 0; i < unique_labels.size(0); ++i) {
        // This is the class we are currently processing
        auto cls = unique_labels[i].item<int64_t>();

        // Roll prototypes to avoid using advanced indexing
        auto rolled_prototypes = torch::roll(prototypes, -cls, 0);

        // Selecting points of the current class
        auto x_cls = torch::index_select(x, 0, torch::nonzero(y == cls).flatten());

        // Computing hinge loss
        auto dot_product = torch::mm(x_cls, (rolled_prototypes[0] - rolled_prototypes.slice(0, 1)).t());
        auto hinge_loss = torch::relu(2 - dot_product);

        // Selecting points exceeding margin and their true labels
        auto exceeding_margin = hinge_loss > 0;
        auto idx_wrong = torch::any(exceeding_margin, -1);
        auto y_true_all = torch::any(exceeding_margin, 0).nonzero().flatten();

        // Reshuffling points exceeding margin between the prototypes
        prototypes_update[cls] += x_cls.index_select(0, torch::nonzero(idx_wrong).flatten()).sum(0);
        for (auto j = 0; j < y_true_all.size(0); ++j) {
            auto y_true = y_true_all[j].item<int64_t>();
            auto cls_to_update = (y_true + 1 + cls) % num_classes;
            
            auto col_mask = exceeding_margin.select(1, y_true);
            auto selected_points = x_cls.index_select(0, torch::nonzero(col_mask).flatten()).sum(0);
            prototypes_update[cls_to_update] -= selected_points;
        }
    }

    // Updating prototypes and returning result
    prototypes = (1 - lr / C) * prototypes + lr * prototypes_update;
    return prototypes;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("step", &step, "MM-HDC prototype update function");
}
