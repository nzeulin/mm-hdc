# MM-HDC: Maximum-Margin Hyperdimensional Computing

This repository provides an implementation of the multi-class maximum-margin hyperdimensional computing (MM-HDC) classifier that adopts the optimization problem formulation of multi-class Weston-Watkins SVM to HDC.

You can try running multi-class MM-HDC on MNIST dataset as follows:
```bash
git clone https://github.com/nzeulin/mm-hdc.git
cd mm-hdc
conda env create -n environment.yaml
conda run -n mm-hdc python main.py --config configs/default_mnist_config.py
```

### Current features
- C++ backend (`libtorch`) to enable fast MM-HDC training.
- Support of floating-point prototypes and hypervectors.

Python-based backend (`_py_step` in `hdc/mmhdc.py`) should be used as a reference, ground-truth implementation only.

### Features to be implemented
- [ ] Training procedure for fixed-point (integer-only) prototypes and hypervectors.
- [ ] Implement C++ backend for fixed-point training procedure.
- [ ] Wrap MM-HDC as a Python package.
 
### Citation

If you use this repository in your research, please cite it as software (paper in progress):

```bibtex
@software{zeulin_2026_mm_hdc,
	author = {Zeulin, Nikita},
	title = {MM-HDC: Maximum-Margin Hyperdimensional Computing},
	year = {2026},
	url = {https://github.com/nzeulin/mm-hdc}
}
```
