# MM-HDC: Maximum-Margin Hyperdimensional Computing

This repository provides an implementation of the multi-class maximum-margin hyperdimensional computing (MM-HDC) classifier that adopts the optimization problem formulation of multi-class Weston-Watkins SVM to HDC.

The algorithm is implemented in the `MultiMMHDC` class, which can be imported as:

```python
from mmhdc import MultiMMHDC
from mmhdc.utils import HDTransform
```

You can install the package in editable mode and run the MNIST example as follows:

```bash
git clone https://github.com/nzeulin/mm-hdc.git
cd mm-hdc
python -m pip install --upgrade pip
python -m pip install ".[examples]"
python example.py --config configs/examples/mnist.py
```

### Current features
- C++ backend (`libtorch`) to enable fast MM-HDC training.
- Support of floating-point prototypes and hypervectors.

Python-based backend (`_py_step` in `src/mmhdc/model.py`) should be used as a reference, ground-truth implementation only.
 
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
