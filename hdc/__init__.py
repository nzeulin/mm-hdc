# Compiling C++ extension for MM-HDC prototype update
import os
from pathlib import Path
from torch.utils.cpp_extension import load
_mmhdc_cpp = load(
    name='mmhdc_cpp',
    extra_cflags=['-O3'],
    is_python_module=True,
    sources=[os.path.join(Path(__file__).parent, 'mmhdc.cpp'),]
)

from hdc.mmhdc import MultiMMHDC, MultiMMHDCInt
from hdc.transform import HDTransform