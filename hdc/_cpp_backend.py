from functools import lru_cache
import os
from pathlib import Path


@lru_cache(maxsize=1)
def get_mmhdc_cpp():
    try:
        from torch.utils.cpp_extension import load

        return load(
            name="mmhdc_cpp",
            extra_cflags=["-O3"],
            is_python_module=True,
            sources=[os.path.join(Path(__file__).parent, "mmhdc.cpp")],
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to build or load the MM-HDC C++ backend. "
            "The 'cpp' backend requires a working C++ toolchain and PyTorch C++ "
            "extension build support. If this environment cannot compile the "
            "extension, use backend='python' instead."
        ) from exc
