"""GPU helpers.

On Windows, the CUDA libraries installed by pip (nvidia-cublas-cu12, nvidia-cudnn-cu12)
live in site-packages/nvidia/*/bin, which is not on the DLL search path. This module
adds those folders so CTranslate2 (used by faster-whisper) can find them.
"""
from __future__ import annotations

import glob
import os
import site


def enable_cuda_libraries() -> list[str]:
    """Make pip-installed NVIDIA DLLs visible (Windows only). Returns the folders added."""
    if os.name != "nt":
        return []
    folders = []
    for base in site.getsitepackages():
        for folder in glob.glob(os.path.join(base, "nvidia", "*", "bin")):
            os.add_dll_directory(folder)
            os.environ["PATH"] = folder + os.pathsep + os.environ["PATH"]
            folders.append(folder)
    return folders


def cuda_device_count() -> int:
    """Number of CUDA GPUs CTranslate2 can use (0 = CPU only)."""
    enable_cuda_libraries()
    import ctranslate2
    return ctranslate2.get_cuda_device_count()