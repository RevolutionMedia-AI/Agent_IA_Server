"""ctypes-based loader for system `librnnoise`.

This provides a minimal `RNNoise` class with a `filter(float_array)` method
that returns a processed float32 numpy array in [-1,1]. The loader is
best-effort and will raise ImportError if the shared library or expected
symbols are not available. Callers should handle failures and fall back
to other denoisers.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import subprocess
import sys
from typing import Optional

import numpy as np

log = logging.getLogger("stt_server.rnnoise_ctypes")


def _find_lib() -> Optional[str]:
    # Try explicit environment override first
    env = os.getenv("RNNOISE_LIB")
    if env:
        return env
    # Try ctypes util
    name = ctypes.util.find_library("rnnoise")
    if name:
        return name
    # Common unix locations
    for candidate in ("/usr/local/lib/librnnoise.so", "/usr/lib/librnnoise.so", "librnnoise.so", "rnnoise.dll"):
        if os.path.exists(candidate):
            return candidate
    return None


class RNNoise:
    """Simple wrapper exposing `filter(arr: np.ndarray[float32]) -> np.ndarray[float32]`.

    Notes:
    - This is a best-effort wrapper. If the underlying C API differs from
      the expected `rnnoise_create` / `rnnoise_destroy` / `rnnoise_process_frame`
      signatures, initialization will raise ImportError.
    - The implementation processes a single frame (one call) with the
      provided sample length. If the native function expects a fixed frame
      length, the call may fail — callers should catch exceptions and
      fall back to another denoiser.
    """

    def __init__(self) -> None:
        libpath = _find_lib()
        if not libpath:
            raise ImportError("librnnoise not found on system (RNNOISE_LIB)")
        # Run a quick probe in a subprocess to ensure the native library
        # can be safely loaded without risking a segmentation fault in the
        # main process. The probe will crash the child process on failure,
        # allowing the parent to fall back safely.
        if os.getenv("RNNOISE_SKIP_PROBE", "false").strip().lower() not in {"1", "true", "yes", "on"}:
            probe = os.path.join(os.path.dirname(__file__), "rnnoise_probe.py")
            try:
                res = subprocess.run([sys.executable, probe, libpath], timeout=4)
                if res.returncode != 0:
                    raise ImportError(f"rnnoise probe failed (code={res.returncode})")
            except subprocess.TimeoutExpired as e:
                raise ImportError("rnnoise probe timed out") from e

        try:
            self._lib = ctypes.CDLL(libpath)
        except Exception as e:
            raise ImportError(f"failed to load librnnoise: {e}")

        # create / destroy
        try:
            self._lib.rnnoise_create.restype = ctypes.c_void_p
            self._lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        except Exception:
            # If these symbols are missing we'll fail below
            pass

        # process frame symbol (best-effort). We assume signature:
        # int rnnoise_process_frame(void *st, float *out, const float *in)
        try:
            self._proc = self._lib.rnnoise_process_frame
            # conservative argtypes
            try:
                self._proc.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
                self._proc.restype = ctypes.c_int
            except Exception:
                # keep raw callable and call via ctypes
                pass
        except AttributeError:
            raise ImportError("librnnoise does not expose rnnoise_process_frame")

        # Create state
        try:
            self._st = self._lib.rnnoise_create()
            if not self._st:
                raise ImportError("rnnoise_create returned NULL")
        except Exception as e:
            raise ImportError(f"failed to create rnnoise state: {e}")

    def __del__(self) -> None:
        try:
            if hasattr(self, "_st") and self._st:
                try:
                    self._lib.rnnoise_destroy(self._st)
                except Exception:
                    pass
        except Exception:
            pass

    def filter(self, arr: np.ndarray) -> np.ndarray:
        """Process a single float32 array and return processed float32 array.

        Input should be a 1-D float32 numpy array with values in [-1,1]. The
        function returns a new float32 array of the same shape. On failure
        this method raises RuntimeError.
        """
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim != 1:
            raise ValueError("RNNoise.filter expects a 1-D float32 array")

        n = a.size
        out = np.empty_like(a)

        # Prepare ctypes pointers
        in_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        try:
            # Call processing function
            res = self._proc(self._st, out_ptr, in_ptr)
            # some implementations return int, some float — ignore value
        except Exception as e:
            log.exception("rnnoise ctypes processing call failed")
            raise RuntimeError("rnnoise processing failed") from e

        return out


__all__ = ["RNNoise"]
