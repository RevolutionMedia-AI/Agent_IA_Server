#!/usr/bin/env python3
import ctypes
import ctypes.util
import sys
from ctypes import c_float, POINTER, c_void_p

def find_lib(path_arg):
    if path_arg:
        return path_arg
    name = ctypes.util.find_library("rnnoise")
    if name:
        return name
    for candidate in ("/usr/local/lib/librnnoise.so", "/usr/lib/librnnoise.so", "librnnoise.so", "rnnoise.dll"):
        try:
            with open(candidate, "rb"):
                return candidate
        except Exception:
            pass
    return None

def main():
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    libpath = find_lib(path_arg)
    if not libpath:
        print("librnnoise not found", file=sys.stderr)
        return 2
    try:
        lib = ctypes.CDLL(libpath)
    except Exception as e:
        print("failed to load librnnoise:", e, file=sys.stderr)
        return 3
    try:
        lib.rnnoise_create.restype = c_void_p
        lib.rnnoise_destroy.argtypes = [c_void_p]
    except Exception:
        pass
    try:
        proc = lib.rnnoise_process_frame
        try:
            proc.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float)]
            proc.restype = ctypes.c_int
        except Exception:
            pass
    except AttributeError:
        print("rnnoise_process_frame not found", file=sys.stderr)
        return 4
    try:
        st = lib.rnnoise_create()
        if not st:
            print("rnnoise_create returned NULL", file=sys.stderr)
            return 5
        # Create a 480-sample zero buffer (common RNNoise frame length)
        zeros = (c_float * 480)(*([0.0] * 480))
        out = (c_float * 480)()
        proc(st, out, zeros)
        lib.rnnoise_destroy(st)
    except Exception as e:
        print("rnnoise probe call failed:", e, file=sys.stderr)
        return 6
    return 0

if __name__ == "__main__":
    sys.exit(main())
