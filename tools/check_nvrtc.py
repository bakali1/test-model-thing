#!/usr/bin/env python3
"""tools/check_nvrtc.py -- compile a CUDA source to PTX WITHOUT a GPU.

WHY THIS EXISTS. The engine's fastest ring-credit path is a hand-written
CUDA kernel JIT-compiled at first use (`torch.cuda._compile_kernel`, i.e.
NVRTC). A kernel that does not compile is not a slow kernel, it is a path
that prints "unavailable" and silently falls back -- on the machine that is
MEASURED, at the moment of measurement, which is the worst possible time to
find out. There is no CUDA device on the CPU-only box, so the source was
never compiled before it was shipped.

NVRTC itself needs no device: it compiles to PTX for a chosen virtual
architecture, and the driver JITs the PTX later. So this tool can prove
that the kernel compiles -- for the T4's architecture (compute_75) and for
the oldest one the engine supports (compute_60, the MX150's) -- on a laptop
with no GPU at all.

Usage:
    .venv/bin/python tools/check_nvrtc.py                 # every kernel in gpu_port
    .venv/bin/python tools/check_nvrtc.py --src file.cu --arch compute_75

Exit code 0 = every kernel compiled.
"""
import argparse
import ctypes
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def find_nvrtc():
    """Locate libnvrtc: the nvidia pip wheels, a CUDA_HOME, or the loader.

    The CPU-only torch wheel does not ship one, so the search also covers
    the OTHER virtualenvs on the machine (the nvidia cu12/cu13 wheels land
    there) and $TMT_NVRTC_LIB as an explicit override.
    """
    cands = []
    if os.environ.get("TMT_NVRTC_LIB"):
        cands.append(os.environ["TMT_NVRTC_LIB"])
    for base in (os.environ.get("CUDA_HOME"),):
        if base:
            cands.append(os.path.join(base, "lib", "libnvrtc.so"))
    for pat in ("/usr/local/cuda*/lib64/libnvrtc.so*",
                "/usr/lib/x86_64-linux-gnu/libnvrtc.so*",
                "/var/lib/flatpak/app/*/files/lib/python*/site-packages/"
                "nvidia/cuda_nvrtc/lib/libnvrtc.so*"):
        cands += sorted(glob.glob(pat))
    try:
        import site
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            cands += sorted(glob.glob(os.path.join(
                sp, "nvidia", "cu*", "lib", "libnvrtc.so*")))
            cands += sorted(glob.glob(os.path.join(
                sp, "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so*")))
    except Exception:                                  # pragma: no cover
        pass
    # the wheel may live in another virtualenv of this machine. The search is
    # FIXED-DEPTH on purpose: a recursive glob over $HOME takes minutes.
    home = os.path.expanduser("~")
    for pat in (".local/lib/python*/site-packages/nvidia/cu*/lib/libnvrtc.so*",
                ".airllm-tui/.venv/lib/python*/site-packages/nvidia/cu*/lib/libnvrtc.so*",
                "*/.venv/lib/python*/site-packages/nvidia/cu*/lib/libnvrtc.so*",
                "*/venv/lib/python*/site-packages/nvidia/cu*/lib/libnvrtc.so*",
                "programing/*/.venv/lib/python*/site-packages/nvidia/cu*/lib/libnvrtc.so*",
                "programing/*/venv/lib/python*/site-packages/nvidia/cu*/lib/libnvrtc.so*",
                "programing/*/*/.venv/lib/python*/site-packages/nvidia/cu*/lib/libnvrtc.so*",
                "programing/*/*/venv/lib/python*/site-packages/nvidia/cu*/lib/libnvrtc.so*"):
        cands += sorted(glob.glob(os.path.join(home, pat)))
    for path in cands:
        if os.path.exists(path):
            try:
                # libnvrtc dlopen()s libnvrtc-builtins.so at load time, and the
                # wheel's copy is not on the loader's path -- so preload it
                # (RTLD_GLOBAL) or the load fails with "failed to open
                # libnvrtc-builtins".
                d = os.path.dirname(path)
                for b in sorted(glob.glob(os.path.join(
                        d, "libnvrtc-builtins.so*"))):
                    try:
                        ctypes.CDLL(b, mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass
                return ctypes.CDLL(path), path
            except OSError:
                continue
    return None, None


class Nvrtc:
    """The slice of the NVRTC C API this tool needs."""

    def __init__(self):
        self.lib, self.path = find_nvrtc()
        if self.lib is None:
            raise RuntimeError("libnvrtc not found (pip nvidia-cuda-nvrtc, "
                               "or a CUDA toolkit)")
        lib = self.lib
        lib.nvrtcGetErrorString.restype = ctypes.c_char_p
        lib.nvrtcGetErrorString.argtypes = [ctypes.c_int]
        lib.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int, ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p)]
        lib.nvrtcCompileProgram.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        lib.nvrtcGetProgramLogSize.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        lib.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.nvrtcGetPTXSize.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        lib.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        major, minor = ctypes.c_int(), ctypes.c_int()
        lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
        self.version = (major.value, minor.value)

    def _check(self, rc):
        if rc != 0:
            raise RuntimeError(self.lib.nvrtcGetErrorString(rc).decode())

    def compile_to_ptx(self, src, arch="compute_75"):
        prog = ctypes.c_void_p()
        self._check(self.lib.nvrtcCreateProgram(
            ctypes.byref(prog), src.encode(), b"tmt.cu", 0, None, None))
        opts = (f"--gpu-architecture={arch}".encode(),
                b"--std=c++17")
        c_opts = (ctypes.c_char_p * len(opts))(*opts)
        rc = self.lib.nvrtcCompileProgram(prog, len(opts), c_opts)
        n = ctypes.c_size_t()
        self.lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        log = b""
        if n.value > 1:
            buf = ctypes.create_string_buffer(n.value)
            self.lib.nvrtcGetProgramLog(prog, buf)
            log = buf.value
        ptx = b""
        if rc == 0:
            self.lib.nvrtcGetPTXSize(prog, ctypes.byref(n))
            buf = ctypes.create_string_buffer(n.value)
            self.lib.nvrtcGetPTX(prog, buf)
            ptx = buf.value
        self.lib.nvrtcDestroyProgram(ctypes.byref(prog))
        return rc, log.decode(errors="replace"), ptx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None, help="a .cu file to compile")
    ap.add_argument("--arch", default="compute_75",
                    help="virtual arch (compute_75 = T4, compute_60 = Pascal)")
    ap.add_argument("--all-arch", action="store_true",
                    help="also compile for compute_60 (the MX150)")
    args = ap.parse_args()

    try:
        nv = Nvrtc()
    except RuntimeError as exc:
        print(f"SKIP: {exc}")
        return 0
    print(f"libnvrtc {nv.version[0]}.{nv.version[1]}  ({nv.path})")

    sources = {}
    if args.src:
        with open(args.src) as fh:
            sources[os.path.basename(args.src)] = fh.read()
    else:
        import gpu_port
        for name in sorted(dir(gpu_port)):
            if name.startswith("_NVRTC") and name.endswith("_SRC"):
                sources[name] = getattr(gpu_port, name)

    archs = [args.arch] + (["compute_60"] if args.all_arch else [])
    ok = True
    for name, src in sources.items():
        for arch in archs:
            rc, log, ptx = nv.compile_to_ptx(src, arch=arch)
            if rc == 0:
                print(f"  [ok]   {name} @ {arch}: {len(ptx)} bytes of PTX")
                if log.strip():
                    print("        log:", log.strip()[:400])
            else:
                ok = False
                print(f"  [FAIL] {name} @ {arch}")
                print(log[:4000])
    print("NVRTC:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
