"""Compiling CUDA at runtime on a machine with no compiler.

Three of this server's hot spots are shapes PyTorch has no good kernel for on
this card - a 4-bit GEMV (`app/engine/nf4gemv.py`), and the RMS norm and rotary
embedding that surround it (`app/engine/fusedops.py`). Writing them means
getting CUDA C onto the GPU, and the two usual routes both need a host C
compiler: Triton shells out to one to build its launcher stub, and
`torch.utils.cpp_extension.load_inline` is a compiler invocation by definition.
This machine has no `cc`, and installing one needs a root it does not have
either.

NVRTC needs nothing but the driver, and it is already here - the `torch` wheel
depends on `nvidia-cuda-nvrtc-cu12`, so `libnvrtc.so.12` sits in site-packages
whether or not anyone meant to install it. It compiles a source string straight
to a cubin for the current architecture, and the driver API - `libcuda.so.1`,
which must be present for torch to work at all - loads and launches it inside
the context torch has already made current, so the kernels share torch's
stream, its allocator and its tensors with nothing marshalled in between.

The cost is that a launch goes through `ctypes` rather than a compiled binding:
about 6 us of Python per call, against the 50-100 us the kernels here run for.
`app/engine/nf4gemv.py` is called 390 times a frame and so builds its argument
block once and rewrites two pointers; anything called less often can afford
`launch()`.

Nothing in here raises on a machine without CUDA. `available()` answers the
question, callers fall back to what they were doing before, and the fallback is
always correct and merely slower.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_nvrtc = None
_driver = None
_unavailable: str | None = None
_arch: str | None = None


def _find_nvrtc() -> str:
    override = os.environ.get("TTS_NVRTC_PATH")
    if override:
        return override
    if os.name == "nt":
        # The Windows torch wheel bundles NVRTC next to its own DLLs instead of
        # depending on the nvidia-cuda-nvrtc package.
        import torch

        root = os.path.join(os.path.dirname(torch.__file__), "lib")
        pattern = "nvrtc64_*.dll"
    else:
        import nvidia.cuda_nvrtc  # noqa: F401 - imported for its location

        root = os.path.join(os.path.dirname(nvidia.cuda_nvrtc.__file__), "lib")
        pattern = "libnvrtc.so*"
    # The `alt` build is the alternate-architecture one; it compiles the same
    # source but is not what torch itself loads.
    candidates = [
        path
        for path in sorted(glob.glob(os.path.join(root, pattern)))
        if "alt" not in os.path.basename(path)
    ]
    if not candidates:
        raise FileNotFoundError(f"no {pattern} under {root}")
    return candidates[0]


def available() -> bool:
    """Whether runtime compilation works here. Cheap after the first call."""
    global _nvrtc, _driver, _unavailable, _arch

    if _nvrtc is not None:
        return True
    if _unavailable is not None:
        return False

    with _lock:
        if _nvrtc is not None:
            return True
        if _unavailable is not None:
            return False
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("no CUDA device")
            # Kernels are launched into whatever context torch has made
            # current, so torch has to have made one.
            torch.zeros(1, device="cuda")
            major, minor = torch.cuda.get_device_capability(0)
            _arch = f"sm_{major}{minor}"
            _nvrtc = ctypes.CDLL(_find_nvrtc())
            _driver = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
            return True
        except Exception as exc:  # noqa: BLE001 - a missing fast path is not fatal
            _unavailable = str(exc)
            logger.warning(
                "runtime CUDA compilation unavailable (%s); hand-written kernels "
                "are off and their slower fallbacks will serve every call",
                exc,
            )
            return False


def driver():
    """The `libcuda.so.1` handle, for callers that launch kernels themselves."""
    return _driver


def build(source: str, names, *, fast_math: bool = True, label: str = "kernel") -> dict:
    """Compile `source` for this device and return `{name: CUfunction}`.

    `fast_math` is right for a kernel whose arithmetic is a dot product of
    values that were quantised to four bits anyway, and wrong for one computing
    a reciprocal square root that everything downstream is scaled by, so it is
    stated per module rather than assumed.
    """
    if not available():
        raise RuntimeError(f"runtime compilation unavailable: {_unavailable}")

    program = ctypes.c_void_p()
    if _nvrtc.nvrtcCreateProgram(
        ctypes.byref(program), source.encode(), f"{label}.cu".encode(), 0, None, None
    ):
        raise RuntimeError(f"nvrtcCreateProgram failed for {label}")

    options = [f"--gpu-architecture={_arch}".encode(), b"-default-device"]
    if fast_math:
        options.append(b"--use_fast_math")
    argv = (ctypes.c_char_p * len(options))(*options)
    status = _nvrtc.nvrtcCompileProgram(program, len(options), argv)

    size = ctypes.c_size_t()
    _nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    _nvrtc.nvrtcGetProgramLog(program, log)
    if status:
        raise RuntimeError(f"{label} failed to compile for {_arch}: {log.value.decode()}")

    _nvrtc.nvrtcGetCUBINSize(program, ctypes.byref(size))
    cubin = ctypes.create_string_buffer(size.value)
    _nvrtc.nvrtcGetCUBIN(program, cubin)
    _nvrtc.nvrtcDestroyProgram(ctypes.byref(program))

    module = ctypes.c_void_p()
    if _driver.cuModuleLoadData(ctypes.byref(module), cubin):
        raise RuntimeError(f"cuModuleLoadData failed for {label}")

    functions = {}
    for name in names:
        function = ctypes.c_void_p()
        if _driver.cuModuleGetFunction(ctypes.byref(function), module, name.encode()):
            raise RuntimeError(f"{label} has no kernel named {name}")
        functions[name] = function
    # The module owns the code the CUfunctions point into, so it has to outlive
    # them; nothing ever unloads one, which is right for kernels built once at
    # start-up and used until the process ends.
    functions["__module__"] = module
    return functions


def pack(values):
    """Build a launch argument block, and the ctypes objects that back it.

    Both must be kept alive for as long as the block is used: the array holds
    bare addresses into the objects, so letting them be collected leaves the
    launch reading freed memory. Callers that launch in a loop keep the pair and
    rewrite the pointers in place.
    """
    argv = (ctypes.c_void_p * len(values))(
        *[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in values]
    )
    return argv, values


def launch(function, grid: int, block: int, argv, *, shared: int = 0) -> None:
    """Launch `function` on torch's current stream."""
    import torch

    status = _driver.cuLaunchKernel(
        function,
        grid, 1, 1,
        block, 1, 1,
        shared,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        argv,
        None,
    )
    if status:
        raise RuntimeError(f"cuLaunchKernel failed: {status}")


#: Conversions every kernel here needs, and cannot get from `cuda_fp16.h`:
#: NVRTC would have to be pointed at the CUDA headers, which are a pip
#: package's layout rather than a promise. `cvt` is one instruction either way.
PRELUDE = r"""
typedef unsigned int uint;
typedef unsigned short half_t;

__device__ __forceinline__ float h2f(half_t h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ half_t f2h(float f) {
  half_t h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}
__device__ __forceinline__ float bf2f(half_t b) {
  // bfloat16 is the top half of a float32 and nothing else.
  uint bits = ((uint)b) << 16;
  return __int_as_float((int)bits);
}
"""
