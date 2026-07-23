"""Tier 2: launch a kernel from Python, on XLA's stream, inside a compiled
program -- the pure-Python form of what a C++ launch shim usually does.

The chain: Triton IR --(plugin)--> PTX --(module_load_data)--> function
--(launch_kernel)--> run, from a Python FFI handler, called from XLA.
No triton package, no subprocess, no C++.
"""
import array
import ctypes
import pathlib

import pytest

import pypjrt
import pypjrt.ffi as ffi
import pypjrt.triton as tri
from pypjrt import errors

pytestmark = pytest.mark.tier2
cuda = pytest.importorskip("pypjrt.cuda")

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
KERNEL = (DATA / "triton_double.mlir").read_text()
F32, N = 11, 64
TIGHT = {"preallocate": False, "memory_fraction": 0.05}

# The Triton kernel doubles its input. Two facts about its compiled PTX, both
# load-bearing and both learned the hard way:
#   - it declares `.reqntid 128`, so it MUST be launched with 128 threads;
#   - Triton appends two scratch/profile pointer params after the two real
#     ones, so the launch passes four, the last two null.
BLOCK = 128
ENTRY = b"double_kernel"

DOUBLE = """
module @m {
  func.func public @main(%x: tensor<64xf32>) -> tensor<64xf32> {
    %0 = stablehlo.custom_call @ptx_double(%x) {api_version = 4 : i32}
        : (tensor<64xf32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
}
"""


@pytest.fixture(scope="module")
def plugin(gpu_plugin_path):
    if not cuda.available():
        pytest.skip("libcuda.so.1 not loadable")
    p = pypjrt.Plugin(gpu_plugin_path)
    if not tri.available(p):
        pytest.skip("plugin does not advertise the Triton extension")
    return p


@pytest.fixture(scope="module")
def ptx(plugin):
    with pypjrt.Client.create(plugin, options=TIGHT) as c, c.devices() as d:
        return tri.compile(plugin, KERNEL, arch=tri.arch_of(d[0]))


# --- module surface, driver-level happy and negative paths -----------------
# These need a current CUDA context, which a live pypjrt client provides.

def test_module_load_get_unload_roundtrip(plugin, ptx):
    with pypjrt.Client.create(plugin, options=TIGHT) as c, c.devices():
        mod = cuda.module_load_data(ptx.asm)
        assert mod != 0
        fn = cuda.module_get_function(mod, "double_kernel")
        assert fn != 0
        cuda.module_unload(mod)


def test_module_load_rejects_garbage(plugin):
    with pypjrt.Client.create(plugin, options=TIGHT) as c, c.devices():
        with pytest.raises(cuda.CudaError):
            cuda.module_load_data(b"this is not ptx or cubin")


def test_get_function_rejects_missing_symbol(plugin, ptx):
    with pypjrt.Client.create(plugin, options=TIGHT) as c, c.devices():
        mod = cuda.module_load_data(ptx.asm)
        try:
            with pytest.raises(cuda.CudaError):
                cuda.module_get_function(mod, "no_such_kernel")
        finally:
            cuda.module_unload(mod)


# --- the real thing: launch on XLA's stream, from a compiled program -------

def test_launch_from_ffi_handler_doubles_input(plugin, ptx):
    """A Python handler loads the module lazily (on XLA's thread, where XLA's
    context is current) and launches the kernel on the stream XLA gave it.
    The result is checked against the arithmetic the kernel claims to do."""
    seen: dict = {}
    state: dict = {}

    @ffi.handler(plugin, "ptx_double")
    def _(call):
        # Lazy, in-handler load: CUmodules are per-context and XLA's context is
        # only guaranteed current here.
        if "fn" not in state:
            mod = cuda.module_load_data(ptx.asm)
            state["fn"] = cuda.module_get_function(mod, ENTRY.decode())
        (x,), (y,) = call.args, call.rets
        stream = call.stream()
        params = [ctypes.c_void_p(x.data), ctypes.c_void_p(y.data),
                  ctypes.c_void_p(0), ctypes.c_void_p(0)]
        cuda.launch_kernel(state["fn"], grid=1, block=BLOCK, params=params,
                           stream=stream, shared_bytes=ptx.smem_bytes)
        seen.update(stream=stream, in_ptr=x.data, out_ptr=y.data, nbytes=x.nbytes)

    with pypjrt.Client.create(plugin, options=TIGHT) as c, c.devices() as devs:
        exe = c.compile(DOUBLE)
        xs = array.array("f", [float(i) for i in range(N)])
        buf = c.buffer_from_host(xs, F32, [N], devs[0])
        (out,) = exe(buf)
        got = array.array("f"); got.frombytes(out.to_host())
        out.close(); buf.close(); exe.close()

    # The handler really ran, on a real stream, with distinct device pointers.
    assert seen["stream"] != 0, "XLA gave no stream"
    assert seen["in_ptr"] and seen["out_ptr"] and seen["in_ptr"] != seen["out_ptr"]
    assert seen["nbytes"] == N * 4
    # And it computed the right answer: every element doubled.
    assert list(got) == [v * 2 for v in xs]


def test_launch_matches_a_stablehlo_oracle(plugin, ptx):
    """The launched kernel and a plain StableHLO `x + x` must agree bit for
    bit -- the launch is not just non-crashing, it is correct."""
    state: dict = {}

    @ffi.handler(plugin, "ptx_double_oracle")
    def _(call):
        if "fn" not in state:
            mod = cuda.module_load_data(ptx.asm)
            state["fn"] = cuda.module_get_function(mod, ENTRY.decode())
        (x,), (y,) = call.args, call.rets
        params = [ctypes.c_void_p(x.data), ctypes.c_void_p(y.data),
                  ctypes.c_void_p(0), ctypes.c_void_p(0)]
        cuda.launch_kernel(state["fn"], 1, BLOCK, params, call.stream(),
                           shared_bytes=ptx.smem_bytes)

    via_kernel = """
    module @m {
      func.func public @main(%x: tensor<64xf32>) -> tensor<64xf32> {
        %0 = stablehlo.custom_call @ptx_double_oracle(%x) {api_version = 4 : i32}
            : (tensor<64xf32>) -> tensor<64xf32>
        return %0 : tensor<64xf32>
      }
    }
    """
    via_hlo = """
    module @m {
      func.func public @main(%x: tensor<64xf32>) -> tensor<64xf32> {
        %0 = stablehlo.add %x, %x : tensor<64xf32>
        return %0 : tensor<64xf32>
      }
    }
    """
    with pypjrt.Client.create(plugin, options=TIGHT) as c, c.devices() as devs:
        xs = array.array("f", [i * 0.25 for i in range(N)])

        def run(src):
            e = c.compile(src)
            a = c.buffer_from_host(xs, F32, [N], devs[0])
            (o,) = e(a)
            g = array.array("f"); g.frombytes(o.to_host())
            o.close(); a.close(); e.close()
            return g

        assert run(via_kernel).tobytes() == run(via_hlo).tobytes()
