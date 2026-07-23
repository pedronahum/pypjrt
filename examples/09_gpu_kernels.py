"""09 — GPU kernels, without the Triton package.

The CUDA plugin can compile Triton IR itself, so a Triton kernel needs neither
the `triton` package nor a Python subprocess. The PTX it returns loads straight
through the CUDA driver, and an FFI handler *launches* it on XLA's own stream --
the whole path, from IR to a running kernel, in pure Python.

Skips cleanly on a plugin that is not an NVIDIA GPU.

    python examples/09_gpu_kernels.py [plugin.so]
"""
import ctypes
import sys

import pypjrt
import pypjrt.ffi as ffi
from pypjrt import errors

F32, N = 11, 64
BLOCK = 128          # the compiled kernel declares `.reqntid 128`

# A minimal `tt` dialect kernel: out[i] = in[i] * 2 over one block of 64.
TRITON_IR = """
module {
  tt.func public @double_kernel(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %off = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %pin = tt.splat %in : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %pi = tt.addptr %pin, %off : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %v = tt.load %pi : tensor<64x!tt.ptr<f32>>
    %two = arith.constant dense<2.000000e+00> : tensor<64xf32>
    %r = arith.mulf %v, %two : tensor<64xf32>
    %pout = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %po = tt.addptr %pout, %off : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %po, %r : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}
"""
DOUBLE = """
module @m {
  func.func public @main(%x: tensor<64xf32>) -> tensor<64xf32> {
    %0 = stablehlo.custom_call @example_ptx_double(%x) {
        api_version = 4 : i32
      } : (tensor<64xf32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
}
"""


def main(plugin_path: str | None = None) -> int:
    try:
        import numpy as np
    except ImportError:
        print("this example needs numpy: pip install 'pypjrt[numpy]'")
        return 0

    plugin = pypjrt.Plugin(plugin_path)
    if not plugin.is_gpu:
        print(f"{plugin.path.name} is not a GPU plugin ({plugin.platform_hint}); skipping.")
        return 0

    import pypjrt.cuda as cuda
    import pypjrt.triton as tri

    if not cuda.available():
        print("libcuda.so.1 is not loadable; skipping.")
        return 0

    options = dict(pypjrt.Client.GPU_DEFAULTS, memory_fraction=0.05)

    if not tri.available(plugin):
        print("plugin does not advertise the Triton extension; skipping.")
        return 0

    with pypjrt.Client.create(plugin, options=options) as client, client.devices() as devices:
        # --- compile Triton IR through the plugin --------------------------
        # The arch string is the *dotted* compute capability ("12.1").
        # `sm_121a`, which is what ptxas wants, is rejected here.
        arch = tri.arch_of(devices[0])
        kernel = tri.compile(plugin, TRITON_IR, arch=arch)
        print(f"triton -> plugin -> {kernel!r}   (arch {arch!r})")
        print(f"  asm starts: {kernel.asm[:48].decode(errors='replace')!r}")
        print("  no `triton` package and no subprocess were involved.")

        # --- an FFI handler that LAUNCHES that kernel on XLA's stream -------
        # The module is loaded lazily, inside the handler: CUmodules are bound
        # to a CUDA context, and XLA's context is only current here.
        state: dict = {}
        seen: dict = {}

        @ffi.handler(plugin, "example_ptx_double")
        def ptx_double(call):
            if "fn" not in state:
                module = cuda.module_load_data(kernel.asm)
                state["fn"] = cuda.module_get_function(module, "double_kernel")
            (x,), (y,) = call.args, call.rets
            stream = call.stream()
            # Triton emits four params: the two real pointers, then two
            # scratch/profile pointers it wants passed as null.
            params = [ctypes.c_void_p(x.data), ctypes.c_void_p(y.data),
                      ctypes.c_void_p(0), ctypes.c_void_p(0)]
            # Launch on XLA's stream. Never synchronize it, never switch context.
            cuda.launch_kernel(state["fn"], grid=1, block=BLOCK, params=params,
                               stream=stream, shared_bytes=kernel.smem_bytes)
            seen.update(stream=stream, in_ptr=x.data, out_ptr=y.data, nbytes=x.nbytes)

        xs = (np.arange(N, dtype=np.float32) * 0.5)
        exe = client.compile(DOUBLE)
        a = client.buffer_from_host(xs, F32, [N], devices[0])
        try:
            (out,) = exe(a)
            got = np.empty(N, np.float32)
            out.to_host(got)
            print(f"\nhandler: stream=0x{seen['stream']:x} "
                  f"in=0x{seen['in_ptr']:x} out=0x{seen['out_ptr']:x} "
                  f"nbytes={seen['nbytes']}")
            print(f"  triton -> PTX -> cuLaunchKernel on XLA's stream")
            print(f"  kernel doubled the input correctly: "
                  f"{np.array_equal(got, xs * 2)}")
            out.close()
        except errors.PjrtError as e:
            print(f"\ncustom call failed: {str(e).splitlines()[0][:70]}")
        a.close(); exe.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
