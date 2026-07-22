"""09 — GPU kernels, without the Triton package.

The CUDA plugin can compile Triton IR itself, so a Triton kernel needs neither
the `triton` package nor a Python subprocess. The PTX it returns loads straight
through the CUDA driver, and an FFI handler can launch it on XLA's own stream.

Skips cleanly on a plugin that is not an NVIDIA GPU.

    python examples/09_gpu_kernels.py [plugin.so]
"""
import ctypes
import sys

import pypjrt
import pypjrt.ffi as ffi
from pypjrt import errors

F32, N = 11, 1024

# A minimal `tt` dialect kernel: out[i] = in[i] * 2 over one block.
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
COPY = """
module @m {
  func.func public @main(%x: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = stablehlo.custom_call @example_dev_copy(%x) {
        api_version = 4 : i32
      } : (tensor<1024xf32>) -> tensor<1024xf32>
    return %0 : tensor<1024xf32>
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

    with pypjrt.Client.create(plugin, options=options) as client, client.devices() as devices:
        # --- compile Triton IR through the plugin --------------------------
        if tri.available(plugin):
            # The arch string is the *dotted* compute capability ("12.1").
            # `sm_121a`, which is what ptxas wants, is rejected here.
            arch = tri.arch_of(devices[0])
            kernel = tri.compile(plugin, TRITON_IR, arch=arch)
            print(f"triton -> plugin -> {kernel!r}   (arch {arch!r})")
            print(f"  asm starts: {kernel.asm[:48].decode(errors='replace')!r}")

            module = ctypes.c_void_p()
            rc = cuda.lib().cuModuleLoadData(ctypes.byref(module), kernel.asm)
            fn = ctypes.c_void_p()
            rc2 = cuda.lib().cuModuleGetFunction(ctypes.byref(fn), module, b"double_kernel")
            print(f"  cuModuleLoadData: {'ok' if rc == 0 else rc}, "
                  f"cuModuleGetFunction: {'ok' if rc2 == 0 else rc2}")
            print("  no `triton` package and no subprocess were involved.")
        else:
            print("plugin does not advertise the Triton extension; skipping that part.")

        # --- an FFI handler doing real device work -------------------------
        seen: dict = {}

        @ffi.handler(plugin, "example_dev_copy")
        def dev_copy(call):
            (x,), (y,) = call.args, call.rets
            stream = call.stream()
            seen.update(stream=stream, in_ptr=x.data, out_ptr=y.data, nbytes=x.nbytes)
            # Launch on XLA's stream. Never synchronize it, never switch context.
            cuda.memcpy_dtod_async(y.data, x.data, x.nbytes, stream)

        xs = (np.arange(N, dtype=np.float32) * 0.5)
        exe = client.compile(COPY)
        a = client.buffer_from_host(xs, F32, [N], devices[0])
        try:
            (out,) = exe(a)
            got = np.empty(N, np.float32)
            out.to_host(got)
            print(f"\nhandler: stream=0x{seen['stream']:x} "
                  f"in=0x{seen['in_ptr']:x} out=0x{seen['out_ptr']:x} "
                  f"nbytes={seen['nbytes']}")
            print(f"device-to-device copy on XLA's stream correct: "
                  f"{np.array_equal(got, xs)}")
            out.close()
        except errors.PjrtError as e:
            print(f"\ncustom call failed: {str(e).splitlines()[0][:70]}")
        a.close(); exe.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
