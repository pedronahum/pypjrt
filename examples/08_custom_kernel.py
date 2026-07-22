"""08 — Your own kernel, called by XLA.

Register a Python function as an XLA FFI handler and XLA will call it from
inside a compiled program, handing over the real buffers and (on an
accelerator) its own stream.

The check at the end is the one worth copying: compile the *same maths* twice --
once as your custom call, once as plain StableHLO -- and diff. It needs no
golden data and catches almost everything.

Note the import: `pypjrt.ffi` names a platform and hands out raw pointers, so
device opacity stops here. That is why it is a separate module.

    python examples/08_custom_kernel.py [plugin.so]
"""
import math
import sys

import pypjrt
import pypjrt.ffi as ffi
from pypjrt import errors

F32, N = 11, 256

# `api_version = 4` selects the typed FFI. Attributes must be a dictionary
# attribute *on the op*: the `mhlo.backend_config` spelling compiles fine and
# silently delivers nothing.
FUSED = """
module @m {
  func.func public @main(%x: tensor<256xf32>, %s: tensor<256xf32>) -> tensor<256xf32> {
    %0 = stablehlo.custom_call @example_scaled_gelu(%x, %s) {
        api_version = 4 : i32,
        backend_config = {alpha = 1.7 : f64, tag = "gelu"}
      } : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    return %0 : tensor<256xf32>
  }
}
"""
DECOMPOSED = """
module @m {
  func.func public @main(%x: tensor<256xf32>, %s: tensor<256xf32>) -> tensor<256xf32> {
    %half = stablehlo.constant dense<0.5> : tensor<256xf32>
    %one  = stablehlo.constant dense<1.0> : tensor<256xf32>
    %a    = stablehlo.constant dense<1.7> : tensor<256xf32>
    %sc   = stablehlo.multiply %x, %s : tensor<256xf32>
    %t    = stablehlo.tanh %sc : tensor<256xf32>
    %p    = stablehlo.add %t, %one : tensor<256xf32>
    %g    = stablehlo.multiply %sc, %p : tensor<256xf32>
    %h    = stablehlo.multiply %g, %half : tensor<256xf32>
    %r    = stablehlo.multiply %h, %a : tensor<256xf32>
    return %r : tensor<256xf32>
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
    if plugin.extension("FFI") is None:
        print("this plugin does not advertise the FFI extension")
        return 0
    options = pypjrt.Client.GPU_DEFAULTS if plugin.is_gpu else None
    seen: dict = {}

    @ffi.handler(plugin, "example_scaled_gelu")
    def scaled_gelu(call):
        # You only ever see EXECUTE: the metadata handshake XLA performs before
        # first use is answered for you. (Miss it and XLA silently drops the
        # registration, which is a memorable afternoon.)
        seen.update(stage=call.stage, attrs=dict(call.attrs), stream=call.stream())
        x, s = call.args
        (y,) = call.rets
        alpha = call.attrs["alpha"]

        if call.stream():
            # On an accelerator `.data` is a *device* pointer: launch on the
            # stream you were given, never dereference it here. See example 09.
            raise RuntimeError("this handler only implements the host path")

        xa, sa, ya = x.as_ctypes(), s.as_ctypes(), y.as_ctypes()
        for i in range(x.size):
            v = xa[i] * sa[i]
            ya[i] = alpha * 0.5 * v * (math.tanh(v) + 1.0)

    xs = np.arange(N, dtype=np.float32) / N
    ss = (1.0 + np.arange(N, dtype=np.float32) / (2 * N)).astype(np.float32)

    with pypjrt.Client.create(plugin, options=options) as client, client.devices() as devices:

        def run(program):
            exe = client.compile(program)
            a = client.buffer_from_host(xs, F32, [N], devices[0])
            b = client.buffer_from_host(ss, F32, [N], devices[0])
            (out,) = exe(a, b)
            got = np.empty(N, np.float32)
            out.to_host(got)
            for h in (out, a, b):
                h.close()
            exe.close()
            return got

        try:
            mine = run(FUSED)
        except errors.PjrtError as e:
            print(f"custom call not runnable here: {str(e).splitlines()[0][:70]}")
            return 0

        print(f"handler saw stage={seen['stage']} (3 == EXECUTE)")
        print(f"decoded attributes: {seen['attrs']}")

        reference = run(DECOMPOSED)
        worst = float(np.max(np.abs(mine - reference) / np.maximum(np.abs(reference), 1e-6)))
        print(f"\ndecompose oracle: max relative error {worst:.3e} "
              f"-> {'agrees' if worst < 1e-4 else 'DISAGREES'}")

        # Without a negative control the result above proves nothing.
        try:
            client.compile(FUSED.replace("example_scaled_gelu", "never_registered"))
            print("negative control FAILED: an unregistered symbol compiled")
        except errors.PjrtError as e:
            print(f"negative control : {str(e).splitlines()[0][:72]}")

        # A handler that raises fails one execution, not the process.
        @ffi.handler(plugin, "example_boom")
        def boom(call):
            raise ValueError("deliberate")

        try:
            run(FUSED.replace("example_scaled_gelu", "example_boom"))
        except errors.PjrtError as e:
            print(f"raising handler  : {type(e).__name__}: {str(e).splitlines()[0][:52]}")
        print("...and the process is still alive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
