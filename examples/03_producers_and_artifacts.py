"""03 — Where StableHLO comes from, and how to ship it.

pypjrt *consumes* StableHLO; it never emits any. This shows the three shapes a
program can arrive in -- hand-written text, text lowered by JAX, and a versioned
portable artifact -- and that all three give the same answer.

Portable artifacts are the one to prefer when you persist a program: they are
versioned bytecode, so producer and plugin can evolve independently.

    python examples/03_producers_and_artifacts.py [plugin.so]
"""
import sys

import pypjrt

HAND_WRITTEN = """
module @m {
  func.func public @main(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
    %0 = stablehlo.multiply %a, %b : tensor<8xf32>
    %1 = stablehlo.tanh %0 : tensor<8xf32>
    return %1 : tensor<8xf32>
  }
}
"""
F32 = 11


def main(plugin_path: str | None = None) -> int:
    try:
        import numpy as np
    except ImportError:
        print("this example needs numpy: pip install 'pypjrt[numpy]'")
        return 0

    plugin = pypjrt.Plugin(plugin_path)
    options = pypjrt.Client.GPU_DEFAULTS if plugin.is_gpu else None
    x = np.arange(1, 9, dtype=np.float32)
    y = (x * 0.25).astype(np.float32)

    with pypjrt.Client.create(plugin, options=options) as client, client.devices() as devices:

        def run(program, label):
            exe = client.compile(program)
            a = client.buffer_from_host(x, F32, [8], devices[0])
            b = client.buffer_from_host(y, F32, [8], devices[0])
            (out,) = exe(a, b)
            got = np.empty(8, np.float32)
            out.to_host(got)
            for h in (out, a, b):
                h.close()
            exe.close()
            kind = "text" if isinstance(program, str) else "bytecode"
            print(f"  {label:<34} {kind:<9} {len(program):>6} -> {got[:3]}")
            return got.tobytes()

        print("same program, three encodings:")
        reference = run(HAND_WRITTEN, "hand-written")

        # --- a producer: JAX lowers to exactly the same language -----------
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            print("\n(install 'pypjrt[jax]' to see the JAX producer and portable artifacts)")
            return 0

        lowered = jax.jit(lambda a, b: jnp.tanh(a * b)).lower(x, y).as_text()
        assert run(lowered, "lowered by jax.jit") == reference

        # --- a portable artifact: versioned bytecode -----------------------
        import jaxlib.mlir.dialects.stablehlo as sh

        # Ask the plugin which version to target rather than guessing.
        target = plugin.stablehlo_target("min")
        artifact = sh.serialize_portable_artifact_str(lowered, target)
        assert run(artifact, f"portable artifact @ {target}") == reference

        print("\nall three produced identical bytes.")
        print("prefer the artifact when persisting: it pins a StableHLO version,")
        print("so the producer and the plugin can move independently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
