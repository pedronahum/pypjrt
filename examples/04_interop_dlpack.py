"""04 — Sharing device memory with other frameworks.

pypjrt is designed to sit *alongside* whatever you already use. DLPack works in
both directions, with no host round-trip: hand a buffer to numpy/jax/torch, or
adopt one of theirs.

    python examples/04_interop_dlpack.py [plugin.so]
"""
import sys

import pypjrt
from pypjrt import errors

SQUARE = """
module @m {
  func.func public @main(%a: tensor<8xf32>) -> tensor<8xf32> {
    %0 = stablehlo.multiply %a, %a : tensor<8xf32>
    return %0 : tensor<8xf32>
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

    with pypjrt.Client.create(plugin, options=options) as client, client.devices() as devices:
        src = np.arange(8, dtype=np.float32)
        buf = client.buffer_from_host(src, F32, [8], devices[0])

        # --- export ---------------------------------------------------------
        device_type, ordinal = buf.__dlpack_device__()
        print(f"buffer          : __dlpack_device__ = ({device_type}, {ordinal})  "
              f"{'CPU' if device_type == 1 else 'CUDA'}")

        if device_type == 1:
            # Host memory: numpy can map it directly, and it really is the same
            # allocation rather than a copy.
            view = np.from_dlpack(buf)
            shared = view.__array_interface__["data"][0] == buf.device_pointer()
            print(f"export -> numpy : {view}")
            print(f"                  same memory as the device buffer: {shared}")
            del view
        else:
            # numpy is host-only, so it cannot consume a CUDA capsule. The
            # capsule is still valid -- a CUDA-aware consumer (torch, cupy,
            # jax-on-gpu) would take it, and we can adopt it back ourselves.
            print("export -> numpy : skipped, numpy cannot map device memory")
            capsule = buf.__dlpack__()
            adopted_back = client.from_dlpack(capsule)
            print(f"export -> pypjrt: adopted back at the same address: "
                  f"{adopted_back.device_pointer() == buf.device_pointer()}")
            adopted_back.close()

        # --- import: adopt someone else's buffer ---------------------------
        try:
            import jax
            import jax.numpy as jnp

            if jax.devices()[0].platform != client.platform_name:
                raise ImportError(
                    f"jax is on {jax.devices()[0].platform!r}, this client is on "
                    f"{client.platform_name!r}; DLPack cannot cross device types")
            foreign = jnp.arange(8, dtype=jnp.float32) * 3
            adopted = client.from_dlpack(foreign)
            print(f"\nimport <- jax   : dims={adopted.dimensions} dtype={adopted.element_type}")

            # ...and it is a first-class buffer: run a program over it.
            exe = client.compile(SQUARE)
            (out,) = exe(adopted)
            back = __import__("jax").dlpack.from_dlpack(out)
            print(f"jax -> pypjrt -> execute -> jax : {np.asarray(back)}")
            del back
            out.close()
            adopted.close()
            exe.close()
        except ImportError as why:
            print(f"\nimport <- jax   : skipped ({why})")
        except errors.PjrtError as e:
            print(f"\nimport refused: {str(e).splitlines()[-1].strip()}")

        # Not every producer qualifies. The backend requires its own alignment,
        # and numpy's default allocator does not guarantee it. Build a
        # deliberately misaligned view so the failure is reproducible rather
        # than a coin flip.
        if device_type != 1:
            buf.close()
            return 0
        base = np.empty(8 + 32, dtype=np.float32)
        pad = (64 - (base.__array_interface__["data"][0] % 64)) // 4
        misaligned = base[pad + 1: pad + 9]
        try:
            client.from_dlpack(misaligned)
            print("\nmisaligned numpy array adopted (unexpected)")
        except errors.PjrtError as e:
            print("\na producer whose allocation is misaligned is refused, with a reason:")
            print(f"  {str(e).splitlines()[-1].strip()}")

        buf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
