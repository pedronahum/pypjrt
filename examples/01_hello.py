"""01 — Hello, accelerator.

The smallest complete thing: load a plugin, compile a program, run it, read the
answer back. Deliberately uses **nothing but the standard library** -- no numpy,
no jax -- because that is the whole claim.

    python examples/01_hello.py [plugin.so]
"""
import array
import sys

import pypjrt

# StableHLO is the input language. You can write it by hand like this, or have
# JAX (or anything else) lower it for you -- see 03_compile_and_run.py.
PROGRAM = """
module @hello {
  func.func public @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %a, %b : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""

F32 = 11  # PJRT_Buffer_Type_F32; pypjrt.typing has named markers for these


def main(plugin_path: str | None = None) -> int:
    # Client.create() finds a plugin for you: an explicit path, then
    # $PYPJRT_PLUGIN, then any installed JAX plugin wheel, then a short search.
    with pypjrt.Client.create(plugin_path) as client:
        print(f"platform : {client.platform_name}")
        print(f"devices  : {client.device_count}")

        executable = client.compile(PROGRAM)

        # Devices are *borrowed*: valid only inside the `with`. Using one after
        # the block raises instead of quietly reading freed memory.
        with client.devices() as devices:
            device = devices[0]
            a = client.buffer_from_host(array.array("f", [1, 2, 3, 4]), F32, [4], device)
            b = client.buffer_from_host(array.array("f", [10, 20, 30, 40]), F32, [4], device)

            (result,) = executable(a, b)

            out = array.array("f")
            out.frombytes(result.to_host())
            print(f"result   : {list(out)}")

            for handle in (result, a, b):
                handle.close()

        executable.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
