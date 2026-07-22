"""07 — Never compile the same program twice.

XLA compilation is slow -- seconds for a real model. A compiled executable can
be serialized and reloaded, so only the first run pays. This shows the cache,
the artifact container, and the guards that stop you loading one somewhere it
would not work.

    python examples/07_artifacts_and_cache.py [plugin.so]
"""
import pathlib
import sys
import tempfile
import time

import pypjrt
from pypjrt import errors

# Big enough that compilation is visibly slower than a cache hit.
PROGRAM = """
module @m {
  func.func public @main(%a: tensor<256x256xf32>, %b: tensor<256x256xf32>)
      -> tensor<256x256xf32> {
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0]
       : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    %1 = stablehlo.tanh %0 : tensor<256x256xf32>
    %2 = stablehlo.dot_general %1, %b, contracting_dims = [1] x [0]
       : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    return %2 : tensor<256x256xf32>
  }
}
"""


def main(plugin_path: str | None = None) -> int:
    plugin = pypjrt.Plugin(plugin_path)
    options = pypjrt.Client.GPU_DEFAULTS if plugin.is_gpu else None
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="pypjrt-example-"))
    cache = pypjrt.CompileCache(workdir / "cache")

    with pypjrt.Client.create(plugin, options=options) as client:
        # --- the cache ------------------------------------------------------
        t0 = time.perf_counter()
        exe = client.compile(PROGRAM, cache=cache)
        cold = time.perf_counter() - t0
        exe.close()

        t0 = time.perf_counter()
        exe = client.compile(PROGRAM, cache=cache)
        warm = time.perf_counter() - t0

        print(f"cold compile : {cold * 1e3:8.1f} ms")
        print(f"warm (cached): {warm * 1e3:8.1f} ms   ({cold / max(warm, 1e-9):.0f}x faster)")
        print(f"cache        : {cache}")

        # The key includes the plugin's XLA version, because an executable
        # compiled against one XLA can silently miscompute under another.
        print(f"\nfingerprint  : {exe.fingerprint()[:24].decode(errors='replace')}...")
        print(f"serialized   : {len(exe.serialize())} bytes")

        # --- a portable artifact on disk ------------------------------------
        artifact = exe.to_artifact(source=PROGRAM.encode())
        path = artifact.write(workdir / "program.pypjrta")
        print(f"\nartifact     : {path.name}, {path.stat().st_size} bytes")
        print(f"  platform   : {artifact.platform}")
        print(f"  xla_version: {artifact.xla_version}")
        print(f"  abi proto  : {len(artifact.abi_proto) // 2} bytes "
              f"({'from the AbiVersion extension' if artifact.abi_proto else 'not advertised here'})")
        exe.close()

        reloaded = client.load_artifact(path)
        print(f"  reloaded   : {reloaded.num_outputs} output(s), no compilation")
        reloaded.close()

        # --- the guards -----------------------------------------------------
        artifact.platform = "some-other-platform"
        artifact.xla_version = 999999
        try:
            client.load_artifact(artifact)
        except errors.PjrtError as e:
            print("\na mismatched artifact is refused, and says why:")
            for reason in str(e).split("; "):
                print(f"  - {reason}")

        # Where the plugin offers it, it answers the compatibility question
        # itself rather than us guessing from recorded strings.
        exe2 = client.compile(PROGRAM)
        verdict = exe2.abi_compatibility()
        print(f"\nplugin's own compatibility verdict: "
              f"{'compatible' if verdict is None else verdict}")
        exe2.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
