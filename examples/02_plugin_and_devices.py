"""02 — What am I actually talking to?

Everything pypjrt knows about a plugin before and after opening a client:
ABI negotiation, advertised extensions, devices, memory spaces, vendor
attributes. Useful when a program misbehaves and you need to know whether the
plugin even supports what you asked for.

    python examples/02_plugin_and_devices.py [plugin.so]
"""
import sys

import pypjrt
from pypjrt import errors


def main(plugin_path: str | None = None) -> int:
    plugin = pypjrt.Plugin(plugin_path)

    # --- what the plugin says about itself, before any client exists --------
    major, minor = plugin.api_version
    print(f"plugin        : {plugin.path.name}")
    print(f"PJRT API      : {major}.{minor}   (headers {plugin.abi.PJRT_API_MAJOR}."
          f"{plugin.abi.PJRT_API_MINOR}, exact match: {plugin.abi_exact})")
    print(f"vtable        : {plugin.n_slots} slots")
    print(f"platform hint : {plugin.platform_hint}  (accelerator: {plugin.is_accelerator})")
    print(f"xla_version   : {plugin.xla_version}")

    if (rng := plugin.stablehlo_version_range) is not None:
        lo, hi = rng
        print(f"StableHLO     : accepts {'.'.join(map(str, lo))} .. {'.'.join(map(str, hi))}")

    # Extensions are optional. Unknown ones are preserved rather than dropped,
    # so a newer plugin degrades to "capability I don't understand".
    print("\nextensions:")
    for e in plugin.extensions:
        print(f"  type {e.type:<3} {e.name or '<unknown to this build>'}")

    # --- devices and memory ------------------------------------------------
    options = pypjrt.Client.GPU_DEFAULTS if plugin.is_gpu else None
    with pypjrt.Client.create(plugin, options=options) as client:
        print(f"\nplatform      : {client.platform_name}  "
              f"(process {client.process_index})")
        with client.devices() as devices:
            for d in devices:
                print(f"\ndevice {d.id}: kind={d.kind!r} local_hw_id={d.local_hardware_id}")
                if d.coords is not None:
                    print(f"  coords        : {d.coords}")
                for m in d.memories():
                    print(f"  memory space  : id={m.id} kind={m.kind!r}")
                try:
                    stats = d.memory_stats()
                    shown = {k: stats[k] for k in ("bytes_in_use", "bytes_limit") if k in stats}
                    print(f"  allocator     : {shown}")
                except errors.PjrtError:
                    print("  allocator     : not reported by this plugin")
                if attrs := d.attributes:
                    print(f"  attributes    : {', '.join(sorted(attrs))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
