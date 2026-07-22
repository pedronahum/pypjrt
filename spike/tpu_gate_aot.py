#!/usr/bin/env python3
"""M4 gate (b): compile for hardware that is not present.

Usage:  python spike/tpu_gate_aot.py [plugin.so] [topology-name ...]

Measured elsewhere: CUDA does the client-free compile but derives the topology
from local hardware; CPU has no AOT compiler at all. TPU is the case the API
exists for.
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pypjrt
from pypjrt import errors
from pypjrt.topology import Topology

ADD = """
module @m {
  func.func public @main(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
    %0 = stablehlo.add %a, %b : tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
"""
plugin = pypjrt.Plugin(sys.argv[1] if len(sys.argv) > 1 else None)
plugin.initialize()
names = sys.argv[2:] or ["v5e:2x2", "v5e:1x1", "v4:2x2x1", "v6e:2x2"]
print(f"plugin {plugin.path.name}  hint={plugin.platform_hint}  api={plugin.api_version}")
print("NOTE: no Client is created anywhere in this script.\n")

ok = False
for name in names:
    try:
        t = Topology.create(plugin, name)
    except errors.PjrtError as e:
        print(f"  create({name!r:12}) -> {type(e).__name__}: {str(e).splitlines()[0][:70]}")
        continue
    descs = t.device_descriptions()
    kinds = sorted({d["kind"] for d in descs})
    print(f"  create({name!r:12}) -> platform={t.platform_name!r} "
          f"version={t.platform_version!r} devices={len(descs)} kinds={kinds}")
    try:
        blob = t.compile(ADD)
        print(f"      client-free compile: {len(blob)} bytes")
        ok = True
    except errors.PjrtError as e:
        print(f"      compile: {type(e).__name__}: {str(e).splitlines()[0][:70]}")
    try:
        s = t.serialize()
        t2 = Topology.deserialize(plugin, s)
        print(f"      serialize/deserialize: {len(s)} bytes, round-trip OK")
        t2.close()
    except errors.PjrtError as e:
        print(f"      serialize/deserialize: {type(e).__name__}: {str(e).splitlines()[0][:60]}")
    t.close()

print(f"\nGATE (b): {'names honoured and compiled -- verify the device count matches the '
                    'requested slice, not local hardware' if ok else 'NOT MET on this plugin'}")
