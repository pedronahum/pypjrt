#!/usr/bin/env python3
"""M8 collective half: a real all_reduce across devices.

Usage:  python spike/tpu_gate_collective.py [plugin.so] [num_replicas]
"""
import array, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pypjrt
from pypjrt.compile_options import CompileOptions

F32, N = 11, 8
plugin = pypjrt.Plugin(sys.argv[1] if len(sys.argv) > 1 else None)
want = int(sys.argv[2]) if len(sys.argv) > 2 else 0
opts = {"preallocate": False, "memory_fraction": 0.5} if plugin.is_gpu else None

with pypjrt.Client.create(plugin, options=opts) as c, c.devices() as devs:
    n = want or c.device_count
    print(f"platform={c.platform_name!r} devices={c.device_count} using {n} replica(s)")
    for i, d in enumerate(devs[:n]):
        print(f"  device {i}: id={d.id} kind={d.kind!r} coords={d.coords}")
    groups = ", ".join(str(i) for i in range(n))
    src = f"""
module @m {{
  func.func public @main(%a: tensor<8xf32>) -> tensor<8xf32> {{
    %0 = "stablehlo.all_reduce"(%a) ({{
      ^bb0(%l: tensor<f32>, %r: tensor<f32>):
        %s = stablehlo.add %l, %r : tensor<f32>
        stablehlo.return %s : tensor<f32>
    }}) {{replica_groups = dense<[[{groups}]]> : tensor<1x{n}xi64>}} : (tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }}
}}
"""
    e = c.compile(src, options=CompileOptions(num_replicas=n))
    print(f"  replicas={e.num_replicas} devices/launch={e.addressable_device_count} "
          f"assignment={e.device_assignment()}")
    bufs = [c.buffer_from_host(array.array("f", [float(10 ** i)] * N), F32, [N], devs[i])
            for i in range(n)]
    outs = e.execute_sharded([[b] for b in bufs])
    want_sum = float(sum(10 ** i for i in range(n)))
    allgood = True
    for i, row in enumerate(outs):
        g = array.array("f"); g.frombytes(row[0].to_host())
        good = all(abs(x - want_sum) < 1e-3 for x in g)
        allgood &= good
        print(f"  replica {i} -> {list(g)[:4]}... expected {want_sum} {'OK' if good else 'WRONG'}")
        row[0].close()
    for b in bufs:
        b.close()
    e.close()
    print(f"\nCOLLECTIVE: {'PASS' if allgood else 'FAIL'} across {n} device(s)")
