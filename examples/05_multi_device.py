"""05 — More than one device.

`Executable` is device-list shaped because that is the shape of the underlying
API: arguments are `[device][argument]`. This shows the three ways to use more
than one device -- replicas, SPMD partitions, and a collective -- and checks the
sharded answer against the single-device one.

    python examples/05_multi_device.py [plugin.so]
"""
import array
import sys

import pypjrt
from pypjrt import errors
from pypjrt.compile_options import CompileOptions

F32, N = 11, 8

SINGLE = """
module @m {
  func.func public @main(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
    %0 = stablehlo.multiply %a, %b : tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
"""
SHARDED = """
module @m {
  func.func public @main(
      %a: tensor<8xf32> {mhlo.sharding = "{devices=[2]<=[2]}"},
      %b: tensor<8xf32> {mhlo.sharding = "{devices=[2]<=[2]}"}
    ) -> (tensor<8xf32> {mhlo.sharding = "{devices=[2]<=[2]}"}) {
    %0 = stablehlo.multiply %a, %b : tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
"""
ALL_REDUCE = """
module @m {
  func.func public @main(%a: tensor<8xf32>) -> tensor<8xf32> {
    %0 = "stablehlo.all_reduce"(%a) ({
      ^bb0(%l: tensor<f32>, %r: tensor<f32>):
        %s = stablehlo.add %l, %r : tensor<f32>
        stablehlo.return %s : tensor<f32>
    }) {replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
"""


def main(plugin_path: str | None = None) -> int:
    plugin = pypjrt.Plugin(plugin_path)
    options = pypjrt.Client.GPU_DEFAULTS if plugin.is_gpu else None
    xs = [float(i + 1) for i in range(N)]
    ys = [float(10 * (i + 1)) for i in range(N)]

    with pypjrt.Client.create(plugin, options=options) as client, client.devices() as devices:
        if client.device_count < 2:
            print(f"this plugin exposes {client.device_count} device; "
                  f"the sharded parts need at least 2.")
            return 0
        print(f"platform {client.platform_name!r}, {client.device_count} devices")
        print(f"device assignment for (2 replicas, 1 partition): "
              f"{client.default_device_assignment(2, 1)}")

        # --- reference: one device, whole tensor ---------------------------
        ref_exe = client.compile(SINGLE)
        a = client.buffer_from_host(array.array("f", xs), F32, [N], devices[0])
        b = client.buffer_from_host(array.array("f", ys), F32, [N], devices[0])
        (o,) = ref_exe(a, b)
        reference = o.to_host()
        for h in (o, a, b):
            h.close()
        ref_exe.close()
        print(f"\nsingle device      : {list(array.array('f', reference))}")

        # --- SPMD: the compiler splits the tensor across 2 devices ---------
        exe = client.compile(SHARDED, options=CompileOptions(
            num_partitions=2, use_spmd_partitioning=True))
        print(f"sharded executable : replicas={exe.num_replicas} "
              f"partitions={exe.num_partitions} devices/launch={exe.addressable_device_count}")
        print(f"                     assignment (replica, partition) = {exe.device_assignment()}")

        half = N // 2
        shards_a = client.buffers_from_host(
            [array.array("f", xs[:half]), array.array("f", xs[half:])], F32, [half], devices[:2])
        shards_b = client.buffers_from_host(
            [array.array("f", ys[:half]), array.array("f", ys[half:])], F32, [half], devices[:2])

        # arguments[device][argument]
        outputs = exe.execute_sharded([[shards_a[0], shards_b[0]],
                                       [shards_a[1], shards_b[1]]])
        joined = b"".join(row[0].to_host() for row in outputs)
        print(f"two devices        : {list(array.array('f', joined))}")
        print(f"                     byte-identical to single device: {joined == bytes(reference)}")
        for row in outputs:
            row[0].close()
        for h in shards_a + shards_b:
            h.close()
        exe.close()

        # --- a collective: every replica ends up with the sum --------------
        try:
            coll = client.compile(ALL_REDUCE, options=CompileOptions(num_replicas=2))
            p0 = client.buffer_from_host(array.array("f", [1.0] * N), F32, [N], devices[0])
            p1 = client.buffer_from_host(array.array("f", [10.0] * N), F32, [N], devices[1])
            rows = coll.execute_sharded([[p0], [p1]])
            for i, row in enumerate(rows):
                got = array.array("f")
                got.frombytes(row[0].to_host())
                print(f"all_reduce replica {i}: {list(got)[:4]}...")
                row[0].close()
            p0.close(); p1.close(); coll.close()
        except errors.PjrtError as e:
            print(f"\nall_reduce unavailable here: {str(e).splitlines()[0][:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
