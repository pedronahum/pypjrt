"""10 — Nothing blocks until you ask for a value.

Every device operation is asynchronous underneath. This example shows the three
places that surfaces:

* an `Event` is a future -- ask whether it is done, attach a callback, or block
  for its *result*, which is where a failure is reported;
* an event can be minted on the host before the work exists, so a consumer can
  wait on a producer that has not started yet;
* a large weight tensor can be streamed into device memory in chunks, with the
  first chunks landing while later ones are still being read.

    python examples/10_async_and_futures.py [plugin.so]
"""
import array
import sys
import threading

import pypjrt
from pypjrt import errors
from pypjrt.transfer import ShapeSpec

F32 = 11
ROWS, COLS, CHUNKS = 64, 64, 4

DOUBLE = """
module @m {
  func.func public @main(%x: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %0 = stablehlo.add %x, %x : tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
"""


def main(plugin_path: str | None = None) -> int:
    plugin = pypjrt.Plugin(plugin_path)
    options = pypjrt.Client.GPU_DEFAULTS if plugin.is_gpu else None

    with pypjrt.Client.create(plugin, options=options) as client, \
            client.devices() as devices:
        device = devices[0]

        # --- 1. a future you can create before the work exists --------------
        # `PJRT_Event_Create` / `Set` let the host mint a completion token. A
        # consumer thread can wait on it now and be released later, which is how
        # you splice work PJRT does not know about into a PJRT dependency chain.
        token = pypjrt.Event.create(plugin)
        print(f"host-minted event  : done={token.done()} before anyone sets it")

        released = threading.Event()
        token.add_done_callback(lambda err: released.set())

        token.set()                       # ... the producer finishes
        released.wait(timeout=5)
        print(f"                     done={token.done()} after set(), "
              f"callback fired={released.is_set()}")
        token.close()

        # An event carries failure, not just completion. This is the difference
        # between "the transfer finished" and "the transfer worked".
        failed = pypjrt.Event.create(plugin)
        failed.set(error_code=3, message="synthetic failure")
        try:
            failed.result()
            print("                     UNEXPECTED: a failed event did not raise")
        except errors.PjrtError as e:
            print(f"failed event       : result() raises -> {e.message}")
        failed.close()

        # --- 2. streaming a large tensor in, chunk by chunk -----------------
        # Allocate the device buffer from a shape spec *before* the bytes exist,
        # then fill it in pieces. A checkpoint reader can start uploading row 0
        # while it is still parsing row N, with no full host staging copy.
        row_bytes = COLS * 4
        rows_per_chunk = ROWS // CHUNKS

        with client.async_transfer([ShapeSpec(F32, (ROWS, COLS))],
                                   device=device) as t:
            print(f"\nasync transfer     : device reserved "
                  f"{t.buffer_size(0)} bytes for {ROWS}x{COLS} f32")
            for i in range(CHUNKS):
                # Standing in for "read the next slice off disk".
                chunk = bytearray()
                for r in range(rows_per_chunk):
                    value = float(i * rows_per_chunk + r)
                    chunk += memoryview(array.array("f", [value] * COLS)).cast("B")
                t.transfer(0, chunk, offset=i * rows_per_chunk * row_bytes,
                           last=(i == CHUNKS - 1))
                print(f"                     chunk {i + 1}/{CHUNKS} at offset "
                      f"{i * rows_per_chunk * row_bytes}")
            (weights,) = t.buffers()

        # --- 3. the result is a real buffer, so just use it -----------------
        exe = client.compile(DOUBLE)
        (out,) = exe(weights)

        # Reading back is the first place anything blocks: `to_host` awaits the
        # execution's completion event for us.
        host = bytearray(ROWS * COLS * 4)
        out.to_host(host)
        got = array.array("f")
        got.frombytes(bytes(host))

        first_row_ok = got[0] == 0.0
        last_row_ok = got[(ROWS - 1) * COLS] == float(ROWS - 1) * 2
        print(f"\nstreamed then run  : row 0 -> {got[0]}, "
              f"row {ROWS - 1} -> {got[(ROWS - 1) * COLS]}")
        print(f"                     every chunk arrived intact: "
              f"{first_row_ok and last_row_ok}")

        out.close()
        weights.close()
        exe.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
