# EXPECT: argument
"""Handles are not interchangeable."""
import pypjrt
from pypjrt.typing import F32


def probe(c: pypjrt.Client, d: pypjrt.Device, data: bytes) -> None:
    b = c.typed_buffer(F32, data, [4], d)
    c.typed_buffer(F32, data, [4], b)         # must fail: Buffer is not a Device
    exe = c.compile("module {}")
    exe.execute_sharded([[d]])                # must fail: Device is not a Buffer
