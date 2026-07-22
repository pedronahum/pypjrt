# EXPECT: assignment
"""A buffer's dtype is part of its static type."""
import pypjrt
from pypjrt.typing import F32, F64
from pypjrt.client import Buffer


def take_f64(b: Buffer[F64]) -> None: ...


def probe(c: pypjrt.Client, d: pypjrt.Device, data: bytes) -> None:
    f32 = c.typed_buffer(F32, data, [4], d)
    wrong: Buffer[F64] = f32          # must fail: Buffer[F32] is not Buffer[F64]
    take_f64(f32)                     # must fail: same, through a parameter
