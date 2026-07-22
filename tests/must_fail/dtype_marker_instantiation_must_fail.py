# EXPECT: abstract
"""DType markers are types, not values."""
from pypjrt.typing import DType, F32


def probe() -> None:
    x: F32 = F32()                    # must fail: markers are never instantiated
    y: DType = DType()                # must fail: same
