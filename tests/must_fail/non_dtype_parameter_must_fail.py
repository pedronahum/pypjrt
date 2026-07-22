# EXPECT: type argument
"""Buffer's parameter is bound to DType."""
from pypjrt.client import Buffer


def probe(b: Buffer[int]) -> None:            # must fail: int is not a DType
    ...
