"""Phantom dtypes: the typed shell over an untyped core.

Python has no linear types and no dependent shapes, so the honest ceiling here
is what a *type checker* can prove -- and the honest floor is that none of it
exists at runtime. Both are deliberate.

What this buys::

    a: Buffer[F32] = client.typed_buffer(F32, data, [4], dev)
    b: Buffer[F64] = a                       # pyright: error

What it does not buy: shapes (there is no type-level arithmetic in Python, and
shape types in stricter languages usually carry rank without extents either), or anything
about lifetime. `tests/must_fail/` pins what is genuinely rejected, so the
claim can never drift ahead of the checker.

`DType` markers are never instantiated -- they carry the PJRT code as a class
attribute so a runtime value and a static type stay in one place.
"""

from __future__ import annotations

import abc
import importlib
from typing import ClassVar, TypeVar

from . import _abi

# Codes come from the generated ABI, never from a literal. Hand-typing this
# table put BF16 at 16 -- which is F8E5M2 in PJRT_Buffer_Type. 16 *is* BF16, but
# in XLA_FFI_DataType, a different enum for the same concept. The plugin duly
# read a bf16 buffer as f8e5m2 and rejected it.
_v = _abi.available()[0]
_M = importlib.import_module(f"{_abi.__name__}.pjrt_{_v[0]}_{_v[1]}")


def _code(name: str) -> int:
    return int(getattr(_M, f"PJRT_Buffer_Type_{name}"))


class DType(abc.ABC):
    """A phantom dtype marker. Never instantiated.

    Abstract *by construction*: no marker implements ``_marker``, so every one
    of them stays abstract and ``F32()`` is a static error, not merely a
    runtime one. A `must_fail` probe pins that.
    """

    code: ClassVar[int]
    name: ClassVar[str]
    itemsize: ClassVar[int]

    @abc.abstractmethod
    def _marker(self) -> None:  # pragma: no cover - deliberately never implemented
        """Unimplemented on purpose. Do not add a body to any subclass."""


class PRED(DType):
    code, name, itemsize = _code("PRED"), "pred", 1


class S8(DType):
    code, name, itemsize = _code("S8"), "s8", 1


class S16(DType):
    code, name, itemsize = _code("S16"), "s16", 2


class S32(DType):
    code, name, itemsize = _code("S32"), "s32", 4


class S64(DType):
    code, name, itemsize = _code("S64"), "s64", 8


class U8(DType):
    code, name, itemsize = _code("U8"), "u8", 1


class U16(DType):
    code, name, itemsize = _code("U16"), "u16", 2


class U32(DType):
    code, name, itemsize = _code("U32"), "u32", 4


class U64(DType):
    code, name, itemsize = _code("U64"), "u64", 8


class F16(DType):
    code, name, itemsize = _code("F16"), "f16", 2


class F32(DType):
    code, name, itemsize = _code("F32"), "f32", 4


class F64(DType):
    code, name, itemsize = _code("F64"), "f64", 8


class BF16(DType):
    code, name, itemsize = _code("BF16"), "bf16", 2


#: Bound so `Buffer[SomethingElse]` is itself a type error.
DT = TypeVar("DT", bound=DType)

ALL: tuple[type[DType], ...] = (
    PRED, S8, S16, S32, S64, U8, U16, U32, U64, F16, F32, F64, BF16,
)
BY_CODE: dict[int, type[DType]] = {d.code: d for d in ALL}


def by_code(code: int) -> type[DType] | None:
    """Recover a marker from a runtime PJRT dtype, or ``None``.

    Refinement from an erased value is a runtime lookup in every host language
    without GADTs -- Swift reaches the same conclusion.
    """
    return BY_CODE.get(code)


__all__ = ["DType", "DT", "ALL", "BY_CODE", "by_code",
           "PRED", "S8", "S16", "S32", "S64", "U8", "U16", "U32", "U64",
           "F16", "F32", "F64", "BF16"]
