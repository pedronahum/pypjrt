"""Tier 0: dtype codes must come from the ABI, never from a literal.

Regression guard for a real bug: BF16 was hand-typed as 16, which is F8E5M2 in
PJRT_Buffer_Type. 16 *is* BF16 -- in XLA_FFI_DataType, a different enum for the
same concept. The plugin read a bf16 buffer as f8e5m2 and rejected it.
"""
import pytest
from pypjrt import _abi
from pypjrt.typing import ALL, BF16, F32, by_code

pytestmark = pytest.mark.tier0
A = _abi.load(0, _abi.available()[0][1])[0]


@pytest.mark.parametrize("marker", ALL, ids=lambda m: m.name)
def test_marker_code_matches_the_abi_enum(marker):
    want = getattr(A, f"PJRT_Buffer_Type_{marker.__name__}")
    assert marker.code == want, (
        f"{marker.__name__}.code is {marker.code} but PJRT_Buffer_Type_"
        f"{marker.__name__} is {want} -- do not hand-type dtype codes")


def test_bf16_is_thirteen_not_sixteen():
    assert BF16.code == 13
    assert A.PJRT_Buffer_Type_F8E5M2 == 16


def test_the_two_dtype_enums_are_not_interchangeable():
    """PJRT_Buffer_Type and XLA_FFI_DataType number the same concepts
    differently. Conflating them is what caused the bug above."""
    assert A.PJRT_Buffer_Type_BF16 == 13
    assert A.XLA_FFI_DataType_BF16 == 16
    assert A.PJRT_Buffer_Type_F32 == A.XLA_FFI_DataType_F32 == 11  # these happen to agree


def test_by_code_roundtrip():
    for m in ALL:
        assert by_code(m.code) is m
    assert by_code(9999) is None


def test_dlpack_table_is_derived_from_the_markers():
    from pypjrt.dlpack import _PJRT_TO_DL, kDLBfloat
    assert _PJRT_TO_DL[BF16.code] == (kDLBfloat, 16)
    assert BF16.code not in (16,), "the f8e5m2 code must not appear as bf16"
    for m in ALL:
        assert m.code in _PJRT_TO_DL, f"{m.name} missing from the DLPack table"
