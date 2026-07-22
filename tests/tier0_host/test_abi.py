"""Tier 0: the ABI itself. Needs no plugin and no hardware."""
import ctypes, pytest
from pypjrt import _abi
from pypjrt.compile_options import CompileOptions, WELL_KNOWN_DEFAULT

pytestmark = pytest.mark.tier0
A = _abi.load(0, _abi.available()[0][1])[0]


def test_import_asserts_ctypes_matches_clang():
    # Importing the generated module runs every _check(). The layout of
    # PJRT_Buffer_MemoryLayout caught a real generator bug this way: an
    # anonymous union member was dropped, shifting `type` from 72 to 16.
    assert A.PJRT_Buffer_MemoryLayout.type.offset == 72
    assert hasattr(A.PJRT_Buffer_MemoryLayout, "tiled")  # spliced via _anonymous_


def test_vtable_shape():
    assert A.VTABLE_OFFSET == 40
    assert len(A.SLOT) >= 136
    assert A.SLOT["PJRT_Error_Destroy"] == 0
    assert list(A.SLOT) == list(A.FUNCTIONS)


def test_struct_size_is_not_sizeof():
    """STRUCT_SIZE is offsetof(last)+sizeof(last). Using sizeof() makes the
    plugin reject the call. 23 structs diverge at PJRT 0.114."""
    diverging = [
        n[: -len("_STRUCT_SIZE")] for n in dir(A)
        if n.endswith("_STRUCT_SIZE")
        and isinstance(getattr(A, n[: -len("_STRUCT_SIZE")], None), type)
        and getattr(A, n) != ctypes.sizeof(getattr(A, n[: -len("_STRUCT_SIZE")]))
    ]
    assert len(diverging) >= 20
    assert A.PJRT_Device_MemoryStats_Args_STRUCT_SIZE == 201
    assert ctypes.sizeof(A.PJRT_Device_MemoryStats_Args) == 208


def test_xla_ffi_call_frame_layout():
    """Pins the offsets spike/pjrt_ffi.py derived by hand, with no GPU."""
    assert ctypes.sizeof(A.XLA_FFI_CallFrame) == 176
    assert (A.XLA_FFI_CallFrame.args.offset, A.XLA_FFI_CallFrame.rets.offset,
            A.XLA_FFI_CallFrame.attrs.offset, A.XLA_FFI_CallFrame.future.offset) == (40, 80, 120, 168)
    assert ctypes.sizeof(A.XLA_FFI_Buffer) == 48
    assert A.XLA_FFI_Buffer.dims.offset == 40
    assert A.XLA_FFI_ExecutionStage_EXECUTE == 3


def test_shape2_execute_is_device_list_shaped():
    for f in ("num_devices", "num_args", "argument_lists", "output_lists"):
        assert hasattr(A.PJRT_LoadedExecutable_Execute_Args, f)


def test_shape_multiprocess_kv_callbacks_exist():
    for f in ("kv_get_callback", "kv_put_callback", "kv_try_get_callback"):
        assert hasattr(A.PJRT_Client_Create_Args, f)


def test_compile_options_default_matches_well_known_bytes():
    assert CompileOptions().encode(A) == WELL_KNOWN_DEFAULT


def test_compile_options_is_not_a_constant():
    """Hardcoding the 6 bytes hardcodes single-device execution."""
    sharded = CompileOptions(num_partitions=4, use_spmd_partitioning=True,
                             use_shardy_partitioner=True).encode(A)
    assert sharded != WELL_KNOWN_DEFAULT
    F = A.PROTO_FIELDS["ExecutableBuildOptionsProto"]
    assert (F["num_replicas"], F["num_partitions"]) == (4, 5)
    assert F["use_spmd_partitioning"] == 6      # NOT 7 -- that's use_auto_spmd
    assert F["use_shardy_partitioner"] == 19    # NOT 34 -- that's a different message
