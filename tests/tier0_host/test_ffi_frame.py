"""Tier 0: hand-build an XLA_FFI_CallFrame in memory and decode it.

The third leg of the M6 evidence set. It pins every call-frame
offset with **no plugin and no GPU**, so an ABI drift is caught by `pytest -q`
on any machine rather than only where hardware exists.
"""
import ctypes, pytest
from pypjrt import _abi
from pypjrt.ffi.frame import CallFrame, FfiBuffer

pytestmark = pytest.mark.tier0
A = _abi.load(0, _abi.available()[0][1])[0]
_VOIDP = ctypes.c_void_p
F32 = 11


class _Frame:
    """A synthetic call frame; holds every allocation alive."""

    def __init__(self, args, rets, attrs=(), stage=None, api=0):
        self.keep = []
        f = A.XLA_FFI_CallFrame()
        ctypes.memset(ctypes.byref(f), 0, ctypes.sizeof(f))
        f.struct_size = A.XLA_FFI_CallFrame_STRUCT_SIZE
        f.stage = A.XLA_FFI_ExecutionStage_EXECUTE if stage is None else stage
        f.api = api
        self._fill_buffers(f.args, args, "args")
        self._fill_buffers(f.rets, rets, "rets")
        self._fill_attrs(f.attrs, attrs)
        self.keep.append(f)
        self.frame = f

    def _fill_buffers(self, slot, specs, field):
        slot.struct_size = A.XLA_FFI_Args_STRUCT_SIZE
        slot.size = len(specs)
        if not specs:
            return
        bufs, ptrs, types = [], (_VOIDP * len(specs))(), (ctypes.c_int32 * len(specs))()
        for i, (dtype, dims, data) in enumerate(specs):
            b = A.XLA_FFI_Buffer()
            ctypes.memset(ctypes.byref(b), 0, ctypes.sizeof(b))
            b.struct_size = A.XLA_FFI_Buffer_STRUCT_SIZE
            b.dtype, b.rank = dtype, len(dims)
            dim_arr = (ctypes.c_int64 * len(dims))(*dims)
            b.dims = ctypes.cast(dim_arr, _VOIDP)
            b.data = data
            self.keep += [b, dim_arr]
            bufs.append(b)
            ptrs[i] = ctypes.cast(ctypes.byref(b), _VOIDP)
            types[i] = A.XLA_FFI_ArgType_BUFFER
        slot.types = ctypes.cast(types, _VOIDP)
        setattr(slot, field, ctypes.cast(ptrs, _VOIDP))
        self.keep += [ptrs, types, bufs]

    def _fill_attrs(self, slot, attrs):
        slot.struct_size = A.XLA_FFI_Attrs_STRUCT_SIZE
        slot.size = len(attrs)
        if not attrs:
            return
        names = (_VOIDP * len(attrs))()
        vals = (_VOIDP * len(attrs))()
        types = (ctypes.c_int32 * len(attrs))()
        for i, (key, kind, payload) in enumerate(attrs):
            kb = ctypes.create_string_buffer(key.encode())
            span = A.XLA_FFI_ByteSpan(ptr=ctypes.cast(kb, _VOIDP), len=len(key))
            names[i] = ctypes.cast(ctypes.byref(span), _VOIDP)
            self.keep += [kb, span]
            if kind == "scalar":
                cval = ctypes.c_float(payload)
                sc = A.XLA_FFI_Scalar(dtype=F32, value=ctypes.cast(ctypes.byref(cval), _VOIDP))
                vals[i] = ctypes.cast(ctypes.byref(sc), _VOIDP)
                types[i] = A.XLA_FFI_AttrType_SCALAR
                self.keep += [cval, sc]
            elif kind == "string":
                sb = ctypes.create_string_buffer(payload.encode())
                sp = A.XLA_FFI_ByteSpan(ptr=ctypes.cast(sb, _VOIDP), len=len(payload))
                vals[i] = ctypes.cast(ctypes.byref(sp), _VOIDP)
                types[i] = A.XLA_FFI_AttrType_STRING
                self.keep += [sb, sp]
            else:  # array of i64
                arr = (ctypes.c_int64 * len(payload))(*payload)
                ar = A.XLA_FFI_Array(dtype=5, size=len(payload),
                                     data=ctypes.cast(arr, _VOIDP))
                vals[i] = ctypes.cast(ctypes.byref(ar), _VOIDP)
                types[i] = A.XLA_FFI_AttrType_ARRAY
                self.keep += [arr, ar]
        slot.names = ctypes.cast(names, _VOIDP)
        slot.attrs = ctypes.cast(vals, _VOIDP)
        slot.types = ctypes.cast(types, _VOIDP)
        self.keep += [names, vals, types]

    def address(self) -> int:
        return ctypes.addressof(self.frame)


def test_call_frame_layout_is_pinned():
    assert ctypes.sizeof(A.XLA_FFI_CallFrame) == 176
    assert (A.XLA_FFI_CallFrame.args.offset, A.XLA_FFI_CallFrame.rets.offset,
            A.XLA_FFI_CallFrame.attrs.offset, A.XLA_FFI_CallFrame.future.offset) == (40, 80, 120, 168)
    assert ctypes.sizeof(A.XLA_FFI_Buffer) == 48 and A.XLA_FFI_Buffer.dims.offset == 40
    assert A.XLA_FFI_Api.XLA_FFI_Error_Create.offset == 48
    assert A.XLA_FFI_Api.XLA_FFI_Stream_Get.offset == 80


def test_decode_synthetic_frame():
    xs = (ctypes.c_float * 6)(1, 2, 3, 4, 5, 6)
    ys = (ctypes.c_float * 6)()
    f = _Frame(args=[(F32, (2, 3), ctypes.addressof(xs))],
               rets=[(F32, (2, 3), ctypes.addressof(ys))],
               attrs=[("alpha", "scalar", 1.5), ("tag", "string", "gelu"),
                      ("taps", "array", [3, 5, 7])])
    call = CallFrame(A, f.address())

    assert call.stage == A.XLA_FFI_ExecutionStage_EXECUTE
    assert len(call.args) == 1 and len(call.rets) == 1
    (a,), (r,) = call.args, call.rets
    assert (a.dtype, a.dims, a.size, a.itemsize, a.nbytes) == (F32, (2, 3), 6, 4, 24)
    assert list(a.as_ctypes()) == [1, 2, 3, 4, 5, 6]

    assert call.attrs["alpha"] == pytest.approx(1.5)
    assert call.attrs["tag"] == "gelu"
    assert call.attrs["taps"] == [3, 5, 7]

    # writing through rets reaches the caller's memory
    out = r.as_ctypes()
    for i in range(r.size):
        out[i] = float(i * 10)
    assert list(ys) == [0, 10, 20, 30, 40, 50]


def test_frame_with_no_args_or_attrs():
    call = CallFrame(A, _Frame(args=[], rets=[], attrs=[]).address())
    assert call.args == [] and call.rets == [] and call.attrs == {}
    assert call.stream() == 0          # api is NULL -> no stream, no crash


def test_non_execute_stage_is_visible():
    f = _Frame(args=[], rets=[], stage=A.XLA_FFI_ExecutionStage_PREPARE)
    assert CallFrame(A, f.address()).stage == A.XLA_FFI_ExecutionStage_PREPARE
