"""Register a *Python function* as an XLA FFI custom-call handler and have XLA
call it from inside a compiled StableHLO program. Pure ctypes, no jaxlib."""
import ctypes, sys, array, re

PLUGIN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PYPJRT_PLUGIN", "")
PLATFORM = (sys.argv[2] if len(sys.argv) > 2 else "cpu").encode()
VOIDP, SIZET, I64, I32 = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64, ctypes.c_int
HDR = "vendor/xla/pjrt/c/pjrt_c_api.h"
SLOT = {m: i for i, m in enumerate(re.findall(
    r"_PJRT_API_STRUCT_FIELD\((\w+)\);",
    re.search(r"typedef struct PJRT_Api \{(.*?)\n\} PJRT_Api;", open(HDR).read(), re.S).group(1)))}

lib = ctypes.CDLL(PLUGIN, mode=ctypes.RTLD_LOCAL)   # keep for process lifetime — never dlclose
lib.GetPjrtApi.restype = VOIDP
api = lib.GetPjrtApi()
vtable = ctypes.cast(api + 40, ctypes.POINTER(VOIDP))
_FN = ctypes.CFUNCTYPE(VOIDP, VOIDP)

def struct_size(cls, last): return getattr(cls, last).offset + getattr(cls, last).size
def A(cls, last, **kw):
    o = cls(); ctypes.memset(ctypes.byref(o), 0, ctypes.sizeof(o))
    o.struct_size = struct_size(cls, last)
    for k, v in kw.items(): setattr(o, k, v)
    return o
class ErrMsg(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("error",VOIDP),("message",VOIDP),("message_size",SIZET)]
class ErrDes(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("error",VOIDP)]
def call(name, args):
    err = _FN(vtable[SLOT[name]])(ctypes.byref(args))
    if err:
        m = A(ErrMsg,"message_size",error=err); _FN(vtable[SLOT["PJRT_Error_Message"]])(ctypes.byref(m))
        msg = ctypes.string_at(m.message, m.message_size).decode(errors="replace")
        d = A(ErrDes,"error",error=err); _FN(vtable[SLOT["PJRT_Error_Destroy"]])(ctypes.byref(d))
        raise RuntimeError(msg)

# ---- walk the PJRT extension chain for the FFI extension (type 5) ------------
class ExtBase(ctypes.Structure): pass
ExtBase._fields_=[("struct_size",SIZET),("type",I32),("_pad",I32),("next",ctypes.POINTER(ExtBase))]
head_ext = ctypes.cast(api + 8, ctypes.POINTER(ctypes.POINTER(ExtBase))).contents
ext, ffi_ext = (head_ext if head_ext else None), None
while ext:
    if ext.contents.type == 5: ffi_ext = ctypes.addressof(ext.contents); break
    ext = ext.contents.next if ext.contents.next else None
if not ffi_ext: sys.exit("no PJRT FFI extension on this plugin")
print(f"PJRT_FFI extension @ 0x{ffi_ext:x}")
# PJRT_FFI_Extension { base[24]; type_register@24; user_data_add@32; register_handler@40 }
register_handler = ctypes.cast(ffi_ext + 40, ctypes.POINTER(VOIDP))[0]

# ---- XLA FFI call-frame layout (from xla/ffi/api/c_api.h, FFI 0.3) ----------
#   CallFrame: struct_size@0 ext@8 api@16 ctx@24 stage:i32@32 args@40 rets@80 attrs@120 future@168
#   Args/Rets: struct_size@0 ext@8 size:i64@16 types@24 values@32           [40]
#   Buffer:    struct_size@0 ext@8 dtype:i32@16 data@24 rank:i64@32 dims@40 [48]
class FfiExtBase(ctypes.Structure): pass
FfiExtBase._fields_=[("struct_size",SIZET),("type",I32),("_pad",I32),("next",ctypes.POINTER(FfiExtBase))]
class FfiArgs(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("size",I64),("types",VOIDP),("values",VOIDP)]
class FfiCallFrame(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",ctypes.POINTER(FfiExtBase)),("api",VOIDP),("ctx",VOIDP),
              ("stage",I32),("_pad",I32),
              ("args",FfiArgs),("rets",FfiArgs),
              ("attrs_ss",SIZET),("attrs_ext",VOIDP),("attrs_size",I64),
              ("attrs_types",VOIDP),("attrs_names",VOIDP),("attrs_vals",VOIDP),
              ("future",VOIDP)]
class FfiBuffer(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("dtype",I32),("_pad",I32),
              ("data",VOIDP),("rank",I64),("dims",ctypes.POINTER(I64))]
class FfiApiVersion(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("major",I32),("minor",I32)]
class FfiMetadata(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("api_version",FfiApiVersion),("traits",ctypes.c_uint32),
              ("_pad",I32),("state_type_id",I64)]

assert ctypes.sizeof(FfiCallFrame) == 176, ctypes.sizeof(FfiCallFrame)
assert FfiCallFrame.args.offset == 40 and FfiCallFrame.rets.offset == 80
assert FfiCallFrame.attrs_ss.offset == 120 and FfiCallFrame.future.offset == 168
assert ctypes.sizeof(FfiBuffer) == 48 and FfiBuffer.dims.offset == 40
print("call-frame layout asserts OK "
      f"(frame={ctypes.sizeof(FfiCallFrame)}, buffer={ctypes.sizeof(FfiBuffer)})")

EXECUTE, METADATA_EXT = 3, 1
seen = {"metadata_probe": 0, "execute": 0, "stages": []}

def _buffers(a):
    vals = ctypes.cast(a.values, ctypes.POINTER(VOIDP))
    return [ctypes.cast(vals[i], ctypes.POINTER(FfiBuffer)).contents for i in range(a.size)]

HANDLER_T = ctypes.CFUNCTYPE(VOIDP, ctypes.POINTER(FfiCallFrame))

def _handler(frame_ptr):
    f = frame_ptr.contents
    # (1) service the metadata probe FIRST — miss this and XLA silently drops us
    e = f.ext
    while e:
        if e.contents.type == METADATA_EXT:
            md = ctypes.cast(ctypes.cast(ctypes.addressof(e.contents) + 24,
                             ctypes.POINTER(VOIDP))[0], ctypes.POINTER(FfiMetadata)).contents
            md.api_version.major, md.api_version.minor = 0, 3
            md.traits = 0
            seen["metadata_probe"] += 1
            return None
        e = e.contents.next if e.contents.next else None
    seen["stages"].append(f.stage)
    if f.stage != EXECUTE:
        return None
    # (2) real work, in Python, on XLA-owned buffers
    (inp,), (out,) = _buffers(f.args), _buffers(f.rets)
    n = 1
    for i in range(inp.rank): n *= inp.dims[i]
    src = (ctypes.c_float * n).from_address(inp.data)
    dst = (ctypes.c_float * n).from_address(out.data)
    for i in range(n):
        dst[i] = src[i] * 2.0 + 1.0
    seen["execute"] += 1
    return None

HANDLER = HANDLER_T(_handler)          # must outlive the process
NAME = b"pypjrt_double"
_keep = [NAME, PLATFORM, HANDLER]

class RegArgs(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("target_name",VOIDP),("target_name_size",SIZET),
              ("handler",VOIDP),("platform_name",VOIDP),("platform_name_size",SIZET),
              ("traits",I32)]
ra = A(RegArgs,"traits",
       target_name=ctypes.cast(ctypes.c_char_p(NAME), VOIDP), target_name_size=len(NAME),
       handler=ctypes.cast(HANDLER, VOIDP),
       platform_name=ctypes.cast(ctypes.c_char_p(PLATFORM), VOIDP),
       platform_name_size=len(PLATFORM))
err = _FN(register_handler)(ctypes.byref(ra))
if err:
    m = A(ErrMsg,"message_size",error=err); _FN(vtable[SLOT["PJRT_Error_Message"]])(ctypes.byref(m))
    raise RuntimeError(ctypes.string_at(m.message, m.message_size).decode())
print(f"registered handler '{NAME.decode()}' for platform '{PLATFORM.decode()}'"
      f"  (metadata probes so far: {seen['metadata_probe']})")

# ---- compile + execute a program that calls it ------------------------------
class CC(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("create_options",VOIDP),("num_options",SIZET),
              ("kv_get",VOIDP),("kv_get_arg",VOIDP),("kv_put",VOIDP),("kv_put_arg",VOIDP),
              ("client",VOIDP),("kv_try_get",VOIDP),("kv_try_get_arg",VOIDP)]
class AD(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("client",VOIDP),("devs",VOIDP),("n",SIZET)]
class Prog(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("code",VOIDP),("code_size",SIZET),
              ("format",VOIDP),("format_size",SIZET)]
class Comp(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("client",VOIDP),("program",VOIDP),
              ("opts",VOIDP),("opts_size",SIZET),("executable",VOIDP)]
class FH(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("client",VOIDP),("data",VOIDP),("type",I32),
              ("_p",I32),("dims",VOIDP),("num_dims",SIZET),("strides",VOIDP),("num_strides",SIZET),
              ("sem",I32),("_p2",I32),("device",VOIDP),("memory",VOIDP),("layout",VOIDP),
              ("done",VOIDP),("buffer",VOIDP)]
class EO(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("send",VOIDP),("recv",VOIDP),("nsend",SIZET),
              ("nrecv",SIZET),("launch_id",I32),("_p",I32),("nondon",VOIDP),("nnondon",SIZET),
              ("context",VOIDP),("call_location",VOIDP),("num_tasks",SIZET),("task_ids",VOIDP),
              ("inc_ids",VOIDP),("multi_slice",VOIDP),("major_minor",ctypes.c_bool)]
class EX(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("executable",VOIDP),("options",VOIDP),
              ("arglists",VOIDP),("ndev",SIZET),("nargs",SIZET),("outlists",VOIDP),
              ("events",VOIDP),("exec_device",VOIDP)]
class TH(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("src",VOIDP),("layout",VOIDP),("dst",VOIDP),
              ("dst_size",SIZET),("event",VOIDP)]
class EV(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("event",VOIDP)]

MLIR = b"""
module @jit_ffi {
  func.func public @main(%a: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.custom_call @pypjrt_double(%a) {api_version = 4 : i32} : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""
OPTS = bytes([0x1a, 0x04, 0x20, 0x01, 0x28, 0x01])

cc = A(CC,"kv_try_get_arg"); call("PJRT_Client_Create", cc)
ad = A(AD,"n",client=cc.client); call("PJRT_Client_AddressableDevices", ad)
dev = ctypes.cast(ad.devs, ctypes.POINTER(VOIDP))[0]
fmt = b"mlir"
pr = A(Prog,"format_size",code=ctypes.cast(ctypes.c_char_p(MLIR),VOIDP),code_size=len(MLIR),
       format=ctypes.cast(ctypes.c_char_p(fmt),VOIDP),format_size=len(fmt))
co = A(Comp,"executable",client=cc.client,program=ctypes.addressof(pr),
       opts=ctypes.cast(ctypes.c_char_p(OPTS),VOIDP),opts_size=len(OPTS))
call("PJRT_Client_Compile", co)
print(f"compiled program containing @{NAME.decode()}")

data = array.array("f",[1.0,2.0,3.0,4.0]); dims=(I64*1)(4)
fh = A(FH,"buffer",client=cc.client,data=ctypes.cast(data.buffer_info()[0],VOIDP),type=11,
       dims=ctypes.cast(dims,VOIDP),num_dims=1,sem=0,device=dev)
call("PJRT_Client_BufferFromHostBuffer", fh)
if fh.done: call("PJRT_Event_Await", A(EV,"event",event=fh.done))

opts = A(EO,"major_minor")
argrow=(VOIDP*1)(fh.buffer); arglists=(VOIDP*1)(ctypes.cast(argrow,VOIDP))
outrow=(VOIDP*1)();          outlists=(VOIDP*1)(ctypes.cast(outrow,VOIDP))
ex = A(EX,"exec_device",executable=co.executable,options=ctypes.addressof(opts),
       arglists=ctypes.cast(arglists,VOIDP),ndev=1,nargs=1,
       outlists=ctypes.cast(outlists,VOIDP))
call("PJRT_LoadedExecutable_Execute", ex)

out = array.array("f",[0.0]*4)
th = A(TH,"event",src=outrow[0],dst=ctypes.cast(out.buffer_info()[0],VOIDP),dst_size=16)
call("PJRT_Buffer_ToHostBuffer", th)
if th.event: call("PJRT_Event_Await", A(EV,"event",event=th.event))

print(f"metadata probes={seen['metadata_probe']}  execute calls={seen['execute']}  stages={seen['stages']}")
print(f"RESULT: {list(out)}")
assert list(out) == [3.0,5.0,7.0,9.0], "MISMATCH"
assert seen["execute"] >= 1, "handler never ran"
print("PASS — a Python function ran as an XLA FFI handler inside a compiled program")
