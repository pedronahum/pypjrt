"""End-to-end PJRT from pure-ctypes Python: compile StableHLO, execute, read back.
No jax, no jaxlib, no numpy, no compiled extension. Python 3.12 stdlib only."""
import ctypes, struct, sys, array

PLUGIN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PYPJRT_PLUGIN", "")
VOIDP, SIZET, I64, I32 = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64, ctypes.c_int

# ---- v-table slot indices, from the header's PJRT_Api field order -------------
SLOT = {n: i for i, n in enumerate("""
PJRT_Error_Destroy PJRT_Error_Message PJRT_Error_GetCode PJRT_Plugin_Initialize
PJRT_Plugin_Attributes PJRT_Event_Destroy PJRT_Event_IsReady PJRT_Event_Error
PJRT_Event_Await PJRT_Event_OnReady PJRT_Client_Create PJRT_Client_Destroy
PJRT_Client_PlatformName PJRT_Client_ProcessIndex PJRT_Client_PlatformVersion
PJRT_Client_Devices PJRT_Client_AddressableDevices PJRT_Client_LookupDevice
PJRT_Client_LookupAddressableDevice PJRT_Client_AddressableMemories
PJRT_Client_Compile PJRT_Client_DefaultDeviceAssignment
PJRT_Client_BufferFromHostBuffer""".split())}
# ToHostBuffer / Execute / Buffer_Destroy are further down; find them by re-parsing.
import re
_h = open("vendor/xla/pjrt/c/pjrt_c_api.h").read()
_body = re.search(r"typedef struct PJRT_Api \{(.*?)\n\} PJRT_Api;", _h, re.S).group(1)
SLOT = {m: i for i, m in enumerate(re.findall(r"_PJRT_API_STRUCT_FIELD\((\w+)\);", _body))}

lib = ctypes.CDLL(PLUGIN, mode=ctypes.RTLD_LOCAL)
lib.GetPjrtApi.restype = VOIDP
api = lib.GetPjrtApi()
vtable = ctypes.cast(api + 3 * 8 + 2 * 8, ctypes.POINTER(VOIDP))  # skip struct_size, ext, version{4 words}
# header = size_t + ptr + PJRT_Api_Version{size_t, ptr, int, int} = 8+8+(8+8+4+4)=40 bytes
vtable = ctypes.cast(api + 40, ctypes.POINTER(VOIDP))

_FN = ctypes.CFUNCTYPE(VOIDP, VOIDP)  # PJRT_Error* fn(Args*)
def call(name, args):
    fn = _FN(vtable[SLOT[name]])
    err = fn(ctypes.byref(args))
    if err:
        check_err(err)
    return None

def struct_size(cls, last):
    return getattr(cls, last).offset + getattr(cls, last).size

def A(cls, last, **kw):
    o = cls(); ctypes.memset(ctypes.byref(o), 0, ctypes.sizeof(o))
    o.struct_size = struct_size(cls, last)
    for k, v in kw.items(): setattr(o, k, v)
    return o

# ---- error boundary ----------------------------------------------------------
class ErrMsgArgs(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("error",VOIDP),("message",VOIDP),("message_size",SIZET)]
class ErrCodeArgs(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("error",VOIDP),("code",I32)]
class ErrDestroyArgs(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("error",VOIDP)]

def check_err(err):
    m = A(ErrMsgArgs,"message_size",error=err); _FN(vtable[SLOT["PJRT_Error_Message"]])(ctypes.byref(m))
    c = A(ErrCodeArgs,"code",error=err);        _FN(vtable[SLOT["PJRT_Error_GetCode"]])(ctypes.byref(c))
    msg = ctypes.string_at(m.message, m.message_size).decode(errors="replace")
    d = A(ErrDestroyArgs,"error",error=err);    _FN(vtable[SLOT["PJRT_Error_Destroy"]])(ctypes.byref(d))
    raise RuntimeError(f"PJRT error (code {c.code}): {msg}")

# ---- structs -----------------------------------------------------------------
class ClientCreateArgs(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("create_options",VOIDP),("num_options",SIZET),
              ("kv_get_callback",VOIDP),("kv_get_user_arg",VOIDP),("kv_put_callback",VOIDP),
              ("kv_put_user_arg",VOIDP),("client",VOIDP),("kv_try_get_callback",VOIDP),
              ("kv_try_get_user_arg",VOIDP)]
class AddrDevArgs(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("client",VOIDP),
              ("addressable_devices",VOIDP),("num_addressable_devices",SIZET)]
class Program(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("code",VOIDP),("code_size",SIZET),
              ("format",VOIDP),("format_size",SIZET)]
class CompileArgs(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("client",VOIDP),("program",VOIDP),
              ("compile_options",VOIDP),("compile_options_size",SIZET),("executable",VOIDP)]
class FromHostArgs(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("client",VOIDP),("data",VOIDP),("type",I32),
              ("dims",VOIDP),("num_dims",SIZET),("byte_strides",VOIDP),("num_byte_strides",SIZET),
              ("host_buffer_semantics",I32),("device",VOIDP),("memory",VOIDP),("device_layout",VOIDP),
              ("done_with_host_buffer",VOIDP),("buffer",VOIDP)]
class ExecOptions(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("send_callbacks",VOIDP),("recv_callbacks",VOIDP),
              ("num_send_ops",SIZET),("num_recv_ops",SIZET),("launch_id",I32),
              ("non_donatable_input_indices",VOIDP),("num_non_donatable_input_indices",SIZET),
              ("context",VOIDP),("call_location",VOIDP),("num_tasks",SIZET),("task_ids",VOIDP),
              ("incarnation_ids",VOIDP),("multi_slice_config",VOIDP),
              ("use_major_to_minor_data_layout_for_callbacks",ctypes.c_bool)]
class ExecuteArgs(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("executable",VOIDP),("options",VOIDP),
              ("argument_lists",VOIDP),("num_devices",SIZET),("num_args",SIZET),
              ("output_lists",VOIDP),("device_complete_events",VOIDP),("execute_device",VOIDP)]
class ToHostArgs(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("src",VOIDP),("host_layout",VOIDP),
              ("dst",VOIDP),("dst_size",SIZET),("event",VOIDP)]
class EventArgs(ctypes.Structure):
    _fields_=[("struct_size",SIZET),("ext",VOIDP),("event",VOIDP)]

F32 = 11
IMMUTABLE_ONLY_DURING_CALL = 0

# ---- the program -------------------------------------------------------------
MLIR = b"""
module @jit_add {
  func.func public @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %a, %b : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""
# CompileOptionsProto: build_options{num_replicas=1, num_partitions=1}
COMPILE_OPTS = bytes([0x1a, 0x04, 0x20, 0x01, 0x28, 0x01])

# ---- run ---------------------------------------------------------------------
cc = A(ClientCreateArgs, "kv_try_get_user_arg"); call("PJRT_Client_Create", cc)
print(f"client created: 0x{cc.client:x}")

ad = A(AddrDevArgs, "num_addressable_devices", client=cc.client)
call("PJRT_Client_AddressableDevices", ad)
devs = ctypes.cast(ad.addressable_devices, ctypes.POINTER(VOIDP))
print(f"addressable devices: {ad.num_addressable_devices}")
dev = devs[0]

fmt = b"mlir"
prog = A(Program, "format_size", code=ctypes.cast(ctypes.c_char_p(MLIR), VOIDP),
         code_size=len(MLIR), format=ctypes.cast(ctypes.c_char_p(fmt), VOIDP), format_size=len(fmt))
ca = A(CompileArgs, "executable", client=cc.client, program=ctypes.addressof(prog),
       compile_options=ctypes.cast(ctypes.c_char_p(COMPILE_OPTS), VOIDP),
       compile_options_size=len(COMPILE_OPTS))
call("PJRT_Client_Compile", ca)
print(f"compiled: 0x{ca.executable:x}")

def upload(vals):
    data = array.array("f", vals)
    dims = (I64 * 1)(len(vals))
    fh = A(FromHostArgs, "buffer", client=cc.client,
           data=ctypes.cast(data.buffer_info()[0], VOIDP), type=F32,
           dims=ctypes.cast(dims, VOIDP), num_dims=1,
           host_buffer_semantics=IMMUTABLE_ONLY_DURING_CALL, device=dev)
    call("PJRT_Client_BufferFromHostBuffer", fh)
    if fh.done_with_host_buffer:
        call("PJRT_Event_Await", A(EventArgs, "event", event=fh.done_with_host_buffer))
    return fh.buffer, data, dims

b0, d0, dm0 = upload([1.0, 2.0, 3.0, 4.0])
b1, d1, dm1 = upload([10.0, 20.0, 30.0, 40.0])
print("uploaded 2 buffers")

opts = A(ExecOptions, "use_major_to_minor_data_layout_for_callbacks")
argrow = (VOIDP * 2)(b0, b1)
arglists = (VOIDP * 1)(ctypes.cast(argrow, VOIDP))
outrow = (VOIDP * 1)()
outlists = (VOIDP * 1)(ctypes.cast(outrow, VOIDP))
ea = A(ExecuteArgs, "execute_device", executable=ca.executable, options=ctypes.addressof(opts),
       argument_lists=ctypes.cast(arglists, VOIDP), num_devices=1, num_args=2,
       output_lists=ctypes.cast(outlists, VOIDP))
call("PJRT_LoadedExecutable_Execute", ea)
print(f"executed -> out buffer 0x{outrow[0]:x}")

out = array.array("f", [0.0] * 4)
th = A(ToHostArgs, "event", src=outrow[0], dst=ctypes.cast(out.buffer_info()[0], VOIDP),
       dst_size=out.itemsize * len(out))
call("PJRT_Buffer_ToHostBuffer", th)
if th.event:
    call("PJRT_Event_Await", A(EventArgs, "event", event=th.event))
print(f"RESULT: {list(out)}")
assert list(out) == [11.0, 22.0, 33.0, 44.0], "MISMATCH"
print("PASS — [1,2,3,4] + [10,20,30,40] == [11,22,33,44] on PJRT, pure ctypes")
