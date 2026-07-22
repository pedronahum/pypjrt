"""A conformance harness for PJRT plugins.

    python -m pypjrt.conform <plugin.so> [--json out.json] [-v]
    python -m pypjrt.conform --diff a.json b.json

Why this exists: a plugin exercised only by jaxlib accretes jaxlib-shaped
assumptions, and today a vendor validates a new plugin by running JAX -- which
conflates plugin bugs with framework bugs. A thin, dependency-free, scriptable
client is the right instrument, and it is the deliverable that serves PJRT's
second stated goal.

The load-bearing rule: **a capability a plugin lacks is UNSUPPORTED, not FAIL.**
Plugins legitimately differ -- this box's CPU plugin advertises 5 extensions and
its CUDA plugin 11, and `PJRT_Device_ClearMemoryStats` simply does not exist
before PJRT 0.106. A harness that cannot tell "absent" from "broken" is useless
to a vendor.
"""

from __future__ import annotations

import argparse
import array
import ctypes
import json
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable

from . import errors
from ._loader import Plugin
from .client import Buffer, Client, Device, Event, Executable

ADD_MLIR = """
module @conform {
  func.func public @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %a, %b : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""
F32 = 11


class Result(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNSUPPORTED = "unsupported"   # plugin does not offer it -- a difference, not a defect
    SKIP = "skip"                 # a prerequisite check failed


@dataclass
class CheckResult:
    id: str
    category: str
    result: Result
    detail: str = ""


@dataclass
class Report:
    plugin: str
    api_version: tuple[int, int] = (0, 0)
    abi_version: tuple[int, int] = (0, 0)
    struct_size: int = 0
    slots: int = 0
    platform: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    extensions: list[dict[str, Any]] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out = {r.value: 0 for r in Result}
        for c in self.checks:
            out[c.result.value] += 1
        return out

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["checks"] = [{**asdict(c), "result": c.result.value} for c in self.checks]
        d["api_version"] = list(self.api_version)
        d["abi_version"] = list(self.abi_version)
        return d


# --------------------------------------------------------------------------
# check registry


_CHECKS: list[tuple[str, str, Callable[["Ctx"], Any], bool]] = []


def check(category: str, id_: str, *, optional: bool = False):
    """Register a check.

    ``optional=True`` marks an entry point that plugins are not required to
    provide, so *any* PJRT error from it is a capability gap rather than a
    defect. Classifying by error code alone is not enough: the CPU plugin
    reports a missing AOT compiler as NotFound, not Unimplemented.
    """
    def deco(fn):
        _CHECKS.append((category, id_, fn, optional))
        return fn
    return deco


class Ctx:
    """Shared state. Built lazily so a failure early degrades the rest to SKIP."""

    def __init__(self, plugin: Plugin, options: dict[str, Any] | None):
        self.plugin = plugin
        self.abi = plugin.abi
        self._options = options
        self._client: Client | None = None
        self._exe: Executable | None = None
        self._dev_ptr: int | None = None
        self._topo_: Any = None
        self._topo_blob_: bytes | None = None
        self._abi_: Any = None
        self._trash: list[Any] = []

    # Accessing a prerequisite that a previous check failed to set IS the skip
    # condition, so the accessor performs the check. That also makes the types
    # non-Optional, which is what a type checker needs to be useful here.

    @property
    def client(self) -> Client:
        if self._client is None:
            raise _Skip("no client")
        return self._client

    @client.setter
    def client(self, v: Client) -> None:
        self._client = v

    @property
    def exe(self) -> Executable:
        if self._exe is None:
            raise _Skip("no compiled executable")
        return self._exe

    @exe.setter
    def exe(self, v: Executable) -> None:
        self._exe = v

    @property
    def dev_ptr(self) -> int:
        if not self._dev_ptr:
            raise _Skip("no device")
        return self._dev_ptr

    @dev_ptr.setter
    def dev_ptr(self, v: int) -> None:
        self._dev_ptr = v

    @property
    def topo(self):
        if self._topo_ is None:
            raise _Skip("no topology")
        return self._topo_

    @topo.setter
    def topo(self, v) -> None:
        self._topo_ = v

    @property
    def topo_blob(self) -> bytes:
        if self._topo_blob_ is None:
            raise _Skip("topology was not serialized")
        return self._topo_blob_

    @topo_blob.setter
    def topo_blob(self, v: bytes) -> None:
        self._topo_blob_ = v

    @property
    def abi_ext(self):
        if self._abi_ is None:
            raise _Skip("no AbiVersion extension")
        return self._abi_

    @abi_ext.setter
    def abi_ext(self, v) -> None:
        self._abi_ = v

    def device(self) -> Device:
        return Device(self.plugin, self.dev_ptr)

    def buffer(self, vals=(1.0, 2.0, 3.0, 4.0)) -> Buffer:
        b = self.client.buffer_from_host(array.array("f", vals), F32, [len(vals)], self.device())
        self._trash.append(b)
        return b

    def close(self):
        for x in reversed(self._trash):
            try:
                x.close()
            except Exception:
                pass
        for x in (self._exe, self._client):
            try:
                if x is not None:
                    x.close()
            except Exception:
                pass


def need(cond: object, why: str = "prerequisite missing"):
    if not cond:
        raise _Skip(why)


class _Skip(Exception):
    pass


# --------------------------------------------------------------------------
# checks -- ABI and plugin


@check("abi", "api_version_readable")
def _(c: Ctx):
    major, minor = c.plugin.api_version
    if major < 0 or minor < 0:
        raise AssertionError(f"nonsense version {major}.{minor}")
    return f"{major}.{minor}"


@check("abi", "struct_size_consistent")
def _(c: Ctx):
    ss, off = c.plugin.api_struct_size, c.abi.VTABLE_OFFSET
    if ss <= off or (ss - off) % ctypes.sizeof(ctypes.c_void_p):
        raise AssertionError(f"PJRT_Api struct_size {ss} inconsistent with header {off}")
    return f"struct_size={ss} -> {c.plugin.n_slots} slots"


@check("abi", "vtable_within_headers")
def _(c: Ctx):
    known = len(c.abi.SLOT)
    if c.plugin.n_slots > known:
        return f"plugin exposes {c.plugin.n_slots} slots, newer than our {known}: re-pin headers"
    return f"{c.plugin.n_slots}/{known} slots"


@check("abi", "unknown_slot_is_clean_error")
def _(c: Ctx):
    """A function past the plugin's vtable must raise, never jump wild."""
    beyond: list[str] = [n for n, sl in c.abi.SLOT.items() if sl >= c.plugin.n_slots]
    if not beyond:
        raise _Skip("plugin exposes the full known vtable")
    try:
        c.plugin.fn(beyond[0])
    except errors.UnsupportedByPlugin:
        return f"{len(beyond)} newer entry point(s) refused cleanly"
    raise AssertionError(f"{beyond[0]} was resolved despite being past the vtable")


@check("plugin", "initialize")
def _(c: Ctx):
    c.plugin.initialize()
    return "ok"


@check("plugin", "attributes")
def _(c: Ctx):
    a = c.plugin.attributes
    return ", ".join(sorted(a)) or "(none)"


@check("plugin", "xla_version_present")
def _(c: Ctx):
    v = c.plugin.xla_version
    need(v is not None, "plugin publishes no xla_version")
    return str(v)


@check("plugin", "stablehlo_version_range")
def _(c: Ctx):
    r = c.plugin.stablehlo_version_range
    if r is None:
        raise _Skip("plugin publishes no stablehlo version range")
    lo, hi = r
    if lo > hi:
        raise AssertionError(f"minimum {lo} exceeds current {hi}")
    return f"{'.'.join(map(str, lo))} .. {'.'.join(map(str, hi))}"


@check("extensions", "chain_walk")
def _(c: Ctx):
    ext = c.plugin.extensions
    names = [e.name or f"type{e.type}?" for e in ext]
    return f"{len(ext)}: " + ", ".join(names) if ext else "(none)"


@check("extensions", "unknown_types_preserved")
def _(c: Ctx):
    unknown = [e for e in c.plugin.extensions if not e.known]
    return (f"{len(unknown)} unknown type(s) kept: "
            + ", ".join(str(e.type) for e in unknown)) if unknown else "all types known"


# -- client -----------------------------------------------------------------


@check("client", "create")
def _(c: Ctx):
    c.client = Client(c.plugin, options=c._options)
    return "ok"


@check("client", "platform_name")
def _(c: Ctx):
    return c.client.platform_name


@check("client", "platform_version")
def _(c: Ctx):
    a = c.plugin.args("PJRT_Client_PlatformVersion_Args", client=c.client.address)
    c.plugin.call("PJRT_Client_PlatformVersion", a)
    return ctypes.string_at(a.platform_version, a.platform_version_size).decode()[:60]


@check("client", "process_index")
def _(c: Ctx):
    return str(c.client.process_index)


@check("client", "addressable_devices")
def _(c: Ctx):
    n = c.client.device_count
    if n < 1:
        raise AssertionError("no addressable devices")
    c.dev_ptr = c.client._addressable()[0]
    return f"{n} device(s)"


@check("client", "all_devices")
def _(c: Ctx):
    a = c.plugin.args("PJRT_Client_Devices_Args", client=c.client.address)
    c.plugin.call("PJRT_Client_Devices", a)
    return f"{a.num_devices} device(s)"


@check("client", "addressable_memories")
def _(c: Ctx):
    a = c.plugin.args("PJRT_Client_AddressableMemories_Args", client=c.client.address)
    c.plugin.call("PJRT_Client_AddressableMemories", a)
    return f"{a.num_addressable_memories} memory space(s)"


@check("client", "topology_description")
def _(c: Ctx):
    a = c.plugin.args("PJRT_Client_TopologyDescription_Args", client=c.client.address)
    c.plugin.call("PJRT_Client_TopologyDescription", a)
    t = c.plugin.args("PJRT_TopologyDescription_PlatformName_Args", topology=a.topology)
    c.plugin.call("PJRT_TopologyDescription_PlatformName", t)
    return ctypes.string_at(t.platform_name, t.platform_name_size).decode()


# -- error boundary ---------------------------------------------------------


@check("errors", "error_code_reported")
def _(c: Ctx):
    """Provoke a real failure and check the code survives the boundary.

    A client that skips PJRT_Error_GetCode has stringly-typed errors.
    """
    try:
        c.client.compile("this is not valid MLIR at all")
    except errors.PjrtError as e:
        if type(e) is errors.Unknown and e.code == 2:
            return "message only (GetCode unavailable or UNKNOWN)"
        return f"{type(e).__name__} (code {e.code})"
    raise AssertionError("invalid MLIR compiled successfully")


# -- device -----------------------------------------------------------------


@check("device", "description")
def _(c: Ctx):
    a = c.plugin.args("PJRT_Device_GetDescription_Args", device=c.dev_ptr)
    c.plugin.call("PJRT_Device_GetDescription", a)
    k = c.plugin.args("PJRT_DeviceDescription_Kind_Args", device_description=a.device_description)
    c.plugin.call("PJRT_DeviceDescription_Kind", k)
    return ctypes.string_at(k.device_kind, k.device_kind_size).decode()


@check("device", "is_addressable")
def _(c: Ctx):
    a = c.plugin.args("PJRT_Device_IsAddressable_Args", device=c.dev_ptr)
    c.plugin.call("PJRT_Device_IsAddressable", a)
    return str(bool(a.is_addressable))


@check("device", "local_hardware_id")
def _(c: Ctx):
    a = c.plugin.args("PJRT_Device_LocalHardwareId_Args", device=c.dev_ptr)
    c.plugin.call("PJRT_Device_LocalHardwareId", a)
    return str(a.local_hardware_id)


@check("device", "default_memory")
def _(c: Ctx):
    a = c.plugin.args("PJRT_Device_DefaultMemory_Args", device=c.dev_ptr)
    c.plugin.call("PJRT_Device_DefaultMemory", a)
    m = c.plugin.args("PJRT_Memory_Kind_Args", memory=a.memory)
    c.plugin.call("PJRT_Memory_Kind", m)
    return ctypes.string_at(m.kind, m.kind_size).decode()


@check("device", "memory_stats")
def _(c: Ctx):
    s = c.device().memory_stats()
    return ", ".join(f"{k}={v}" for k, v in list(s.items())[:3])


@check("device", "clear_memory_stats")
def _(c: Ctx):
    c.device().clear_memory_stats()
    return "ok"


# -- compile ----------------------------------------------------------------


@check("device", "memory_spaces", optional=True)
def _(c: Ctx):
    mems = c.device().memories()
    return ", ".join(f"{m.id}:{m.kind}" for m in mems) if mems else "(none)"


@check("device", "execute_context", optional=True)
def _(c: Ctx):
    from .client import ExecuteContext
    ctx = ExecuteContext.create(c.plugin)
    ctx.close()
    return "ok"


@check("triton", "compile", optional=True)
def _(c: Ctx):
    """Compile Triton IR through the plugin, no triton package or subprocess."""
    from . import triton as tri
    if not tri.available(c.plugin):
        raise _Skip("plugin does not advertise the Triton extension")
    import pathlib
    # src/pypjrt/conform.py -> parents[2] is the repo root
    src = pathlib.Path(__file__).resolve().parents[2] / "tests/data/triton_double.mlir"
    if not src.exists():
        raise _Skip("triton fixture not available")
    k = tri.compile(c.plugin, src.read_text(), arch=tri.arch_of(c.device()))
    return f"{len(k.asm)} bytes of asm, smem={k.smem_bytes}"


@check("compile", "stablehlo_text")
def _(c: Ctx):
    c.exe = c.client.compile(ADD_MLIR)
    return "ok"


@check("compile", "num_outputs")
def _(c: Ctx):
    return str(c.exe.num_outputs)


@check("compile", "output_element_types")
def _(c: Ctx):
    a = c.plugin.args("PJRT_Executable_OutputElementTypes_Args", executable=c.exe._executable())
    c.plugin.call("PJRT_Executable_OutputElementTypes", a)
    p = ctypes.cast(a.output_types, ctypes.POINTER(ctypes.c_int32))
    return ",".join(str(p[i]) for i in range(a.num_output_types))


@check("compile", "output_dimensions")
def _(c: Ctx):
    a = c.plugin.args("PJRT_Executable_OutputDimensions_Args", executable=c.exe._executable())
    c.plugin.call("PJRT_Executable_OutputDimensions", a)
    return f"{a.num_outputs} output(s)"


@check("compile", "fingerprint", optional=True)
def _(c: Ctx):
    """A cache key computed by the plugin -- better than hashing MLIR text."""
    a = c.plugin.args("PJRT_Executable_Fingerprint_Args", executable=c.exe._executable())
    c.plugin.call("PJRT_Executable_Fingerprint", a)
    fp = ctypes.string_at(a.executable_fingerprint, a.executable_fingerprint_size)
    return fp.hex()[:16] if fp else "(empty)"


@check("compile", "compiled_memory_stats")
def _(c: Ctx):
    s = c.exe.compiled_memory_stats()
    return f"peak={s.get('peak_memory_in_bytes', 0)}B"


@check("compile", "cost_analysis")
def _(c: Ctx):
    ca = c.exe.cost_analysis()
    return f"{len(ca)} properties" + (f", flops={ca['flops']:.0f}" if "flops" in ca else "")


@check("compile", "serialize")
def _(c: Ctx):
    """Half of a persistent compile cache. Bound by no source repo."""
    a = c.plugin.args("PJRT_Executable_Serialize_Args", executable=c.exe._executable())
    c.plugin.call("PJRT_Executable_Serialize", a)
    n = int(a.serialized_bytes_size)
    if a.serialized_executable and a.serialized_executable_deleter:
        ctypes.CFUNCTYPE(None, ctypes.c_void_p)(a.serialized_executable_deleter)(
            a.serialized_executable)
    return f"{n} bytes"


@check("compile", "optimized_program", optional=True)
def _(c: Ctx):
    """Post-optimisation HLO -- the best debugging affordance in the API."""
    prog = c.plugin.args("PJRT_Program")
    a = c.plugin.args("PJRT_Executable_OptimizedProgram_Args",
                      executable=c.exe._executable(), program=ctypes.addressof(prog))
    c.plugin.call("PJRT_Executable_OptimizedProgram", a)
    return f"{prog.code_size} bytes available"


@check("compile", "deserialize_and_load")
def _(c: Ctx):
    """The other half of a persistent compile cache."""
    blob = c.exe.serialize()
    e2 = c.client.deserialize_executable(blob)
    n = e2.num_outputs
    e2.close()
    return f"reloaded {len(blob)} bytes, {n} output(s)"


@check("device", "attributes", optional=True)
def _(c: Ctx):
    """Vendor facts, and the only route to a device's mesh position.

    libtpu reports coords / core_on_chip / num_cores / slice_index here and
    nowhere else; the CUDA plugin reports coords, slice_index, core_count.
    """
    at = c.device().attributes
    return ", ".join(sorted(at)[:6]) if at else "(none reported)"


@check("device", "coords")
def _(c: Ctx):
    co = c.device().coords
    need(co is not None, "plugin reports no coords attribute")
    return str(co)


@check("topology", "from_client", optional=True)
def _(c: Ctx):
    from .topology import Topology
    t = Topology.from_client(c.client)
    c.topo = t
    return f"{t.platform_name!r} {t.platform_version!r}"


@check("topology", "device_descriptions", optional=True)
def _(c: Ctx):
    t = c.topo
    d = t.device_descriptions()
    return f"{len(d)} device(s), kinds={sorted({x['kind'] for x in d})}"


@check("topology", "fingerprint", optional=True)
def _(c: Ctx):
    t = c.topo
    return f"{t.fingerprint():#x}"


@check("topology", "serialize", optional=True)
def _(c: Ctx):
    t = c.topo
    c.topo_blob = t.serialize()
    return f"{len(c.topo_blob)} bytes"


@check("topology", "deserialize", optional=True)
def _(c: Ctx):
    """Ship a topology to a machine that lacks the hardware."""
    blob = c.topo_blob
    from .topology import Topology
    t = Topology.deserialize(c.plugin, blob)
    t.close()
    return "ok"


@check("topology", "create_by_name", optional=True)
def _(c: Ctx):
    """Describe hardware by name, with no client.

    On the CUDA plugin this succeeds but *ignores* the name and returns the
    local topology -- so it cannot describe absent hardware. TPU is the case
    this exists for.
    """
    from .topology import Topology
    t = Topology.create(c.plugin, "v5e:2x2")
    got = f"{t.platform_name!r}/{t.platform_version!r}"
    t.close()
    return f"{got} (name honoured only if this is not the local device)"


@check("topology", "client_free_compile", optional=True)
def _(c: Ctx):
    """PJRT_Compile against a topology with no client -- build artifacts in CI."""
    t = c.topo
    return f"{len(t.compile(ADD_MLIR))} bytes"


@check("abi_version", "extension_present")
def _(c: Ctx):
    from .artifact import AbiVersion
    av = AbiVersion.probe(c.plugin)
    need(av is not None, "plugin does not advertise the AbiVersion extension")
    c.abi_ext = av
    return "ok"


@check("abi_version", "executable_proto")
def _(c: Ctx):
    blob = c.abi_ext.executable_proto(c.exe._executable())
    return f"{len(blob)} bytes"


@check("abi_version", "is_compatible_with_executable")
def _(c: Ctx):
    """The artifact guard, answered by the plugin rather than guessed by us."""
    verdict = c.abi_ext.check(c.client.address, c.exe._executable())
    return "compatible" if verdict is None else f"INCOMPATIBLE: {verdict[:110]}"


@check("transfer", "async_host_to_device", optional=True)
def _(c: Ctx):
    """Chunked upload: stream a large array in without a host staging copy."""
    from .transfer import ShapeSpec
    with c.client.async_transfer([ShapeSpec(F32, (4,))], device=c.device()) as t:
        n, size = len(t), t.buffer_size(0)
        t.transfer(0, array.array("f", [1.0, 2.0, 3.0, 4.0]))
        b = t.retrieve(0)
        b.close()
    return f"{n} buffer(s), {size} bytes reserved"


@check("layouts", "buffer_and_default", optional=True)
def _(c: Ctx):
    from . import extensions as ext
    if not ext.layouts_available(c.plugin):
        raise _Skip("plugin does not advertise Layouts")
    b = c.buffer()
    with ext.buffer_layout(b) as bl:
        return bl.serialize().decode(errors="replace")


@check("layouts", "executable_parameters", optional=True)
def _(c: Ctx):
    from . import extensions as ext
    if not ext.layouts_available(c.plugin):
        raise _Skip("plugin does not advertise Layouts")
    ls = ext.executable_layouts(c.exe, "parameters")
    return ", ".join(l.serialize().decode(errors="replace") for l in ls)


@check("stream", "external_ready_events", optional=True)
def _(c: Ctx):
    from . import extensions as ext
    if not ext.stream_available(c.plugin):
        raise _Skip("plugin does not advertise Stream")
    st = ext.device_stream(c.device())
    ext.wait_for_buffer(c.buffer(), st)
    return f"stream 0x{st:x}"


# -- buffers and execution --------------------------------------------------


@check("buffer", "from_host")
def _(c: Ctx):
    b = c.buffer()
    return f"dtype={b.element_type} dims={b.dimensions} nbytes={b.nbytes}"


@check("buffer", "device_and_memory")
def _(c: Ctx):
    b = c.buffer()
    d = c.plugin.args("PJRT_Buffer_Device_Args", buffer=b.address)
    c.plugin.call("PJRT_Buffer_Device", d)
    m = c.plugin.args("PJRT_Buffer_Memory_Args", buffer=b.address)
    c.plugin.call("PJRT_Buffer_Memory", m)
    return "ok"


@check("buffer", "is_on_cpu")
def _(c: Ctx):
    a = c.plugin.args("PJRT_Buffer_IsOnCpu_Args", buffer=c.buffer().address)
    c.plugin.call("PJRT_Buffer_IsOnCpu", a)
    return str(bool(a.is_on_cpu))


@check("buffer", "memory_layout")
def _(c: Ctx):
    a = c.plugin.args("PJRT_Buffer_GetMemoryLayout_Args", buffer=c.buffer().address)
    c.plugin.call("PJRT_Buffer_GetMemoryLayout", a)
    return f"type={int(a.layout.type)}"


@check("buffer", "ready_event")
def _(c: Ctx):
    a = c.plugin.args("PJRT_Buffer_ReadyEvent_Args", buffer=c.buffer().address)
    c.plugin.call("PJRT_Buffer_ReadyEvent", a)
    Event(c.plugin, a.event).consume()
    return "ok"


@check("buffer", "opaque_device_pointer", optional=True)
def _(c: Ctx):
    """Half of zero-copy export (M7)."""
    a = c.plugin.args("PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args",
                      buffer=c.buffer().address)
    c.plugin.call("PJRT_Buffer_OpaqueDeviceMemoryDataPointer", a)
    return "non-null" if a.device_memory_ptr else "null"


@check("buffer", "external_reference_count", optional=True)
def _(c: Ctx):
    """The other half -- pinning device memory for DLPack."""
    b = c.buffer()
    inc = c.plugin.args("PJRT_Buffer_IncreaseExternalReferenceCount_Args", buffer=b.address)
    c.plugin.call("PJRT_Buffer_IncreaseExternalReferenceCount", inc)
    dec = c.plugin.args("PJRT_Buffer_DecreaseExternalReferenceCount_Args", buffer=b.address)
    c.plugin.call("PJRT_Buffer_DecreaseExternalReferenceCount", dec)
    return "ok"


@check("execute", "single_device")
def _(c: Ctx):
    a = c.buffer((1.0, 2.0, 3.0, 4.0))
    b = c.buffer((10.0, 20.0, 30.0, 40.0))
    (out,) = c.exe(a, b)
    c._trash.append(out)
    got = array.array("f")
    got.frombytes(out.to_host())
    if list(got) != [11.0, 22.0, 33.0, 44.0]:
        raise AssertionError(f"wrong result {list(got)}")
    return "[11, 22, 33, 44]"


@check("execute", "device_complete_events")
def _(c: Ctx):
    outs = c.exe.execute_sharded([[c.buffer(), c.buffer()]])
    for row in outs:
        for b in row:
            c._trash.append(b)
    return f"{len(outs)} device row(s)"


# --------------------------------------------------------------------------


def run(plugin: Plugin, options: dict[str, Any] | None = None) -> Report:
    rep = Report(
        plugin=plugin.path.name,
        api_version=plugin.api_version,
        abi_version=(plugin.abi.PJRT_API_MAJOR, plugin.abi.PJRT_API_MINOR),
        struct_size=plugin.api_struct_size,
        slots=plugin.n_slots,
        extensions=[{"type": e.type, "name": e.name} for e in plugin.extensions],
    )
    ctx = Ctx(plugin, options)
    try:
        try:
            rep.attributes = plugin.attributes
        except errors.PjrtError:
            pass
        for category, id_, fn, optional in _CHECKS:
            full = f"{category}.{id_}"
            try:
                detail = fn(ctx)
                rep.checks.append(CheckResult(full, category, Result.PASS, str(detail or "")))
            except _Skip as e:
                rep.checks.append(CheckResult(full, category, Result.SKIP, str(e)))
            except (errors.UnsupportedByPlugin, errors.Unimplemented) as e:
                rep.checks.append(CheckResult(full, category, Result.UNSUPPORTED,
                                              str(e).splitlines()[0][:160]))
            except errors.PjrtError as e:
                rep.checks.append(CheckResult(
                    full, category,
                    Result.UNSUPPORTED if optional else Result.FAIL,
                    f"{type(e).__name__}: {str(e).splitlines()[0][:150]}"))
            except Exception as e:  # noqa: BLE001 -- a check must never kill the harness
                rep.checks.append(CheckResult(full, category, Result.FAIL,
                                              f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"))
        if ctx._client is not None:
            rep.platform = ctx._client.platform_name
    finally:
        ctx.close()
    return rep


def render(rep: Report, verbose: bool = False) -> str:
    out: list[str] = []
    a, b = rep.api_version, rep.abi_version
    out.append(f"pypjrt conformance — {rep.plugin}")
    out.append(f"  plugin API {a[0]}.{a[1]} · headers {b[0]}.{b[1]} · "
               f"struct_size {rep.struct_size} · {rep.slots} slots · "
               f"{len(rep.extensions)} extensions"
               + (f" · platform {rep.platform!r}" if rep.platform else ""))
    if rep.extensions:
        out.append("  extensions: " + ", ".join(e["name"] or f"type{e['type']}?"
                                                for e in rep.extensions))
    out.append("")
    by_cat: dict[str, list[CheckResult]] = {}
    for c in rep.checks:
        by_cat.setdefault(c.category, []).append(c)
    for cat, cs in by_cat.items():
        n = {r: sum(1 for c in cs if c.result is r) for r in Result}
        bits = [f"{n[Result.PASS]} pass"]
        for r, label in ((Result.UNSUPPORTED, "unsupported"), (Result.SKIP, "skip"),
                         (Result.FAIL, "FAIL")):
            if n[r]:
                bits.append(f"{n[r]} {label}")
        out.append(f"  {cat:<11} {', '.join(bits)}")
        if verbose:
            for c in cs:
                out.append(f"      {c.result.value:<12} {c.id:<38} {c.detail}")
    counts = rep.counts()
    for r, header in ((Result.FAIL, "FAILURES"), (Result.UNSUPPORTED, "UNSUPPORTED"),
                      (Result.SKIP, "SKIPPED")):
        rows = [c for c in rep.checks if c.result is r]
        if rows and (r is not Result.UNSUPPORTED or not verbose):
            out.append("")
            out.append(f"  {header} ({len(rows)}):")
            for c in rows:
                out.append(f"    {c.id:<38} {c.detail}")
    out.append("")
    out.append(f"  {counts['pass']} pass · {counts['unsupported']} unsupported · "
               f"{counts['skip']} skip · {counts['fail']} FAIL")
    return "\n".join(out)


def diff(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Compare two reports. Capability differences are expected and informative."""
    ra = {c["id"]: c["result"] for c in a["checks"]}
    rb = {c["id"]: c["result"] for c in b["checks"]}
    na, nb = a["plugin"], b["plugin"]
    out = [f"{na}  vs  {nb}", ""]
    ea = {e["name"] or f"type{e['type']}" for e in a["extensions"]}
    eb = {e["name"] or f"type{e['type']}" for e in b["extensions"]}
    if ea - eb:
        out.append(f"  extensions only in {na}: {', '.join(sorted(ea - eb))}")
    if eb - ea:
        out.append(f"  extensions only in {nb}: {', '.join(sorted(eb - ea))}")
    diffs = [(k, ra.get(k, "-"), rb.get(k, "-")) for k in sorted(set(ra) | set(rb))
             if ra.get(k) != rb.get(k)]
    if diffs:
        out.append("")
        out.append(f"  {len(diffs)} check(s) differ:")
        for k, x, y in diffs:
            out.append(f"    {k:<38} {x:<12} {y}")
    else:
        out.append("  identical results")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pypjrt.conform",
                                 description="Conformance-probe a PJRT plugin.")
    ap.add_argument("plugin", nargs="?", help="path to a PJRT plugin .so")
    ap.add_argument("--json", metavar="FILE", help="write the report as JSON")
    ap.add_argument("-v", "--verbose", action="store_true", help="show every check")
    ap.add_argument("--diff", nargs=2, metavar=("A.json", "B.json"),
                    help="compare two saved reports")
    ap.add_argument("--memory-fraction", type=float, default=None,
                    help="accelerator allocator cap (default 0.5)")
    args = ap.parse_args(argv)

    if args.diff:
        with open(args.diff[0]) as f1, open(args.diff[1]) as f2:
            print(diff(json.load(f1), json.load(f2)))
        return 0

    if not args.plugin:
        ap.error("a plugin path is required (or use --diff)")

    plugin = Plugin(args.plugin)
    options = None
    if plugin.is_accelerator:
        options = dict(Client.GPU_DEFAULTS)
        if args.memory_fraction is not None:
            options["memory_fraction"] = args.memory_fraction

    rep = run(plugin, options)
    print(render(rep, args.verbose))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep.to_json(), f, indent=2)
        print(f"\n  wrote {args.json}")
    # Only real failures are non-zero. UNSUPPORTED is a capability difference.
    return 1 if rep.counts()["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
