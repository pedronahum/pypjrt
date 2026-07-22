"""Plugin loading, ABI negotiation, and the call boundary.

Everything below is the productionised form of ``spike/pjrt_e2e.py``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from . import _abi, errors
from ._lock import dispatching, pjrt_call

_FN = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)

# RTLD_LOCAL so CPU/CUDA/TPU plugins can coexist in one process.
_DLOPEN_FLAGS = os.RTLD_NOW | os.RTLD_LOCAL
_RTLD_DEEPBIND = getattr(os, "RTLD_DEEPBIND", 0)


# --------------------------------------------------------------------------
# discovery


def find_plugin(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Locate a PJRT plugin .so.

    Precedence: explicit -> $PYPJRT_PLUGIN -> installed jax plugins -> a short
    search path. We do *not* hardcode an absolute path, and we do *not* shell
    out to ``python3 -c`` to glob for one. We are already in Python.
    """
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"no PJRT plugin at {p}")
        return p

    for var in ("PYPJRT_PLUGIN", "TPU_LIBRARY_PATH"):
        if env := os.environ.get(var):
            p = Path(env).expanduser()
            if p.exists():
                return p
            raise FileNotFoundError(f"${var} points at {p}, which does not exist")

    for p in _jax_plugin_candidates():
        if p.exists():
            return p

    for p in _libtpu_candidates():
        if p.exists():
            return p

    for d in (Path.home() / "lib", Path("/usr/local/lib"), Path("/usr/lib")):
        for name in ("libpjrt_c_api_cpu_plugin.so", "pjrt_c_api_cpu_plugin.so"):
            if (p := d / name).exists():
                return p

    raise FileNotFoundError(
        "no PJRT plugin found. Set $PYPJRT_PLUGIN, pass one explicitly, or "
        "install a jax plugin wheel (e.g. jax-cuda12-plugin)."
    )


def _libtpu_candidates() -> Iterator[Path]:
    """libtpu is not packaged like the GPU plugins.

    It ships as ``libtpu/libtpu.so`` in its own wheel and is conventionally
    located through ``$TPU_LIBRARY_PATH``; on a TPU VM it may also sit in
    ``/lib``. Globbing ``jax_plugins/*/xla_*_plugin.so`` never finds it.
    """
    import importlib.util
    try:
        spec = importlib.util.find_spec("libtpu")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        for root in spec.submodule_search_locations:
            yield from sorted(Path(root).glob("libtpu*.so"))
    for d in (Path("/lib"), Path("/usr/lib"), Path.home() / "lib"):
        yield d / "libtpu.so"


def _jax_plugin_candidates() -> Iterator[Path]:
    """Installed jax plugin .so files, without importing jax."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("jax_plugins")
    except (ImportError, ValueError):
        return
    if spec is None or not spec.submodule_search_locations:
        return
    for root in spec.submodule_search_locations:
        yield from sorted(Path(root).glob("*/xla_*_plugin.so"))
        yield from sorted(Path(root).glob("*/*.so"))


# --------------------------------------------------------------------------
# PJRT_NamedValue decoding -- used by Plugin_Attributes and GetCostAnalysis


def read_named_values(abi, ptr: int, count: int) -> dict[str, Any]:
    """Decode a ``PJRT_NamedValue[]`` into a dict."""
    if not ptr or count <= 0:
        return {}
    arr = ctypes.cast(ptr, ctypes.POINTER(abi.PJRT_NamedValue))
    out: dict[str, Any] = {}
    for i in range(count):
        nv = arr[i]
        name = ctypes.string_at(nv.name, nv.name_size).decode(errors="replace")
        t = int(nv.type)
        if t == abi.PJRT_NamedValue_kString:
            out[name] = ctypes.string_at(nv.string_value, nv.value_size).decode(errors="replace")
        elif t == abi.PJRT_NamedValue_kInt64:
            out[name] = int(nv.int64_value)
        elif t == abi.PJRT_NamedValue_kInt64List:
            p64 = ctypes.cast(nv.int64_array_value, ctypes.POINTER(ctypes.c_int64))
            out[name] = tuple(int(p64[j]) for j in range(nv.value_size))
        elif t == abi.PJRT_NamedValue_kFloat:
            out[name] = float(nv.float_value)
        elif t == abi.PJRT_NamedValue_kBool:
            out[name] = bool(nv.bool_value)
    return out


# --------------------------------------------------------------------------
# extensions


@dataclass(frozen=True)
class Extension:
    type: int
    version: int
    address: int
    name: str  # "" when this build does not know the type

    @property
    def known(self) -> bool:
        return bool(self.name)


# --------------------------------------------------------------------------


class Plugin:
    """A loaded PJRT plugin: the shared object plus its negotiated ABI.

    The ``CDLL`` is held for the process lifetime and never ``dlclose``d.
    Closing it tears down the plugin's *static* XLA FFI registry and silently
    unregisters every handler already installed. The symptom is a custom call
    that resolved a moment ago suddenly reporting an unknown target.
    """

    _loaded: dict[str, "Plugin"] = {}

    def __init__(self, path: str | os.PathLike[str] | None = None, *, deepbind: bool = False):
        self.path = find_plugin(path)
        key = str(self.path.resolve())
        if (prev := Plugin._loaded.get(key)) is not None:
            self.__dict__ = prev.__dict__  # one dlopen per .so per process
            return

        flags = _DLOPEN_FLAGS | (_RTLD_DEEPBIND if deepbind else 0)
        self._lib = ctypes.CDLL(key, mode=flags)
        get_api = self._lib.GetPjrtApi
        get_api.restype = ctypes.c_void_p
        get_api.argtypes = []
        api = get_api()
        if not api:
            raise errors.IncompatiblePlugin(f"{self.path}: GetPjrtApi() returned NULL")
        self.api: int = api

        # Read struct_size and the version BEFORE trusting any other offset.
        ext_off, major_off, minor_off = _abi.bootstrap_offsets()
        self._ext_off = ext_off
        self.api_struct_size: int = ctypes.cast(
            api, ctypes.POINTER(ctypes.c_size_t))[0]
        self.api_version: tuple[int, int] = (
            int(ctypes.cast(api + major_off, ctypes.POINTER(ctypes.c_int32))[0]),
            int(ctypes.cast(api + minor_off, ctypes.POINTER(ctypes.c_int32))[0]),
        )

        self.abi, self.abi_exact = _abi.load(*self.api_version)

        n_slots = (self.api_struct_size - self.abi.VTABLE_OFFSET) // ctypes.sizeof(ctypes.c_void_p)
        if n_slots < 1:
            raise errors.IncompatiblePlugin(
                f"{self.path}: PJRT_Api struct_size {self.api_struct_size} is smaller "
                f"than the {self.abi.VTABLE_OFFSET}-byte header"
            )
        self.n_slots: int = n_slots
        self._vtable = ctypes.cast(api + self.abi.VTABLE_OFFSET, ctypes.POINTER(ctypes.c_void_p))
        self._fn_cache: dict[str, Any] = {}

        self.extensions: tuple[Extension, ...] = self._walk_extensions()
        Plugin._loaded[key] = self

    # -- extension chain ---------------------------------------------------

    def _walk_extensions(self) -> tuple[Extension, ...]:
        """Walk ``PJRT_Api.extension_start``, preserving unknown types.

        Unknown ``(type, version)`` pairs are kept rather than dropped, so a
        newer plugin degrades to "capability not understood" instead of a crash.
        """
        names = {
            v: k[len("PJRT_Extension_Type_"):]
            for k, v in vars(self.abi).items()
            if k.startswith("PJRT_Extension_Type_") and isinstance(v, int)
        }
        out: list[Extension] = []
        node = ctypes.cast(self.api + self._ext_off, ctypes.POINTER(ctypes.c_void_p))[0]
        seen: set[int] = set()
        while node and node not in seen:
            seen.add(node)
            base = ctypes.cast(node, ctypes.POINTER(self.abi.PJRT_Extension_Base)).contents
            out.append(Extension(type=int(base.type), version=int(base.struct_size),
                                 address=int(node), name=names.get(int(base.type), "")))
            node = base.next
        return tuple(out)

    def extension(self, name_or_type: str | int) -> Extension | None:
        """Capability probe. Returns ``None`` when absent -- never raises.

        The core must never *assume* an extension; plugins legitimately differ.
        """
        if isinstance(name_or_type, int):
            return next((e for e in self.extensions if e.type == name_or_type), None)
        want = name_or_type.lower()
        return next((e for e in self.extensions if e.name.lower() == want), None)

    def require_extension(self, name_or_type: str | int) -> Extension:
        ext = self.extension(name_or_type)
        if ext is None:
            have = ", ".join(e.name or f"type{e.type}" for e in self.extensions) or "none"
            raise errors.UnsupportedByPlugin(
                f"{self.path.name} does not advertise the {name_or_type!r} extension "
                f"(it has: {have})"
            )
        return ext

    # -- the call boundary -------------------------------------------------

    def fn(self, name: str):
        if (f := self._fn_cache.get(name)) is not None:
            return f
        slot = self.abi.SLOT.get(name)
        if slot is None:
            raise errors.Unimplemented(f"{name} is not in the PJRT {self.abi.PJRT_API_MAJOR}."
                                       f"{self.abi.PJRT_API_MINOR} vtable")
        if slot >= self.n_slots:
            raise errors.UnsupportedByPlugin(
                f"{name} is at slot {slot} but {self.path.name} exposes only "
                f"{self.n_slots} slots (plugin API {self.api_version[0]}.{self.api_version[1]})"
            )
        ptr = self._vtable[slot]
        if not ptr:
            raise errors.UnsupportedByPlugin(f"{name} is NULL in this plugin's vtable")
        f = _FN(ptr)
        self._fn_cache[name] = f
        return f

    def args(self, struct_name: str, **kw) -> Any:
        """Build a zeroed ``PJRT_*_Args`` with the correct ``struct_size``.

        ``struct_size`` is ``offsetof(last) + sizeof(last)``, NOT
        ``sizeof(struct)``; they differ for 23 structs at PJRT 0.114. The
        generated module carries the right value.
        """
        cls = getattr(self.abi, struct_name)
        obj = cls()
        ctypes.memset(ctypes.byref(obj), 0, ctypes.sizeof(obj))
        obj.struct_size = getattr(self.abi, f"{struct_name}_STRUCT_SIZE")
        for k, v in kw.items():
            setattr(obj, k, v)
        return obj

    def call(self, name: str, args: Any, *, reentrant: bool = False) -> None:
        """Invoke a PJRT entry point, converting ``PJRT_Error*`` into an exception."""
        f = self.fn(name)
        with pjrt_call():
            if reentrant:
                with dispatching():
                    err = f(ctypes.byref(args))
            else:
                err = f(ctypes.byref(args))
        if err:
            raise self._to_exception(err)

    def _to_exception(self, err: int) -> errors.PjrtError:
        """Read code + message, destroy the handle, build the exception.

        The handle is destroyed on every path -- it must never escape.
        """
        code, message = 2, "<no message>"
        try:
            m = self.args("PJRT_Error_Message_Args", error=err)
            self.fn("PJRT_Error_Message")(ctypes.byref(m))
            if m.message:
                message = ctypes.string_at(m.message, m.message_size).decode(errors="replace")
            try:
                c = self.args("PJRT_Error_GetCode_Args", error=err)
                if not self.fn("PJRT_Error_GetCode")(ctypes.byref(c)):
                    code = int(c.code)
            except errors.PjrtError:
                pass  # GetCode itself is optional in principle
        finally:
            d = self.args("PJRT_Error_Destroy_Args", error=err)
            self.fn("PJRT_Error_Destroy")(ctypes.byref(d))
        return errors.make(code, message)

    # -- misc --------------------------------------------------------------

    @property
    def attributes(self) -> dict[str, Any]:
        """``PJRT_Plugin_Attributes`` -- queryable *before* a client exists.

        Carries three things worth having: ``xla_version`` (which belongs in any
        executable-cache key, because an executable compiled against one XLA
        can silently miscompute under another), the
        ``stablehlo_{current,minimum}_version`` range this plugin accepts, and a
        vendor marker such as ``cuda_version`` -- which is how we can tell an
        accelerator plugin from a CPU one before calling Client_Create.
        """
        if (cached := getattr(self, "_attrs", None)) is not None:
            return cached
        a = self.args("PJRT_Plugin_Attributes_Args")
        self.call("PJRT_Plugin_Attributes", a)
        attrs = read_named_values(self.abi, a.attributes, a.num_attributes)
        self._attrs = attrs
        return attrs

    #: Attribute keys that identify a GPU-family plugin.
    _GPU_MARKERS = ("cuda_version", "rocm_version", "sycl_version", "hip_version")

    @property
    def platform_hint(self) -> str:
        """``"gpu"`` / ``"tpu"`` / ``"cpu"`` / ``"unknown"``, before a client.

        Layered, because no single signal covers every vendor: attribute
        markers first, then extensions that only one family advertises, then
        the artifact name as a last resort. A TPU plugin publishes none of the
        GPU markers and none of the GPU extensions, so keying only on those
        (as the first version of this did) silently classified libtpu as a
        non-accelerator and disabled the live-client guard.
        """
        try:
            attrs = self.attributes
        except errors.PjrtError:
            attrs = {}
        if any(k in attrs for k in self._GPU_MARKERS):
            return "gpu"
        if any(self.extension(n) for n in ("TpuTopology", "TpuExecutable", "Megascale")):
            return "tpu"
        if self.extension("Gpu_Custom_Call") is not None:
            return "gpu"
        name = self.path.name.lower()
        if "tpu" in name:
            return "tpu"
        if any(k in name for k in ("cuda", "rocm", "gpu")):
            return "gpu"
        if "cpu" in name:
            return "cpu"
        return "unknown"

    @property
    def is_accelerator(self) -> bool:
        """Whether this plugin manages an accelerator of any kind.

        Drives the live-client guard (a second client competes for the same
        device on TPU exactly as it does on GPU) and the default FFI platform.
        """
        return self.platform_hint in ("gpu", "tpu")

    @property
    def is_gpu(self) -> bool:
        """Specifically a GPU-family plugin.

        Only these accept the ``preallocate`` / ``memory_fraction`` allocator
        create-options; a TPU plugin would reject them.
        """
        return self.platform_hint == "gpu"

    @property
    def stablehlo_version_range(self) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        """``(minimum, current)`` StableHLO versions this plugin accepts.

        The target range for portable artifacts.
        """
        a = self.attributes
        lo, hi = a.get("stablehlo_minimum_version"), a.get("stablehlo_current_version")
        if isinstance(lo, tuple) and isinstance(hi, tuple):
            return lo, hi
        return None

    def stablehlo_target(self, prefer: str = "min") -> str:
        """The StableHLO version a producer should serialize a portable
        artifact to for this plugin.

        We do not emit StableHLO -- but choosing a target version *is* a
        compatibility negotiation with the plugin, which is below the line.
        Recipe for a producer::

            import jaxlib.mlir.dialects.stablehlo as sh
            blob = sh.serialize_portable_artifact_str(text, plugin.stablehlo_target())
            exe = client.compile(blob)       # Executable accepts bytes

        ``prefer="min"`` picks the oldest version the plugin admits, which is
        the most portable; ``"max"`` picks the newest.
        """
        r = self.stablehlo_version_range
        if r is None:
            raise errors.UnsupportedByPlugin(
                f"{self.path.name} publishes no stablehlo version range")
        lo, hi = r
        return ".".join(str(x) for x in (lo if prefer == "min" else hi))

    @property
    def xla_version(self) -> int | None:
        v = self.attributes.get("xla_version")
        return int(v) if isinstance(v, int) else None

    def initialize(self) -> None:
        """``PJRT_Plugin_Initialize`` -- the header says it must be called
        before anything else. Skipping it happens to work only while the
        plugin implements it as a no-op, as the JAX CUDA plugin does."""
        self.call("PJRT_Plugin_Initialize", self.args("PJRT_Plugin_Initialize_Args"))

    def __repr__(self) -> str:
        exact = "" if self.abi_exact else f" (using ABI {self.abi.PJRT_API_MAJOR}.{self.abi.PJRT_API_MINOR})"
        return (f"<Plugin {self.path.name} api={self.api_version[0]}.{self.api_version[1]}"
                f"{exact} slots={self.n_slots} ext={len(self.extensions)}>")
