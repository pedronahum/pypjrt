"""Topologies, and compiling for hardware you do not have.

``PJRT_Compile`` takes a ``PJRT_TopologyDescription`` and an **optional**
client, so an artifact can be built on a machine that has no such device --
build for a TPU slice in CI on a CPU box. Combined with
``TopologyDescription_Serialize``/``Deserialize`` you can ship the topology
itself. Almost nobody binds any of this, and instead builds a whole AOT
artifact format around a client-bound compile.

Support is uneven and worth probing rather than assuming. On this box:

* CPU plugin  -- ``NotFound: No compiler factory for platform: cpu``
* CUDA plugin -- ``PJRT_Compile`` works against a client-derived topology, but
  ``Deserialize`` is ``Unimplemented``, so a topology cannot be shipped.

TPU is the case this API exists for: ``create(name=...)`` is meant to work with
no hardware present.
"""

from __future__ import annotations

import ctypes
from typing import Any

from . import errors
from ._loader import Plugin, read_named_values
from .compile_options import CompileOptions

_VOIDP = ctypes.c_void_p


class Topology:
    """A device topology, with or without a live client."""

    def __init__(self, plugin: Plugin, ptr: int, *, owned: bool = True):
        self._plugin = plugin
        self._ptr = ptr
        self._owned = owned
        self._closed = False

    # -- construction ------------------------------------------------------

    @classmethod
    def create(cls, plugin: Plugin, name: str, *,
               options: dict[str, Any] | None = None) -> "Topology":
        """Describe a topology by name, with no client and no hardware.

        The name is plugin-specific -- e.g. a TPU slice such as ``"v5e:2x2"``.
        """
        from .client import _named_values
        nb = ctypes.create_string_buffer(name.encode())
        named, keep = _named_values(plugin, dict(options or {}))
        a = plugin.args("PJRT_TopologyDescription_Create_Args",
                        topology_name=ctypes.cast(nb, _VOIDP), topology_name_size=len(name))
        if named is not None:
            a.create_options = ctypes.cast(named, _VOIDP)
            a.num_options = len(options or {})
        plugin.call("PJRT_TopologyDescription_Create", a)
        del keep, nb
        return cls(plugin, a.topology)

    @classmethod
    def from_client(cls, client) -> "Topology":
        """The topology of a live client. Owned by the client, not by us."""
        a = client._plugin.args("PJRT_Client_TopologyDescription_Args",
                                client=client._check())
        client._plugin.call("PJRT_Client_TopologyDescription", a)
        return cls(client._plugin, a.topology, owned=False)

    @classmethod
    def deserialize(cls, plugin: Plugin, blob: bytes) -> "Topology":
        """Adopt a topology serialized elsewhere -- the ship-it-to-CI half."""
        buf = ctypes.create_string_buffer(blob, len(blob))
        a = plugin.args("PJRT_TopologyDescription_Deserialize_Args",
                        serialized_topology=ctypes.cast(buf, _VOIDP),
                        serialized_topology_size=len(blob))
        plugin.call("PJRT_TopologyDescription_Deserialize", a)
        return cls(plugin, a.topology)

    # -- inspection --------------------------------------------------------

    def _check(self) -> int:
        if self._closed:
            raise errors.HandleClosed("Topology is closed")
        return self._ptr

    @property
    def platform_name(self) -> str:
        a = self._plugin.args("PJRT_TopologyDescription_PlatformName_Args",
                              topology=self._check())
        self._plugin.call("PJRT_TopologyDescription_PlatformName", a)
        return ctypes.string_at(a.platform_name, a.platform_name_size).decode()

    @property
    def platform_version(self) -> str:
        a = self._plugin.args("PJRT_TopologyDescription_PlatformVersion_Args",
                              topology=self._check())
        self._plugin.call("PJRT_TopologyDescription_PlatformVersion", a)
        return ctypes.string_at(a.platform_version, a.platform_version_size).decode()

    @property
    def attributes(self) -> dict[str, Any]:
        a = self._plugin.args("PJRT_TopologyDescription_Attributes_Args",
                              topology=self._check())
        self._plugin.call("PJRT_TopologyDescription_Attributes", a)
        return read_named_values(self._plugin.abi, a.attributes, a.num_attributes)

    def device_descriptions(self) -> list[dict[str, Any]]:
        """Every device in the topology, including ones not present locally."""
        a = self._plugin.args("PJRT_TopologyDescription_GetDeviceDescriptions_Args",
                              topology=self._check())
        self._plugin.call("PJRT_TopologyDescription_GetDeviceDescriptions", a)
        ptrs = ctypes.cast(a.descriptions, ctypes.POINTER(_VOIDP))
        out = []
        for i in range(a.num_descriptions):
            d = int(ptrs[i])
            entry: dict[str, Any] = {}
            for fn, args_name, field, key in (
                ("PJRT_DeviceDescription_Id", "PJRT_DeviceDescription_Id_Args", "id", "id"),
                ("PJRT_DeviceDescription_ProcessIndex",
                 "PJRT_DeviceDescription_ProcessIndex_Args", "process_index", "process_index"),
            ):
                x = self._plugin.args(args_name, device_description=d)
                self._plugin.call(fn, x)
                entry[key] = int(getattr(x, field))
            k = self._plugin.args("PJRT_DeviceDescription_Kind_Args", device_description=d)
            self._plugin.call("PJRT_DeviceDescription_Kind", k)
            entry["kind"] = ctypes.string_at(k.device_kind, k.device_kind_size).decode()
            at = self._plugin.args("PJRT_DeviceDescription_Attributes_Args",
                                   device_description=d)
            try:
                self._plugin.call("PJRT_DeviceDescription_Attributes", at)
                entry["attributes"] = read_named_values(self._plugin.abi, at.attributes,
                                                        at.num_attributes)
            except errors.PjrtError:
                entry["attributes"] = {}
            out.append(entry)
        return out

    def memory_space_kind_ids(self) -> list[int]:
        a = self._plugin.args("PJRT_TopologyDescription_GetMemorySpaceKindIds_Args",
                              topology=self._check())
        self._plugin.call("PJRT_TopologyDescription_GetMemorySpaceKindIds", a)
        if not a.memory_space_kind_ids:
            return []
        p32 = ctypes.cast(a.memory_space_kind_ids, ctypes.POINTER(ctypes.c_int32))
        return [int(p32[i]) for i in range(a.num_memory_space_kind_ids)]

    def serialize(self) -> bytes:
        a = self._plugin.args("PJRT_TopologyDescription_Serialize_Args",
                              topology=self._check())
        self._plugin.call("PJRT_TopologyDescription_Serialize", a)
        try:
            return ctypes.string_at(a.serialized_bytes, a.serialized_bytes_size)
        finally:
            if a.serialized_topology and a.serialized_topology_deleter:
                ctypes.CFUNCTYPE(None, _VOIDP)(a.serialized_topology_deleter)(
                    a.serialized_topology)

    def fingerprint(self) -> int:
        """A ``uint64`` identifying this topology -- not a byte span."""
        a = self._plugin.args("PJRT_TopologyDescription_Fingerprint_Args",
                              topology=self._check())
        self._plugin.call("PJRT_TopologyDescription_Fingerprint", a)
        return int(a.fingerprint)

    # -- the point of all this --------------------------------------------

    def compile(self, program: str | bytes, *, options: CompileOptions | None = None,
                client=None) -> bytes:
        """AOT-compile for this topology and return the serialized executable.

        ``client`` is optional: pass one only if the plugin needs it. The
        result is bytes rather than an ``Executable`` because the whole point
        is that the target device may not be present.
        """
        code = program.encode() if isinstance(program, str) else bytes(program)
        if options is None:
            options = CompileOptions(use_shardy_partitioner=b"sdy." in code)
        opts = options.encode(self._plugin.abi)
        fmt = b"mlir"
        cb = ctypes.create_string_buffer(code, len(code))
        fb = ctypes.create_string_buffer(fmt, len(fmt))
        ob = ctypes.create_string_buffer(opts, len(opts))
        prog = self._plugin.args("PJRT_Program",
                                 code=ctypes.cast(cb, _VOIDP), code_size=len(code),
                                 format=ctypes.cast(fb, _VOIDP), format_size=len(fmt))
        a = self._plugin.args("PJRT_Compile_Args", topology=self._check(),
                              program=ctypes.addressof(prog),
                              compile_options=ctypes.cast(ob, _VOIDP),
                              compile_options_size=len(opts))
        if client is not None:
            a.client = client._check()
        self._plugin.call("PJRT_Compile", a)
        exe = int(a.executable)
        try:
            s = self._plugin.args("PJRT_Executable_Serialize_Args", executable=exe)
            self._plugin.call("PJRT_Executable_Serialize", s)
            try:
                return ctypes.string_at(s.serialized_bytes, s.serialized_bytes_size)
            finally:
                if s.serialized_executable and s.serialized_executable_deleter:
                    ctypes.CFUNCTYPE(None, _VOIDP)(s.serialized_executable_deleter)(
                        s.serialized_executable)
        finally:
            self._plugin.call("PJRT_Executable_Destroy",
                              self._plugin.args("PJRT_Executable_Destroy_Args",
                                                executable=exe))

    # -- lifetime ----------------------------------------------------------

    def close(self) -> None:
        if self._closed or not self._owned:
            self._closed = True
            return
        self._closed = True
        self._plugin.call("PJRT_TopologyDescription_Destroy",
                          self._plugin.args("PJRT_TopologyDescription_Destroy_Args",
                                            topology=self._ptr))

    def __enter__(self) -> "Topology":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        try:
            return f"<Topology {self.platform_name!r} {self.platform_version!r}>"
        except errors.PjrtError:
            return f"<Topology 0x{self._ptr:x}>"
