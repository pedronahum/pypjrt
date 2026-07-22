"""AOT artifacts: a serialized executable plus everything needed to reject it.

The idea worth keeping from hand-rolled AOT formats is that a mismatched load
fails with a diagnostic instead of crashing inside XLA. Those formats hand-roll
the guard from ``host_arch`` / ``sm_arch`` / ``cuda_version`` strings; PJRT
ships the supported mechanism -- the **AbiVersion
extension**, whose ``IsCompatibleWithExecutable`` answers the question the
plugin itself is authoritative on. We use it where advertised and
fall back to recorded identifiers where it is not.

Container layout, little-endian, deliberately trivial to inspect:

    magic   8   b"PYPJRTA\\0"
    version u32
    n_meta  u32   JSON metadata, utf-8
    n_exec  u64   serialized PJRT executable
    n_src   u64   StableHLO source (fallback / provenance; may be empty)
    <meta><exec><src>
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import errors

MAGIC = b"PYPJRTA\0"
VERSION = 1
_HEADER = struct.Struct("<8sIIQQ")


class ArtifactMismatch(errors.PjrtError):
    """The artifact cannot be loaded here, and we can say why."""


@dataclass
class Artifact:
    executable: bytes
    platform: str = ""
    api_version: tuple[int, int] = (0, 0)
    xla_version: int | None = None
    fingerprint: str = ""            # hex, from PJRT_Executable_Fingerprint
    abi_proto: str = ""              # hex, from the AbiVersion extension
    compile_options: str = ""        # hex, the exact CompileOptionsProto used
    source_sha256: str = ""
    source: bytes = b""
    output_types: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- container ---------------------------------------------------------

    def to_bytes(self) -> bytes:
        meta = json.dumps({
            "platform": self.platform,
            "api_version": list(self.api_version),
            "xla_version": self.xla_version,
            "fingerprint": self.fingerprint,
            "abi_proto": self.abi_proto,
            "compile_options": self.compile_options,
            "source_sha256": self.source_sha256,
            "output_types": self.output_types,
            "metadata": self.metadata,
        }, sort_keys=True).encode()
        return (_HEADER.pack(MAGIC, VERSION, len(meta), len(self.executable), len(self.source))
                + meta + self.executable + self.source)

    @classmethod
    def from_bytes(cls, blob: bytes) -> "Artifact":
        if len(blob) < _HEADER.size:
            raise ArtifactMismatch("not a pypjrt artifact: too short")
        magic, ver, n_meta, n_exec, n_src = _HEADER.unpack_from(blob)
        if magic != MAGIC:
            raise ArtifactMismatch(f"not a pypjrt artifact: bad magic {magic!r}")
        if ver != VERSION:
            raise ArtifactMismatch(f"artifact format version {ver}, this build writes {VERSION}")
        o = _HEADER.size
        meta = json.loads(blob[o:o + n_meta]); o += n_meta
        exe = blob[o:o + n_exec]; o += n_exec
        src = blob[o:o + n_src]
        return cls(executable=exe, source=src,
                   platform=meta["platform"], api_version=tuple(meta["api_version"]),
                   xla_version=meta["xla_version"], fingerprint=meta["fingerprint"],
                   abi_proto=meta["abi_proto"], compile_options=meta["compile_options"],
                   source_sha256=meta["source_sha256"],
                   output_types=meta.get("output_types", []),
                   metadata=meta.get("metadata", {}))

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_bytes(self.to_bytes())
        return p

    @classmethod
    def read(cls, path: str | Path) -> "Artifact":
        return cls.from_bytes(Path(path).read_bytes())

    # -- guards ------------------------------------------------------------

    def check_compatible(self, plugin, *, platform: str | None = None,
                         strict: bool = True) -> list[str]:
        """Reject an artifact that cannot run here, with a reason.

        Returns the list of problems (empty when clean). ``strict`` raises
        instead. The authoritative check is the plugin's own
        ``IsCompatibleWithExecutable``; the recorded identifiers are the
        fallback for plugins that do not advertise the extension.
        """
        problems: list[str] = []
        # The platform name needs a live client, so the caller supplies it.
        # Deriving it from plugin attributes silently never fired -- a guard
        # that quietly does nothing is worse than no guard.
        if self.platform and platform and platform != self.platform:
            problems.append(f"platform: artifact built for {self.platform!r}, "
                            f"this client is {platform!r}")
        if self.api_version != (0, 0) and self.api_version[0] != plugin.api_version[0]:
            problems.append(f"PJRT API major: artifact {self.api_version[0]}, "
                            f"plugin {plugin.api_version[0]}")
        if self.xla_version is not None and plugin.xla_version is not None \
                and self.xla_version != plugin.xla_version:
            problems.append(f"xla_version: artifact {self.xla_version}, "
                            f"plugin {plugin.xla_version} -- a serialized executable "
                            f"compiled against one XLA can silently miscompute under another")
        if strict and problems:
            raise ArtifactMismatch("; ".join(problems))
        return problems


# --------------------------------------------------------------------------
# the AbiVersion extension


class AbiVersion:
    """Wrapper over ``PJRT_Extension_Type_AbiVersion``.

    ``IsCompatibleWithExecutable`` is the artifact guard, answered by the plugin
    rather than guessed by us. Absent on plugins that do not advertise it, so
    every entry point here is behind a capability probe.
    """

    # PJRT_AbiVersion_Extension: base[24] then the function-pointer table.
    _OFF = {
        "client_runtime_abi_version": 24,
        "executable_get_abi_version": 32,
        "runtime_abi_version_destroy": 40,
        "is_compatible_with_runtime": 48,
        "is_compatible_with_executable": 56,
        "runtime_abi_version_to_proto": 64,
        "runtime_abi_version_platform_id": 72,
        "executable_abi_version_destroy": 80,
        "executable_abi_version_to_proto": 88,
        "executable_abi_version_platform_id": 96,
    }
    _FN = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)

    def __init__(self, plugin):
        self._p = plugin
        self._ext = plugin.require_extension("AbiVersion")

    @classmethod
    def probe(cls, plugin) -> "AbiVersion | None":
        return cls(plugin) if plugin.extension("AbiVersion") else None

    def _call(self, name: str, args) -> None:
        ptr = ctypes.cast(self._ext.address + self._OFF[name],
                          ctypes.POINTER(ctypes.c_void_p))[0]
        if not ptr:
            raise errors.UnsupportedByPlugin(f"AbiVersion.{name} is NULL")
        err = self._FN(ptr)(ctypes.byref(args))
        if err:
            raise self._p._to_exception(err)

    def runtime_version_for(self, client_address: int) -> int:
        a = self._p.args("PJRT_Client_RuntimeAbiVersion_Args", client=client_address)
        self._call("client_runtime_abi_version", a)
        return int(a.abi_version)

    def executable_version(self, executable_address: int) -> int:
        a = self._p.args("PJRT_Executable_GetAbiVersion_Args", executable=executable_address)
        self._call("executable_get_abi_version", a)
        return int(a.abi_version)

    def executable_proto(self, executable_address: int) -> bytes:
        v = self.executable_version(executable_address)
        a = self._p.args("PJRT_ExecutableAbiVersion_ToProto_Args", abi_version=v)
        self._call("executable_abi_version_to_proto", a)
        blob = ctypes.string_at(a.serialized_proto, a.serialized_proto_size)
        if a.serialized_proto_holder and a.serialized_proto_deleter:
            ctypes.CFUNCTYPE(None, ctypes.c_void_p)(a.serialized_proto_deleter)(
                a.serialized_proto_holder)
        self._call("executable_abi_version_destroy",
                   self._p.args("PJRT_ExecutableAbiVersion_Destroy_Args", abi_version=v))
        return blob

    def check(self, client_address: int, executable_address: int) -> str | None:
        """``None`` when compatible, else the plugin's own explanation.

        On this box the CUDA plugin reports a real skew here:
        "CUDA toolkit version mismatch. Running with version 13.0.0, but
        executable requires >= 13.2.0".
        """
        rt = self.runtime_version_for(client_address)
        ev = self.executable_version(executable_address)
        try:
            self._call("is_compatible_with_executable", self._p.args(
                "PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable_Args",
                abi_version=rt, executable_abi_version=ev))
            return None
        except errors.PjrtError as e:
            return e.message
        finally:
            for fn, argname, v in (("runtime_abi_version_destroy",
                                    "PJRT_RuntimeAbiVersion_Destroy_Args", rt),
                                   ("executable_abi_version_destroy",
                                    "PJRT_ExecutableAbiVersion_Destroy_Args", ev)):
                try:
                    self._call(fn, self._p.args(argname, abi_version=v))
                except Exception:
                    pass


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
