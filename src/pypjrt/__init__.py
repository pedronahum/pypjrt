"""pypjrt -- a dependency-free Python client for the PJRT C API and the XLA FFI.

pypjrt is everything BELOW StableHLO. It consumes StableHLO (text or portable
artifact bytes) from any producer and owns compilation, buffers, execution,
custom calls and artifacts. It does not build StableHLO, differentiate, or
propagate shardings.
"""

from __future__ import annotations

from . import errors
from ._loader import Extension, Plugin, find_plugin
from .client import (AsyncTrackingEvent, Buffer, Client, CopyToDeviceStream, Device,
                     Event, ExecuteContext, Executable, Memory)
from .artifact import AbiVersion, Artifact, ArtifactMismatch
from .cache import CompileCache
from .compile_options import CompileOptions
from .kv import FileStore, InMemoryStore, KeyValueStore
from .session import Program, Session, Slot
from .topology import Topology
from .transfer import AsyncTransfer, ShapeSpec
from . import typing
from .typing import DType

# Single source of truth is pyproject.toml; duplicating it here means the two
# drift and the wheel reports a version it is not. Falls back for a source tree
# that was never installed.
from importlib.metadata import PackageNotFoundError as _NotFound, version as _version

try:
    __version__ = _version("pypjrt")
except _NotFound:  # pragma: no cover - running from a bare checkout
    __version__ = "0.0.0.dev0"

del _version, _NotFound
__all__ = [
    "Plugin", "Extension", "find_plugin", "CompileOptions", "errors",
    "Client", "Device", "Buffer", "Event", "Executable", "Memory", "ExecuteContext",
    "Artifact", "ArtifactMismatch", "AbiVersion", "CompileCache",
    "Session", "Program", "Slot", "Topology", "AsyncTransfer", "ShapeSpec", "typing", "DType",
    "KeyValueStore", "InMemoryStore", "FileStore",
]
