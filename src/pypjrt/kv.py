"""Key-value rendezvous for multi-controller (SPMD) clients.

``PJRT_Client_Create_Args`` carries three callbacks -- ``kv_get``, ``kv_put``
and ``kv_try_get`` -- and the header says plainly that *"PJRT client can use
these callbacks to share information between processes/nodes."* You cannot form
a multi-host client without them: it is not orchestration policy, it is a
required argument to client creation.

**It is usually left unimplemented**, or stubbed with an in-process map that
cannot rendezvous anything -- which looks like it works right up until a second
process joins. In Python a real backend over a shared directory is an
afternoon's work, so there is no excuse for the stub.

This is deliberately **multi-controller**: every process runs the same program
and they rendezvous here. Single-controller (Pathways) is a different system
"""

from __future__ import annotations

import ctypes
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Protocol

from . import errors

_VOIDP = ctypes.c_void_p


class KeyValueStore(Protocol):
    """The rendezvous contract. ``get`` blocks; ``try_get`` does not."""

    def get(self, key: str, timeout_ms: int) -> bytes: ...
    def try_get(self, key: str) -> bytes | None: ...
    def put(self, key: str, value: bytes) -> None: ...


class InMemoryStore:
    """Single-process backend. Useful for tests and for ``num_nodes == 1``."""

    def __init__(self):
        self._d: dict[str, bytes] = {}
        self._cv = threading.Condition()

    def get(self, key: str, timeout_ms: int) -> bytes:
        deadline = time.monotonic() + max(timeout_ms, 0) / 1000.0
        with self._cv:
            while key not in self._d:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise KeyError(key)
                self._cv.wait(remaining)
            return self._d[key]

    def try_get(self, key: str) -> bytes | None:
        with self._cv:
            return self._d.get(key)

    def put(self, key: str, value: bytes) -> None:
        with self._cv:
            self._d[key] = value
            self._cv.notify_all()


class FileStore:
    """Rendezvous through a shared directory. Works across processes on a host.

    Writes are atomic (``os.replace`` within the directory), so a reader never
    observes a torn value. Keys are hex-encoded, so any byte string is a legal
    key regardless of the filesystem.
    """

    def __init__(self, directory: str | os.PathLike[str] | None = None,
                 poll_interval: float = 0.005):
        self.dir = Path(directory) if directory else Path(tempfile.mkdtemp(prefix="pypjrt-kv-"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.poll_interval = poll_interval
        self.gets = 0
        self.puts = 0

    def _path(self, key: str) -> Path:
        return self.dir / (key.encode().hex() + ".val")

    def try_get(self, key: str) -> bytes | None:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError:
            return None

    def get(self, key: str, timeout_ms: int) -> bytes:
        deadline = time.monotonic() + max(timeout_ms, 0) / 1000.0
        while True:
            if (v := self.try_get(key)) is not None:
                self.gets += 1
                return v
            if time.monotonic() >= deadline:
                raise KeyError(key)
            time.sleep(self.poll_interval)

    def put(self, key: str, value: bytes) -> None:
        p = self._path(key)
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(value)
            os.replace(tmp, p)          # atomic within a directory
            self.puts += 1
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def __repr__(self) -> str:
        return f"<FileStore {self.dir} puts={self.puts} gets={self.gets}>"


# ---------------------------------------------------------------------------
# the C callbacks


#: Value buffers handed to the plugin, freed by its deleter. Keyed by address.
_VALUES: dict[int, ctypes.Array] = {}
_VALUES_LOCK = threading.Lock()

_GET_T = ctypes.CFUNCTYPE(_VOIDP, _VOIDP)
_PUT_T = ctypes.CFUNCTYPE(_VOIDP, _VOIDP)
_DELETER_T = ctypes.CFUNCTYPE(None, _VOIDP)


@_DELETER_T
def _value_deleter(ptr):
    if ptr:
        with _VALUES_LOCK:
            _VALUES.pop(int(ptr), None)


def _fail(args, code: int, message: str) -> int:
    """Build a PJRT_Error through the callback_error the plugin gave us."""
    if not args.callback_error:
        return 0
    msg = message.encode()
    fn = ctypes.CFUNCTYPE(_VOIDP, ctypes.c_int32, ctypes.c_char_p, ctypes.c_size_t)(
        args.callback_error)
    return int(fn(code, msg, len(msg)) or 0)


class KvBridge:
    """Holds the trampolines for one client. Must outlive the client."""

    def __init__(self, plugin, store: KeyValueStore):
        self.plugin = plugin
        self.store = store
        abi = plugin.abi
        self.calls = {"get": 0, "try_get": 0, "put": 0}

        def _emit_value(args, value: bytes, deleter_field: str) -> int:
            buf = ctypes.create_string_buffer(value, len(value))
            addr = ctypes.addressof(buf)
            with _VALUES_LOCK:
                _VALUES[addr] = buf
            args.value = ctypes.cast(buf, _VOIDP)
            args.value_size = len(value)
            setattr(args, deleter_field, ctypes.cast(_value_deleter, _VOIDP))
            return 0

        def on_get(ptr):
            a = ctypes.cast(ptr, ctypes.POINTER(abi.PJRT_KeyValueGetCallback_Args)).contents
            self.calls["get"] += 1
            key = ctypes.string_at(a.key, a.key_size).decode(errors="replace")
            try:
                v = store.get(key, int(a.timeout_in_ms))
            except KeyError:
                return _fail(a, 5, f"key {key!r} not found within "
                                   f"{a.timeout_in_ms} ms")     # NOT_FOUND
            except Exception as e:  # noqa: BLE001 -- never unwind into the plugin
                return _fail(a, 13, f"{type(e).__name__}: {e}")  # INTERNAL
            return _emit_value(a, v, "value_deleter_callback")

        def on_try_get(ptr):
            a = ctypes.cast(ptr, ctypes.POINTER(abi.PJRT_KeyValueTryGetCallback_Args)).contents
            self.calls["try_get"] += 1
            key = ctypes.string_at(a.key, a.key_size).decode(errors="replace")
            try:
                v = store.try_get(key)
            except Exception as e:  # noqa: BLE001
                return _fail(a, 13, f"{type(e).__name__}: {e}")
            if v is None:
                return _fail(a, 5, f"key {key!r} not present")
            return _emit_value(a, v, "value_deleter_callback")

        def on_put(ptr):
            a = ctypes.cast(ptr, ctypes.POINTER(abi.PJRT_KeyValuePutCallback_Args)).contents
            self.calls["put"] += 1
            key = ctypes.string_at(a.key, a.key_size).decode(errors="replace")
            val = ctypes.string_at(a.value, a.value_size)
            try:
                store.put(key, val)
            except Exception as e:  # noqa: BLE001
                return _fail(a, 13, f"{type(e).__name__}: {e}")
            return 0

        # Held for the life of this object, which the Client holds.
        self.get_cb = _GET_T(on_get)
        self.try_get_cb = _GET_T(on_try_get)
        self.put_cb = _PUT_T(on_put)

    def apply(self, args) -> None:
        """Install the callbacks on a ``PJRT_Client_Create_Args``."""
        args.kv_get_callback = ctypes.cast(self.get_cb, _VOIDP)
        args.kv_try_get_callback = ctypes.cast(self.try_get_cb, _VOIDP)
        args.kv_put_callback = ctypes.cast(self.put_cb, _VOIDP)
