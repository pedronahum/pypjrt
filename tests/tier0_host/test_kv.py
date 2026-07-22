"""Tier 0: the rendezvous store and its C callbacks. No plugin needed."""
import ctypes, threading, time, pytest
from pypjrt import _abi
from pypjrt.kv import FileStore, InMemoryStore, KvBridge

pytestmark = pytest.mark.tier0
A = _abi.load(0, _abi.available()[0][1])[0]
_VOIDP = ctypes.c_void_p


class _FakePlugin:
    abi = A


# -- store semantics --------------------------------------------------------


@pytest.fixture(params=["memory", "file"])
def store(request, tmp_path):
    return InMemoryStore() if request.param == "memory" else FileStore(tmp_path)


def test_put_then_get(store):
    store.put("k", b"v")
    assert store.get("k", 1000) == b"v"
    assert store.try_get("k") == b"v"


def test_try_get_missing_is_none_not_a_wait(store):
    t0 = time.monotonic()
    assert store.try_get("absent") is None
    assert time.monotonic() - t0 < 0.5


def test_get_times_out(store):
    t0 = time.monotonic()
    with pytest.raises(KeyError):
        store.get("absent", 60)
    assert time.monotonic() - t0 >= 0.05


def test_get_blocks_until_another_thread_puts(store):
    def writer():
        time.sleep(0.05)
        store.put("late", b"arrived")
    threading.Thread(target=writer, daemon=True).start()
    assert store.get("late", 5000) == b"arrived"


def test_file_store_is_shared_between_independent_handles(tmp_path):
    """Two processes see one directory; two FileStore objects model that."""
    a, b = FileStore(tmp_path), FileStore(tmp_path)
    a.put("shared", b"\x00\x01\x02")
    assert b.get("shared", 1000) == b"\x00\x01\x02"


def test_binary_safe_keys_and_values(tmp_path):
    s = FileStore(tmp_path)
    s.put("a/b/c:d e", b"\x00\xff\n\r")
    assert s.get("a/b/c:d e", 1000) == b"\x00\xff\n\r"


# -- the C callbacks, invoked synthetically ---------------------------------


def _args(cls_name, **kw):
    cls = getattr(A, cls_name)
    o = cls()
    ctypes.memset(ctypes.byref(o), 0, ctypes.sizeof(o))
    o.struct_size = getattr(A, f"{cls_name}_STRUCT_SIZE")
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def _invoke(cb, args) -> int:
    return int(ctypes.CFUNCTYPE(_VOIDP, _VOIDP)(
        ctypes.cast(cb, _VOIDP).value)(ctypes.byref(args)) or 0)


def test_put_callback_writes_through_to_the_store():
    store = InMemoryStore()
    bridge = KvBridge(_FakePlugin(), store)
    key, val = b"topology/0", b"\x01\x02\x03"
    kb, vb = ctypes.create_string_buffer(key), ctypes.create_string_buffer(val)
    a = _args("PJRT_KeyValuePutCallback_Args",
              key=ctypes.cast(kb, _VOIDP), key_size=len(key),
              value=ctypes.cast(vb, _VOIDP), value_size=len(val))
    assert _invoke(bridge.put_cb, a) == 0
    assert store.try_get("topology/0") == val
    assert bridge.calls["put"] == 1


def test_get_callback_returns_the_value_and_a_deleter():
    store = InMemoryStore()
    store.put("k", b"hello")
    bridge = KvBridge(_FakePlugin(), store)
    kb = ctypes.create_string_buffer(b"k")
    a = _args("PJRT_KeyValueGetCallback_Args",
              key=ctypes.cast(kb, _VOIDP), key_size=1, timeout_in_ms=1000)
    assert _invoke(bridge.get_cb, a) == 0
    assert a.value_size == 5
    assert ctypes.string_at(a.value, a.value_size) == b"hello"
    assert a.value_deleter_callback, "plugin has no way to free the value"
    # the deleter must not crash and must release our buffer
    from pypjrt.kv import _VALUES
    assert int(a.value) in _VALUES
    ctypes.CFUNCTYPE(None, _VOIDP)(ctypes.cast(a.value_deleter_callback, _VOIDP).value)(a.value)
    assert int(a.value) not in _VALUES


def test_missing_key_produces_an_error_not_a_crash():
    calls = []

    @ctypes.CFUNCTYPE(_VOIDP, ctypes.c_int32, ctypes.c_char_p, ctypes.c_size_t)
    def fake_error(code, msg, n):
        calls.append((code, msg[:n].decode()))
        return 0x1234        # a non-null "PJRT_Error*"

    bridge = KvBridge(_FakePlugin(), InMemoryStore())
    kb = ctypes.create_string_buffer(b"nope")
    a = _args("PJRT_KeyValueGetCallback_Args",
              key=ctypes.cast(kb, _VOIDP), key_size=4, timeout_in_ms=10,
              callback_error=ctypes.cast(fake_error, _VOIDP))
    rc = _invoke(bridge.get_cb, a)
    assert rc == 0x1234, "the callback must return the error the plugin built"
    assert calls and calls[0][0] == 5           # NOT_FOUND
    assert "not found" in calls[0][1]


def test_store_exception_becomes_an_internal_error():
    class Broken:
        def get(self, k, t): raise RuntimeError("disk on fire")
        def try_get(self, k): raise RuntimeError("disk on fire")
        def put(self, k, v): raise RuntimeError("disk on fire")

    seen = []

    @ctypes.CFUNCTYPE(_VOIDP, ctypes.c_int32, ctypes.c_char_p, ctypes.c_size_t)
    def fake_error(code, msg, n):
        seen.append((code, msg[:n].decode()))
        return 0x99

    bridge = KvBridge(_FakePlugin(), Broken())
    kb = ctypes.create_string_buffer(b"k")
    a = _args("PJRT_KeyValuePutCallback_Args",
              key=ctypes.cast(kb, _VOIDP), key_size=1,
              value=ctypes.cast(kb, _VOIDP), value_size=1,
              callback_error=ctypes.cast(fake_error, _VOIDP))
    assert _invoke(bridge.put_cb, a) == 0x99
    assert seen[0][0] == 13 and "disk on fire" in seen[0][1]   # INTERNAL
