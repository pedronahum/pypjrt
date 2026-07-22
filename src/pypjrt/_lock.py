"""One process-wide lock for PJRT calls, and the rule that goes with it.

Why a lock at all: the PJRT GPU client is *not* thread-safe for concurrent buffer/event operations, and that a
finalizer-driven ``PJRT_Buffer_Destroy`` racing ``Execute`` produces
"UnboundedWorkQueue deleted with pending work" / heap corruption. Python's
``__del__`` runs at arbitrary points on arbitrary threads, so this applies with
full force.

THE RULE: an XLA FFI handler must never take this lock and must never call a
PJRT API. ``Execute`` holds the lock while XLA dispatches handlers from its own
worker threads -- and, as spike/pjrt_ffi.py demonstrated, sometimes on the
*registering* thread inside ``PJRT_FFI_Register_Handler`` itself. A thread-local
re-entrancy guard handles that; we make it a design constraint
and check it.

ctypes releases the GIL around ``CFUNCTYPE`` foreign calls, so holding this lock
across a blocking ``Execute`` does not stall unrelated Python threads.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

_PJRT = threading.RLock()

# Set while we are inside a call that XLA may re-enter (Execute, handler
# registration). Checked by the FFI layer to catch a handler reaching back in.
_in_dispatch = threading.local()


def in_dispatch() -> bool:
    return getattr(_in_dispatch, "flag", False)


@contextmanager
def pjrt_call():
    """Serialise a PJRT API call."""
    if in_dispatch():
        raise RuntimeError(
            "PJRT API called from inside an XLA FFI handler. Handlers must "
            "decode their call frame, launch, and return -- they may not call "
            "back into PJRT (see pypjrt/_lock.py)."
        )
    with _PJRT:
        yield


@contextmanager
def dispatching():
    """Mark a region XLA may synchronously re-enter (Execute, registration)."""
    prev = in_dispatch()
    _in_dispatch.flag = True
    try:
        yield
    finally:
        _in_dispatch.flag = prev
