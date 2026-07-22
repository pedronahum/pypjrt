"""The PJRT error boundary.

Every entry point that can fail returns ``PJRT_Error*``. The contract is: read
the code, read the message, ``PJRT_Error_Destroy``, then raise. An error handle
must never escape into Python -- leaking one leaks plugin memory, and holding
one past ``Destroy`` is a use-after-free.

A client that does not bind ``PJRT_Error_GetCode`` has stringly-typed errors
and callers cannot branch on them. We bind it.
"""

from __future__ import annotations


class PjrtError(RuntimeError):
    """Base for every error raised across the PJRT boundary."""

    code: int = 2  # UNKNOWN

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class Cancelled(PjrtError): code = 1
class Unknown(PjrtError): code = 2
class InvalidArgument(PjrtError): code = 3
class DeadlineExceeded(PjrtError): code = 4
class NotFound(PjrtError): code = 5
class AlreadyExists(PjrtError): code = 6
class PermissionDenied(PjrtError): code = 7
class ResourceExhausted(PjrtError): code = 8
class FailedPrecondition(PjrtError): code = 9
class Aborted(PjrtError): code = 10
class OutOfRange(PjrtError): code = 11
class Unimplemented(PjrtError): code = 12
class Internal(PjrtError): code = 13
class Unavailable(PjrtError): code = 14
class DataLoss(PjrtError): code = 15
class Unauthenticated(PjrtError): code = 16


BY_CODE: dict[int, type[PjrtError]] = {
    c.code: c
    for c in (
        Cancelled, Unknown, InvalidArgument, DeadlineExceeded, NotFound,
        AlreadyExists, PermissionDenied, ResourceExhausted, FailedPrecondition,
        Aborted, OutOfRange, Unimplemented, Internal, Unavailable, DataLoss,
        Unauthenticated,
    )
}


def make(code: int, message: str) -> PjrtError:
    return BY_CODE.get(code, Unknown)(message, code)


class IncompatiblePlugin(PjrtError):
    """The plugin's ABI version cannot be spoken by this build."""


class UnsupportedByPlugin(PjrtError):
    """A capability this plugin does not advertise.

    Raised by capability probes rather than by crashing on a missing extension
    -- plugins legitimately differ.
    """


class HandleClosed(PjrtError):
    """Use of a handle after ``close()``, or of a borrowed handle after its
    ``with`` block. Python has no linear types; this is the runtime equivalent,
    and it is what a linear type would have bought us."""
