"""XLA FFI custom calls -- **device opacity is surrendered here**.

`pypjrt` itself never names a device: the same code drives a CPU plugin and an
NVIDIA GB10. This subpackage is where that stops. Registering a handler means
naming a platform ("Host", "CUDA"), receiving raw device pointers, and taking
XLA's stream. That is inherent to the feature, not to this design -- but it is
made visible at the import, so a reviewer seeing `pypjrt.ffi` in a diff knows
portability was traded on that line.

Nothing in `pypjrt` core imports this package.
"""

from .frame import CallFrame, FfiBuffer
from .registry import handler, register, registered

__all__ = ["CallFrame", "FfiBuffer", "register", "handler", "registered"]
