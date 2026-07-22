"""Name-keyed programs over a slot registry.

Born from a real bug -- a 16-step decode loop that
spawned one compiler subprocess per step, paying ~18 s of artifact load
and a 15 GiB safetensors mmap *per step* to drown a millisecond kernel.

Four ideas, all of them below the StableHLO line and all of them cheap here:

1. **Name-keyed I/O**, validated against a declared schema -- not positional.
2. **Two-level slot resolution**: per-call inputs, then the session registry.
   A Gemma-class model registers 719 weight inputs once; a step passes only
   tokens / cache / position.
3. **Lazy materialisation**: a binding uploads on first reference and caches, so
   a 9 GB checkpoint only moves the slots a program actually names.
4. **Output buffers feed back as input buffers with zero host roundtrip** -- the
   invariant that makes autoregressive decode viable (~735 MiB KV cache).

PJRT does not expose argument *names* (they live in the producer's MLIR), so a
schema is declared by the caller. We record; we do not infer.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from . import errors
from .client import Buffer, Client, Device, Executable

_VOIDP = ctypes.c_void_p


@dataclass
class Slot:
    """A named input, resolved per call or from the registry."""
    name: str
    dtype: int | None = None
    dims: tuple[int, ...] | None = None

    def validate(self, buf: Buffer) -> None:
        if self.dtype is not None and buf.element_type != self.dtype:
            raise errors.InvalidArgument(
                f"slot {self.name!r}: expected dtype {self.dtype}, got {buf.element_type}")
        if self.dims is not None and tuple(buf.dimensions) != self.dims:
            raise errors.InvalidArgument(
                f"slot {self.name!r}: expected dims {self.dims}, got {tuple(buf.dimensions)}")


class Program:
    """An executable plus its named input/output schema.

    The per-call ctypes argument arrays are built **once** and reused, so a step
    writes only what changed. That is worth ~10 us/call even in a compiled
    language; it is worth proportionally more here, where filling an args struct
    costs more than the foreign call itself.
    """

    def __init__(self, session: "Session", executable: Executable,
                 inputs: Sequence[Slot], outputs: Sequence[str] | None = None):
        self.session = session
        self.executable = executable
        self.inputs = list(inputs)
        n_out = executable.num_outputs
        if outputs is None:
            outputs = [f"out{i}" for i in range(n_out)]
        if len(outputs) != n_out:
            raise errors.InvalidArgument(
                f"{len(outputs)} output name(s) for an executable with {n_out} output(s)")
        self.outputs = list(outputs)
        self._n_dev = executable.addressable_device_count
        self._ctx: dict[str, Any] = {}

    # -- the pre-allocated execute context ---------------------------------

    def _context(self):
        """Build the Args/outer/inner pointer chain once; reuse it per call."""
        if self._ctx:
            return self._ctx
        n_dev, n_arg, n_out = self._n_dev, len(self.inputs), len(self.outputs)
        rows = [(_VOIDP * n_arg)() for _ in range(n_dev)]
        arglists = (_VOIDP * n_dev)(*[ctypes.cast(r, _VOIDP) for r in rows])
        outrows = [(_VOIDP * n_out)() for _ in range(n_dev)]
        outlists = (_VOIDP * n_dev)(*[ctypes.cast(r, _VOIDP) for r in outrows])
        self._ctx = dict(rows=rows, arglists=arglists, outrows=outrows, outlists=outlists)
        return self._ctx

    # -- calling -----------------------------------------------------------

    def __call__(self, __device_args: Sequence[dict[str, Buffer]] | None = None,
                 **inputs: Buffer) -> dict[str, Buffer]:
        """Run with name-keyed inputs; returns name-keyed outputs.

        Single-device form: ``prog(x=buf, y=buf)``. Multi-device form: pass a
        list of per-device dicts.
        """
        per_device = __device_args if __device_args is not None else [inputs]
        if len(per_device) != self._n_dev:
            raise errors.InvalidArgument(
                f"program expects {self._n_dev} device(s), got {len(per_device)}")

        resolved: list[list[Buffer]] = []
        for d, given in enumerate(per_device):
            unknown = set(given) - {s.name for s in self.inputs}
            if unknown:
                raise errors.InvalidArgument(
                    f"unknown input(s) {sorted(unknown)}; "
                    f"this program takes {[s.name for s in self.inputs]}")
            row: list[Buffer] = []
            for slot in self.inputs:
                buf = given.get(slot.name)
                if buf is None:
                    buf = self.session.resolve(slot.name, device_index=d)
                if buf is None:
                    raise errors.InvalidArgument(
                        f"no buffer for input {slot.name!r}: pass it, or register it "
                        f"on the session with set_global()/bind()")
                slot.validate(buf)
                row.append(buf)
            resolved.append(row)

        outs = self.executable.execute_sharded(resolved)
        if self._n_dev == 1:
            return dict(zip(self.outputs, outs[0]))
        return [dict(zip(self.outputs, row)) for row in outs]  # type: ignore[return-value]

    @property
    def donate_alias_count(self) -> int:
        return self.executable.donate_alias_count

    def close(self) -> None:
        self.executable.close()


class Session:
    """A client plus a registry of named buffers.

    Register weights once; a step passes only what changes.
    """

    def __init__(self, client: Client):
        self.client = client
        self._globals: dict[str, Buffer] = {}
        self._lazy: dict[str, Callable[[Device], Buffer]] = {}
        self._devices: list[Device] | None = None

    # -- lifetime ----------------------------------------------------------

    def __enter__(self) -> "Session":
        self._devices = [Device(self.client._plugin, p)
                         for p in self.client._addressable()]
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        for b in self._globals.values():
            b.close()
        self._globals.clear()
        self._lazy.clear()
        for d in self._devices or ():
            d._invalidate()
        self._devices = None

    def device(self, index: int = 0) -> Device:
        if self._devices is None:
            raise errors.HandleClosed("Session is not open; use `with Session(client) as s:`")
        return self._devices[index]

    @property
    def device_count(self) -> int:
        return len(self._devices or ())

    # -- the registry ------------------------------------------------------

    def set_global(self, name: str, buffer: Buffer) -> None:
        """Register a materialised buffer under a name."""
        if (old := self._globals.pop(name, None)) is not None:
            old.close()
        self._globals[name] = buffer

    def bind(self, name: str, loader: Callable[[Device], Buffer]) -> None:
        """Register a *lazy* binding, materialised on first reference.

        The point of the design: a checkpoint with hundreds of tensors only
        uploads the slots a program actually names.
        """
        self._lazy[name] = loader

    def bind_many(self, loaders: dict[str, Callable[[Device], Buffer]]) -> None:
        self._lazy.update(loaders)

    def resolve(self, name: str, device_index: int = 0) -> Buffer | None:
        """Per-call inputs are checked by the caller; this is level two."""
        if (b := self._globals.get(name)) is not None:
            return b
        if (loader := self._lazy.pop(name, None)) is not None:
            b = loader(self.device(device_index))
            self._globals[name] = b
            return b
        return None

    @property
    def resident(self) -> list[str]:
        """Names currently materialised on device."""
        return sorted(self._globals)

    @property
    def pending(self) -> list[str]:
        """Names bound but not yet materialised."""
        return sorted(self._lazy)

    # -- programs ----------------------------------------------------------

    def program(self, source: str | bytes, inputs: Iterable[str | Slot], *,
                outputs: Sequence[str] | None = None,
                options=None, cache=None) -> Program:
        """Compile and give the arguments names."""
        slots = [s if isinstance(s, Slot) else Slot(s) for s in inputs]
        exe = self.client.compile(source, options=options, cache=cache)
        return Program(self, exe, slots, outputs)

    def program_from_executable(self, executable: Executable, inputs: Iterable[str | Slot],
                                *, outputs: Sequence[str] | None = None) -> Program:
        slots = [s if isinstance(s, Slot) else Slot(s) for s in inputs]
        return Program(self, executable, slots, outputs)

    def feed_back(self, mapping: dict[str, Buffer]) -> None:
        """Install output buffers as the next step's inputs, with no host copy.

        The load-bearing invariant for autoregressive decode: the KV cache never
        leaves the device.
        """
        for name, buf in mapping.items():
            self.set_global(name, buf)
