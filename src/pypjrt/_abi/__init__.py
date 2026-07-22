"""ABI module selection.

Generated modules are versioned data. A plugin reports its own
``PJRT_Api_Version``; we pick the matching generated module, or the nearest
older one, or refuse. That is a *negotiation* -- an assertion would be the
common failure mode: reinterpreting ``PJRT_Api`` at some fixed size and never
reading ``struct_size`` or the version at all.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from types import ModuleType

_PAT = re.compile(r"^pjrt_(\d+)_(\d+)$")


def available() -> list[tuple[int, int]]:
    """Every generated ABI version present, newest first."""
    out = []
    for m in pkgutil.iter_modules(__path__):
        if (hit := _PAT.match(m.name)):
            out.append((int(hit.group(1)), int(hit.group(2))))
    return sorted(out, reverse=True)


def load(major: int, minor: int) -> tuple[ModuleType, bool]:
    """Select an ABI module for a plugin reporting ``(major, minor)``.

    Returns ``(module, exact)``. Raises on an unusable major version.

    Forward compatibility runs one way: a caller built against *newer* headers
    passes larger ``struct_size`` values, and an older plugin reads only the
    prefix it knows. The reverse is not safe, so we never select a module newer
    than the plugin unless nothing older exists.
    """
    have = available()
    if not have:
        raise RuntimeError("no generated ABI modules; run tools/gen_abi.py")

    if not any(M == major for M, _ in have):
        raise IncompatiblePlugin(
            f"plugin reports PJRT API major {major}; this build only knows "
            f"major(s) {sorted({M for M, _ in have})}. A major bump is an "
            f"ABI break -- re-vendor headers and regenerate."
        )

    same_major = [v for v in have if v[0] == major]
    for v in same_major:  # newest first
        if v[1] <= minor:
            return importlib.import_module(f"{__name__}.pjrt_{v[0]}_{v[1]}"), v[1] == minor
    oldest = same_major[-1]
    return importlib.import_module(f"{__name__}.pjrt_{oldest[0]}_{oldest[1]}"), False


def bootstrap_offsets() -> tuple[int, int, int]:
    """Byte offsets of ``PJRT_Api``'s header fields, for use *before* an ABI
    module has been selected.

    Chicken-and-egg: we must read ``pjrt_api_version`` to choose a module, but
    the offsets live in a module. The header prefix (struct_size,
    extension_start, pjrt_api_version{struct_size, ext, major, minor}) is the
    bootstrap contract and has been stable across every PJRT version -- but we
    still take it from generated layout rather than hardcoding it.

    Returns ``(extension_start, major_version, minor_version)``.
    """
    have = available()
    if not have:
        raise RuntimeError("no generated ABI modules; run tools/gen_abi.py")
    m = importlib.import_module(f"{__name__}.pjrt_{have[0][0]}_{have[0][1]}")
    v_off = m.PJRT_Api.pjrt_api_version.offset
    return (m.PJRT_Api.extension_start.offset,
            v_off + m.PJRT_Api_Version.major_version.offset,
            v_off + m.PJRT_Api_Version.minor_version.offset)


class IncompatiblePlugin(RuntimeError):
    """The plugin's ABI cannot be spoken by any generated module."""
