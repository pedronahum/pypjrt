"""A persistent compile cache.

Nobody binds ``PJRT_Executable_Serialize`` / ``DeserializeAndLoad``, so every
client re-pays full XLA compilation on every process start -- tens of seconds
for a Gemma-class model. An in-memory dict keyed on the *StableHLO text* is not
a cache either: it re-renders the text on every timed
iteration.

The key includes ``xla_version``: a serialized executable compiled against one
XLA can silently miscompute under another.
Where the plugin offers ``PJRT_Executable_Fingerprint`` we record it too, since
that is a key the plugin itself computed.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from .artifact import Artifact, ArtifactMismatch


def default_dir() -> Path:
    if env := os.environ.get("PYPJRT_CACHE_DIR"):
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "pypjrt" / "executables"


class CompileCache:
    """Disk-backed, content-addressed. Safe to share between processes."""

    def __init__(self, directory: str | Path | None = None):
        self.dir = Path(directory) if directory else default_dir()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(program: bytes, options: bytes, platform: str,
            xla_version: int | None, api_major: int) -> str:
        h = hashlib.sha256()
        for part in (program, b"\0", options, b"\0", platform.encode(), b"\0",
                     str(xla_version).encode(), b"\0", str(api_major).encode()):
            h.update(part)
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.pypjrta"

    def load(self, key: str) -> Artifact | None:
        p = self._path(key)
        if not p.exists():
            self.misses += 1
            return None
        try:
            a = Artifact.read(p)
        except (ArtifactMismatch, OSError, ValueError):
            # A corrupt or stale-format entry is a miss, never a crash.
            self.misses += 1
            try:
                p.unlink()
            except OSError:
                pass
            return None
        self.hits += 1
        return a

    def store(self, key: str, artifact: Artifact) -> Path:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Atomic within a directory, so concurrent writers cannot tear an entry.
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(artifact.to_bytes())
            os.replace(tmp, p)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return p

    def clear(self) -> int:
        n = 0
        if self.dir.exists():
            for p in self.dir.rglob("*.pypjrta"):
                p.unlink()
                n += 1
        return n

    def __repr__(self) -> str:
        return f"<CompileCache {self.dir} hits={self.hits} misses={self.misses}>"
