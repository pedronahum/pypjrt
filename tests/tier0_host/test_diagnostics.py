"""Tier 0: diagnostic formatting. No plugin needed."""
import pytest
from pypjrt.client import _OOM_KEYS, _fmt_bytes

pytestmark = pytest.mark.tier0


@pytest.mark.parametrize("n,want", [
    (0, "0 B"), (512, "512 B"), (1024, "1.0 KiB"),
    (1536, "1.5 KiB"), (2 << 30, "2.0 GiB"),
])
def test_fmt_bytes(n, want):
    assert _fmt_bytes(n) == want


def test_oom_keys_lead_with_the_actionable_three():
    """The M1 gate: the message must carry these three."""
    assert _OOM_KEYS[:3] == ("bytes_in_use", "bytes_limit", "largest_free_block_bytes")
