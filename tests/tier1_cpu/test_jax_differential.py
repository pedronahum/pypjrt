"""Tier 1: the differential oracle -- jax produces StableHLO, pypjrt consumes it.

This is the structural advantage no source repo had: jaxlib is a reference PJRT
client on the same machine using the same plugin, so every behavioural question
becomes a two-line differential test. A non-Python client has to shell out to a
JAX venv and compare pre-recorded values; we compare live.

It is also an end-to-end proof of the boundary: pypjrt consumes
StableHLO from a producer it knows nothing about.
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import pytest, pypjrt
from pypjrt.compile_options import CompileOptions

jax = pytest.importorskip("jax", reason="the jax differential oracle needs the [jax] extra")
np = pytest.importorskip("numpy")
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

pytestmark = pytest.mark.tier1
F32 = 11


def _run(client, mlir, per_device_args, options=None):
    with client.devices() as devs:
        exe = client.compile(mlir, options=options)
        rows = [[client.buffer_from_host(a, F32, list(a.shape), devs[d])
                 for a in args] for d, args in enumerate(per_device_args)]
        outs = exe.execute_sharded(rows)
        res = []
        for row in outs:
            host = np.empty(row[0].dimensions, dtype=np.float32)
            row[0].to_host(host.data.cast("B"))
            res.append(host)
            row[0].close()
        for r in rows:
            for b in r:
                b.close()
        exe.close()
        return res


@pytest.fixture(scope="module")
def client(cpu_plugin_path):
    c = pypjrt.Client.create(pypjrt.Plugin(cpu_plugin_path))
    yield c
    c.close()


def test_single_device_matches_jax_byte_for_byte(client):
    f = lambda a, b: jnp.tanh(a * b + 1.0)
    x = np.arange(1, 9, dtype=np.float32)
    y = (x * 0.5).astype(np.float32)
    mlir = jax.jit(f).lower(x, y).as_text()
    want = np.asarray(jax.jit(f)(x, y))
    got, = _run(client, mlir, [[x, y]])
    assert got.tobytes() == want.tobytes()


def test_sharded_matches_jax_byte_for_byte(client):
    """jax lowers a NamedSharding program; pypjrt runs it across 4 devices."""
    if client.device_count < 4 or len(jax.devices()) < 4:
        pytest.skip("need 4 devices on both sides")
    f = lambda a, b: jnp.tanh(a * b + 1.0)
    mesh = Mesh(np.array(jax.devices()[:4]), ("x",))
    sh = NamedSharding(mesh, P("x"))
    xs = np.arange(1, 17, dtype=np.float32)
    ys = (xs * 0.25).astype(np.float32)
    jf = jax.jit(f, in_shardings=(sh, sh), out_shardings=sh)
    mlir = jf.lower(xs, ys).as_text()
    want = np.asarray(jf(xs, ys))
    k = len(xs) // 4
    got = _run(client, mlir,
               [[xs[i*k:(i+1)*k], ys[i*k:(i+1)*k]] for i in range(4)],
               CompileOptions(num_partitions=4, use_spmd_partitioning=True,
                              use_shardy_partitioner=True))
    assert np.concatenate(got).tobytes() == want.tobytes()


@pytest.fixture(scope="module")
def shardy_module():
    """A jax-lowered module that genuinely targets Shardy.

    Only a *sharded* lowering emits `sdy.` -- a single-device one does not.
    """
    if len(jax.devices()) < 4:
        pytest.skip("need 4 jax devices to produce a Shardy module")
    f = lambda a, b: a * b
    mesh = Mesh(np.array(jax.devices()[:4]), ("x",))
    sh = NamedSharding(mesh, P("x"))
    xs = np.arange(1, 17, dtype=np.float32)
    mlir = jax.jit(f, in_shardings=(sh, sh), out_shardings=sh).lower(xs, xs).as_text()
    if "sdy." not in mlir:
        pytest.skip("this jax build does not lower to Shardy")
    return mlir, xs


def test_shardy_module_auto_matched_when_no_options_given(client, shardy_module):
    """jax >= 0.11 lowers targeting Shardy. With no explicit CompileOptions we
    match the *partitioner* to the producer, instead of failing deep inside the
    GSPMD partitioner.

    We do NOT infer the mesh size from the module -- that is the producer's
    knowledge, and reading it back out would mean parsing IR, which is above our
    line. So options=None yields a single-device executable, and
    XLA collapses the sharded program correctly (pinned below).
    """
    mlir, xs = shardy_module
    exe = client.compile(mlir)          # options=None -> Shardy auto-matched
    assert (exe.num_replicas, exe.num_partitions) == (1, 1)
    exe.close()


def test_collapsing_a_sharded_module_to_one_device_is_still_correct(client, shardy_module):
    """A 4-way-sharded module compiled with num_partitions=1 runs on the full
    tensor and agrees with jax byte-for-byte. Worth pinning: it means the
    conservative default cannot silently produce wrong numbers."""
    mlir, xs = shardy_module
    want = np.asarray(jax.jit(lambda a, b: a * b)(xs, xs))
    got, = _run(client, mlir, [[xs, xs]])
    assert got.tobytes() == want.tobytes()


def test_explicit_options_are_honoured_not_overridden(client, shardy_module):
    """Auto-matching applies only when the caller expressed no preference; an
    explicit CompileOptions is passed through as given, and the resulting
    partitioner mismatch is reported with an actionable hint."""
    mlir, _ = shardy_module
    with pytest.raises(pypjrt.errors.PjrtError, match="use_shardy_partitioner=True"):
        client.compile(mlir, options=CompileOptions(
            num_partitions=4, use_spmd_partitioning=True, use_shardy_partitioner=False))
