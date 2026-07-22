# pypjrt

[![ci](https://github.com/pedronahum/pypjrt/actions/workflows/ci.yml/badge.svg)](https://github.com/pedronahum/pypjrt/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/pedronahum/pypjrt)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)](pyproject.toml)

**Run compiled ML programs on CPUs, GPUs and TPUs from Python — with zero dependencies.**

pypjrt is a pure-Python client for [PJRT](https://openxla.org/xla/pjrt), the hardware-agnostic
runtime interface underneath JAX, TensorFlow and PyTorch/XLA. It loads a vendor's PJRT plugin,
compiles StableHLO, moves buffers, runs programs across devices, and lets you register custom
GPU kernels — all through `ctypes`, with **nothing in `install_requires`**.

```python
import pypjrt

with pypjrt.Client.create() as client, client.devices() as devices:
    exe = client.compile(stablehlo_text)
    a = client.buffer_from_host(xs, 11, [8], devices[0])
    (out,) = exe(a)
    print(out.to_host())
```

---

## Why this exists

PJRT's premise is that **frameworks call one API and hardware vendors implement it**, so neither
has to know about the other. That works well — in C++. In Python there is exactly one PJRT client:
the one buried inside `jaxlib` as a ~500 MB compiled `xla_extension.so`.

So today, if you want to run a compiled program on an accelerator from Python, your options are to
adopt an entire framework or to write C++. There is no small, reusable way to just *call PJRT*.

pypjrt is that missing piece. The client half of PJRT is common infrastructure — it just happened
to ship inside a framework. Unbundling it means the cost of talking to accelerator hardware drops
from "take a 500 MB dependency" to "install a pure-Python wheel".

## What you get

**No dependencies, no build step.** The core imports nothing outside the standard library. No
compiler on your machine, no per-platform wheels, no linking against XLA. About 5,600 lines of
hand-written Python plus a generated ABI module.

**Genuinely hardware-agnostic.** The same code path drives an XLA CPU plugin and an NVIDIA GB10,
differing only by a file path. Capability differences are *negotiated*, never assumed: every
extension goes through a probe that returns `None` when absent, and version skew is handled by
selecting a matching ABI rather than asserting. One build currently speaks to plugins reporting
PJRT 0.104 and 0.108 against headers pinned at 0.114.

**Composes instead of competing.** Zero-copy DLPack in **both** directions — export a buffer to
torch/jax/numpy, or adopt theirs without a host copy. pypjrt is designed to sit alongside the
framework you already use, not to replace it.

**Custom kernels, from Python.** Register a Python function as an XLA FFI handler and XLA will
call it inside a compiled program, with real device pointers and its own stream. Or compile Triton
IR through the plugin itself — no `triton` package, no subprocess.

**A persistent compile cache.** Serialize compiled executables and skip XLA compilation on
restart. Measured on an NVIDIA GB10: **6.2 s cold → 66 ms warm**. Cache keys include the plugin's
XLA version, because an executable compiled against one XLA can silently miscompute under another.

**Diagnostics that tell you what went wrong.** Allocation failures carry the device's own
allocator state instead of a bare `RESOURCE_EXHAUSTED`:

```
Out of memory while trying to allocate 64.00MiB.
  device memory: device 0: bytes_in_use=2.4 GiB, bytes_limit=2.4 GiB,
    largest_free_block_bytes=0 B, peak_bytes_in_use=2.4 GiB, num_allocs=204
  client create-options: {'preallocate': False, 'memory_fraction': 0.02}
  hint: this client is already capped; the workload needs more memory than it was given
```

**A conformance harness for plugin authors.** Today a vendor validates a new PJRT plugin by
running JAX, which conflates plugin bugs with framework bugs. `pypjrt.conform` is a thin,
scriptable, dependency-free second opinion.

## Install

```sh
pip install git+https://github.com/pedronahum/pypjrt      # pre-1.0; not on PyPI yet
```

You also need a **PJRT plugin** — the vendor's shared library. The easiest source is a JAX plugin
wheel, which pypjrt discovers automatically:

```sh
pip install jax-cuda12-plugin     # or jax-cuda13-plugin, libtpu, ...
```

Otherwise point at one explicitly:

```sh
export PYPJRT_PLUGIN=/path/to/pjrt_plugin.so    # or TPU_LIBRARY_PATH for libtpu
```

Optional extras: `pypjrt[numpy]` for array interop, `pypjrt[jax]` if you want JAX to *produce*
the StableHLO you run.

## Usage

Ten runnable examples in [`examples/`](examples/) go from "hello" to a training loop with custom
kernels; each is standalone and takes an optional plugin path. `python examples/01_hello.py` uses
nothing but the standard library.

### Compile and run

pypjrt consumes StableHLO from any producer — hand-written, or lowered by JAX:

```python
import jax, jax.numpy as jnp, numpy as np, pypjrt

f = lambda a, b: jnp.tanh(a * b + 1.0)
x = np.arange(8, dtype=np.float32)
mlir = jax.jit(f).lower(x, x).as_text()          # any producer will do

with pypjrt.Client.create() as client, client.devices() as devices:
    exe = client.compile(mlir)
    a = client.buffer_from_host(x, 11, [8], devices[0])
    (out,) = exe(a, a)
    result = np.empty(8, np.float32)
    out.to_host(result)
```

Portable artifacts (versioned StableHLO bytecode) work too — `compile()` takes `str` or `bytes`,
and `plugin.stablehlo_target()` tells a producer which version to serialize to.

### Zero-copy interop

```python
import numpy as np
arr = np.from_dlpack(buffer)                 # shares device memory, no copy
adopted = client.from_dlpack(jax_array)      # and back the other way
```

### Multiple devices

`Executable` is device-list shaped, because that is the shape of the underlying API:

```python
opts = pypjrt.CompileOptions(num_partitions=2, use_spmd_partitioning=True)
exe = client.compile(sharded_mlir, options=opts)
outputs = exe.execute_sharded([[shard_a], [shard_b]])
```

### Named programs and donation

```python
from pypjrt.session import Session, Slot

with Session(client) as s:
    step = s.program(training_step, [Slot("theta"), Slot("grad")], outputs=["theta"])
    s.bind_many(lazy_weight_loaders)      # only slots the program names are uploaded
    for _ in range(steps):
        s.feed_back(step())               # output becomes the next input, no host copy
```

### Custom kernels

```python
import pypjrt.ffi as ffi, pypjrt.cuda as cuda

@ffi.handler(plugin, "my_kernel")
def my_kernel(call):
    (x,), (y,) = call.args, call.rets
    cuda.memcpy_dtod_async(y.data, x.data, x.nbytes, call.stream())
```

Then reference it from StableHLO as
`stablehlo.custom_call @my_kernel(...) {api_version = 4 : i32}`.

Device-specific code lives in clearly-named modules — `pypjrt.ffi`, `pypjrt.cuda`,
`pypjrt.triton` — so that trading away portability is visible at the import. The core never names
a device.

### Checking a plugin

```sh
python -m pypjrt.conform /path/to/plugin.so --json report.json
python -m pypjrt.conform --diff cpu.json gpu.json
```

Capabilities a plugin lacks are reported as *unsupported*, not as failures — a harness that can't
tell "absent" from "broken" is no use to a vendor. Only real defects set a non-zero exit code.

## Scope

pypjrt is **everything below StableHLO**, and deliberately nothing above it.

**In scope:** plugin loading and ABI negotiation, compilation, buffers and memory spaces, single-
and multi-device execution, donation, AOT artifacts and caching, XLA FFI custom calls, DLPack,
async completion, and multi-process rendezvous.

**Out of scope, permanently:** emitting StableHLO, automatic differentiation, sharding
*propagation*, graph optimisation, kernel authoring DSLs, and model libraries. Those already exist
in Python and are better there. Keeping the line sharp is what makes pypjrt a runtime that
frameworks can build on rather than another framework.

## Compatibility

| | |
|---|---|
| Python | 3.10+ |
| PJRT C API | headers pinned at 0.114; older plugins negotiated automatically |
| XLA FFI | 0.3 |
| Verified against | XLA CPU plugin (0.108) and NVIDIA CUDA plugin (0.104) on GB10 / aarch64 |
| Platforms | Linux. macOS untested, Windows unsupported |

132 of 138 core PJRT entry points are bound; the remainder are recorded decisions rather than
omissions, and a test pins the list so it cannot drift.

## Status

Pre-1.0 and honest about it. The API is settled in shape — in particular `Event` is future-shaped
and `Executable` is device-list-shaped, both fixed early so they would not need breaking later —
but names may still move before 1.0.

250 tests run across three tiers: host-only (no plugin required), CPU-plugin, and GPU. The suite
treats a skipped test in a required tier as a failure, type-checks the package with pyright,
regenerates the ABI from the pinned headers and fails on any diff, and keeps a directory of
snippets that **must not** type-check.

Two capabilities remain unverified for want of hardware rather than code: compiling for a device
that is not present, and a collective across processes. Both need more than one accelerator.

## Development

```sh
git clone https://github.com/pedronahum/pypjrt && cd pypjrt
uv venv .venv && uv pip install --python .venv/bin/python -e '.[dev]'

./local-ci.sh                                    # everything, on a box with plugins
PYPJRT_GPU_PLUGIN=/path/to/gpu.so ./local-ci.sh  # including the GPU tier
```

The ABI module is generated from the vendored OpenXLA headers with libclang and committed —
no struct offset or enum value is ever written by hand:

```sh
python tools/gen_abi.py
```

## Contributing

Issues and pull requests are welcome at
[github.com/pedronahum/pypjrt](https://github.com/pedronahum/pypjrt). Reports from a plugin this
has never been run against are especially useful — attach the output of
`python -m pypjrt.conform /path/to/plugin.so -v`, which is a complete, dependency-free description
of what that plugin does and does not offer.

Run `./local-ci.sh` before opening a pull request; it runs everything CI runs plus the tiers that
need real hardware.

## License

Apache-2.0. Vendored OpenXLA headers under `vendor/` are Apache-2.0, copyright The OpenXLA
Authors.
