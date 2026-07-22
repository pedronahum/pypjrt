# Examples

Each file is standalone and runnable. They are ordered so that reading them in
sequence takes you from "hello" to a training loop with custom kernels.

```sh
pip install -e '.[dev]'          # or just: pip install pypjrt numpy
python examples/01_hello.py      # every example takes an optional plugin path
python examples/01_hello.py /path/to/pjrt_plugin.so
```

Without an argument each example finds a plugin the same way the library does:
`$PYPJRT_PLUGIN`, then an installed JAX plugin wheel, then a short search path.

| | | needs |
|---|---|---|
| [`01_hello.py`](01_hello.py) | Compile a program, run it, read the result. **Standard library only** — no numpy, no jax. | any plugin |
| [`02_plugin_and_devices.py`](02_plugin_and_devices.py) | What you are talking to: ABI negotiation, extensions, devices, memory spaces, allocator state. | any plugin |
| [`03_producers_and_artifacts.py`](03_producers_and_artifacts.py) | Where StableHLO comes from — hand-written, lowered by JAX, or a versioned portable artifact — and that all three agree. | numpy, jax |
| [`04_interop_dlpack.py`](04_interop_dlpack.py) | Zero-copy sharing in both directions, and why some producers are refused. | numpy, jax |
| [`05_multi_device.py`](05_multi_device.py) | Replicas, SPMD partitions and a collective, checked against the single-device answer. | ≥2 devices |
| [`06_training_loop.py`](06_training_loop.py) | Named slots, lazy weight loading, donation and output-to-input feedback. Resumes from its own checkpoint. | any plugin |
| [`07_artifacts_and_cache.py`](07_artifacts_and_cache.py) | Skip compilation on restart; artifact guards that refuse a program which would not run here. | any plugin |
| [`08_custom_kernel.py`](08_custom_kernel.py) | A Python function called by XLA inside a compiled program, checked against a decompose oracle. | FFI extension |
| [`09_gpu_kernels.py`](09_gpu_kernels.py) | Triton IR compiled by the plugin, PTX loaded through the CUDA driver, a handler launching on XLA's stream. | NVIDIA GPU |
| [`10_async_and_futures.py`](10_async_and_futures.py) | Events as futures, a completion token minted before the work exists, and a tensor streamed to the device in chunks. | any plugin |

Run them all at once, which is also what CI does so they cannot rot:

```sh
python examples/run_all.py
python examples/run_all.py /path/to/gpu_plugin.so
```

Examples degrade rather than crash: one that needs two devices, or an extension
the plugin does not advertise, says so and exits cleanly.
