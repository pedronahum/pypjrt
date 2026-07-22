"""06 — A training loop that survives being killed.

Puts the pieces together: a Session with named inputs, weights bound lazily so
only what a program asks for is uploaded, buffer donation so the optimiser
updates in place, and output-to-input feedback so nothing round-trips through
the host between steps.

Run it twice -- the second run resumes from the checkpoint the first wrote.

    python examples/06_training_loop.py [plugin.so]
"""
import array
import json
import pathlib
import sys
import tempfile

import pypjrt
from pypjrt.session import Session, Slot

F32, N, STEPS = 11, 16, 10
CHECKPOINT = pathlib.Path(tempfile.gettempdir()) / "pypjrt_example_theta.json"

# `tf.aliasing_output = 0` marks %theta as donated: its device memory is reused
# for output 0 instead of allocating fresh. This is what jax.jit(donate_argnums)
# emits, and it is why a training step does not grow memory every iteration.
STEP = """
module @m {
  func.func public @main(%theta: tensor<16xf32> {tf.aliasing_output = 0 : i32},
                         %grad: tensor<16xf32>,
                         %lr: tensor<f32>) -> tensor<16xf32> {
    %b = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %d = stablehlo.multiply %b, %grad : tensor<16xf32>
    %0 = stablehlo.subtract %theta, %d : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}
"""


def main(plugin_path: str | None = None) -> int:
    plugin = pypjrt.Plugin(plugin_path)
    options = pypjrt.Client.GPU_DEFAULTS if plugin.is_gpu else None

    resumed = CHECKPOINT.exists()
    theta0 = (array.array("f", json.loads(CHECKPOINT.read_text())) if resumed
              else array.array("f", [1.0] * N))
    print("resuming from checkpoint" if resumed else "starting fresh")

    with pypjrt.Client.create(plugin, options=options) as client, Session(client) as session:
        # Names come from you: PJRT does not expose argument names, so a Session
        # records the schema rather than inventing one.
        step = session.program(
            STEP,
            [Slot("theta", F32, (N,)), Slot("grad", F32, (N,)), Slot("lr", F32, ())],
            outputs=["theta"],
        )

        # Lazy bindings: a loader runs only if a program actually names its slot.
        # With a real checkpoint this is what stops you uploading 9 GB to use 2.
        loaded = []

        def loader(name, values, dims):
            def load(device):
                loaded.append(name)
                return client.buffer_from_host(array.array("f", values), F32, dims, device)
            return load

        session.bind_many({
            "theta": loader("theta", theta0, [N]),
            "grad": loader("grad", [0.5] * N, [N]),
            "lr": loader("lr", [0.01], []),
            "unused": loader("unused", [0.0] * N, [N]),   # never referenced
        })
        print(f"bound but not yet on device : {session.pending}")

        for _ in range(STEPS):
            # Every input resolves from the registry; the output is installed
            # as the next step's `theta` without touching the host.
            session.feed_back(step())

        print(f"materialised during the run : {sorted(loaded)}  "
              f"(note 'unused' was never uploaded)")
        print(f"donations                   : {step.donate_alias_count} "
              f"of {STEPS} steps reused their input buffer")

        final = session._globals["theta"].to_host()
        values = array.array("f", final)
        print(f"theta[0] after {STEPS} steps    : {values[0]:.4f}")
        CHECKPOINT.write_text(json.dumps(list(values)))
        print(f"checkpoint written to       : {CHECKPOINT}")
        print("\nrun this again to resume from it "
              f"(delete {CHECKPOINT.name} to start over).")
        step.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
