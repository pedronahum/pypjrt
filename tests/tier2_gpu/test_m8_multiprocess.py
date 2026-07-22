"""Tier 2: M8 -- two real processes rendezvous through a shared KV store."""
import json, pathlib, subprocess, sys, textwrap
import pytest, pypjrt

pytestmark = pytest.mark.tier2

_NODE = textwrap.dedent('''
    import sys, json; sys.path.insert(0, {src!r})
    import pypjrt
    from pypjrt.kv import FileStore
    node, nnodes, kvdir, plug, out = (int(sys.argv[1]), int(sys.argv[2]),
                                      sys.argv[3], sys.argv[4], sys.argv[5])
    store = FileStore(kvdir)
    p = pypjrt.Plugin(plug)
    opts = {{"preallocate": False, "memory_fraction": 0.05,
             "num_nodes": nnodes, "node_id": node}}
    with pypjrt.Client(p, options=opts, kv_store=store) as c:
        json.dump({{"process_index": c.process_index, "devices": c.device_count,
                    "kv": c.kv_calls, "puts": store.puts, "gets": store.gets}},
                  open(out, "w"))
''')


def _spawn(tmp_path, node, nnodes, kvdir, plugin):
    src = str(pathlib.Path(__file__).resolve().parents[2] / "src")
    script = tmp_path / f"node{node}.py"
    script.write_text(_NODE.format(src=src))
    out = tmp_path / f"node{node}.json"
    return subprocess.Popen(
        [sys.executable, str(script), str(node), str(nnodes), str(kvdir),
         str(plugin), str(out)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True), out


def test_m8_gate_two_processes_form_a_distributed_client(tmp_path, gpu_plugin_path):
    """The gate's rendezvous half: two processes, one shared store, distinct
    process indices, and XLA's real topology exchange running through Python
    callbacks. No source repo has this."""
    kv = tmp_path / "kv"
    kv.mkdir()
    procs = [_spawn(tmp_path, i, 2, kv, gpu_plugin_path) for i in range(2)]
    results = []
    try:
        for proc, out in procs:
            rc = proc.wait(timeout=300)
            assert rc == 0, f"node exited {rc}: {proc.stderr.read()[-700:]}"
            results.append(json.loads(out.read_text()))
    finally:
        for proc, _ in procs:
            if proc.poll() is None:
                proc.kill()

    assert sorted(r["process_index"] for r in results) == [0, 1]
    assert all(r["devices"] >= 1 for r in results)
    assert sum(r["kv"]["put"] for r in results) > 0, "no rendezvous traffic"
    assert sum(r["kv"]["get"] for r in results) > 0

    # XLA's own topology keys, written through our callbacks
    keys = {bytes.fromhex(p.stem).decode() for p in kv.glob("*.val")}
    assert any("global_topology" in k for k in keys), keys
    assert sum("local_topology" in k for k in keys) == 2, keys
