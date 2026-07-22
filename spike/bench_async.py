"""Sync vs async completion, with reps so the spread is visible."""
import sys, array, statistics, threading, time; sys.path.insert(0, 'src')
import pypjrt
from pypjrt.client import Event
F32, N = 11, 512
MUL = """
module @m {
  func.func public @main(%a: tensor<512xf32>, %b: tensor<512xf32>) -> tensor<512xf32> {
    %0 = stablehlo.multiply %a, %b : tensor<512xf32>
    return %0 : tensor<512xf32>
  }
}
"""
p = pypjrt.Plugin(sys.argv[1] if len(sys.argv) > 1 else None)
opts = {"preallocate": False, "memory_fraction": 0.05} if p.is_accelerator else None
REPS, ITER = 7, 200
with pypjrt.Client.create(p, options=opts) as c, c.devices() as devs:
    exe = c.compile(MUL)
    xs = array.array("f", [1.5] * N)
    a = c.buffer_from_host(xs, F32, [N], devs[0])
    b = c.buffer_from_host(xs, F32, [N], devs[0])
    def ready(buf):
        r = p.args("PJRT_Buffer_ReadyEvent_Args", buffer=buf.address)
        p.call("PJRT_Buffer_ReadyEvent", r); return Event(p, r.event)
    def sync_step():
        (o,) = exe(a, b); e = ready(o); e.result(); e.close(); o.close()
    def async_step():
        (o,) = exe(a, b); e = ready(o); f = threading.Event()
        e.add_done_callback(lambda m: f.set()); f.wait(5); e.close(); o.close()
    def rep(fn):
        for _ in range(30): fn()
        out = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            for _ in range(ITER): fn()
            out.append((time.perf_counter() - t0) / ITER * 1e6)
        return out
    s, y = rep(sync_step), rep(async_step)
    for name, v in (("result() [blocking]", s), ("add_done_callback  ", y)):
        print(f"  {name}: median {statistics.median(v):7.1f}  min {min(v):7.1f}  "
              f"max {max(v):7.1f}  spread {max(v)-min(v):6.1f} us")
    d = statistics.median(y) - statistics.median(s)
    spread = max(max(s)-min(s), max(y)-min(y))
    print(f"  delta(median) = {d:+.1f} us; run-to-run spread = {spread:.1f} us -> "
          f"{'INCONCLUSIVE (delta within noise)' if abs(d) < spread else 'significant'}")
    a.close(); b.close(); exe.close()
