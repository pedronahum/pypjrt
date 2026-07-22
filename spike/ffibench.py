import ctypes, ctypes.util, time, statistics

libc = ctypes.CDLL("libc.so.6", use_errno=False)
# getpid is a trivial syscall-ish call; use it to measure ctypes dispatch overhead
f = libc.getpid
f.restype = ctypes.c_int
f.argtypes = []

def bench(fn, n=200000):
    ts=[]
    for _ in range(5):
        t0=time.perf_counter_ns()
        for _ in range(n): fn()
        t1=time.perf_counter_ns()
        ts.append((t1-t0)/n)
    return min(ts)

print(f"ctypes no-arg call:        {bench(f):8.1f} ns")

# with a struct pointer arg, closer to PJRT_Xxx(api, &args)
class Args(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_size_t), ("extension_start", ctypes.c_void_p),
                ("a", ctypes.c_int), ("b", ctypes.c_int)]
memcpy = libc.memcpy
memcpy.restype = ctypes.c_void_p
memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
src = Args(); dst = Args()
ps, pd = ctypes.byref(src), ctypes.byref(dst)
sz = ctypes.sizeof(Args)
print(f"ctypes ptr-arg call:       {bench(lambda: memcpy(pd, ps, sz)):8.1f} ns")

# pre-bound pointers (avoid byref each call)
pd2 = ctypes.cast(ctypes.pointer(dst), ctypes.c_void_p)
ps2 = ctypes.cast(ctypes.pointer(src), ctypes.c_void_p)
print(f"ctypes prebound-ptr call:  {bench(lambda: memcpy(pd2, ps2, sz)):8.1f} ns")

# python function call baseline
def noop(): pass
print(f"pure-python call baseline: {bench(noop):8.1f} ns")

# struct field write cost (args must be zeroed+filled per call)
def fill():
    src.struct_size = sz; src.extension_start = None; src.a = 1; src.b = 2
print(f"struct fill (4 fields):    {bench(fill):8.1f} ns")
