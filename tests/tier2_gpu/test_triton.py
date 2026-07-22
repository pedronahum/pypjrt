"""Tier 2: compile Triton IR through the plugin -- no triton package, no subprocess."""
import ctypes, pathlib, pytest, pypjrt
import pypjrt.triton as tri
from pypjrt import errors

pytestmark = pytest.mark.tier2
KERNEL = (pathlib.Path(__file__).resolve().parents[1] / "data" / "triton_double.mlir").read_text()
TIGHT = {"preallocate": False, "memory_fraction": 0.05}


@pytest.fixture(scope="module")
def plugin(gpu_plugin_path):
    p = pypjrt.Plugin(gpu_plugin_path)
    if not tri.available(p):
        pytest.skip("plugin does not advertise the Triton extension")
    return p


@pytest.fixture(scope="module")
def arch(plugin):
    with pypjrt.Client.create(plugin, options=TIGHT) as c, c.devices() as devs:
        return tri.arch_of(devs[0])


def test_arch_is_the_dotted_compute_capability(arch):
    """`sm_121a` and `sm_121` are both rejected; the plugin wants "12.1"."""
    assert "." in arch and not arch.startswith("sm_")


def test_compiles_triton_ir_to_ptx(plugin, arch):
    k = tri.compile(plugin, KERNEL, arch=arch)
    assert k.asm and k.smem_bytes >= 0
    assert b".version" in k.asm and b"NVPTX" in k.asm


def test_negative_control_invalid_ir(plugin, arch):
    with pytest.raises(errors.PjrtError, match="[Pp]arse"):
        tri.compile(plugin, "this is not triton ir", arch=arch)


def test_negative_control_bad_arch(plugin):
    with pytest.raises(errors.PjrtError, match="architecture"):
        tri.compile(plugin, KERNEL, arch="sm_121a")


def test_ptx_loads_and_resolves_through_the_cuda_driver(plugin, arch):
    """Closes the loop: plugin-compiled PTX is launchable, so a Triton kernel
    needs neither the triton package nor a Python subprocess."""
    cuda = pytest.importorskip("pypjrt.cuda")
    if not cuda.available():
        pytest.skip("libcuda.so.1 not loadable")
    k = tri.compile(plugin, KERNEL, arch=arch)
    lib = cuda.lib()
    mod = ctypes.c_void_p()
    assert lib.cuModuleLoadData(ctypes.byref(mod), k.asm) == 0
    fn = ctypes.c_void_p()
    assert lib.cuModuleGetFunction(ctypes.byref(fn), mod, b"double_kernel") == 0
    assert fn.value
