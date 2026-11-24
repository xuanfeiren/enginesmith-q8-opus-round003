"""dev only: compile (and run once, on the experiment core) one NKI kernel through NKI's documented route.

usage: kc.py MODULE KERNEL SPEC_NAME OUT.neff
  MODULE: python module in kernels/ or dev/; SPEC_NAME: a name from build_neffs.neff_specs() or from MODULE.SPECS
Runs inside the GPU lock. The kernel runs with random inputs (zeros for int/u8)."""
import os, sys, time, shutil, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "../kernels"), HERE]
mod, kname, spec, out = sys.argv[1:5]
art = tempfile.mkdtemp(prefix="kc-", dir="/tmp/q8x")
os.environ["NKI_ARTIFACTS_DIR"] = art
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "34")
import numpy as np, ml_dtypes, importlib
import build_neffs
m = importlib.import_module(mod)
specs = dict(build_neffs.neff_specs()); specs.update(getattr(m, "SPECS", {}))
_, args = specs[spec]
rng = np.random.default_rng(0)
def mk(shape, dt):
    if dt == "bfloat16": return (rng.standard_normal(shape) * 0.02).astype(ml_dtypes.bfloat16)
    if dt == "float32": return (rng.standard_normal(shape)).astype(np.float32)
    return np.zeros(shape, dtype=dt)
kw = {n: mk(s, d) for n, s, d in args}
t0 = time.time()
try:
    getattr(m, kname)[2](**kw)
    print("ran ok", flush=True)
except Exception as e:
    print("EXC", type(e).__name__, str(e)[-2000:], flush=True)
neff = os.path.join(art, "kernel.neff")
if os.path.exists(neff):
    shutil.copy(neff, out); print(f"wrote {out} in {time.time()-t0:.0f}s")
    shutil.rmtree(art, ignore_errors=True)
else:
    print("no neff; artifacts in", art); sys.exit(1)
