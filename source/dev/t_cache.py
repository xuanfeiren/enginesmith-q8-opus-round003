"""dev: check exp.pre_cache2: kernel writes caches in place; we read them back through a second kernel? No: nki
standalone returns only outputs, so this test compiles a wrapper that copies the caches to outputs."""
import os, sys, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, "../kernels"), HERE]
art = tempfile.mkdtemp(prefix="t-", dir="/tmp/q8x"); os.environ["NKI_ARTIFACTS_DIR"] = art
import numpy as np, ml_dtypes, exp
import nki, nki.language as nl, nki.isa as nisa
bf = ml_dtypes.bfloat16
T = int(os.environ.get("T", 512)); SMAX = 4608; NKV, D = 8, 128
r = np.random.default_rng(3)
ks = [r.standard_normal((4, 1, T, D)).astype(bf) for _ in range(2)]
vs = [r.standard_normal((4, 1, T, D)).astype(bf) for _ in range(2)]
kc = np.zeros((NKV, D, SMAX), bf); vc = np.zeros((NKV, SMAX, D), bf)
res = exp.pre_cache2_t[2](ks[0], ks[1], vs[0], vs[1], kc, vc)
kco, vco = res
kref = np.concatenate([k[:, 0] for k in ks], 0)   # [8, T, D]
vref = np.concatenate([v[:, 0] for v in vs], 0)
print("T", T, "k err", np.abs(kco[:, :, :T].astype(np.float32) - kref.transpose(0, 2, 1).astype(np.float32)).max(),
      "v err", np.abs(vco[:, :T].astype(np.float32) - vref.astype(np.float32)).max(),
      "beyond T zero", np.abs(kco[:, :, T:].astype(np.float32)).max(), np.abs(vco[:, T:].astype(np.float32)).max())
shutil.copy(art + "/kernel.neff", f"/tmp/q8x/pre_cache2_{T}.neff")
