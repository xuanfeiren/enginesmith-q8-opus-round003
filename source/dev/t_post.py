"""dev: check exp.pre_post against numpy (HF rounding points)."""
import os, sys, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, "../kernels"), HERE]
art = tempfile.mkdtemp(prefix="t-", dir="/tmp/q8x"); os.environ["NKI_ARTIFACTS_DIR"] = art
import numpy as np, ml_dtypes, qwen3 as exp
bf = ml_dtypes.bfloat16; H, I, NQ, D = 4096, 12288, 32, 128
T = int(os.environ.get("T", 256))
r = np.random.default_rng(4)
f = lambda t: np.asarray(t).astype(np.float32); B = lambda t: np.asarray(t, dtype=np.float32).astype(bf)
x = B(r.standard_normal((1, T, H))); a = B(r.standard_normal((NQ, D, T)) * 0.5)
g2 = B(1 + 0.1 * r.standard_normal(H)); g2c = np.ascontiguousarray(g2.reshape(-1, 128).T)
wo = B(r.standard_normal((H, H)) * 0.02); wg = B(r.standard_normal((H, I)) * 0.02); wu = B(r.standard_normal((H, I)) * 0.02); wd = B(r.standard_normal((I, H)) * 0.02)
out = f(exp.pre_post[2](x, a, wo, g2c, wg, wu, wd)).reshape(T, H)
shutil.copy(art + "/kernel.neff", f"/tmp/q8x/pre_post_{T}.neff")
A = f(a).transpose(2, 0, 1).reshape(T, NQ * D)        # [T, heads*d]
o = B(A @ f(wo)); h = B(f(x[0]) + f(o))
hn = B(f(h) / np.sqrt((f(h) ** 2).mean(-1, keepdims=True) + 1e-6)); hn = B(f(hn) * f(g2))
g = B(f(hn) @ f(wg)); u = B(f(hn) @ f(wu)); s = B(f(g) / (1 + np.exp(-f(g)))); p = B(f(s) * f(u))
ref = f(B(f(h) + f(B(f(p) @ f(wd)))))
err = np.abs(out - ref)
print(f"T={T} max abs err", err.max(), "mean", err.mean(), "ref absmax", np.abs(ref).max(), "frac exact", (err == 0).mean())
