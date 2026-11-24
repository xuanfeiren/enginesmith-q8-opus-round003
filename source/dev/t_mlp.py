"""dev: numeric check of exp.mlp_dec against numpy (bf16 rounding points as HF)."""
import os, sys, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, "../kernels"), HERE]
art = tempfile.mkdtemp(prefix="t-", dir="/tmp/q8x"); os.environ["NKI_ARTIFACTS_DIR"] = art
import numpy as np, ml_dtypes, exp
bf = ml_dtypes.bfloat16; H, I = 4096, 12288
r = np.random.default_rng(1)
x = r.standard_normal((1, 1, H)).astype(bf); a = (r.standard_normal((1, 1, H)) * 0.5).astype(bf)
g2 = (1 + 0.1 * r.standard_normal((1, H))).astype(bf)
wg = (r.standard_normal((H, I)) * 0.02).astype(bf); wu = (r.standard_normal((H, I)) * 0.02).astype(bf)
wd = (r.standard_normal((I, H)) * 0.02).astype(bf)
K = os.environ.get("K", "mlp_dec")
g2c = np.ascontiguousarray(g2.reshape(32, 128).T)
out = (exp.mlp_dec2[2](x, a, g2c, wg, wu, wd) if K == "mlp_dec2" else exp.mlp_dec[2](x, a, g2, wg, wu, wd))
if isinstance(out, (tuple, list)): out, dh, dhn, dg, dp = out
out = out.astype(np.float32).reshape(-1)
shutil.copy(art + "/kernel.neff", f"/tmp/q8x/{K}.neff")
f = lambda t: t.astype(np.float32)
h = (f(x) + f(a)).astype(bf).reshape(-1)
hn = (f(h) / np.sqrt((f(h) ** 2).mean() + 1e-6)).astype(bf)
hn = (f(hn) * f(g2[0])).astype(bf)
g = (f(hn) @ f(wg)).astype(bf); u = (f(hn) @ f(wu)).astype(bf)
s = (f(g) / (1 + np.exp(-f(g)))).astype(bf); p = (f(s) * f(u)).astype(bf)
d = (f(p) @ f(wd)).astype(bf)
ref = f((f(h) + f(d)).astype(bf))
if os.environ.get("KDEBUG"):
    col = lambda v: v.reshape(-1, 128).T
    print("h", np.abs(f(dh) - f(col(h))).max(), "hn", np.abs(f(dhn) - f(col(hn))).max())
    gg = np.concatenate([f(dg[0]).T.reshape(-1), f(dg[1]).T.reshape(-1)])
    print("p", np.abs(gg - f(p)).max(), np.abs(f(p)).max())
    pp = (f(dp[0]) + f(dp[1])).T.reshape(-1)
    print("down", np.abs(pp - f(p) @ f(wd)).max())
err = np.abs(out - ref)
print("max abs err", err.max(), "mean", err.mean(), "ref absmax", np.abs(ref).max(), "frac exact", (err == 0).mean())
if os.environ.get("KDEBUG"):
    pr = f(p).reshape(2, 48, 128)
    for c in range(2):
        e = np.abs(f(dg[c]).T - pr[c]).max(axis=1)
        print("core", c, "err per tile", np.round(e[:12], 2), "...")
    # is the kernel's p a permutation/scale of ref? check correlation of tile 0
    k0 = f(dg[0])[:, 0]; 
    print("corr tile0", np.corrcoef(k0, pr[0, 0])[0, 1])
    print("down from kernel p", np.abs(pp - gg @ f(wd)).max())
