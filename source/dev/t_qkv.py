"""dev: check exp.pre_qkv2 (+ pre_attn2 when ATT=1) against numpy."""
import os, sys, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, "../kernels"), HERE]
art = tempfile.mkdtemp(prefix="t-", dir="/tmp/q8x"); os.environ["NKI_ARTIFACTS_DIR"] = art
import numpy as np, ml_dtypes, qwen3 as exp
bf = ml_dtypes.bfloat16; H, NQ, NKV, D, SMAX = 4096, 32, 8, 128, 4608
T = int(os.environ.get("T", 256))
r = np.random.default_rng(5)
f = lambda t: np.asarray(t).astype(np.float32); B = lambda t: np.asarray(t, dtype=np.float32).astype(bf)
x = B(r.standard_normal((1, T, H)))
g1 = B(1 + 0.1 * r.standard_normal(H)); g1c = np.ascontiguousarray(g1.reshape(-1, 128).T)
qn = B(1 + 0.1 * r.standard_normal(D)); kn = B(1 + 0.1 * r.standard_normal(D))
wqkv = B(r.standard_normal((H, (NQ + 2 * NKV) * D)) * 0.02)
inv = (1.0 / (1e6 ** (np.arange(0, D, 2, dtype=np.float64) / D))).astype(np.float32)
ang = np.arange(SMAX, dtype=np.float32)[:, None] * inv[None, :]
rc = B(np.concatenate([np.cos(ang), np.cos(ang)], 1)); rs = B(np.concatenate([np.sin(ang), np.sin(ang)], 1))
rot = np.zeros((D, D), np.float32)
for m in range(D): rot[m + 64 if m < 64 else m - 64, m] = -1 if m < 64 else 1
kc = np.zeros((NKV, D, SMAX), bf); vc = np.zeros((NKV, SMAX, D), bf)
qo, ko, vo = exp.pre_qkv2[2](x, g1c, wqkv, qn.reshape(D, 1), kn.reshape(D, 1), np.ascontiguousarray(rc.T), np.ascontiguousarray(rs.T), B(rot), kc, vc)
shutil.copy(art + "/kernel.neff", f"/tmp/q8x/pre_qkv2_{T}.neff")
def rms(v, g):
    vn = B(f(v) / np.sqrt((f(v) ** 2).mean(-1, keepdims=True) + 1e-6)); return B(f(vn) * f(g))
xn = rms(x[0], g1); qkv = B(f(xn) @ f(wqkv))
q = qkv[:, :NQ * D].reshape(T, NQ, D); k = qkv[:, NQ * D:(NQ + NKV) * D].reshape(T, NKV, D); v = qkv[:, (NQ + NKV) * D:].reshape(T, NKV, D)
qh = rms(q, qn); kh = rms(k, kn)
rh = lambda t: np.concatenate([-f(t)[..., 64:], f(t)[..., :64]], -1)
cs, sn = f(rc[:T])[:, None, :], f(rs[:T])[:, None, :]
qe = B(f(B(f(qh) * cs)) + f(B(rh(qh) * sn))); ke = B(f(B(f(kh) * cs)) + f(B(rh(kh) * sn)))
print(f"T={T} q err", np.abs(f(qo) - f(qe).transpose(1, 2, 0)).max(), "k err", np.abs(f(ko) - f(ke).transpose(1, 2, 0)).max(),
      "v err", np.abs(f(vo) - f(v).transpose(1, 0, 2)).max(), "absmax q", np.abs(f(qe)).max(),
      "q exact", (f(qo) == f(qe).transpose(1, 2, 0)).mean())
if os.environ.get("ATT"):
    a0, a1 = exp.pre_attn2[2](qo, ko, vo)
    A = np.concatenate([f(a0), f(a1)], 0)   # [NQ, D, T]
    att = np.zeros((NQ, D, T), np.float32)
    for hq in range(NQ):
        s = f(qe[:, hq]) @ f(ke[:, hq // 4]).T / np.sqrt(D)
        s[np.triu_indices(T, 1)] = -np.inf
        p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        att[hq] = (p @ f(v[:, hq // 4])).T
    print("attn err", np.abs(A - att).max(), "absmax", np.abs(att).max())
