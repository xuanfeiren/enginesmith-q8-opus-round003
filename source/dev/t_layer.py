"""dev: numeric check of exp.layer_dec2 (one decode layer) against numpy with HF's bf16 rounding points."""
import os, sys, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, "../kernels"), HERE]
art = tempfile.mkdtemp(prefix="t-", dir="/tmp/q8x"); os.environ["NKI_ARTIFACTS_DIR"] = art
import numpy as np, ml_dtypes
def perm_qkv(w):
    """[..., in, out] q|k|v columns -> per-core groups (the engine's layout)"""
    q, k, v = w[..., :4096], w[..., 4096:5120], w[..., 5120:]
    return np.ascontiguousarray(np.concatenate([q[..., :2048], k[..., :512], v[..., :512], q[..., 2048:], k[..., 512:], v[..., 512:]], -1))
import qwen3 as exp
bf = ml_dtypes.bfloat16; H, I, NQ, NKV, D, SMAX = 4096, 12288, 32, 8, 128, 4608
NL = int(os.environ.get("NL", 1)); S = int(os.environ.get("S", 512)); POS = int(os.environ.get("POS", 300))
r = np.random.default_rng(2)
f = lambda t: np.asarray(t).astype(np.float32)
B = lambda t: np.asarray(t, dtype=np.float32).astype(bf)
x = B(r.standard_normal((1, 1, H)))
g1s = B(1 + 0.1 * r.standard_normal((NL, H))); g2s = B(1 + 0.1 * r.standard_normal((NL, H)))
qns = B(1 + 0.1 * r.standard_normal((NL, D))); kns = B(1 + 0.1 * r.standard_normal((NL, D)))
wqkvs = B(r.standard_normal((NL, H, (NQ + 2 * NKV) * D)) * 0.02); wos = B(r.standard_normal((NL, NQ * D, H)) * 0.02)
wgs = B(r.standard_normal((NL, H, I)) * 0.02); wus = B(r.standard_normal((NL, H, I)) * 0.02); wds = B(r.standard_normal((NL, I, H)) * 0.02)
kcs = B(r.standard_normal((NL, NKV, D, SMAX)) * 2); vcs = B(r.standard_normal((NL, NKV, SMAX, D)))
inv = (1.0 / (1e6 ** (np.arange(0, D, 2, dtype=np.float64) / D))).astype(np.float32)
ang = np.arange(SMAX, dtype=np.float32)[:, None] * inv[None, :]
rcos = B(np.concatenate([np.cos(ang), np.cos(ang)], 1)); rsin = B(np.concatenate([np.sin(ang), np.sin(ang)], 1))
rot = np.zeros((D, D), np.float32)
for m in range(D):
    if m < 64: rot[m + 64, m] = -1
    else: rot[m - 64, m] = 1
rot = B(rot)
col = lambda v: np.ascontiguousarray(np.asarray(v).reshape(-1, 128).T)
pos = np.array([[POS]], np.int32); posf = np.array([[POS]], np.float32)
colL = lambda g: np.stack([col(v) for v in g])
res = exp.layer_dec[2](x, colL(g1s), perm_qkv(wqkvs), qns.reshape(NL, D, 1), kns.reshape(NL, D, 1), rcos, rsin, rot, kcs.copy(), vcs.copy(),
                        pos, posf, wos, colL(g2s), wgs, wus, wds, np.zeros((1, S), np.uint8))
shutil.copy(art + "/kernel.neff", f"/tmp/q8x/layer_dec_{NL}_{S}.neff")
if isinstance(res, (tuple, list)): out, dq, da, dh = res
else: out = res
out = f(out).reshape(-1)
# reference
def rms(v, g):
    vn = B(f(v) / np.sqrt((f(v) ** 2).mean(-1, keepdims=True) + 1e-6)); return B(f(vn) * f(g))
xv = x.reshape(-1)
for L in range(NL):
  g1, g2, qn, kn, wqkv, wo, wg, wu, wd, kc, vc = g1s[L], g2s[L], qns[L], kns[L], wqkvs[L], wos[L], wgs[L], wus[L], wds[L], kcs[L], vcs[L]
  xn = rms(xv, g1)
  qkv = B(f(xn) @ f(wqkv)); q = qkv[:NQ * D].reshape(NQ, D); k = qkv[NQ * D:(NQ + NKV) * D].reshape(NKV, D); v = qkv[(NQ + NKV) * D:].reshape(NKV, D)
  qh = rms(q, qn); kh = rms(k, kn)
  rh = lambda t: np.concatenate([-f(t)[:, 64:], f(t)[:, :64]], 1)
  cs, sn = f(rcos[POS]), f(rsin[POS])
  qe = B(f(B(f(qh) * cs)) + f(B(rh(qh) * sn))); ke = B(f(B(f(kh) * cs)) + f(B(rh(kh) * sn)))
  K = f(kc).copy(); K[:, :, POS] = f(ke); V = f(vc).copy(); V[:, POS, :] = f(v)
  att = np.zeros((NQ, D), np.float32)
  for hq in range(NQ):
      hk = hq // 4
      s = (f(qe[hq]) @ K[hk, :, :S]) / np.sqrt(D)
      s[POS + 1:] = -np.inf
      p = np.exp(s - s.max()); p = B(p / p.sum())
      att[hq] = f(B(f(p) @ V[hk, :S]))
  o = B(f(B(att.reshape(-1))) @ f(wo)); h = B(f(xv) + f(o))
  hn = rms(h, g2)
  g = B(f(hn) @ f(wg)); u = B(f(hn) @ f(wu)); sl = B(f(g) / (1 + np.exp(-f(g)))); pm = B(f(sl) * f(u))
  dd = B(f(pm) @ f(wd)); ref = f(B(f(h) + f(dd)))
  xv = ref.astype(bf)
if os.environ.get("KDEBUG"):
    for c in range(2):
        qk_k = f(dq[c]).T  # [24, 128]
        print("core", c, "q err", np.abs(qk_k[:16] - f(qe[16 * c:16 * c + 16])).max(), "k err", np.abs(qk_k[16:20] - f(ke[4 * c:4 * c + 4])).max(),
              "v err", np.abs(qk_k[20:24] - f(v[4 * c:4 * c + 4])).max(), "att err", np.abs(f(da[c]).T - att[16 * c:16 * c + 16]).max())
    print("h err", np.abs(f(dh).T.reshape(-1) - f(h)).max())
err = np.abs(out - ref)
print(f"NL={NL} S={S} pos={POS} max abs err", err.max(), "mean", err.mean(), "ref absmax", np.abs(ref).max(), "frac exact", (err == 0).mean())
