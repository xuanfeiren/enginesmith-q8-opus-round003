"""dev: chained check of the X-layout prefill (embed_x, pre_qkv3, pre_attn2, pre_post3, x_row) for one layer."""
import os, sys, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, "../kernels"), HERE]
import numpy as np, ml_dtypes
def perm_qkv(w):
    """[..., in, out] q|k|v columns -> per-core groups (the engine's layout)"""
    q, k, v = w[..., :4096], w[..., 4096:5120], w[..., 5120:]
    return np.ascontiguousarray(np.concatenate([q[..., :2048], k[..., :512], v[..., :512], q[..., 2048:], k[..., 512:], v[..., 512:]], -1))
bf = ml_dtypes.bfloat16; H, I, NQ, NKV, D, SMAX, V = 4096, 12288, 32, 8, 128, 4608, 2048
T = int(os.environ.get("T", 256)); P = int(os.environ.get("P", T - 3))
def run(kname, *args):
    art = tempfile.mkdtemp(prefix="x-", dir="/tmp/q8x"); os.environ["NKI_ARTIFACTS_DIR"] = art
    import qwen3
    r = getattr(qwen3, kname)[2](*args)
    shutil.copy(art + "/kernel.neff", f"/tmp/q8x/{kname}_{T}.neff")
    return r
r = np.random.default_rng(6)
f = lambda t: np.asarray(t).astype(np.float32); B = lambda t: np.asarray(t, dtype=np.float32).astype(bf)
emb = B(r.standard_normal((V, H))); toks = r.integers(0, V, (T, 1)).astype(np.int32)
g1 = B(1 + 0.1 * r.standard_normal(H)); g2 = B(1 + 0.1 * r.standard_normal(H))
col = lambda v: np.ascontiguousarray(np.asarray(v).reshape(-1, 128).T)
qn = B(1 + 0.1 * r.standard_normal(D)); kn = B(1 + 0.1 * r.standard_normal(D))
wqkv = B(r.standard_normal((H, (NQ + 2 * NKV) * D)) * 0.02); wo = B(r.standard_normal((H, H)) * 0.02)
wg = B(r.standard_normal((H, I)) * 0.02); wu = B(r.standard_normal((H, I)) * 0.02); wd = B(r.standard_normal((I, H)) * 0.02)
inv = (1.0 / (1e6 ** (np.arange(0, D, 2, dtype=np.float64) / D))).astype(np.float32)
ang = np.arange(SMAX, dtype=np.float32)[:, None] * inv[None, :]
rc = B(np.concatenate([np.cos(ang), np.cos(ang)], 1)); rs = B(np.concatenate([np.sin(ang), np.sin(ang)], 1))
rot = np.zeros((D, D), np.float32)
for m in range(D): rot[m + 64 if m < 64 else m - 64, m] = -1 if m < 64 else 1
kc = np.zeros((NKV, D, SMAX), bf); vc = np.zeros((NKV, SMAX, D), bf)
X = run("embed_x", toks, emb)
TB = min(T, 512); NB = T // TB
x = f(emb[toks[:, 0]])
Xr = f(X).transpose(0, 3, 2, 1).reshape(T, H)
print("embed_x err", np.abs(Xr - x).max())
S1 = lambda t: np.ascontiguousarray(np.asarray(t)[None])
rcT, rsT = np.ascontiguousarray(rc.T), np.ascontiguousarray(rs.T)
qo, ko, vo = run("pre_qkv3", X, S1(col(g1)), S1(perm_qkv(wqkv)), S1(qn.reshape(D, 1)), S1(kn.reshape(D, 1)), rcT, rsT, B(rot), S1(kc), S1(vc))
a0, a1 = run("pre_attn2", qo, ko, vo)
a = np.concatenate([a0, a1], 0)
X2 = run("pre_post3", X, a, S1(wo), S1(col(g2)), S1(wg), S1(wu), S1(wd))
b, t = divmod(P, TB)
row = run("x_row", X2, np.array([[b * 128 * (H // 128) * TB + t]], np.int32))
def rms(v, g):
    vn = B(f(v) / np.sqrt((f(v) ** 2).mean(-1, keepdims=True) + 1e-6)); return B(f(vn) * f(g))
xn = rms(x, g1); qkv = B(f(xn) @ f(wqkv))
q = qkv[:, :NQ * D].reshape(T, NQ, D); k = qkv[:, NQ * D:(NQ + NKV) * D].reshape(T, NKV, D); v = qkv[:, (NQ + NKV) * D:].reshape(T, NKV, D)
qh = rms(q, qn); kh = rms(k, kn)
rh = lambda t: np.concatenate([-f(t)[..., 64:], f(t)[..., :64]], -1)
cs, sn = f(rc[:T])[:, None, :], f(rs[:T])[:, None, :]
qe = B(f(B(f(qh) * cs)) + f(B(rh(qh) * sn))); ke = B(f(B(f(kh) * cs)) + f(B(rh(kh) * sn)))
print("q err", np.abs(f(qo) - f(qe).transpose(1, 2, 0)).max(), "k err", np.abs(f(ko) - f(ke).transpose(1, 2, 0)).max(), "v err", np.abs(f(vo) - f(v).transpose(1, 0, 2)).max())
att = np.zeros((T, NQ, D), np.float32)
for hq in range(NQ):
    s = f(qe[:, hq]) @ f(ke[:, hq // 4]).T / np.sqrt(D); s[np.triu_indices(T, 1)] = -np.inf
    p = np.exp(s - s.max(1, keepdims=True)); p = B(p / p.sum(1, keepdims=True)); att[:, hq] = f(B(f(p) @ f(v[:, hq // 4])))
o = B(B(att.reshape(T, -1)) @ f(wo)); h = B(f(x) + f(o)); hn = rms(h, g2)
g = B(f(hn) @ f(wg)); u = B(f(hn) @ f(wu)); sl = B(f(g) / (1 + np.exp(-f(g)))); pm = B(f(sl) * f(u))
ref = f(B(f(h) + f(B(f(pm) @ f(wd)))))
X2r = f(X2).transpose(0, 3, 2, 1).reshape(T, H)
err = np.abs(X2r - ref)
print(f"T={T} layer max abs err", err.max(), "mean", err.mean(), "frac exact", (err == 0).mean(), "| x_row err", np.abs(f(row).reshape(-1) - X2r[P]).max())
Xold = run("pre_post", B(x)[None], a, wo, col(g2), wg, wu, wd)
d2 = np.abs(f(Xold).reshape(T, H) - X2r)
print("post3 vs post: max", d2.max(), "frac exact", (d2 == 0).mean(), "| old vs ref max", np.abs(f(Xold).reshape(T, H) - ref).max(), "mean", np.abs(f(Xold).reshape(T, H) - ref).mean())
if os.environ.get("QKVA"):
    kc2 = np.zeros((NKV, D, SMAX), bf); vc2 = np.zeros((NKV, SMAX, D), bf)
    a2 = run("pre_qkva", X, S1(col(g1)), S1(perm_qkv(wqkv)), S1(qn.reshape(D, 1)), S1(kn.reshape(D, 1)), rcT, rsT, B(rot), S1(kc2), S1(vc2))
    attT = att.transpose(1, 2, 0)   # [NQ, D, T]
    print("qkva vs attn2: max", np.abs(f(a2) - f(a)).max(), "frac exact", (f(a2) == f(a)).mean(),
          "| qkva vs numpy max", np.abs(f(a2) - attT).max(), "attn2 vs numpy max", np.abs(f(a) - attT).max(), "absmax", np.abs(attT).max())

if os.environ.get("LAYERS"):
    # two layers: the second with other weights (permuted copies), pre_layers vs two single-layer passes
    perm = lambda w: np.ascontiguousarray(w[::-1])
    W = dict(g1c=np.stack([col(g1), col(g2)]), wqkv=perm_qkv(np.stack([wqkv, perm(wqkv)])), qn=np.stack([qn.reshape(D, 1), kn.reshape(D, 1)]),
             kn=np.stack([kn.reshape(D, 1), qn.reshape(D, 1)]), wo=np.stack([wo, perm(wo)]), g2c=np.stack([col(g2), col(g1)]),
             wg=np.stack([wg, perm(wg)]), wu=np.stack([wu, perm(wu)]), wd=np.stack([wd, perm(wd)]))
    kcs = np.zeros((2, NKV, D, SMAX), bf); vcs = np.zeros((2, NKV, SMAX, D), bf)
    Y = run("pre_layers", X, W["g1c"], W["wqkv"], W["qn"], W["kn"], rcT, rsT, B(rot), kcs, vcs, W["wo"], W["g2c"], W["wg"], W["wu"], W["wd"])
    Z = X
    for L_ in range(2):
        g = lambda k: np.ascontiguousarray(W[k][L_:L_ + 1])
        aa = run("pre_qkva", Z, g("g1c"), g("wqkv"), g("qn"), g("kn"), rcT, rsT, B(rot), S1(kc), S1(vc))
        Z = run("pre_post3", Z, aa, g("wo"), g("g2c"), g("wg"), g("wu"), g("wd"))
    dz = np.abs(f(Y) - f(Z))
    print("pre_layers vs 2 x (qkva + post3): max", dz.max(), "frac exact", (dz == 0).mean())
