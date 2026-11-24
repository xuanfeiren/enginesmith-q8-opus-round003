"""dev: GEMV micro-benchmarks, y = x @ W, W [K, N] natural ([in, out]) layout."""
import os, sys, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, "../kernels"), HERE]
import nki, nki.language as nl, nki.isa as nisa

KPD = int(os.environ.get("KPD", "8"))   # k rows per DMA per partition
MODE = os.environ.get("MODE", "pmaj")
NSPL = int(os.environ.get("NSPL", "1"))
DGEM = {"hw": nisa.dge_mode.hwdge, "un": nisa.dge_mode.unknown, "none": nisa.dge_mode.none}[os.environ.get("DGEM", "un")]

@nki.jit
def gemv_pm(x, w):
    """x [1, K], w [K, N]; p-major: k = p*KT + kt. Output y [1, N] in p-major too: y[p*NT + nt]. One core: all N."""
    K, N = w.shape
    KT = K // 128
    NT = N // 128
    nprog = nl.num_programs()
    pid = nl.program_id(0) if nprog > 1 else 0
    KC = KT // nprog       # split K between cores
    xs = nl.ndarray((128, KT), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=x.reshape((128, KT)))
    acc = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.psum)
    w3 = w.reshape((128, KT, N))
    for kb in range(KC // KPD):
        k0 = pid * KC + kb * KPD
        wt = nl.ndarray((128, KPD, N), dtype=w.dtype, buffer=nl.sbuf)
        for q in range(NSPL):
            pp = 128 // NSPL
            nisa.dma_copy(dst=wt[q * pp:(q + 1) * pp], src=w3[q * pp:(q + 1) * pp, k0:k0 + KPD, 0:N], dge_mode=DGEM)
        for kk in range(KPD):
            for nt in range(NT):
                if MODE == "pmaj":
                    st = wt[0:128, kk, nt:N:NT]
                else:
                    st = wt[0:128, kk, nt * 128:(nt + 1) * 128]
                nisa.nc_matmul(dst=acc[0:128, nt:nt + 1], stationary=st, moving=xs[0:128, k0 + kk:k0 + kk + 1])
    part = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=part, src=acc)
    out = nl.ndarray((nprog, 128, NT), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out[pid], src=part)
    return out

if __name__ == "__main__":
    art = tempfile.mkdtemp(prefix="g-", dir="/tmp/q8x"); os.environ["NKI_ARTIFACTS_DIR"] = art
    import numpy as np, ml_dtypes
    bf = ml_dtypes.bfloat16
    K, N = int(os.environ.get("K", 4096)), int(os.environ.get("N", 4096))
    r = np.random.default_rng(0)
    x = r.standard_normal((1, K)).astype(bf); w = (r.standard_normal((K, N)) * 0.02).astype(bf)
    out = gemv_pm[2](x, w)
    y = out.sum(0)  # [128, NT]
    ref = x.astype(np.float32) @ w.astype(np.float32)
    NT = N // 128
    ref_l = ref.reshape(128, NT) if MODE == "pmaj" else ref.reshape(NT, 128).T
    print(MODE, "max err", np.abs(y - ref_l).max(), np.abs(ref).max())
    shutil.copy(art + "/kernel.neff", f"/tmp/q8x/gemv_{MODE}.neff")
