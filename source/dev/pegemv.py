"""dev: PE-only GEMV rate (weights resident in SBUF, no DMA in the loop).
MODE=ws: weight-stationary (128x128 W tile stationary, x column moving, out column in PSUM)
MODE=mv: x-stationary (x column [128,1] stationary, W [128, CW] moving, out row [1, CW] in PSUM)"""
import os, sys, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, "../kernels"), HERE]
import nki, nki.language as nl, nki.isa as nisa
MODE = os.environ.get("MODE", "ws")
KT = 8
N = 4096
REP = int(os.environ.get("REP", 20))
CW = int(os.environ.get("CW", 512))


@nki.jit
def peg(w, x):
    ws = nl.ndarray((128, KT, N), dtype=w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=ws, src=w)
    xs = nl.ndarray((128, KT), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=x)
    out = nl.ndarray((2, 128, N // 128), dtype=nl.float32, buffer=nl.shared_hbm)
    pid = nl.program_id(0)
    tot = nl.ndarray((128, N // 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=tot, value=0.0)
    NT = N // 128
    for r in range(REP):
        if MODE == "ws":
            acc = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.psum)
            for k in range(KT):
                for nt in range(NT):
                    nisa.nc_matmul(dst=acc[0:128, nt:nt + 1], stationary=ws[0:128, k, nt * 128:(nt + 1) * 128],
                                   moving=xs[0:128, k:k + 1])
            nisa.tensor_tensor(dst=tot, data1=tot, data2=acc, op=nl.add)
        else:
            row = nl.ndarray((1, N), dtype=nl.float32, buffer=nl.sbuf)
            for c in range(N // CW):
                acc = nl.ndarray((1, CW), dtype=nl.float32, buffer=nl.psum)
                for k in range(KT):
                    nisa.nc_matmul(dst=acc, stationary=xs[0:128, k:k + 1], moving=ws[0:128, k, c * CW:(c + 1) * CW])
                nisa.tensor_copy(dst=row[0:1, c * CW:(c + 1) * CW], src=acc)
            nisa.tensor_tensor(dst=tot[0:1, 0:NT], data1=tot[0:1, 0:NT], data2=row[0:1, 0:NT], op=nl.add)
    nisa.dma_copy(dst=out[pid], src=tot)
    return out


if __name__ == "__main__":
    art = tempfile.mkdtemp(prefix="peg-", dir="/tmp/q8x"); os.environ["NKI_ARTIFACTS_DIR"] = art
    import numpy as np, ml_dtypes
    peg[2](np.ones((128, KT, N), ml_dtypes.bfloat16), np.ones((128, KT), ml_dtypes.bfloat16))
    shutil.copy(art + "/kernel.neff", f"/tmp/q8x/peg_{MODE}_{CW}_{REP}.neff")
