"""dev: tensor-engine peak: N matmuls [128x128 stationary] x [128 x 512 moving] from SBUF, no weight DMA."""
import os, sys, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, "../kernels"), HERE]
import nki, nki.language as nl, nki.isa as nisa
MW = int(os.environ.get("MW", 512))
NM = int(os.environ.get("NM", 2048)); NS = int(os.environ.get("NS", 8))

@nki.jit
def pe(a, b):
    st = nl.ndarray((128, NS, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=st, src=a)
    mv = nl.ndarray((128, 4, MW), dtype=b.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=mv, src=b)
    out = nl.ndarray((2, 128, MW), dtype=nl.float32, buffer=nl.shared_hbm)
    pid = nl.program_id(0)
    acc = nl.ndarray((128, MW), dtype=nl.float32, buffer=nl.psum)
    for i in range(NM):
        nisa.nc_matmul(dst=acc, stationary=st[0:128, i % NS, 0:128], moving=mv[0:128, i % 4, 0:MW])
    r = nl.ndarray((128, MW), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r, src=acc)
    nisa.dma_copy(dst=out[pid], src=r)
    return out

if __name__ == "__main__":
    art = tempfile.mkdtemp(prefix="pe-", dir="/tmp/q8x"); os.environ["NKI_ARTIFACTS_DIR"] = art
    import numpy as np, ml_dtypes
    rg = np.random.default_rng(0)
    if os.environ.get("RND"):
        a = rg.standard_normal((128, NS, 128)).astype(ml_dtypes.bfloat16); b = rg.standard_normal((128, 4, MW)).astype(ml_dtypes.bfloat16)
    else:
        a = np.ones((128, NS, 128), ml_dtypes.bfloat16); b = np.ones((128, 4, MW), ml_dtypes.bfloat16)
    pe[2](a, b)
    shutil.copy(art + "/kernel.neff", f"/tmp/q8x/pe_{NM}.neff")
