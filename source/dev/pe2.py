"""dev: PE rate in the kernels' loop shape: GN accumulators [128, MW], K=16 matmuls each, distinct stationaries."""
import os, sys, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, "../kernels"), HERE]
import nki, nki.language as nl, nki.isa as nisa
MW = int(os.environ.get("MW", 512)); GN = int(os.environ.get("GN", 4)); REP = int(os.environ.get("REP", 32))

@nki.jit
def pe2(a, b):
    st = nl.ndarray((128, 16, GN * 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=st, src=a)
    mv = nl.ndarray((128, 16, MW), dtype=b.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=mv, src=b)
    out = nl.ndarray((2, 128, MW), dtype=nl.float32, buffer=nl.shared_hbm)
    pid = nl.program_id(0)
    tot = nl.ndarray((128, MW), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=tot, value=0.0)
    for r in range(REP):
        acc = []
        for _v in range(GN):
            acc.append(nl.ndarray((128, MW), dtype=nl.float32, buffer=nl.psum))
        for u in range(GN):
            for k in range(16):
                nisa.nc_matmul(dst=acc[u], stationary=st[0:128, k, u * 128:(u + 1) * 128], moving=mv[0:128, k, 0:MW])
        for u in range(GN):
            nisa.tensor_tensor(dst=tot, data1=tot, data2=acc[u], op=nl.add)
    nisa.dma_copy(dst=out[pid], src=tot)
    return out

if __name__ == "__main__":
    art = tempfile.mkdtemp(prefix="pe2-", dir="/tmp/q8x"); os.environ["NKI_ARTIFACTS_DIR"] = art
    import numpy as np, ml_dtypes
    pe2[2](np.ones((128, 16, GN * 128), ml_dtypes.bfloat16), np.ones((128, 16, MW), ml_dtypes.bfloat16))
    shutil.copy(art + "/kernel.neff", "/tmp/q8x/pe2.neff")
