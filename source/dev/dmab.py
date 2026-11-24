"""dev: DMA throughput of weight slabs [128 p, KTB rows-of-tiles, W cols] from a [4096, N] bf16 tensor."""
import os, sys, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, "../kernels"), HERE]
import nki, nki.language as nl, nki.isa as nisa
W = int(os.environ.get("W", 512)); KTB = int(os.environ.get("KTB", 32)); N = 12288

@nki.jit
def dmab(w):
    K = w.shape[0]; KT = K // 128
    pid = nl.program_id(0)
    acc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    NC = (N // 2) // W
    for c in range(NC):
        for kb in range(KT // KTB):
            t = nl.ndarray((128, KTB, W), dtype=w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=t, src=w.ap(pattern=[[N, 128], [128 * N, KTB], [1, W]], offset=pid * (N // 2) + c * W + kb * KTB * 128 * N), dge_mode=nisa.dge_mode.hwdge)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=t[0:128, 0, 0:1], op=nl.add)
    out = nl.ndarray((2, 128, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out[pid], src=acc)
    return out

if __name__ == "__main__":
    art = tempfile.mkdtemp(prefix="d-", dir="/tmp/q8x"); os.environ["NKI_ARTIFACTS_DIR"] = art
    import numpy as np, ml_dtypes
    dmab[2](np.zeros((4096, N), ml_dtypes.bfloat16))
    shutil.copy(art + "/kernel.neff", "/tmp/q8x/dmab.neff")
