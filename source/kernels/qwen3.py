"""NKI kernels of the Qwen3-8B engine (build time only: traced and compiled to NEFFs by build_neffs.py).

Every kernel derives its sizes (prompt bucket T, attention bucket S) from its input shapes, so one source compiles
all buckets. Layer kernels wrap the nkilib kernels vendored in kernels/nkilib (Apache-2.0, see THIRD_PARTY.md).

Layouts (bf16 unless noted):
  x            [1, T, H]          hidden states (decode: T = 1)
  wqkv         [H, (NQ+2NKV)*D]   q|k|v projections, transposed from the checkpoint ([out, in] -> [in, out])
  wo           [NQ*D, H]          wg, wu [H, I]   wd [I, H]
  kc, vc       [1, NKV, SMAX, D]  KV cache of one layer
  wlm          [2, VC, 128, KT, 512]  LM head, transposed and swizzled: (core, chunk, partition, k-tile, column)
"""
import os as _os
import nki
import nki.language as nl
import nki.isa as nisa

from nkilib.experimental.transformer.attention_block_tkg import attention_block_tkg
from nkilib.core.mlp.mlp import mlp
from nkilib.core.qkv.qkv import qkv
from nkilib.core.attention.attention_cte import attention_cte
from nkilib.core.output_projection.output_projection_cte.output_projection_cte import output_projection_cte
from nkilib.core.max.cascaded_max import cascaded_max
from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.logging import get_logger
from nkilib.core.utils.common_types import QuantizationType, NormType, ActFnType, QKVOutputLayout, QKNormConfig

H = 4096
NQ = 32
NKV = 8
D = 128
I = 12288
EPS = 1e-6
G = 2        # head groups in prefill (nkilib CTE kernels: at most 4096 projection columns / 17 heads per call)
# MLP intermediate chunk in prefill: nkilib mlp_cte takes at most 4096; its token tile shrinks as the chunk grows,
# so large buckets use 2048 (measured faster at T >= 1024) and small ones 4096 (fewer executions).
LMC = 512    # LM head columns per chunk
# ---- schedule / engine switches (session 003). The defaults are the build; Q8_<NAME> environment variables override
# them for development only (dev/var.sh compiles + profiles a variant; build_neffs.py refuses to run with any Q8_ set,
# because the NEFF cache key does not include them). Measurements: NOTES.md, "Session 003".
def _sw(name, default):
    v = _os.environ.get("Q8_" + name)
    if v is None:
        return default
    return type(default)(v)


ATT_PIPE = _sw("ATT_PIPE", 3)        # _qkva/_qkva2 attention: software-pipeline lag of the 3 stages (0: off)
ATT_NORM = _sw("ATT_NORM", "vec")    # softmax normalization engine: vec | act
ATT_MAX = _sw("ATT_MAX", "pe")       # pe: diagonal mask on the PE + one tensor_scalar_reduce pass; ref: as round 002
ATT_CPT = _sw("ATT_CPT", "auto")     # engine of the P^T PSUM -> SBUF copy: auto | act | vec
ATT_COUT = _sw("ATT_COUT", "auto")   # engine of the attention output PSUM -> SBUF copy
ATT_QOVL = _sw("ATT_QOVL", 0)        # _qkva2 (TB <= 256): attention overlapped with the q projections (_qkva2_ovl)
PO_GN = _sw("PO_GN", 0)              # _post4: o-proj output tiles per group (0: gn_of(TB))
GU_FUSE = _sw("GU_FUSE", 1)          # decode MLP: gate + up of a k-tile in one buffer / one PSUM tile
GU_PF = _sw("GU_PF", 0)              # decode: fused gate/up slabs prefetched before the attention
ATT_OVL = _sw("ATT_OVL", 0)          # _qkva: block b's attention spread over block b+1's QKV units (neutral)
Q4 = _sw("Q4", 1)                    # QKV in head groups with the q/k norm + rope lagged (_qkv4)
Q4_LAG = _sw("Q4_LAG", 2)            # ... lag in head groups
Q4_GN = _sw("Q4_GN", 0)              # ... heads per group (0: gn_of(TB) / 2)
Q4_DRAIN = _sw("Q4_DRAIN", "auto")   # ... engine of the projection PSUM drains
POST5 = _sw("POST5", 1)              # pre_layers / pre_post3: per-group sendrecv swaps (_post5_blk)
POST5_LAG = _sw("POST5_LAG", 3)      # ... swapped groups consumed this many groups later
P4_LAG = _sw("P4_LAG", -1)           # _post4 o-proj: consume swaps (+ squares) this many groups later (-1: after the loop)
FIN_ENG = _sw("FIN_ENG", "auto")     # engines of _fin (auto | vec | pool | alt), the sum-of-squares adds, hn = h * rstd,
SSQ_ENG = _sw("SSQ_ENG", "auto")     # and the gamma multiply (auto | act | vec)
HN_ENG = _sw("HN_ENG", "auto")
HNG_ENG = _sw("HNG_ENG", "auto")
QN_ENG = _sw("QN_ENG", "auto")       # _qkv3_norm: engine of x * rstd (auto | pool | alt), gamma (auto | act), adds
QN_G = _sw("QN_G", "auto")
QN_SSQ = _sw("QN_SSQ", "auto")
DQ_KG = _sw("DQ_KG", 2)              # decode QKV: k-tiles per weight DMA
DD_KG = _sw("DD_KG", 2)              # decode down: rows per weight DMA
DO_KG = _sw("DO_KG", 1)              # decode o-proj: rows per weight DMA
_ENG = {"auto": nisa.engine.unknown, "act": nisa.engine.scalar, "vec": nisa.engine.vector}
_ENG2 = {"auto": nisa.engine.unknown, "vec": nisa.engine.vector, "pool": nisa.engine.gpsimd, "act": nisa.engine.scalar,
         "alt": nisa.engine.unknown}

# ----------------------------------------------------------------------------------------------- decode MLP
# Column layout: a vector v of n*128 elements lives in SBUF as [128, n] with v[t*128 + p] at [p, t]. GEMVs are
# weight-stationary: a [128 k, 128 n] tile of W [in, out] is the stationary operand, the input column the moving one,
# and the output tile lands in PSUM already in column layout. (x stationary with 512-wide moving weights is bound by
# the PE's moving rate, ~358 GB/s per core: 541 us for the MLP against 515 us for this form, measured.)
DGE = nisa.dge_mode.hwdge


def _rows_to_col(dst_sb, src_rows, n):
    """src_rows [n, 128] SBUF (row-major vector of n*128) -> dst_sb [128, n] column layout (v[t*128+p] at [p, t])."""
    pt = nl.ndarray((128, n), dtype=src_rows.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=pt, data=src_rows, engine=nisa.engine.tensor)
    nisa.tensor_copy(dst=dst_sb, src=pt)


def _col_rmsnorm(dst, col, gcol, ones_f, n):
    """dst [128, n] bf16 = bf16(col * rsqrt(mean(col^2) + eps)) * gcol; col [128, n] bf16 (column layout of n*128)."""
    sq = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    ps = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=sq, op=nl.square, data=col, reduce_op=nl.add, reduce_res=ps)
    tp = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=tp, stationary=ones_f, moving=ps, is_stationary_onezero=True)
    epsb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=epsb, value=EPS)
    rstd = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=tp, bias=epsb, scale=1.0 / (n * 128))
    nisa.activation(dst=dst, op=nl.copy, data=col, scale=rstd)
    nisa.tensor_tensor(dst=dst, data1=dst, data2=gcol, op=nl.multiply)


# ----------------------------------------------------------------------------------------------- decode
@nki.jit
def embed_dec(tok, pos, emb, cos_tab, sin_tab):
    """tok, pos: int32 [1, 1]. Returns x [1, 1, H], cos/sin [D/2, 1, 1], pos as f32 [1, 1], pos as u32 [1, 1]."""
    tok_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    pos_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tok_sb, src=tok)
    nisa.dma_copy(dst=pos_sb, src=pos)
    xr = nl.ndarray((1, H), dtype=emb.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xr, src=emb.ap(pattern=[[H, 1], [1, H]], offset=0, scalar_offset=tok_sb, indirect_dim=0))
    x = nl.ndarray((1, 1, H), dtype=emb.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=x.reshape((1, H)), src=xr)
    HD = D // 2
    cr = nl.ndarray((1, HD), dtype=cos_tab.dtype, buffer=nl.sbuf)
    sr = nl.ndarray((1, HD), dtype=sin_tab.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=cr, src=cos_tab.ap(pattern=[[HD, 1], [1, HD]], offset=0, scalar_offset=pos_sb, indirect_dim=0))
    nisa.dma_copy(dst=sr, src=sin_tab.ap(pattern=[[HD, 1], [1, HD]], offset=0, scalar_offset=pos_sb, indirect_dim=0))
    cos = nl.ndarray((HD, 1, 1), dtype=cos_tab.dtype, buffer=nl.shared_hbm)
    sin = nl.ndarray((HD, 1, 1), dtype=sin_tab.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=cos.reshape((1, HD)), src=cr)
    nisa.dma_copy(dst=sin.reshape((1, HD)), src=sr)
    pf_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    pu_sb = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pf_sb, src=pos_sb)
    nisa.tensor_copy(dst=pu_sb, src=pos_sb)
    posf = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    posu = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=posf, src=pf_sb)
    nisa.dma_copy(dst=posu, src=pu_sb)
    return x, cos, sin, posf, posu


def _gu_slab(wg, wu, l, kt, pid):
    """decode fused gate/up weight slab of k-tile kt: [128, 2, I/2] (gate | up columns of this core)"""
    IC = I // 2
    c0 = pid * IC
    wgu = nl.ndarray((128, 2, IC), dtype=wg.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=wgu[0:128, 0, 0:IC], src=wg[l, kt * 128:(kt + 1) * 128, c0:c0 + IC], dge_mode=DGE)
    nisa.dma_copy(dst=wgu[0:128, 1, 0:IC], src=wu[l, kt * 128:(kt + 1) * 128, c0:c0 + IC], dge_mode=DGE)
    return wgu


def _mlp_col(h, l, g2c, wg, wu, wd, ones_f, pid, pf=None):
    """MLP of the column h [128, KT] bf16 -> returns d + h (column, bf16)."""
    KT = H // 128
    IC = I // 2
    NTI = IC // 128
    gc = nl.ndarray((128, KT), dtype=g2c.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gc, src=g2c[l])
    hn = nl.ndarray((128, KT), dtype=h.dtype, buffer=nl.sbuf)
    _col_rmsnorm(hn, h, gc, ones_f, KT)
    c0 = pid * IC
    if GU_FUSE:
        # gate and up of one k-tile in ONE buffer and ONE PSUM tile ([128, 2, NTI]: gate at [:, 0], up at [:, 1]),
        # matmuls alternating gate/up per output tile: one accumulation chain, so the scheduler keeps the slab order
        gup = nl.ndarray((128, 2, NTI), dtype=nl.float32, buffer=nl.psum)
        for kt in range(KT):
            if pf is not None and kt < len(pf):
                wgu = pf[kt]
            else:
                wgu = _gu_slab(wg, wu, l, kt, pid)
            for nt in range(NTI):
                for wi in range(2):
                    nisa.nc_matmul(dst=gup[0:128, wi, nt:nt + 1], stationary=wgu[0:128, wi, nt * 128:(nt + 1) * 128],
                                   moving=hn[0:128, kt:kt + 1])
        gus = nl.ndarray((128, 2, NTI), dtype=h.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gus, src=gup)
        gs = nl.ndarray((128, NTI), dtype=h.dtype, buffer=nl.sbuf)
        nisa.activation(dst=gs, op=nl.silu, data=gus[0:128, 0, 0:NTI])
        nisa.tensor_tensor(dst=gs, data1=gs, data2=gus[0:128, 1, 0:NTI], op=nl.multiply)
    gps = nl.ndarray((128, NTI), dtype=nl.float32, buffer=nl.psum)
    ups = nl.ndarray((128, NTI), dtype=nl.float32, buffer=nl.psum)
    for kt in range(0 if GU_FUSE else KT):
        wgt = nl.ndarray((128, IC), dtype=wg.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wgt, src=wg[l, kt * 128:(kt + 1) * 128, c0:c0 + IC], dge_mode=DGE)
        for nt in range(NTI):
            nisa.nc_matmul(dst=gps[0:128, nt:nt + 1], stationary=wgt[0:128, nt * 128:(nt + 1) * 128],
                           moving=hn[0:128, kt:kt + 1])
        wut = nl.ndarray((128, IC), dtype=wu.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wut, src=wu[l, kt * 128:(kt + 1) * 128, c0:c0 + IC], dge_mode=DGE)
        for nt in range(NTI):
            nisa.nc_matmul(dst=ups[0:128, nt:nt + 1], stationary=wut[0:128, nt * 128:(nt + 1) * 128],
                           moving=hn[0:128, kt:kt + 1])
    if not GU_FUSE:
        gs = nl.ndarray((128, NTI), dtype=h.dtype, buffer=nl.sbuf)
        us = nl.ndarray((128, NTI), dtype=h.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gs, src=gps)
        nisa.tensor_copy(dst=us, src=ups)
        nisa.activation(dst=gs, op=nl.silu, data=gs)
        nisa.tensor_tensor(dst=gs, data1=gs, data2=us, op=nl.multiply)
    dps = nl.ndarray((128, KT), dtype=nl.float32, buffer=nl.psum)
    for kg in range(NTI // DD_KG):
        r0 = c0 + kg * DD_KG * 128
        wdt = nl.ndarray((128, DD_KG, H), dtype=wd.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wdt, src=wd.ap(pattern=[[H, 128], [128 * H, DD_KG], [1, H]], offset=l * I * H + r0 * H), dge_mode=DGE)
        for k in range(DD_KG):
            kt = kg * DD_KG + k
            for nt in range(KT):
                nisa.nc_matmul(dst=dps[0:128, nt:nt + 1], stationary=wdt[0:128, k, nt * 128:(nt + 1) * 128],
                               moving=gs[0:128, kt:kt + 1])
    d = _xsum(dps, KT, h.dtype, pid, 1)
    nisa.tensor_tensor(dst=d, data1=h, data2=d, op=nl.add)
    return d


def _xsum(ps, n, dtype, pid, pipe):
    """ps [128, n] fp32 PSUM partial of this core -> bf16 [128, n] SBUF sum of both cores' partials."""
    part = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=part, src=ps)
    oth = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.sendrecv(src=part, dst=oth, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=pipe,
                  dma_engine=nisa.dma_engine.gpsimd_dma)
    tot = nl.ndarray((128, n), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tot, data1=part, data2=oth, op=nl.add)   # a + b == b + a: both cores agree
    return tot


def _qkv_slab(wqkv, l, kg, pid, dt):
    """decode QKV weight slab kg (DQ_KG k-tiles x this core's columns) of layer l"""
    QH = NQ // 2
    KH = NKV // 2
    NC_ = QH + 2 * KH
    QKVW = (NQ + 2 * NKV) * D
    w = nl.ndarray((128, DQ_KG, NC_ * D), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=w, src=wqkv.ap(pattern=[[QKVW, 128], [128 * QKVW, DQ_KG], [1, NC_ * D]],
                                     offset=l * H * QKVW + kg * DQ_KG * 128 * QKVW + pid * NC_ * D), dge_mode=DGE)
    return w


def _layer(xc, l, g1c, wqkv, qn, kn, csf, snf, rot_sb, kc, vc, pos_sb, msk, wo, g2c, wg, wu, wd, S, ones_f, pid):
    """Decoder layer l (index into the stacked weights) for one token: column xc [128, H/128] -> column out."""
    dt = xc.dtype
    SMAX = kc.shape[3]
    KT = H // 128
    QH = NQ // 2        # q heads per core
    KH = NKV // 2       # kv heads per core
    GQ = QH // KH       # q heads per kv head
    NC_ = QH + 2 * KH   # qkv column tiles per core
    gc = nl.ndarray((128, KT), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=gc, src=g1c[l])
    xn = nl.ndarray((128, KT), dtype=dt, buffer=nl.sbuf)
    _col_rmsnorm(xn, xc, gc, ones_f, KT)
    # ---- qkv (weight-stationary): column tiles 0..QH-1 q, QH..QH+KH-1 k, then v
    qkv_ps = nl.ndarray((128, NC_), dtype=nl.float32, buffer=nl.psum)
    # wqkv columns are grouped per core (host layout): [core c: q heads 16c.. | k heads 4c.. | v heads 4c..], so
    # this core's NC_ column tiles are one contiguous 6 KB piece per row
    c0 = pid * NC_ * D
    if DQ_KG > 1:
        for kg in range(KT // DQ_KG):
            w = _qkv_slab(wqkv, l, kg, pid, dt)
            for k in range(DQ_KG):
                kt = kg * DQ_KG + k
                for t in range(NC_):
                    nisa.nc_matmul(dst=qkv_ps[0:128, t:t + 1], stationary=w[0:128, k, t * D:(t + 1) * D], moving=xn[0:128, kt:kt + 1])
    for kt in range(KT if DQ_KG == 1 else 0):
        w = nl.ndarray((128, NC_ * D), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=w, src=wqkv[l, kt * 128:(kt + 1) * 128, c0:c0 + NC_ * D], dge_mode=DGE)
        for t in range(NC_):
            nisa.nc_matmul(dst=qkv_ps[0:128, t:t + 1], stationary=w[0:128, t * D:(t + 1) * D], moving=xn[0:128, kt:kt + 1])
    qkv = nl.ndarray((128, NC_), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qkv, src=qkv_ps)
    # ---- q/k rmsnorm per head (over the partitions)
    NQK = QH + KH
    sq = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=qkv[0:128, 0:NQK])
    ss = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ss, stationary=ones_f, moving=sq, is_stationary_onezero=True)
    epsb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=epsb, value=EPS)
    rstd = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=ss, bias=epsb, scale=1.0 / D)
    nqk = nl.ndarray((128, NQK), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=nqk, data1=qkv[0:128, 0:NQK], data2=rstd, op=nl.multiply)
    gq = nl.ndarray((128, 1), dtype=dt, buffer=nl.sbuf)
    gk = nl.ndarray((128, 1), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=gq, src=qn[l])
    nisa.dma_copy(dst=gk, src=kn[l])
    gqf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    gkf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gqf, src=gq)
    nisa.tensor_copy(dst=gkf, src=gk)
    nisa.tensor_scalar(dst=nqk[0:128, 0:QH], data=nqk[0:128, 0:QH], op0=nl.multiply, operand0=gqf)
    nisa.tensor_scalar(dst=nqk[0:128, QH:NQK], data=nqk[0:128, QH:NQK], op0=nl.multiply, operand0=gkf)
    # ---- rope: e = bf16(n * cos) + bf16(rotate_half(n) * sin)
    rh = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=rh, stationary=rot_sb, moving=nqk)
    ea = nl.ndarray((128, NQK), dtype=dt, buffer=nl.sbuf)
    eb = nl.ndarray((128, NQK), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ea, data=nqk, op0=nl.multiply, operand0=csf)
    nisa.tensor_scalar(dst=eb, data=rh, op0=nl.multiply, operand0=snf)
    qk = nl.ndarray((128, NQK), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=qk, data1=ea, data2=eb, op=nl.add)
    # ---- cache write at pos: K column per head into kc[h, :, pos]; V as rows into vc[h, pos, :]
    nisa.dma_copy(dst=kc.ap(pattern=[[SMAX, D], [D * SMAX, KH]], offset=(l * NKV + pid * KH) * D * SMAX,
                            scalar_offset=pos_sb, indirect_dim=3), src=qk[0:128, QH:NQK])
    vr_ps = nl.ndarray((KH, 128), dtype=dt, buffer=nl.psum)
    nisa.nc_transpose(dst=vr_ps, data=qkv[0:128, NQK:NC_], engine=nisa.engine.tensor)
    vr = nl.ndarray((KH, 128), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vr, src=vr_ps)
    nisa.dma_copy(dst=vc.ap(pattern=[[SMAX * D, KH], [1, D]], offset=(l * NKV + pid * KH) * SMAX * D,
                            scalar_offset=pos_sb, indirect_dim=2), src=vr)
    # the first GU_PF gate/up slabs queued before the o-proj slabs: they stream during the attention, when the o-proj
    # slab pool is full
    pf = []
    if GU_FUSE:
        for kt in range(GU_PF):
            pf.append(_gu_slab(wg, wu, l, kt, pid))
    # ---- attention over the bucket: scores [QH, S] for all local q heads (q of other groups zeroed per kv head)
    qm = nl.ndarray((128, KH, QH), dtype=dt, buffer=nl.sbuf)
    nisa.memset(dst=qm, value=0.0)
    for j in range(KH):
        nisa.tensor_copy(dst=qm[0:128, j, j * GQ:(j + 1) * GQ], src=qk[0:128, j * GQ:(j + 1) * GQ])
    kts = []
    for j in range(KH):
        kt_sb = nl.ndarray((128, S), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=kt_sb, src=kc[l, pid * KH + j, 0:D, 0:S])
        kts.append(kt_sb)
    sc = nl.ndarray((QH, S), dtype=nl.float32, buffer=nl.sbuf)
    for c in range((S + 511) // 512):   # S: a multiple of 128; key chunks of 512 (the last one may be shorter)
        c0 = c * 512
        cw = min(512, S - c0)
        sp = nl.ndarray((QH, 512), dtype=nl.float32, buffer=nl.psum)
        for j in range(KH):
            nisa.nc_matmul(dst=sp[0:QH, 0:cw], stationary=qm[0:128, j, 0:QH], moving=kts[j][0:128, c0:c0 + cw])
        nisa.scalar_tensor_tensor(dst=sc[0:QH, c0:c0 + cw], data=sp[0:QH, 0:cw], op0=nl.multiply,
                                  operand0=1.0 / (D ** 0.5), op1=nl.add, operand1=msk[0:QH, c0:c0 + cw])
    mx = nl.ndarray((QH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mx, op=nl.maximum, data=sc, axis=1, negate=True)
    ex = nl.ndarray((QH, S), dtype=nl.float32, buffer=nl.sbuf)
    sm = nl.ndarray((QH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=ex, op=nl.exp, data=sc, bias=mx, scale=1.0, reduce_op=nl.add, reduce_res=sm)
    rs = nl.ndarray((QH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rs, data=sm)
    pb = nl.ndarray((QH, S), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=pb, data=ex, op0=nl.multiply, operand0=rs)
    NT = S // 128
    pT_ps = nl.ndarray((128, NT, QH), dtype=dt, buffer=nl.psum)
    for t in range(NT):
        nisa.nc_transpose(dst=pT_ps[0:128, t, 0:QH], data=pb[0:QH, t * 128:(t + 1) * 128], engine=nisa.engine.tensor)
    pT = nl.ndarray((128, NT, QH), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pT, src=pT_ps)
    ac_ps = nl.ndarray((128, QH), dtype=dt, buffer=nl.psum)
    for j in range(KH):
        vt = nl.ndarray((128, NT, D), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=vt, src=vc.ap(pattern=[[D, 128], [128 * D, NT], [1, D]], offset=(l * NKV + pid * KH + j) * SMAX * D))
        o_ps = nl.ndarray((GQ, D), dtype=nl.float32, buffer=nl.psum)
        for t in range(NT):
            nisa.nc_matmul(dst=o_ps, stationary=pT[0:128, t, j * GQ:(j + 1) * GQ], moving=vt[0:128, t, 0:D])
        o_sb = nl.ndarray((GQ, D), dtype=dt, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_sb, src=o_ps)
        nisa.nc_transpose(dst=ac_ps[0:128, j * GQ:(j + 1) * GQ], data=o_sb, engine=nisa.engine.tensor)
    ac = nl.ndarray((128, QH), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ac, src=ac_ps)
    # ---- o-proj: K = this core's heads (rows pid*QH*D ..), N = H; partial sums exchanged
    op_ps = nl.ndarray((128, KT), dtype=nl.float32, buffer=nl.psum)
    for kg in range(QH // DO_KG):
        r0 = pid * QH * D + kg * DO_KG * 128
        wot = nl.ndarray((128, DO_KG, H), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=wot, src=wo.ap(pattern=[[H, 128], [128 * H, DO_KG], [1, H]], offset=l * NQ * D * H + r0 * H), dge_mode=DGE)
        for k in range(DO_KG):
            kt = kg * DO_KG + k
            for nt in range(KT):
                nisa.nc_matmul(dst=op_ps[0:128, nt:nt + 1], stationary=wot[0:128, k, nt * 128:(nt + 1) * 128],
                               moving=ac[0:128, kt:kt + 1])
    o = _xsum(op_ps, KT, dt, pid, 0)
    h = nl.ndarray((128, KT), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h, data1=xc, data2=o, op=nl.add)
    # ---- mlp, output row
    y = _mlp_col(h, l, g2c, wg, wu, wd, ones_f, pid, pf)
    return y


@nki.jit
def layer_dec(x, g1c, wqkv, qn, kn, rcos, rsin, rot, kc, vc, pos, posf, wo, g2c, wg, wu, wd, sctx):
    """NL consecutive decoder layers for one token (NL = leading dim of the stacked per-layer tensors), all on chip
    between the input and output rows. x [1,1,H] row; g1c/g2c [NL, 128, H/128] gammas in column layout;
    qn/kn [NL, D, 1]; rcos/rsin [SMAX, D] (halves repeated); rot [D, D] bf16 rotate-half matrix (rot[k, m] = -1 if
    k = m + 64, +1 if k = m - 64); kc [NL, NKV, D, SMAX] (K cache, transposed); vc [NL, NKV, SMAX, D];
    pos int32 [1, 1], posf f32 [1, 1]; sctx [1, S] carries the attention bucket S. Core c: q heads 16c..,
    kv heads 4c.., half of gate/up columns; o-proj and down are split along K and summed by one exchange each."""
    S = sctx.shape[1]
    NL = wqkv.shape[0]
    pid = nl.program_id(0)
    KT = H // 128
    QH = NQ // 2
    ones_f = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    xr = nl.ndarray((KT, 128), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xr, src=x.reshape((KT, 128)))
    xc = nl.ndarray((128, KT), dtype=x.dtype, buffer=nl.sbuf)
    _rows_to_col(xc, xr, KT)
    pos_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pos_sb, src=pos)
    cs = nl.ndarray((128, 1), dtype=rcos.dtype, buffer=nl.sbuf)
    sn = nl.ndarray((128, 1), dtype=rsin.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=cs, src=rcos.ap(pattern=[[1, D], [1, 1]], offset=0, scalar_offset=pos_sb, indirect_dim=0))
    nisa.dma_copy(dst=sn, src=rsin.ap(pattern=[[1, D], [1, 1]], offset=0, scalar_offset=pos_sb, indirect_dim=0))
    csf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    snf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=csf, src=cs)
    nisa.tensor_copy(dst=snf, src=sn)
    rot_sb = nl.ndarray((D, D), dtype=rot.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_sb, src=rot)
    posq = nl.ndarray((QH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=posq, src=posf.ap(pattern=[[0, QH], [1, 1]], offset=0))
    io = nl.ndarray((QH, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[1, S]], offset=0, channel_multiplier=0)
    msk = nl.ndarray((QH, S), dtype=nl.float32, buffer=nl.sbuf)   # -30000 beyond pos (causal + bucket padding)
    nisa.tensor_scalar(dst=msk, data=io, op0=nl.greater, operand0=posq, op1=nl.multiply, operand1=-30000.0)
    for l in range(NL):
        xc = _layer(xc, l, g1c, wqkv, qn, kn, csf, snf, rot_sb, kc, vc, pos_sb, msk, wo, g2c, wg, wu, wd, S, ones_f, pid)
    yr_ps = nl.ndarray((KT, 128), dtype=x.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=yr_ps, data=xc, engine=nisa.engine.tensor)
    yr = nl.ndarray((KT, 128), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=yr, src=yr_ps)
    out = nl.ndarray((1, 1, H), dtype=x.dtype, buffer=nl.shared_hbm)
    KHH = KT // 2
    nisa.dma_copy(dst=out.reshape((KT, 128))[pid * KHH:(pid + 1) * KHH, :], src=yr[pid * KHH:(pid + 1) * KHH, :])
    return out


@nki.jit
def lm_head(x, gf, wlm):
    """logits [1, 2*VC*LMC] f32 = rmsnorm(x[0, 0]) * gf @ W. wlm [2, VC, 128, KT, LMC] (see module doc).
    Each core computes its half of the (padded) vocabulary."""
    NC, VC, P, KT, CW = wlm.shape
    nprog = nl.num_programs()
    pid = nl.program_id(0) if nprog > 1 else 0
    VP = NC * VC * CW
    # rmsnorm on one partition
    xr = nl.ndarray((1, H), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xr, src=x.reshape((1, H)))
    sq = nl.ndarray((1, H), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=sq, op=nl.square, data=xr, reduce_op=nl.add, reduce_res=ss)
    epsb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=epsb, value=EPS)
    rstd = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=ss, bias=epsb, scale=1.0 / H)
    # normalized x (bf16, as the reference rounds it before the head), then * gamma, in column layout
    xn = nl.ndarray((1, H), dtype=x.dtype, buffer=nl.sbuf)
    nisa.activation(dst=xn, op=nl.copy, data=xr, scale=rstd)
    gr = nl.ndarray((1, H), dtype=gf.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gr, src=gf)
    nisa.tensor_tensor(dst=xn, data1=xn, data2=gr, op=nl.multiply)
    xh = nl.ndarray((1, H), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=xh, src=xn)
    xT = nl.ndarray((P, KT), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xT, src=xh.ap(pattern=[[1, P], [P, KT]], offset=0))
    logits = nl.ndarray((1, VP), dtype=nl.float32, buffer=nl.shared_hbm)
    wflat = wlm.reshape((NC * VC * P, KT * CW))
    for c in range(VC):
        wt = nl.ndarray((P, KT * CW), dtype=wlm.dtype, buffer=nl.sbuf)
        r0 = (pid * VC + c) * P
        nisa.dma_copy(dst=wt, src=wflat[r0:r0 + P, :])
        acc = nl.ndarray((1, CW), dtype=nl.float32, buffer=nl.psum)
        for kt in range(KT):
            nisa.nc_matmul(dst=acc, stationary=xT[0:P, kt:kt + 1], moving=wt[0:P, kt * CW:(kt + 1) * CW])
        lsb = nl.ndarray((1, CW), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=lsb, src=acc)
        col = (pid * VC + c) * CW
        nisa.dma_copy(dst=logits[0:1, col:col + CW], src=lsb)
    return logits


argmax = cascaded_max  # logits [1, V] f32 -> (max value [1, 1], index [1, 1])


# ----------------------------------------------------------------------------------------------- prefill
@nki.jit
def embed_pre(toks, emb):
    """toks int32 [T, 1] -> x [1, T, H]."""
    T = toks.shape[0]
    x = nl.ndarray((1, T, H), dtype=emb.dtype, buffer=nl.shared_hbm)
    for t in range(T // 128):
        ids = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ids, src=toks[t * 128:(t + 1) * 128, :])
        rows = nl.ndarray((128, H), dtype=emb.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=rows, src=emb.ap(pattern=[[H, 128], [1, H]], offset=0, vector_offset=ids, indirect_dim=0))
        nisa.dma_copy(dst=x.reshape((T, H))[t * 128:(t + 1) * 128, :], src=rows)
    return x


def _proj(x, g1, w, nh, gam, cos, sin, rope):
    T = x.shape[1]
    o = nl.ndarray((nh, 1, T, D), dtype=x.dtype, buffer=nl.shared_hbm)
    if rope:
        return qkv(output_hbm=o, input=x, fused_qkv_weights=w, output_layout=QKVOutputLayout.NBSd,
                   fused_norm_type=NormType.RMS_NORM, gamma_norm_weights=g1, norm_eps=EPS,
                   fused_rope=True, cos_cache=cos, sin_cache=sin, d_head=D, num_q_heads=nh, num_kv_heads=0,
                   qk_norm_pre_rope=QKNormConfig(k_norm=None), qk_norm_pre_rope_q_gamma=gam)
    return qkv(output_hbm=o, input=x, fused_qkv_weights=w, output_layout=QKVOutputLayout.NBSd,
               fused_norm_type=NormType.RMS_NORM, gamma_norm_weights=g1, norm_eps=EPS, d_head=D)


@nki.jit
def pre_qkv(x, g1, wqkv, qn, kn, cos, sin):
    """x [1, T, H]; cos/sin [1, T, D]. Returns q0, q1 [NQ/G, 1, T, D], k0, k1, v0, v1 [NKV/G, 1, T, D]
    (group g holds q heads g*NQ/G.. and the kv heads they use)."""
    qd = NQ * D // G
    kd = NKV * D // G
    q0 = _proj(x, g1, wqkv[:, 0:qd], NQ // G, qn, cos, sin, True)
    q1 = _proj(x, g1, wqkv[:, qd:2 * qd], NQ // G, qn, cos, sin, True)
    k0 = _proj(x, g1, wqkv[:, 2 * qd:2 * qd + kd], NKV // G, kn, cos, sin, True)
    k1 = _proj(x, g1, wqkv[:, 2 * qd + kd:2 * qd + 2 * kd], NKV // G, kn, cos, sin, True)
    v0 = _proj(x, g1, wqkv[:, 2 * qd + 2 * kd:2 * qd + 3 * kd], NKV // G, None, cos, sin, False)
    v1 = _proj(x, g1, wqkv[:, 2 * qd + 3 * kd:2 * qd + 4 * kd], NKV // G, None, cos, sin, False)
    return q0, q1, k0, k1, v0, v1


@nki.jit
def pre_attn(q0, q1, k0, k1, v0, v1):
    """Causal attention of each head group. Returns a0, a1 [NQ/G, D, T]."""
    T = q0.shape[2]
    nk = NKV // G
    a0 = attention_cte(q=q0.reshape((NQ // G, T, D)), k=k0.reshape((nk, T, D)), v=v0.reshape((nk, T, D)),
                       scale=1.0 / (D ** 0.5), causal_mask=True, tp_q=True, tp_k=True, tp_out=True)
    a1 = attention_cte(q=q1.reshape((NQ // G, T, D)), k=k1.reshape((nk, T, D)), v=v1.reshape((nk, T, D)),
                       scale=1.0 / (D ** 0.5), causal_mask=True, tp_q=True, tp_k=True, tp_out=True)
    return a0, a1


def _pre_cache_body(k0, k1, v0, v1, kc, vc):
    """Writes the prompt's K (transposed: kc [NKV, D, SMAX]) and V (vc [NKV, SMAX, D]) for positions 0..T-1 in
    place. k_g, v_g [NKV/2, 1, T, D] (group g = kv heads 4g..4g+3). Core pid handles heads 2*pid, 2*pid + 1 of each
    group. Returns a dummy [1, 1]."""
    T = k0.shape[2]
    pid = nl.program_id(0)
    nk = NKV // G
    ks = [k0, k1]
    vs = [v0, v1]
    for gi in range(G):
        for i in range(2):
            hh = 2 * pid + i
            h = gi * nk + hh
            nisa.dma_copy(dst=vc[h, 0:T, 0:D], src=vs[gi][hh, 0, 0:T, 0:D])
            kT = nl.ndarray((128, T), dtype=k0.dtype, buffer=nl.sbuf)
            for t4 in range((T // 128 + 3) // 4):
                n4 = min(4, T // 128 - t4 * 4)
                tp = nl.ndarray((128, 512), dtype=k0.dtype, buffer=nl.psum)
                for u in range(n4):
                    t = t4 * 4 + u
                    kt_sb = nl.ndarray((128, D), dtype=k0.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=kt_sb, src=ks[gi][hh, 0, t * 128:(t + 1) * 128, 0:D])
                    nisa.nc_transpose(dst=tp[0:128, u * 128:(u + 1) * 128], data=kt_sb, engine=nisa.engine.tensor)
                nisa.tensor_copy(dst=kT[0:128, t4 * 512:t4 * 512 + n4 * 128], src=tp[0:128, 0:n4 * 128])
            nisa.dma_copy(dst=kc[h, 0:D, 0:T], src=kT)


@nki.jit
def pre_cache(k0, k1, v0, v1, kc, vc):
    _pre_cache_body(k0, k1, v0, v1, kc, vc)
    d = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=d, value=0.0)
    out = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=d)
    return out


def _sync(t):
    """Both physical cores wait until t (written by an nkilib kernel, partly by each core) is complete. Without it
    a consumer on one core races the other core's writes (measured: nondeterministic prefill at T=1024)."""
    if nl.num_programs() > 1:
        nisa.core_barrier(data=t, cores=(0, 1))


def _addrows(dst, srcs, T):
    """dst [T, H] = sum of srcs (each [T, H] HBM), left to right in bf16; row tiles split between the cores."""
    nprog = nl.num_programs()
    pid = nl.program_id(0) if nprog > 1 else 0
    NT = T // 128
    per = (NT + nprog - 1) // nprog
    t0 = pid * per
    t1 = min(NT, t0 + per)
    for t in range(t0, t1):
        acc = nl.ndarray((128, H), dtype=dst.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=acc, src=srcs[0][t * 128:(t + 1) * 128, :])
        for j in range(1, len(srcs)):
            tb = nl.ndarray((128, H), dtype=dst.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=tb, src=srcs[j][t * 128:(t + 1) * 128, :])
            nisa.tensor_tensor(dst=acc, data1=acc, data2=tb, op=nl.add)
        nisa.dma_copy(dst=dst[t * 128:(t + 1) * 128, :], src=acc)


@nki.jit
def pre_oproj(x, a0, a1, wo):
    """h [1, T, H] = x + attention output projection (one projection per head group, summed)."""
    T = x.shape[1]
    qd = NQ * D // G
    o0 = output_projection_cte(attention=a0.reshape((1, NQ // G, D, T)), weight=wo[0:qd, :])
    o1 = output_projection_cte(attention=a1.reshape((1, NQ // G, D, T)), weight=wo[qd:2 * qd, :])
    _sync(o0)
    _sync(o1)
    h = nl.ndarray((1, T, H), dtype=x.dtype, buffer=nl.shared_hbm)
    _addrows(h.reshape((T, H)), [x.reshape((T, H)), o0.reshape((T, H)), o1.reshape((T, H))], T)
    return h


@nki.jit
def pre_mlp(h, acc, g2, wg, wu, wd, chunk):
    """out [1, T, H] = acc + MLP of h over intermediate chunk c of NCH (chunk [NCH, c+1] carries both)."""
    T = h.shape[1]
    c = chunk.shape[1] - 1
    ICH = I // chunk.shape[0]
    sbm = SbufManager(0, 200 * 1024, get_logger("mlp"))
    sbm.set_name_prefix(f"mlp{c}_")
    res = mlp(sbm=sbm, hidden_tensor=h, gate_proj_weights_tensor=wg, up_proj_weights_tensor=wu,
              down_proj_weights_tensor=wd, normalization_weights_tensor=g2,
              inter_chunk_offset=c * ICH, inter_chunk_size=ICH,
              activation_fn=ActFnType.SiLU, normalization_type=NormType.RMS_NORM, eps=EPS)
    _sync(res[0])
    out = nl.ndarray((1, T, H), dtype=h.dtype, buffer=nl.shared_hbm)
    _addrows(out.reshape((T, H)), [acc.reshape((T, H)), res[0].reshape((T, H))], T)
    return out


# ----------------------------------------------------------------------------------------------- prefill, small T
# T <= 512: weights split between the cores (heads, intermediate columns), tokens on the moving side; nkilib's CTE
# kernels split tokens instead, so each core streams every weight (measured 2x the HBM reads at T=256).
def _xpose_rows_in(dstT, src_rows_hbm, T, n):
    """src [T, n*128] HBM rows -> dstT [128, n, T] SBUF (dstT[p, j, t] = src[t, j*128 + p]), PE transposes."""
    NTT = T // 128
    rows = []
    for tt in range(NTT):
        r = nl.ndarray((128, n * 128), dtype=dstT.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=r, src=src_rows_hbm[tt * 128:(tt + 1) * 128, 0:n * 128])
        rows.append(r)
    for j in range(n):
        tp = nl.ndarray((128, T), dtype=dstT.dtype, buffer=nl.psum)
        for tt in range(NTT):
            nisa.nc_transpose(dst=tp[0:128, tt * 128:(tt + 1) * 128], data=rows[tt][0:128, j * 128:(j + 1) * 128],
                              engine=nisa.engine.tensor)
        nisa.tensor_copy(dst=dstT[0:128, j, 0:T], src=tp)


def _xsum_hbm(part, n, T, pid, lo, hi):
    """part [128, n, T] fp32 SBUF partial of this core; returns fp32 [128, hi-lo, T] SBUF = sum of both cores'
    partials for tiles lo..hi-1 (exchange through shared HBM + core barrier)."""
    buf = nl.ndarray((2, 128, n, T), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=buf[pid], src=part)
    nisa.core_barrier(data=buf, cores=(0, 1))
    oth = nl.ndarray((128, hi - lo, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=oth, src=buf[1 - pid, 0:128, lo:hi, 0:T])
    tot = nl.ndarray((128, hi - lo, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tot, data1=part[0:128, lo:hi, 0:T], data2=oth, op=nl.add)
    return tot


@nki.jit
def pre_post(x, a, wo, g2c, wg, wu, wd):
    """Prefill, small T (<= 512): out [1, T, H] = h + MLP(h), h = x + o_proj(attention).
    x [1, T, H]; a [NQ, D, T] attention output (head, d, token); g2c [128, H/128]; weights [in, out].
    Core c: o-proj over its 16 heads (partials summed through HBM), MLP over intermediate columns c*I/2..
    (down partials summed through HBM), all in the transposed [feature, token] layout."""
    T = x.shape[1]
    TB = min(T, 256)            # token block: weights are re-streamed per block, SBUF holds one block
    out = nl.ndarray((1, T, H), dtype=x.dtype, buffer=nl.shared_hbm)
    o2 = out.reshape((T, H))
    for b in range(T // TB):
        _post_blk(x, a, wo, g2c, wg, wu, wd, o2, T, b * TB, TB)
    return out


def _post_blk(x, a, wo, g2c, wg, wu, wd, o2, T, t0, TB):
    """pre_post for tokens t0..t0+TB-1."""
    pid = nl.program_id(0)
    KT = H // 128
    QH = NQ // 2
    IC = I // 2
    NI = IC // 128
    GN = 8                      # output tiles per weight DMA: 1024 columns = 2 KB pieces (1 KB: 435 vs 550 GB/s)
    KG = 16                     # contraction tiles per weight DMA
    ones_f = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    # ---- o-proj partial over this core's heads: oT[n, t] = sum_k wo[k, n] a[k, t]
    at = nl.ndarray((128, QH, TB), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=at, src=a.ap(pattern=[[T, 128], [D * T, QH], [1, TB]], offset=pid * QH * D * T + t0))
    op = nl.ndarray((128, KT, TB), dtype=nl.float32, buffer=nl.sbuf)
    for g in range(KT // GN):
        w = nl.ndarray((128, QH, GN * 128), dtype=wo.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w, src=wo.ap(pattern=[[H, 128], [128 * H, QH], [1, GN * 128]], offset=pid * QH * D * H + g * GN * 128), dge_mode=DGE)
        acc = []
        for _v in range(GN // 2):
            acc.append(nl.ndarray((128, 2, TB), dtype=nl.float32, buffer=nl.psum))
        for u in range(GN):
            for k in range(QH):
                nisa.nc_matmul(dst=acc[u // 2][0:128, u % 2, 0:TB], stationary=w[0:128, k, u * 128:(u + 1) * 128], moving=at[0:128, k, 0:TB])
        for v in range(GN // 2):
            nisa.tensor_copy(dst=op[0:128, g * GN + 2 * v:g * GN + 2 * v + 2, 0:TB], src=acc[v])
    o = _xsum_hbm(op, KT, TB, pid, 0, KT)
    ob = nl.ndarray((128, KT, TB), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ob, src=o)
    # ---- h = x + o (transposed), rmsnorm over features
    hT = nl.ndarray((128, KT, TB), dtype=x.dtype, buffer=nl.sbuf)
    _xpose_rows_in(hT, x.reshape((T, H))[t0:t0 + TB, 0:H], TB, KT)
    nisa.tensor_tensor(dst=hT, data1=hT, data2=ob, op=nl.add)
    sq = nl.ndarray((128, KT, TB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=hT)
    s1 = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=s1, op=nl.add, data=sq.ap(pattern=[[KT * TB, 128], [1, TB], [TB, KT]], offset=0), axis=2)
    ssp = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ssp, stationary=ones_f, moving=s1, is_stationary_onezero=True)
    epsb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=epsb, value=EPS)
    rstd = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=ssp, bias=epsb, scale=1.0 / H)
    gc = nl.ndarray((128, KT), dtype=g2c.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gc, src=g2c)
    gcf = nl.ndarray((128, KT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gcf, src=gc)
    hn = nl.ndarray((128, KT, TB), dtype=x.dtype, buffer=nl.sbuf)
    for j in range(KT):
        nisa.tensor_tensor(dst=hn[0:128, j, 0:TB], data1=hT[0:128, j, 0:TB], data2=rstd, op=nl.multiply)
        nisa.tensor_scalar(dst=hn[0:128, j, 0:TB], data=hn[0:128, j, 0:TB], op0=nl.multiply, operand0=gcf[0:128, j:j + 1])
    # ---- gate/up over this core's intermediate tiles, act = silu(g) * u
    act = nl.ndarray((128, NI, TB), dtype=x.dtype, buffer=nl.sbuf)
    for g in range(NI // GN):
        c0 = pid * IC + g * GN * 128
        gu = []
        for wi in range(2):
            wmat = wg if wi == 0 else wu
            acc = []
            for _v in range(GN // 2):
                acc.append(nl.ndarray((128, 2, TB), dtype=nl.float32, buffer=nl.psum))
            for kg in range(KT // KG):
                wt = nl.ndarray((128, KG, GN * 128), dtype=wmat.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=wt, src=wmat.ap(pattern=[[I, 128], [128 * I, KG], [1, GN * 128]], offset=kg * KG * 128 * I + c0), dge_mode=DGE)
                for u in range(GN):
                    for k in range(KG):
                        nisa.nc_matmul(dst=acc[u // 2][0:128, u % 2, 0:TB], stationary=wt[0:128, k, u * 128:(u + 1) * 128],
                                       moving=hn[0:128, kg * KG + k, 0:TB])
            r = nl.ndarray((128, GN, TB), dtype=x.dtype, buffer=nl.sbuf)
            for v in range(GN // 2):
                nisa.tensor_copy(dst=r[0:128, 2 * v:2 * v + 2, 0:TB], src=acc[v])
            gu.append(r)
        nisa.activation(dst=gu[0], op=nl.silu, data=gu[0])
        nisa.tensor_tensor(dst=act[0:128, g * GN:(g + 1) * GN, 0:TB], data1=gu[0], data2=gu[1], op=nl.multiply)
    # ---- down partial over this core's intermediate rows
    yp = nl.ndarray((128, KT, TB), dtype=nl.float32, buffer=nl.sbuf)
    for g in range(KT // GN):
        acc = []
        for _v in range(GN // 2):
            acc.append(nl.ndarray((128, 2, TB), dtype=nl.float32, buffer=nl.psum))
        for kg in range(NI // KG):
            wdt = nl.ndarray((128, KG, GN * 128), dtype=wd.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=wdt, src=wd.ap(pattern=[[H, 128], [128 * H, KG], [1, GN * 128]], offset=(pid * IC + kg * KG * 128) * H + g * GN * 128), dge_mode=DGE)
            for u in range(GN):
                for k in range(KG):
                    nisa.nc_matmul(dst=acc[u // 2][0:128, u % 2, 0:TB], stationary=wdt[0:128, k, u * 128:(u + 1) * 128],
                                   moving=act[0:128, kg * KG + k, 0:TB])
        for v in range(GN // 2):
            nisa.tensor_copy(dst=yp[0:128, g * GN + 2 * v:g * GN + 2 * v + 2, 0:TB], src=acc[v])
    # ---- this core finishes feature tiles pid*KT/2..: out = h + y, back to rows
    KH2 = KT // 2
    y = _xsum_hbm(yp, KT, TB, pid, pid * KH2, (pid + 1) * KH2)
    yb = nl.ndarray((128, KH2, TB), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=yb, src=y)
    nisa.tensor_tensor(dst=yb, data1=yb, data2=hT[0:128, pid * KH2:(pid + 1) * KH2, 0:TB], op=nl.add)
    for tt in range(TB // 128):
        tp = nl.ndarray((128, KH2 * 128), dtype=x.dtype, buffer=nl.psum)
        for j in range(KH2):
            nisa.nc_transpose(dst=tp[0:128, j * 128:(j + 1) * 128], data=yb[0:128, j, tt * 128:(tt + 1) * 128],
                              engine=nisa.engine.tensor)
        r = nl.ndarray((128, KH2 * 128), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=r, src=tp)
        nisa.dma_copy(dst=o2[t0 + tt * 128:t0 + (tt + 1) * 128, pid * KH2 * 128:(pid + 1) * KH2 * 128], src=r)



def _rms_T(hn, hT, g, ones_f, T, n):
    """hn [128, n, T] = bf16(hT * rsqrt(mean over features + eps)) * gamma; hT [128, n, T] bf16 (feature j*128+p
    at [p, j, t]); g [128, n] gamma in column layout (SBUF, bf16)."""
    sq = nl.ndarray((128, n, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=hT)
    s1 = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=s1, op=nl.add, data=sq.ap(pattern=[[n * T, 128], [1, T], [T, n]], offset=0), axis=2)
    ssp = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ssp, stationary=ones_f, moving=s1, is_stationary_onezero=True)
    epsb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=epsb, value=EPS)
    rstd = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=ssp, bias=epsb, scale=1.0 / (n * 128))
    gf = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gf, src=g)
    for j in range(n):
        nisa.tensor_tensor(dst=hn[0:128, j, 0:T], data1=hT[0:128, j, 0:T], data2=rstd, op=nl.multiply)
        nisa.tensor_scalar(dst=hn[0:128, j, 0:T], data=hn[0:128, j, 0:T], op0=nl.multiply, operand0=gf[0:128, j:j + 1])


@nki.jit
def pre_qkv2(x, g1c, wqkv, qn, kn, rcT, rsT, rot, kc, vc):
    """Prefill, small T: rmsnorm, q/k/v projections, q/k rmsnorm, rope; writes K (transposed) and V into the layer's
    caches (positions 0..T-1) and returns q [NQ, D, T], k [NKV, D, T], v [NKV, T, D] for attention_cte
    (tp_q=False, tp_k=False). rcT/rsT [D, SMAX]: rope tables transposed (halves repeated); qn/kn [D, 1];
    rot [D, D] rotate-half matrix. Core c: q heads 16c.., kv heads 4c.."""
    T = x.shape[1]
    TB = min(T, 256)
    qo = nl.ndarray((NQ, D, T), dtype=x.dtype, buffer=nl.shared_hbm)
    ko = nl.ndarray((NKV, D, T), dtype=x.dtype, buffer=nl.shared_hbm)
    vo = nl.ndarray((NKV, T, D), dtype=x.dtype, buffer=nl.shared_hbm)
    for b in range(T // TB):
        _qkv_blk(x, g1c, wqkv, qn, kn, rcT, rsT, rot, kc, vc, qo, ko, vo, T, b * TB, TB)
    return qo, ko, vo


def _qkv_blk(x, g1c, wqkv, qn, kn, rcT, rsT, rot, kc, vc, qo, ko, vo, T, t0, TB):
    """pre_qkv2 for tokens t0..t0+TB-1."""
    SMAX = kc.shape[2]
    pid = nl.program_id(0)
    KT = H // 128
    QH = NQ // 2
    KH = NKV // 2
    NQK = QH + KH
    ones_f = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    xT = nl.ndarray((128, KT, TB), dtype=x.dtype, buffer=nl.sbuf)
    _xpose_rows_in(xT, x.reshape((T, H))[t0:t0 + TB, 0:H], TB, KT)
    gc = nl.ndarray((128, KT), dtype=g1c.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gc, src=g1c)
    xn = nl.ndarray((128, KT, TB), dtype=x.dtype, buffer=nl.sbuf)
    _rms_T(xn, xT, gc, ones_f, TB, KT)
    QKVW = (NQ + 2 * NKV) * D
    # ---- q and k heads (transposed outputs [d, TB])
    qk = nl.ndarray((128, NQK, TB), dtype=x.dtype, buffer=nl.sbuf)
    GN = 8                                    # heads per weight DMA (1024 columns: 2 KB pieces)
    KG = 16
    for g in range((NQK + GN - 1) // GN):     # 8 q heads, 8 q heads, then the 4 k heads
        nh = min(GN, NQK - g * GN)
        c0 = pid * QH * D + g * GN * D if g < QH // GN else NQ * D + pid * KH * D
        acc = []
        for _v in range(GN // 2):
            acc.append(nl.ndarray((128, 2, TB), dtype=nl.float32, buffer=nl.psum))
        for kg in range(KT // KG):
            w = nl.ndarray((128, KG, GN * 128), dtype=wqkv.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=w[0:128, 0:KG, 0:nh * 128], src=wqkv.ap(pattern=[[QKVW, 128], [128 * QKVW, KG], [1, nh * 128]],
                          offset=kg * KG * 128 * QKVW + c0), dge_mode=DGE)
            for u in range(nh):
                for k in range(KG):
                    nisa.nc_matmul(dst=acc[u // 2][0:128, u % 2, 0:TB], stationary=w[0:128, k, u * 128:(u + 1) * 128],
                                   moving=xn[0:128, kg * KG + k, 0:TB])
        for v in range(nh // 2):
            nisa.tensor_copy(dst=qk[0:128, g * GN + 2 * v:g * GN + 2 * v + 2, 0:TB], src=acc[v])
    # ---- v rows [t, (head, d)]
    wv = nl.ndarray((128, KT, KH * D), dtype=wqkv.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=wv, src=wqkv.ap(pattern=[[QKVW, 128], [128 * QKVW, KT], [1, KH * D]], offset=(NQ + NKV) * D + pid * KH * D), dge_mode=DGE)
    vr = nl.ndarray((128, TB // 128, KH * D), dtype=x.dtype, buffer=nl.sbuf)
    for tt in range(TB // 128):
        ps = nl.ndarray((128, KH * D), dtype=nl.float32, buffer=nl.psum)
        for k in range(KT):
            nisa.nc_matmul(dst=ps, stationary=xn[0:128, k, tt * 128:(tt + 1) * 128], moving=wv[0:128, k, 0:KH * D])
        nisa.tensor_copy(dst=vr[0:128, tt, 0:KH * D], src=ps)
    # ---- q/k rmsnorm over d (partitions), gamma
    sq = nl.ndarray((128, NQK, TB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=qk)
    rstd = nl.ndarray((128, NQK, TB), dtype=nl.float32, buffer=nl.sbuf)
    epsb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=epsb, value=EPS)
    for j in range(NQK):
        ssp = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ssp, stationary=ones_f, moving=sq[0:128, j, 0:TB], is_stationary_onezero=True)
        nisa.activation(dst=rstd[0:128, j, 0:TB], op=nl.rsqrt, data=ssp, bias=epsb, scale=1.0 / D)
    nqk = nl.ndarray((128, NQK, TB), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=nqk, data1=qk, data2=rstd, op=nl.multiply)
    gq = nl.ndarray((128, 1), dtype=qn.dtype, buffer=nl.sbuf)
    gk = nl.ndarray((128, 1), dtype=kn.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gq, src=qn)
    nisa.dma_copy(dst=gk, src=kn)
    gqf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    gkf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gqf, src=gq)
    nisa.tensor_copy(dst=gkf, src=gk)
    nisa.tensor_scalar(dst=nqk[0:128, 0:QH, 0:TB], data=nqk[0:128, 0:QH, 0:TB], op0=nl.multiply, operand0=gqf)
    nisa.tensor_scalar(dst=nqk[0:128, QH:NQK, 0:TB], data=nqk[0:128, QH:NQK, 0:TB], op0=nl.multiply, operand0=gkf)
    # ---- rope: e = bf16(n * cos) + bf16(rotate_half(n) * sin), cos/sin [d, t]
    ct = nl.ndarray((128, TB), dtype=rcT.dtype, buffer=nl.sbuf)
    st = nl.ndarray((128, TB), dtype=rsT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=ct, src=rcT[0:D, t0:t0 + TB])
    nisa.dma_copy(dst=st, src=rsT[0:D, t0:t0 + TB])
    rot_sb = nl.ndarray((D, D), dtype=rot.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_sb, src=rot)
    e = nl.ndarray((128, NQK, TB), dtype=x.dtype, buffer=nl.sbuf)
    for j in range(NQK):
        rp = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=rp, stationary=rot_sb, moving=nqk[0:128, j, 0:TB])
        ea = nl.ndarray((128, TB), dtype=x.dtype, buffer=nl.sbuf)
        eb = nl.ndarray((128, TB), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ea, data1=nqk[0:128, j, 0:TB], data2=ct, op=nl.multiply)
        nisa.tensor_tensor(dst=eb, data1=rp, data2=st, op=nl.multiply)
        nisa.tensor_tensor(dst=e[0:128, j, 0:TB], data1=ea, data2=eb, op=nl.add)
    # ---- outputs and cache writes
    nisa.dma_copy(dst=qo.ap(pattern=[[T, 128], [D * T, QH], [1, TB]], offset=pid * QH * D * T + t0), src=e[0:128, 0:QH, 0:TB])
    nisa.dma_copy(dst=ko.ap(pattern=[[T, 128], [D * T, KH], [1, TB]], offset=pid * KH * D * T + t0), src=e[0:128, QH:NQK, 0:TB])
    nisa.dma_copy(dst=kc.ap(pattern=[[SMAX, 128], [D * SMAX, KH], [1, TB]], offset=pid * KH * D * SMAX + t0), src=e[0:128, QH:NQK, 0:TB])
    for tt in range(TB // 128):
        nisa.dma_copy(dst=vo.ap(pattern=[[D, 128], [T * D, KH], [1, D]], offset=pid * KH * T * D + (t0 + tt * 128) * D),
                      src=vr[0:128, tt, 0:KH * D])
        nisa.dma_copy(dst=vc.ap(pattern=[[D, 128], [SMAX * D, KH], [1, D]], offset=pid * KH * SMAX * D + (t0 + tt * 128) * D),
                      src=vr[0:128, tt, 0:KH * D])
    return qo, ko, vo



@nki.jit
def pre_attn2(q, k, v):
    """Causal attention of each head group on the pre_qkv2 layouts. Returns a0, a1 [NQ/2, D, T]."""
    T = q.shape[2]
    nq = NQ // 2
    nk = NKV // 2
    a0 = attention_cte(q=q[0:nq], k=k[0:nk], v=v[0:nk], scale=1.0 / (D ** 0.5), causal_mask=True,
                       tp_q=False, tp_k=False, tp_out=True)
    a1 = attention_cte(q=q[nq:NQ], k=k[nk:NKV], v=v[nk:NKV], scale=1.0 / (D ** 0.5), causal_mask=True,
                       tp_q=False, tp_k=False, tp_out=True)
    return a0, a1


# ----------------------------------------------------------------------------------------------- prefill, X layout
# The prefill residual stream lives in HBM as X [NB, 128, H/128, TB]: X[b, p, j, t] = x[b*TB + t, j*128 + p]
# (token block b, feature j*128 + p), so a block loads contiguously in the [feature, token] layout the kernels use.
def tb_of(T):
    """token block of bucket T (src/engine.cpp tb_of)"""
    return min(T, 512)


def gn_of(TB):
    """output tiles per weight DMA: [128, TB] fp32 accumulators x GN must leave PSUM room (16 KB per partition)"""
    return 8 if TB <= 256 else 4


@nki.jit
def embed_x(toks, emb):
    """toks int32 [T, 1] -> X [T/TB, 128, H/128, TB] (embedding rows, transposed per block)."""
    T = toks.shape[0]
    TB = tb_of(T)
    KT = H // 128
    X = nl.ndarray((T // TB, 128, KT, TB), dtype=emb.dtype, buffer=nl.shared_hbm)
    for t in range(T // 128):
        ids = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ids, src=toks[t * 128:(t + 1) * 128, :])
        rows = nl.ndarray((128, H), dtype=emb.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=rows, src=emb.ap(pattern=[[H, 128], [1, H]], offset=0, vector_offset=ids, indirect_dim=0))
        st = nl.ndarray((128, KT, 128), dtype=emb.dtype, buffer=nl.sbuf)
        for j4 in range(KT // 4):
            tp = nl.ndarray((128, 4, 128), dtype=emb.dtype, buffer=nl.psum)
            for u in range(4):
                j = j4 * 4 + u
                nisa.nc_transpose(dst=tp[0:128, u, 0:128], data=rows[0:128, j * 128:(j + 1) * 128], engine=nisa.engine.tensor)
            nisa.tensor_copy(dst=st[0:128, j4 * 4:(j4 + 1) * 4, 0:128], src=tp)
        b = (t * 128) // TB
        o = (t * 128) % TB
        nisa.dma_copy(dst=X[b, 0:128, 0:KT, o:o + 128], src=st)
    return X


@nki.jit
def x_row(X, off):
    """The residual row of one token from X -> [1, 1, H] (input of the LM head). off int32 [1, 1]: the element
    offset of that token's feature 0, b * 128 * KT * TB + t (computed by the host)."""
    NB, P, KT, TB = X.shape
    off_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=off_sb, src=off)
    col = nl.ndarray((128, KT), dtype=X.dtype, buffer=nl.sbuf)
    Xf = X.reshape((NB * 128 * KT * TB, 1))
    nisa.dma_copy(dst=col, src=Xf.ap(pattern=[[KT * TB, 128], [TB, KT]], offset=0, scalar_offset=off_sb, indirect_dim=0))
    rp = nl.ndarray((KT, 128), dtype=X.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=rp, data=col, engine=nisa.engine.tensor)
    r = nl.ndarray((KT, 128), dtype=X.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r, src=rp)
    out = nl.ndarray((1, 1, H), dtype=X.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out.reshape((KT, 128)), src=r)
    return out


def _ssq_acc(s1, hT, j0, j1, TB):
    """_ssq_rstd's sum of squares over feature tiles j0..j1-1, continuing s1 (None: start at tile j0 = 0)."""
    for j in range(j0, j1):
        if s1 is None:
            s1 = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=s1, op=nl.square, data=hT[0:128, j, 0:TB])
        else:
            sq = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=sq, op=nl.square, data=hT[0:128, j, 0:TB])
            nisa.tensor_tensor(dst=s1, data1=s1, data2=sq, op=nl.add)
    return s1


def _ssq_fin(s1, ones_f, TB, n):
    """rstd from the per-partition sum of squares s1 (the rest of _ssq_rstd)."""
    ssp = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ssp, stationary=ones_f, moving=s1, is_stationary_onezero=True)
    epsb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=epsb, value=EPS)
    rstd = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=ssp, bias=epsb, scale=1.0 / (n * 128))
    return rstd


def _ssq_rstd(hT, ones_f, TB, n, add_eng="cfg"):
    """rstd [128, TB] fp32 = rsqrt(mean_features(hT^2) + eps) on every partition; hT [128, n, TB] bf16.
    Squares on the scalar engine, the sum over the n feature tiles on the vector engine, one PE matmul with ones for
    the sum over partitions (32 accumulating fp32 matmuls kept the PE from the projections)."""
    s1 = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=s1, op=nl.square, data=hT[0:128, 0, 0:TB])
    for j in range(1, n):
        sq = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sq, op=nl.square, data=hT[0:128, j, 0:TB])
        nisa.tensor_tensor(dst=s1, data1=s1, data2=sq, op=nl.add, engine=_ENG2[SSQ_ENG if add_eng == "cfg" else add_eng])
    ssp = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ssp, stationary=ones_f, moving=s1, is_stationary_onezero=True)
    epsb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=epsb, value=EPS)
    rstd = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=ssp, bias=epsb, scale=1.0 / (n * 128))
    return rstd


@nki.jit
def pre_post3(X, a, wo, g2c, wg, wu, wd):
    """X' = h + MLP(h), h = X + o_proj(attention), in the X layout; a [NQ, D, T] attention output.
    Core c: o-proj over its 16 heads, MLP over intermediate columns c*I/2..; fp32 partials summed through HBM."""
    NB, P_, KT, TB = X.shape
    T = NB * TB
    Xo = nl.ndarray((NB, 128, KT, TB), dtype=X.dtype, buffer=nl.shared_hbm)
    for b in range(NB):
        if POST5:
            _post5_blk(X, a, wo, g2c, wg, wu, wd, Xo, T, b, TB, 0)
        else:
            _post3_blk(X, a, wo, g2c, wg, wu, wd, Xo, T, b, TB, 0)
    return Xo


def _post3_blk(X, a, wo, g2c, wg, wu, wd, Xo, T, b, TB, l):
    pid = nl.program_id(0)
    dt = X.dtype
    KT = H // 128
    KH2 = KT // 2
    QH = NQ // 2
    IC = I // 2
    NI = IC // 128
    GN = gn_of(TB)
    GU = GN                        # gate/up tiles per weight DMA (12 at TB=256, 3 KB pieces: 0.76 -> 0.92 ms, slower)
    KG = 16
    t0 = b * TB
    ones_f = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    obuf = nl.ndarray((2, 128, KT, TB), dtype=nl.float32, buffer=nl.shared_hbm)
    ybuf = nl.ndarray((2, 128, KH2, TB), dtype=nl.float32, buffer=nl.shared_hbm)
    # ---- o-proj partial over this core's heads -> obuf[pid]
    at = nl.ndarray((128, QH, TB), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=at, src=a.ap(pattern=[[T, 128], [D * T, QH], [1, TB]], offset=pid * QH * D * T + t0))
    for g in range(KT // GN):
        w = nl.ndarray((128, QH, GN * 128), dtype=wo.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w, src=wo.ap(pattern=[[H, 128], [128 * H, QH], [1, GN * 128]], offset=l * H * H + pid * QH * D * H + g * GN * 128), dge_mode=DGE)
        acc = []
        for _v in range(GN):
            acc.append(nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum))
        for u in range(GN):
            for k in range(QH):
                nisa.nc_matmul(dst=acc[u], stationary=w[0:128, k, u * 128:(u + 1) * 128], moving=at[0:128, k, 0:TB])
        stg = nl.ndarray((128, GN, TB), dtype=nl.float32, buffer=nl.sbuf)
        for u in range(GN):
            nisa.tensor_copy(dst=stg[0:128, u, 0:TB], src=acc[u])
        nisa.dma_copy(dst=obuf[pid, 0:128, g * GN:(g + 1) * GN, 0:TB], src=stg)
    nisa.core_barrier(data=obuf, cores=(0, 1))
    # ---- h = x + bf16(o0 + o1); keep this core's half of h for the final residual
    hT = nl.ndarray((128, KT, TB), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=hT, src=X[b])
    for g in range(KT // 8):
        o0 = nl.ndarray((128, 8, TB), dtype=nl.float32, buffer=nl.sbuf)
        o1 = nl.ndarray((128, 8, TB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=o0, src=obuf[0, 0:128, g * 8:(g + 1) * 8, 0:TB])
        nisa.dma_copy(dst=o1, src=obuf[1, 0:128, g * 8:(g + 1) * 8, 0:TB])
        ob = nl.ndarray((128, 8, TB), dtype=dt, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ob, data1=o0, data2=o1, op=nl.add)
        nisa.tensor_tensor(dst=hT[0:128, g * 8:(g + 1) * 8, 0:TB], data1=hT[0:128, g * 8:(g + 1) * 8, 0:TB], data2=ob, op=nl.add)
    hh = nl.ndarray((128, KH2, TB), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=hh, src=hT[0:128, pid * KH2:(pid + 1) * KH2, 0:TB])
    rstd = _ssq_rstd(hT, ones_f, TB, KT)
    gc = nl.ndarray((128, KT), dtype=g2c.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gc, src=g2c[l])
    gcf = nl.ndarray((128, KT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gcf, src=gc)
    hn = nl.ndarray((128, KT, TB), dtype=dt, buffer=nl.sbuf)
    for j in range(KT):
        nisa.tensor_tensor(dst=hn[0:128, j, 0:TB], data1=hT[0:128, j, 0:TB], data2=rstd, op=nl.multiply)
        nisa.tensor_scalar(dst=hn[0:128, j, 0:TB], data=hn[0:128, j, 0:TB], op0=nl.multiply, operand0=gcf[0:128, j:j + 1])
    # ---- gate/up -> act = silu(g) * u
    act = nl.ndarray((128, NI, TB), dtype=dt, buffer=nl.sbuf)
    for g in range(NI // GU):
        c0 = pid * IC + g * GU * 128
        gu = []
        for wi in range(2):
            wmat = wg if wi == 0 else wu
            acc = []
            for _v in range(GU):
                acc.append(nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum))
            for kg in range(KT // KG):
                wt = nl.ndarray((128, KG, GU * 128), dtype=wmat.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=wt, src=wmat.ap(pattern=[[I, 128], [128 * I, KG], [1, GU * 128]], offset=l * H * I + kg * KG * 128 * I + c0), dge_mode=DGE)
                for u in range(GU):
                    for k in range(KG):
                        nisa.nc_matmul(dst=acc[u], stationary=wt[0:128, k, u * 128:(u + 1) * 128], moving=hn[0:128, kg * KG + k, 0:TB])
            r = nl.ndarray((128, GU, TB), dtype=dt, buffer=nl.sbuf)
            for u in range(GU):
                nisa.tensor_copy(dst=r[0:128, u, 0:TB], src=acc[u])
            gu.append(r)
        nisa.activation(dst=gu[0], op=nl.silu, data=gu[0])
        nisa.tensor_tensor(dst=act[0:128, g * GU:(g + 1) * GU, 0:TB], data1=gu[0], data2=gu[1], op=nl.multiply)
    # ---- down partial -> ybuf (each core sends the other core's half, keeps its own in SBUF)
    yown = nl.ndarray((128, KH2, TB), dtype=nl.float32, buffer=nl.sbuf)
    for g in range(KT // GN):
        acc = []
        for _v in range(GN):
            acc.append(nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum))
        for kg in range(NI // KG):
            wdt = nl.ndarray((128, KG, GN * 128), dtype=wd.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=wdt, src=wd.ap(pattern=[[H, 128], [128 * H, KG], [1, GN * 128]], offset=l * I * H + (pid * IC + kg * KG * 128) * H + g * GN * 128), dge_mode=DGE)
            for u in range(GN):
                for k in range(KG):
                    nisa.nc_matmul(dst=acc[u], stationary=wdt[0:128, k, u * 128:(u + 1) * 128], moving=act[0:128, kg * KG + k, 0:TB])
        # tiles g*GN.. belong to core (g*GN) // KH2
        owner = (g * GN) // KH2
        lo = g * GN - owner * KH2
        if owner == pid:
            for u in range(GN):
                nisa.tensor_copy(dst=yown[0:128, lo + u, 0:TB], src=acc[u])
        else:
            stg = nl.ndarray((128, GN, TB), dtype=nl.float32, buffer=nl.sbuf)
            for u in range(GN):
                nisa.tensor_copy(dst=stg[0:128, u, 0:TB], src=acc[u])
            nisa.dma_copy(dst=ybuf[pid, 0:128, lo:lo + GN, 0:TB], src=stg)
    nisa.core_barrier(data=ybuf, cores=(0, 1))
    yo = nl.ndarray((128, KH2, TB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=yo, src=ybuf[1 - pid])
    yb = nl.ndarray((128, KH2, TB), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=yb, data1=yown, data2=yo, op=nl.add)
    nisa.tensor_tensor(dst=yb, data1=hh, data2=yb, op=nl.add)
    nisa.dma_copy(dst=Xo[b, 0:128, pid * KH2:(pid + 1) * KH2, 0:TB], src=yb)


def _post5_blk(X, a, wo, g2c, wg, wu, wd, Xo, T, b, TB, l):
    """_post3_blk with the fp32 partial sums swapped between the cores by nisa.sendrecv (SBUF -> SBUF) instead of
    an HBM buffer + core_barrier: o-proj per output group (both cores need all of h), down per pair of groups
    (g owned by core 0, g + KT/(2 GN) by core 1: each core sends the partial of the other core's group). Same
    arithmetic and rounding points: h = x + bf16(o0 + o1), out = h + bf16(y0 + y1)."""
    pid = nl.program_id(0)
    dt = X.dtype
    KT = H // 128
    KH2 = KT // 2
    QH = NQ // 2
    IC = I // 2
    NI = IC // 128
    GN = gn_of(TB)
    GU = GN
    KG = 16
    t0 = b * TB
    ones_f = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    at = nl.ndarray((128, QH, TB), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=at, src=a.ap(pattern=[[T, 128], [D * T, QH], [1, TB]], offset=pid * QH * D * T + t0))
    xb = nl.ndarray((128, KT, TB), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=xb, src=X[b])
    # ---- o-proj partial per group, swapped; h = x + bf16(o0 + o1)
    hT = nl.ndarray((128, KT, TB), dtype=dt, buffer=nl.sbuf)
    pend = []
    for g in range(KT // GN):
        w = nl.ndarray((128, QH, GN * 128), dtype=wo.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w, src=wo.ap(pattern=[[H, 128], [128 * H, QH], [1, GN * 128]], offset=l * H * H + pid * QH * D * H + g * GN * 128), dge_mode=DGE)
        acc = []
        for _v in range(GN):
            acc.append(nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum))
        for u in range(GN):
            for k in range(QH):
                nisa.nc_matmul(dst=acc[u], stationary=w[0:128, k, u * 128:(u + 1) * 128], moving=at[0:128, k, 0:TB])
        stg = nl.ndarray((128, GN, TB), dtype=nl.float32, buffer=nl.sbuf)
        for u in range(GN):
            nisa.tensor_copy(dst=stg[0:128, u, 0:TB], src=acc[u])
        pend.append((g, stg, _swap(stg, g)))
        if len(pend) > POST5_LAG:
            _fin(pend[0:1], hT, xb, GN, TB)
            pend = pend[1:]
    _fin(pend, hT, xb, GN, TB)
    rstd = _ssq_rstd(hT, ones_f, TB, KT)
    gc = nl.ndarray((128, KT), dtype=g2c.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gc, src=g2c[l])
    gcf = nl.ndarray((128, KT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gcf, src=gc)
    hn = nl.ndarray((128, KT, TB), dtype=dt, buffer=nl.sbuf)
    for j in range(KT):
        nisa.tensor_tensor(dst=hn[0:128, j, 0:TB], data1=hT[0:128, j, 0:TB], data2=rstd, op=nl.multiply)
        nisa.tensor_scalar(dst=hn[0:128, j, 0:TB], data=hn[0:128, j, 0:TB], op0=nl.multiply, operand0=gcf[0:128, j:j + 1])
    # ---- gate/up -> act = silu(g) * u
    act = nl.ndarray((128, NI, TB), dtype=dt, buffer=nl.sbuf)
    for g in range(NI // GU):
        c0 = pid * IC + g * GU * 128
        gu = []
        for wi in range(2):
            wmat = wg if wi == 0 else wu
            acc = []
            for _v in range(GU):
                acc.append(nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum))
            for kg in range(KT // KG):
                wt = nl.ndarray((128, KG, GU * 128), dtype=wmat.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=wt, src=wmat.ap(pattern=[[I, 128], [128 * I, KG], [1, GU * 128]], offset=l * H * I + kg * KG * 128 * I + c0), dge_mode=DGE)
                for u in range(GU):
                    for k in range(KG):
                        nisa.nc_matmul(dst=acc[u], stationary=wt[0:128, k, u * 128:(u + 1) * 128], moving=hn[0:128, kg * KG + k, 0:TB])
            r = nl.ndarray((128, GU, TB), dtype=dt, buffer=nl.sbuf)
            for u in range(GU):
                nisa.tensor_copy(dst=r[0:128, u, 0:TB], src=acc[u])
            gu.append(r)
        nisa.activation(dst=gu[0], op=nl.silu, data=gu[0])
        nisa.tensor_tensor(dst=act[0:128, g * GU:(g + 1) * GU, 0:TB], data1=gu[0], data2=gu[1], op=nl.multiply)
    # ---- down partials in the group order 0, NG/2, 1, NG/2 + 1, ...; each pair swapped (own group <- other core)
    NG = KT // GN
    HG = NG // 2
    yb = nl.ndarray((128, KH2, TB), dtype=dt, buffer=nl.sbuf)
    stgs = []
    pend = []
    for gi in range(NG):
        g = (gi // 2) + (gi % 2) * HG
        acc = []
        for _v in range(GN):
            acc.append(nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum))
        for kg in range(NI // KG):
            wdt = nl.ndarray((128, KG, GN * 128), dtype=wd.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=wdt, src=wd.ap(pattern=[[H, 128], [128 * H, KG], [1, GN * 128]], offset=l * I * H + (pid * IC + kg * KG * 128) * H + g * GN * 128), dge_mode=DGE)
            for u in range(GN):
                for k in range(KG):
                    nisa.nc_matmul(dst=acc[u], stationary=wdt[0:128, k, u * 128:(u + 1) * 128], moving=act[0:128, kg * KG + k, 0:TB])
        stg = nl.ndarray((128, GN, TB), dtype=nl.float32, buffer=nl.sbuf)
        for u in range(GN):
            nisa.tensor_copy(dst=stg[0:128, u, 0:TB], src=acc[u])
        stgs.append(stg)
        if gi % 2 == 1:
            p = gi // 2                  # pair (p, p + HG): core 0 owns group p, core 1 group p + HG
            own = stgs[0] if pid == 0 else stgs[1]
            other = stgs[1] if pid == 0 else stgs[0]
            oth = nl.ndarray((128, GN, TB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.sendrecv(src=other, dst=oth, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=p % 2)
            pend.append((p, own, oth))
            stgs = []
            if len(pend) > POST5_LAG:
                _fin(pend[0:1], yb, hT[0:128, pid * KH2:(pid + 1) * KH2, 0:TB], GN, TB)
                pend = pend[1:]
    _fin(pend, yb, hT[0:128, pid * KH2:(pid + 1) * KH2, 0:TB], GN, TB)
    nisa.dma_copy(dst=Xo[b, 0:128, pid * KH2:(pid + 1) * KH2, 0:TB], src=yb)


def _q4_proj(xn, wqkv, l, g, GN, TB, qk):
    """projections of head group g (GN heads; the last group holds the k heads) of one block into qk [128, NQK, TB]
    (weight slabs of more than one group per DMA were slower: 1.976 -> 2.19 ms for 2 groups at T=4096)"""
    pid = nl.program_id(0)
    KT = H // 128
    QH = NQ // 2
    KH = NKV // 2
    NQK = QH + KH
    QKVW = (NQ + 2 * NKV) * D
    NQK2 = QH + 2 * KH
    KG = 16
    nh = min(GN, NQK - g * GN)
    c0 = pid * NQK2 * D + g * GN * D
    acc = []
    for _v in range(nh):
        acc.append(nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum))
    for kg in range(KT // KG):
        w = nl.ndarray((128, KG, GN * 128), dtype=wqkv.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w[0:128, 0:KG, 0:nh * 128], src=wqkv.ap(pattern=[[QKVW, 128], [128 * QKVW, KG], [1, nh * 128]],
                      offset=l * H * QKVW + kg * KG * 128 * QKVW + c0), dge_mode=DGE)
        for u in range(nh):
            for k in range(KG):
                nisa.nc_matmul(dst=acc[u], stationary=w[0:128, k, u * 128:(u + 1) * 128], moving=xn[0:128, kg * KG + k, 0:TB])
    for u in range(nh):
        nisa.tensor_copy(dst=qk[0:128, g * GN + u, 0:TB], src=acc[u], engine=_ENG2[Q4_DRAIN])


def _q4_v(xn, wqkv, l, TB, X):
    """v rows [t, (head, d)] of one block"""
    pid = nl.program_id(0)
    KT = H // 128
    QH = NQ // 2
    KH = NKV // 2
    QKVW = (NQ + 2 * NKV) * D
    NQK2 = QH + 2 * KH
    wv = nl.ndarray((128, KT, KH * D), dtype=wqkv.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=wv, src=wqkv.ap(pattern=[[QKVW, 128], [128 * QKVW, KT], [1, KH * D]], offset=l * H * QKVW + pid * NQK2 * D + (QH + KH) * D), dge_mode=DGE)
    vr = nl.ndarray((128, TB // 128, KH * D), dtype=X.dtype, buffer=nl.sbuf)
    for tt in range(TB // 128):
        ps = nl.ndarray((128, KH * D), dtype=nl.float32, buffer=nl.psum)
        for k in range(KT):
            nisa.nc_matmul(dst=ps, stationary=xn[0:128, k, tt * 128:(tt + 1) * 128], moving=wv[0:128, k, 0:KH * D])
        nisa.tensor_copy(dst=vr[0:128, tt, 0:KH * D], src=ps, engine=_ENG2[Q4_DRAIN])
    return vr


def _q4_post(qk, e, g, GN, TB, cst):
    """q/k rmsnorm (gamma) and rope of head group g: e[:, heads] (same arithmetic as _qkv3_blk).
    cst = [ones_f, epsb, gqf, gkf, ct, st, rot_sb]"""
    QH = NQ // 2
    KH = NKV // 2
    NQK = QH + KH
    dt = e.dtype
    h0 = g * GN
    nh = min(GN, NQK - h0)
    sq = nl.ndarray((128, GN, TB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq[0:128, 0:nh, 0:TB], op=nl.square, data=qk[0:128, h0:h0 + nh, 0:TB])
    rstd = nl.ndarray((128, GN, TB), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(nh):
        ssp = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ssp, stationary=cst[0], moving=sq[0:128, j, 0:TB], is_stationary_onezero=True)
        nisa.activation(dst=rstd[0:128, j, 0:TB], op=nl.rsqrt, data=ssp, bias=cst[1], scale=1.0 / D)
    nqk = nl.ndarray((128, GN, TB), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=nqk[0:128, 0:nh, 0:TB], data1=qk[0:128, h0:h0 + nh, 0:TB], data2=rstd[0:128, 0:nh, 0:TB], op=nl.multiply)
    gam = cst[2] if h0 < QH else cst[3]
    nisa.tensor_scalar(dst=nqk[0:128, 0:nh, 0:TB], data=nqk[0:128, 0:nh, 0:TB], op0=nl.multiply, operand0=gam)
    for j in range(nh):
        rp = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=rp, stationary=cst[6], moving=nqk[0:128, j, 0:TB])
        ea = nl.ndarray((128, TB), dtype=dt, buffer=nl.sbuf)
        eb = nl.ndarray((128, TB), dtype=dt, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ea, data1=nqk[0:128, j, 0:TB], data2=cst[4], op=nl.multiply)
        nisa.tensor_tensor(dst=eb, data1=rp, data2=cst[5], op=nl.multiply)
        nisa.tensor_tensor(dst=e[0:128, h0 + j, 0:TB], data1=ea, data2=eb, op=nl.add)


def _qkv4(X, g1c, wqkv, qn, kn, rcT, rsT, rot, kc, vc, qo, ko, vo, l, bsel=None, xn_given=None, fill=None):
    """pre_qkv3's work for every block of X, software pipelined over head groups: the q/k norm + rope of head
    group u is issued after the projections of group u + Q4_LAG (crossing into the next block), so the PE runs the
    next projections while the vector and scalar engines post-process. Returns the per-block (e, vr) lists.
    bsel: only block bsel of X; xn_given: the normalized input of a single block (X unused, block 0)."""
    if xn_given is not None:
        NB = 1
        TB = xn_given.shape[2]
        dt = xn_given.dtype
    else:
        NB = X.shape[0]
        TB = X.shape[3]
        dt = X.dtype
    T = NB * TB                     # (qo/ko/vo are only passed with every block)
    blocks = []
    if bsel is not None:
        blocks.append(bsel)
    else:
        for b in range(NB):
            blocks.append(b)
    SMAX = kc.shape[3]
    pid = nl.program_id(0)
    QH = NQ // 2
    KH = NKV // 2
    NQK = QH + KH
    GN = gn_of(TB) // 2 if Q4_GN == 0 else Q4_GN
    NG = (NQK + GN - 1) // GN
    ones_f = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    epsb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=epsb, value=EPS)
    gq = nl.ndarray((128, 1), dtype=qn.dtype, buffer=nl.sbuf)
    gk = nl.ndarray((128, 1), dtype=kn.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gq, src=qn[l])
    nisa.dma_copy(dst=gk, src=kn[l])
    gqf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    gkf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gqf, src=gq)
    nisa.tensor_copy(dst=gkf, src=gk)
    rot_sb = nl.ndarray((D, D), dtype=rot.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_sb, src=rot)
    qks = []
    es = []
    vrs = []
    csts = []
    units = []
    for bi in range(len(blocks)):
        for g in range(NG):
            units.append((bi, g))
    N = len(units)
    if xn_given is not None:
        xn_next = xn_given
    else:
        xn_next = _qkv3_norm(X, g1c, blocks[0], TB, l)
    xn = None
    for i in range(N + Q4_LAG):
        if i < N:
            bi = units[i][0]
            b = blocks[bi]
            g = units[i][1]
            if g == 0:
                xn = xn_next
                if bi + 1 < len(blocks):
                    xn_next = _qkv3_norm(X, g1c, blocks[bi + 1], TB, l)
                qks.append(nl.ndarray((128, NQK, TB), dtype=dt, buffer=nl.sbuf))
                es.append(nl.ndarray((128, NQK, TB), dtype=dt, buffer=nl.sbuf))
                t0 = b * TB
                ct = nl.ndarray((128, TB), dtype=rcT.dtype, buffer=nl.sbuf)
                st = nl.ndarray((128, TB), dtype=rsT.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=ct, src=rcT[0:D, t0:t0 + TB])
                nisa.dma_copy(dst=st, src=rsT[0:D, t0:t0 + TB])
                csts.append([ones_f, epsb, gqf, gkf, ct, st, rot_sb])
            _q4_proj(xn, wqkv, l, g, GN, TB, qks[bi])
            if g == NG - 1:
                vr = _q4_v(xn, wqkv, l, TB, xn)
                vrs.append(vr)
                t0 = b * TB
                for tt in range(TB // 128):
                    nisa.dma_copy(dst=vc.ap(pattern=[[D, 128], [SMAX * D, KH], [1, D]], offset=(l * NKV + pid * KH) * SMAX * D + (t0 + tt * 128) * D),
                                  src=vr[0:128, tt, 0:KH * D])
                    if vo is not None:
                        nisa.dma_copy(dst=vo.ap(pattern=[[D, 128], [T * D, KH], [1, D]], offset=pid * KH * T * D + (t0 + tt * 128) * D),
                                      src=vr[0:128, tt, 0:KH * D])
        j = i - Q4_LAG
        if 0 <= j < N:
            bi = units[j][0]
            b = blocks[bi]
            g = units[j][1]
            _q4_post(qks[bi], es[bi], g, GN, TB, csts[bi])
            h0 = g * GN
            nh = min(GN, NQK - h0)
            t0 = b * TB
            e = es[bi]
            if h0 >= QH:   # the k heads: cache (and k output)
                nisa.dma_copy(dst=kc.ap(pattern=[[SMAX, 128], [D * SMAX, KH], [1, TB]], offset=(l * NKV + pid * KH) * D * SMAX + t0), src=e[0:128, QH:NQK, 0:TB])
                if ko is not None:
                    nisa.dma_copy(dst=ko.ap(pattern=[[T, 128], [D * T, KH], [1, TB]], offset=pid * KH * D * T + t0), src=e[0:128, QH:NQK, 0:TB])
            elif qo is not None:
                nisa.dma_copy(dst=qo.ap(pattern=[[T, 128], [D * T, nh], [1, TB]], offset=(pid * QH + h0) * D * T + t0), src=e[0:128, h0:h0 + nh, 0:TB])
        if fill is not None and i < N:
            # steps of another block's attention pipeline, spread over the projection units (vector / scalar work
            # while the PE projects); the caller runs the remaining steps
            per = (_att_nsteps(fill) + N - 1) // N
            for _k in range(per):
                if fill[14][0] < _att_nsteps(fill):
                    _att_step(fill)
    return es, vrs


@nki.jit
def pre_qkv3(X, g1c, wqkv, qn, kn, rcT, rsT, rot, kc, vc):
    """pre_qkv2 on the X layout: returns q [NQ, D, T], k [NKV, D, T], v [NKV, T, D] and writes the caches."""
    NB, P_, KT, TB = X.shape
    T = NB * TB
    qo = nl.ndarray((NQ, D, T), dtype=X.dtype, buffer=nl.shared_hbm)
    ko = nl.ndarray((NKV, D, T), dtype=X.dtype, buffer=nl.shared_hbm)
    vo = nl.ndarray((NKV, T, D), dtype=X.dtype, buffer=nl.shared_hbm)
    # software pipelined: block b+1's rmsnorm is issued before block b's projections, so the vector and scalar
    # engines prepare the next block while the PE works (each engine runs its instructions in program order)
    # software pipelined: block b+1's rmsnorm is issued before block b's projections, so the vector and scalar
    # engines prepare the next block while the PE works (each engine runs its instructions in program order).
    # (Issuing block b+1's projections before block b's q/k norm and rope as well: 0.552 -> 0.558 ms at T=1024.)
    if Q4:
        _qkv4(X, g1c, wqkv, qn, kn, rcT, rsT, rot, kc, vc, qo, ko, vo, 0)
        return qo, ko, vo
    xn_next = _qkv3_norm(X, g1c, 0, TB, 0)
    for b in range(NB):
        xn = xn_next
        if b + 1 < NB:
            xn_next = _qkv3_norm(X, g1c, b + 1, TB, 0)
        _qkv3_blk(xn, X, wqkv, qn, kn, rcT, rsT, rot, kc, vc, qo, ko, vo, T, b, TB, 0)
    return qo, ko, vo


def _qkv3_norm(X, g1c, b, TB, l):
    """rmsnorm(x) * g1 of token block b: [128, KT, TB] bf16 (the moving operand of the projections)."""
    KT = H // 128
    ones_f = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    xT = nl.ndarray((128, KT, TB), dtype=X.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xT, src=X[b])
    gc = nl.ndarray((128, KT), dtype=g1c.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gc, src=g1c[l])
    gf = nl.ndarray((128, KT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gf, src=gc)
    rstd = _ssq_rstd(xT, ones_f, TB, KT, QN_SSQ)
    xn = nl.ndarray((128, KT, TB), dtype=X.dtype, buffer=nl.sbuf)
    for j in range(KT):
        eng = nisa.engine.unknown
        if QN_ENG == "pool" or (QN_ENG == "alt" and j % 2 == 1):
            eng = nisa.engine.gpsimd
        nisa.tensor_tensor(dst=xn[0:128, j, 0:TB], data1=xT[0:128, j, 0:TB], data2=rstd, op=nl.multiply, engine=eng)
        if QN_G == "act":
            nisa.activation(dst=xn[0:128, j, 0:TB], op=nl.copy, data=xn[0:128, j, 0:TB], scale=gf[0:128, j:j + 1])
        else:
            nisa.tensor_scalar(dst=xn[0:128, j, 0:TB], data=xn[0:128, j, 0:TB], op0=nl.multiply, operand0=gf[0:128, j:j + 1], engine=eng)
    return xn


def _qkv3_blk(xn, X, wqkv, qn, kn, rcT, rsT, rot, kc, vc, qo, ko, vo, T, b, TB, l):
    """pre_qkv3 for token block b, given its normalized input xn."""
    t0 = b * TB
    SMAX = kc.shape[3]
    pid = nl.program_id(0)
    KT = H // 128
    QH = NQ // 2
    KH = NKV // 2
    NQK = QH + KH
    ones_f = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    QKVW = (NQ + 2 * NKV) * D
    NQK2 = QH + 2 * KH                        # column tiles per core in wqkv
    # ---- q and k heads (transposed outputs [d, TB])
    qk = nl.ndarray((128, NQK, TB), dtype=X.dtype, buffer=nl.sbuf)
    GN = gn_of(TB)                            # heads per weight DMA
    KG = 16
    for g in range((NQK + GN - 1) // GN):     # 8 q heads, 8 q heads, then the 4 k heads
        nh = min(GN, NQK - g * GN)
        c0 = pid * NQK2 * D + g * GN * D       # per-core column groups (see _layer); k heads follow q
        acc = []
        for _v in range(GN):
            acc.append(nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum))
        for kg in range(KT // KG):
            w = nl.ndarray((128, KG, GN * 128), dtype=wqkv.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=w[0:128, 0:KG, 0:nh * 128], src=wqkv.ap(pattern=[[QKVW, 128], [128 * QKVW, KG], [1, nh * 128]],
                          offset=l * H * QKVW + kg * KG * 128 * QKVW + c0), dge_mode=DGE)
            for u in range(nh):
                for k in range(KG):
                    nisa.nc_matmul(dst=acc[u], stationary=w[0:128, k, u * 128:(u + 1) * 128],
                                   moving=xn[0:128, kg * KG + k, 0:TB])
        for u in range(nh):
            nisa.tensor_copy(dst=qk[0:128, g * GN + u, 0:TB], src=acc[u])
    # ---- v rows [t, (head, d)]
    wv = nl.ndarray((128, KT, KH * D), dtype=wqkv.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=wv, src=wqkv.ap(pattern=[[QKVW, 128], [128 * QKVW, KT], [1, KH * D]], offset=l * H * QKVW + pid * NQK2 * D + (QH + KH) * D), dge_mode=DGE)
    vr = nl.ndarray((128, TB // 128, KH * D), dtype=X.dtype, buffer=nl.sbuf)
    for tt in range(TB // 128):
        ps = nl.ndarray((128, KH * D), dtype=nl.float32, buffer=nl.psum)
        for k in range(KT):
            nisa.nc_matmul(dst=ps, stationary=xn[0:128, k, tt * 128:(tt + 1) * 128], moving=wv[0:128, k, 0:KH * D])
        nisa.tensor_copy(dst=vr[0:128, tt, 0:KH * D], src=ps)
    # ---- q/k rmsnorm over d (partitions), gamma
    sq = nl.ndarray((128, NQK, TB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=qk)
    rstd = nl.ndarray((128, NQK, TB), dtype=nl.float32, buffer=nl.sbuf)
    epsb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=epsb, value=EPS)
    for j in range(NQK):
        ssp = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ssp, stationary=ones_f, moving=sq[0:128, j, 0:TB], is_stationary_onezero=True)
        nisa.activation(dst=rstd[0:128, j, 0:TB], op=nl.rsqrt, data=ssp, bias=epsb, scale=1.0 / D)
    nqk = nl.ndarray((128, NQK, TB), dtype=X.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=nqk, data1=qk, data2=rstd, op=nl.multiply)
    gq = nl.ndarray((128, 1), dtype=qn.dtype, buffer=nl.sbuf)
    gk = nl.ndarray((128, 1), dtype=kn.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gq, src=qn[l])
    nisa.dma_copy(dst=gk, src=kn[l])
    gqf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    gkf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gqf, src=gq)
    nisa.tensor_copy(dst=gkf, src=gk)
    nisa.tensor_scalar(dst=nqk[0:128, 0:QH, 0:TB], data=nqk[0:128, 0:QH, 0:TB], op0=nl.multiply, operand0=gqf)
    nisa.tensor_scalar(dst=nqk[0:128, QH:NQK, 0:TB], data=nqk[0:128, QH:NQK, 0:TB], op0=nl.multiply, operand0=gkf)
    # ---- rope: e = bf16(n * cos) + bf16(rotate_half(n) * sin), cos/sin [d, t]
    ct = nl.ndarray((128, TB), dtype=rcT.dtype, buffer=nl.sbuf)
    st = nl.ndarray((128, TB), dtype=rsT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=ct, src=rcT[0:D, t0:t0 + TB])
    nisa.dma_copy(dst=st, src=rsT[0:D, t0:t0 + TB])
    rot_sb = nl.ndarray((D, D), dtype=rot.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_sb, src=rot)
    e = nl.ndarray((128, NQK, TB), dtype=X.dtype, buffer=nl.sbuf)
    for j in range(NQK):
        rp = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=rp, stationary=rot_sb, moving=nqk[0:128, j, 0:TB])
        ea = nl.ndarray((128, TB), dtype=X.dtype, buffer=nl.sbuf)
        eb = nl.ndarray((128, TB), dtype=X.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ea, data1=nqk[0:128, j, 0:TB], data2=ct, op=nl.multiply)
        nisa.tensor_tensor(dst=eb, data1=rp, data2=st, op=nl.multiply)
        nisa.tensor_tensor(dst=e[0:128, j, 0:TB], data1=ea, data2=eb, op=nl.add)
    # ---- cache writes (always) and q/k/v outputs (when the attention runs in another kernel)
    nisa.dma_copy(dst=kc.ap(pattern=[[SMAX, 128], [D * SMAX, KH], [1, TB]], offset=(l * NKV + pid * KH) * D * SMAX + t0), src=e[0:128, QH:NQK, 0:TB])
    for tt in range(TB // 128):
        nisa.dma_copy(dst=vc.ap(pattern=[[D, 128], [SMAX * D, KH], [1, D]], offset=(l * NKV + pid * KH) * SMAX * D + (t0 + tt * 128) * D),
                      src=vr[0:128, tt, 0:KH * D])
    if qo is not None:
        nisa.dma_copy(dst=qo.ap(pattern=[[T, 128], [D * T, QH], [1, TB]], offset=pid * QH * D * T + t0), src=e[0:128, 0:QH, 0:TB])
        nisa.dma_copy(dst=ko.ap(pattern=[[T, 128], [D * T, KH], [1, TB]], offset=pid * KH * D * T + t0), src=e[0:128, QH:NQK, 0:TB])
        for tt in range(TB // 128):
            nisa.dma_copy(dst=vo.ap(pattern=[[D, 128], [T * D, KH], [1, D]], offset=pid * KH * T * D + (t0 + tt * 128) * D),
                          src=vr[0:128, tt, 0:KH * D])
    return e, vr



def _ident(dt):
    """[128, 128] identity matrix in SBUF"""
    io = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[1, 128]], offset=0, channel_multiplier=-1)     # c - p
    idf = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=idf, data=io, op0=nl.equal, operand0=0.0)
    idb = nl.ndarray((128, 128), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=idb, src=idf)
    return idb


def _att_a(e, es, b, h, qt, TB, T, dmask, ident=None, dmask_b=None):
    """attention stage A of (block b, head h, q tile qt): scaled, masked scores sc [128, nk] and -max.
    With ident (ATT_MAX == "pe"): sc = -(s / sqrt(d)) with the diagonal mask added on the PE, and min(sc) = -max,
    in one vector pass per key block (stage B then uses exp(-1 * sc + min))."""
    QH = NQ // 2
    GQ = QH // (NKV // 2)
    j = h // GQ
    NTB = TB // 128
    nk = (b * NTB + qt + 1) * 128
    sc = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.sbuf)
    if ident is not None:
        nmn = []
        for kb in range(b + 1):
            w = TB if kb < b else (qt + 1) * 128
            sp = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
            if kb < b or w > 128:
                wf = w if kb < b else w - 128
                nisa.nc_matmul(dst=sp[0:128, 0:wf], stationary=e[0:128, h, qt * 128:(qt + 1) * 128],
                               moving=es[kb][0:128, QH + j, 0:wf])
            if kb == b:
                nisa.nc_matmul(dst=sp[0:128, w - 128:w], stationary=e[0:128, h, qt * 128:(qt + 1) * 128],
                               moving=es[kb][0:128, QH + j, w - 128:w])
                nisa.nc_matmul(dst=sp[0:128, w - 128:w], stationary=ident, moving=dmask_b)
            m_ = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar_reduce(dst=sc[0:128, kb * TB:kb * TB + w], data=sp[0:128, 0:w], op0=nl.multiply,
                                      operand0=-1.0 / (D ** 0.5), reduce_op=nl.minimum, reduce_res=m_)
            nmn.append(m_)
        mx = nmn[0]
        if len(nmn) > 1:
            mx = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=mx, data1=nmn[0], data2=nmn[1], op=nl.minimum)
        return [sc, mx, nk, -1.0]
    for kb in range(b + 1):
        w = TB if kb < b else (qt + 1) * 128
        sp = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=sp[0:128, 0:w], stationary=e[0:128, h, qt * 128:(qt + 1) * 128],
                       moving=es[kb][0:128, QH + j, 0:w])
        wf = w if kb < b else w - 128
        if wf > 0:
            nisa.tensor_scalar(dst=sc[0:128, kb * TB:kb * TB + wf], data=sp[0:128, 0:wf], op0=nl.multiply, operand0=1.0 / (D ** 0.5))
        if kb == b:
            nisa.scalar_tensor_tensor(dst=sc[0:128, kb * TB + w - 128:kb * TB + w], data=sp[0:128, w - 128:w],
                                      op0=nl.multiply, operand0=1.0 / (D ** 0.5), op1=nl.add, operand1=dmask)
    mx = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mx, op=nl.maximum, data=sc[0:128, 0:nk], axis=1, negate=True)
    return [sc, mx, nk, 1.0]


def _att_b(st, T, dt):
    """stage B: P = softmax rows in bf16"""
    sc = st[0]
    mx = st[1]
    nk = st[2]
    ex = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.sbuf)
    sm = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=ex[0:128, 0:nk], op=nl.exp, data=sc[0:128, 0:nk], bias=mx, scale=st[3],
                           reduce_op=nl.add, reduce_res=sm)
    rs = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rs, data=sm)
    pb = nl.ndarray((128, T), dtype=dt, buffer=nl.sbuf)
    if ATT_NORM == "act":
        nisa.activation(dst=pb[0:128, 0:nk], op=nl.copy, data=ex[0:128, 0:nk], scale=rs)
    else:
        nisa.tensor_scalar(dst=pb[0:128, 0:nk], data=ex[0:128, 0:nk], op0=nl.multiply, operand0=rs, engine=nisa.engine.vector)
    return [pb, nk]


def _att_c(st, vrs, h, dst, TB, T, dt):
    """stage C: out^T [d, q tile] = V^T P^T into dst [128, 128]"""
    pb = st[0]
    nk = st[1]
    QH = NQ // 2
    GQ = QH // (NKV // 2)
    j = h // GQ
    NTB = TB // 128
    nkt = nk // 128
    pt_ps = nl.ndarray((128, T // 128, 128), dtype=dt, buffer=nl.psum)
    for kt in range(nkt):
        nisa.nc_transpose(dst=pt_ps[0:128, kt, 0:128], data=pb[0:128, kt * 128:(kt + 1) * 128], engine=nisa.engine.tensor)
    pt = nl.ndarray((128, T // 128, 128), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pt[0:128, 0:nkt, 0:128], src=pt_ps[0:128, 0:nkt, 0:128], engine=_ENG[ATT_CPT])
    op_ = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
    for kt in range(nkt):
        nisa.nc_matmul(dst=op_, stationary=vrs[kt // NTB][0:128, kt % NTB, j * D:(j + 1) * D], moving=pt[0:128, kt, 0:128])
    nisa.tensor_copy(dst=dst, src=op_, engine=_ENG[ATT_COUT])


def _att_new(e, es, vrs, b, TB, T, dt, dmask, ident, dmask_b, outs):
    """state of the software-pipelined attention of block b (see _att_pipe), advanced by _att_step"""
    QH = NQ // 2
    NTB = TB // 128
    its = []
    for h in range(QH):
        for qt in range(NTB):
            its.append((h, qt))
    return [e, es, vrs, b, TB, T, dt, dmask, ident, dmask_b, outs, its, [], [], [0]]


def _att_nsteps(st):
    return len(st[11]) + 2 * ATT_PIPE


def _att_step(st):
    """one step i of the pipeline: stage A of iteration i, B of i - LAG, C of i - 2 LAG"""
    e = st[0]
    es = st[1]
    vrs = st[2]
    b = st[3]
    TB = st[4]
    T = st[5]
    dt = st[6]
    its = st[11]
    sa = st[12]
    sb = st[13]
    i = st[14][0]
    N = len(its)
    LAG = ATT_PIPE
    if i < N:
        if ATT_MAX == "pe":
            sa.append(_att_a(e, es, b, its[i][0], its[i][1], TB, T, st[7], st[8], st[9]))
        else:
            sa.append(_att_a(e, es, b, its[i][0], its[i][1], TB, T, st[7]))
    ib = i - LAG
    if 0 <= ib < N:
        sb.append(_att_b(sa[ib], T, dt))
    ic = i - 2 * LAG
    if 0 <= ic < N:
        hh = its[ic][0]
        qq = its[ic][1]
        _att_c(sb[ic], vrs, hh, st[10][hh][0:128, qq * 128:(qq + 1) * 128], TB, T, dt)
    st[14][0] = i + 1


def _att_pipe(e, es, vrs, b, TB, T, dt, dmask, ident, dmask_b, outs):
    """the attention of block b, software pipelined in program order: stage A of iteration i, stage B of i - LAG,
    stage C of i - 2 LAG, so every engine's in-order queue alternates between independent iterations.
    outs[h]: [128, TB] destination of head h ([d, query])."""
    st = _att_new(e, es, vrs, b, TB, T, dt, dmask, ident, dmask_b, outs)
    for _i in range(_att_nsteps(st)):
        _att_step(st)


def _qkva(X, g1c, wqkv, qn, kn, rcT, rsT, rot, kc, vc, l):
    """T <= 1024 (one or two token blocks): pre_qkv3 plus causal attention on chip. Returns a [NQ, D, T] (the
    attention output per head, [d, token]: the input of pre_post3) and writes the caches. Core c: q heads 16c..,
    kv heads 4c..; the K/V of every block stay in SBUF for the later blocks' queries.
    (attention_cte's fixed costs dominated at small T: 0.18 ms per layer at T=256, 0.47 at 1024.)"""
    NB, P_, KT, TB = X.shape
    T = NB * TB
    pid = nl.program_id(0)
    QH = NQ // 2
    KH = NKV // 2
    GQ = QH // KH
    NTB = TB // 128
    ao = nl.ndarray((NQ, D, T), dtype=X.dtype, buffer=nl.shared_hbm)
    # causal mask of a diagonal 128 x 128 key tile: -30000 where key c > query p
    io = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[1, 128]], offset=0, channel_multiplier=-1)
    dmask = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=dmask, data=io, op0=nl.greater, operand0=0.0, op1=nl.multiply, operand1=-30000.0)
    dmask_b = nl.ndarray((128, 128), dtype=X.dtype, buffer=nl.sbuf)     # bf16 copy (-29952 where masked)
    nisa.tensor_copy(dst=dmask_b, src=dmask)
    ident = _ident(X.dtype)
    es = []
    vrs = []
    pfill = None
    for b in range(NB):
        if Q4:
            e4, v4 = _qkv4(X, g1c, wqkv, qn, kn, rcT, rsT, rot, kc, vc, None, None, None, l, bsel=b, fill=pfill)
            e = e4[0]
            vr = v4[0]
        else:
            xn = _qkv3_norm(X, g1c, b, TB, l)
            e, vr = _qkv3_blk(xn, X, wqkv, qn, kn, rcT, rsT, rot, kc, vc, None, None, None, T, b, TB, l)
        es.append(e)
        vrs.append(vr)
        if pfill is not None:
            for _i in range(_att_nsteps(pfill) - pfill[14][0]):
                _att_step(pfill)
            for h in range(QH):
                nisa.dma_copy(dst=ao[pid * QH + h, 0:D, (b - 1) * TB:b * TB], src=pfill[10][h])
            pfill = None
        if ATT_PIPE > 0 and ATT_OVL and Q4 and b + 1 < NB:
            # this block's attention runs inside the next block's QKV projections (fill)
            oTs = []
            for h in range(QH):
                oTs.append(nl.ndarray((128, TB), dtype=X.dtype, buffer=nl.sbuf))
            pfill = _att_new(e, es, vrs, b, TB, T, X.dtype, dmask, ident, dmask_b, oTs)
            continue
        if ATT_PIPE > 0:
            oTs = []
            for h in range(QH):
                oTs.append(nl.ndarray((128, TB), dtype=X.dtype, buffer=nl.sbuf))
            _att_pipe(e, es, vrs, b, TB, T, X.dtype, dmask, ident, dmask_b, oTs)
            for h in range(QH):
                nisa.dma_copy(dst=ao[pid * QH + h, 0:D, b * TB:(b + 1) * TB], src=oTs[h])
            continue
        for h in range(QH):
            j = h // GQ                     # local kv head
            oT = nl.ndarray((128, TB), dtype=X.dtype, buffer=nl.sbuf)     # [d, query]
            for qt in range(NTB):
                gq = b * NTB + qt           # global query tile
                nk = (gq + 1) * 128         # keys 0 .. nk-1 can be visible
                sc = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.sbuf)   # scaled, masked scores
                mx = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
                if ATT_MAX == "pe":
                    # raw scores of every visible key in PSUM, the causal mask of the diagonal tile added by the PE
                    # (identity x mask, exact: 0 where visible), then ONE vector pass: sc = -(s / sqrt(d)) and its
                    # min (= -max) together. exp(-1 * sc + min) == exp(s / sqrt(d) - max) as before.
                    nmn = []
                    for kb in range(b + 1):
                        w = TB if kb < b else (qt + 1) * 128
                        sp = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
                        if kb < b or w > 128:
                            wf = w if kb < b else w - 128
                            nisa.nc_matmul(dst=sp[0:128, 0:wf], stationary=e[0:128, h, qt * 128:(qt + 1) * 128],
                                           moving=es[kb][0:128, QH + j, 0:wf])
                        if kb == b:
                            nisa.nc_matmul(dst=sp[0:128, w - 128:w], stationary=e[0:128, h, qt * 128:(qt + 1) * 128],
                                           moving=es[kb][0:128, QH + j, w - 128:w])
                            nisa.nc_matmul(dst=sp[0:128, w - 128:w], stationary=ident, moving=dmask_b)
                        m_ = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_scalar_reduce(dst=sc[0:128, kb * TB:kb * TB + w], data=sp[0:128, 0:w], op0=nl.multiply,
                                                  operand0=-1.0 / (D ** 0.5), reduce_op=nl.minimum, reduce_res=m_)
                        nmn.append(m_)
                    if len(nmn) == 1:
                        mx = nmn[0]
                    else:
                        nisa.tensor_tensor(dst=mx, data1=nmn[0], data2=nmn[1], op=nl.minimum)
                    esc = -1.0
                else:
                    for kb in range(b + 1):
                        w = TB if kb < b else (qt + 1) * 128
                        sp = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=sp[0:128, 0:w], stationary=e[0:128, h, qt * 128:(qt + 1) * 128],
                                       moving=es[kb][0:128, QH + j, 0:w])
                        # PSUM -> SBUF with the 1/sqrt(d) scale; the diagonal key tile also gets the causal mask
                        wf = w if kb < b else w - 128
                        if wf > 0:
                            nisa.tensor_scalar(dst=sc[0:128, kb * TB:kb * TB + wf], data=sp[0:128, 0:wf], op0=nl.multiply, operand0=1.0 / (D ** 0.5))
                        if kb == b:
                            nisa.scalar_tensor_tensor(dst=sc[0:128, kb * TB + w - 128:kb * TB + w], data=sp[0:128, w - 128:w],
                                                      op0=nl.multiply, operand0=1.0 / (D ** 0.5), op1=nl.add, operand1=dmask)
                    nisa.tensor_reduce(dst=mx, op=nl.maximum, data=sc[0:128, 0:nk], axis=1, negate=True)
                    esc = 1.0
                ex = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.sbuf)
                sm = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation_reduce(dst=ex[0:128, 0:nk], op=nl.exp, data=sc[0:128, 0:nk], bias=mx, scale=esc,
                                       reduce_op=nl.add, reduce_res=sm)
                rs = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.reciprocal(dst=rs, data=sm)
                pb = nl.ndarray((128, T), dtype=X.dtype, buffer=nl.sbuf)
                if ATT_NORM == "act":
                    nisa.activation(dst=pb[0:128, 0:nk], op=nl.copy, data=ex[0:128, 0:nk], scale=rs)
                else:
                    nisa.tensor_scalar(dst=pb[0:128, 0:nk], data=ex[0:128, 0:nk], op0=nl.multiply, operand0=rs,
                                       engine=nisa.engine.gpsimd if ATT_NORM == "pool" else nisa.engine.vector)
                # P^T per key tile, then out^T [d, q] = sum_k (v[k, d] stationary) x P^T[k, q]
                pt_ps = nl.ndarray((128, T // 128, 128), dtype=X.dtype, buffer=nl.psum)
                for kt in range(gq + 1):
                    nisa.nc_transpose(dst=pt_ps[0:128, kt, 0:128], data=pb[0:128, kt * 128:(kt + 1) * 128], engine=nisa.engine.tensor)
                pt = nl.ndarray((128, T // 128, 128), dtype=X.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pt[0:128, 0:gq + 1, 0:128], src=pt_ps[0:128, 0:gq + 1, 0:128])
                op_ = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
                for kt in range(gq + 1):
                    nisa.nc_matmul(dst=op_, stationary=vrs[kt // NTB][0:128, kt % NTB, j * D:(j + 1) * D], moving=pt[0:128, kt, 0:128])
                nisa.tensor_copy(dst=oT[0:128, qt * 128:(qt + 1) * 128], src=op_)
            nisa.dma_copy(dst=ao[pid * QH + h, 0:D, b * TB:(b + 1) * TB], src=oT)
    return ao


@nki.jit
def pre_qkva(X, g1c, wqkv, qn, kn, rcT, rsT, rot, kc, vc):
    return _qkva(X, g1c, wqkv, qn, kn, rcT, rsT, rot, kc, vc, 0)


@nki.jit
def pre_layers(X, g1c, wqkv, qn, kn, rcT, rsT, rot, kc, vc, wo, g2c, wg, wu, wd):
    """T <= 512: NL consecutive prefill layers (NL = leading dim of the stacked weights) in one NEFF:
    per layer _qkva (attention on chip) then pre_post3's body; a barrier makes each layer's output (each core writes
    half of the features) visible to both cores before the next layer reads it."""
    NB, P_, KT, TB = X.shape
    T = NB * TB
    NL = wqkv.shape[0]
    for l in range(NL):
        a = _qkva(X, g1c, wqkv, qn, kn, rcT, rsT, rot, kc, vc, l)
        Xo = nl.ndarray((NB, 128, KT, TB), dtype=X.dtype, buffer=nl.shared_hbm)
        for b in range(NB):
            if POST5:
                _post5_blk(X, a, wo, g2c, wg, wu, wd, Xo, T, b, TB, l)
            else:
                _post3_blk(X, a, wo, g2c, wg, wu, wd, Xo, T, b, TB, l)
        nisa.core_barrier(data=Xo, cores=(0, 1))
        X = Xo
    return X


# ----------------------------------------------------------------------------------------------- prefill, resident
# pre_layers2 (T <= 512, one token block): pre_layers with the residual stream and the attention output resident in
# SBUF across the phases and layers of one NEFF, so no HBM round trip and no core barrier sits between a layer's
# phases. The per-core fp32 partial sums of o-proj and down are swapped per output group with nisa.sendrecv
# (SBUF -> SBUF) right after the group's PSUM drain; the sums are consumed after the group loop (consuming each one
# in program order blocked the vector engine on the swap and stalled the PSUM drains). Same arithmetic and rounding
# points as pre_layers: own + other == other + own in fp32, so both cores hold identical layer outputs.
# Measured (5-layer NEFF, T=256, neuron-explorer): pre_layers 4.97 ms -> 4.36 ms; T=128 4.10 -> 3.69; T=512
# 8.55 -> 8.48; at T=1024 (two blocks) it was slower (17.07 -> 17.59), so 1024 keeps pre_layers.
DN_GN = 4   # down: output tiles per group. With 8 (gn_of(256)) the scheduler ran each accumulator over all of the
            # group's weight slabs (u-outer), pinning the whole group's slabs and stalling the weight DMA: 4.48 -> 4.36


def _swap(stg, g):
    """the other core's partial [128, n, TB] fp32 of the same output tiles (both cores call it)."""
    pid = nl.program_id(0)
    oth = nl.ndarray(stg.shape, dtype=stg.dtype, buffer=nl.sbuf)
    # (dma_engine=gpsimd_dma: refused, at most 1 KB per partition)
    nisa.sendrecv(src=stg, dst=oth, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=g % 2)
    return oth


def _fin(pend, out, base, GN, TB):
    """out[group g] = base[group g] + bf16(own + other) for every swapped group."""
    for pe_ in pend:
        g = pe_[0]
        stg = pe_[1]
        oth = pe_[2]
        eng = _ENG2[FIN_ENG]
        if FIN_ENG == "alt":
            eng = nisa.engine.gpsimd if g % 2 == 1 else nisa.engine.vector
        sb = nl.ndarray((128, GN, TB), dtype=out.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sb, data1=stg, data2=oth, op=nl.add, engine=eng)
        nisa.tensor_tensor(dst=out[0:128, g * GN:(g + 1) * GN, 0:TB], data1=base[0:128, g * GN:(g + 1) * GN, 0:TB], data2=sb, op=nl.add, engine=eng)


def _norm_sb(xT, g1c, TB, l):
    """_qkv3_norm of a block already in SBUF."""
    KT = H // 128
    ones_f = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    gc = nl.ndarray((128, KT), dtype=g1c.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gc, src=g1c[l])
    gf = nl.ndarray((128, KT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gf, src=gc)
    rstd = _ssq_rstd(xT, ones_f, TB, KT)
    xn = nl.ndarray((128, KT, TB), dtype=xT.dtype, buffer=nl.sbuf)
    for j in range(KT):
        nisa.tensor_tensor(dst=xn[0:128, j, 0:TB], data1=xT[0:128, j, 0:TB], data2=rstd, op=nl.multiply, engine=_ENG2[HN_ENG])
        if HNG_ENG == "act":
            nisa.activation(dst=xn[0:128, j, 0:TB], op=nl.copy, data=xn[0:128, j, 0:TB], scale=gf[0:128, j:j + 1])
        else:
            nisa.tensor_scalar(dst=xn[0:128, j, 0:TB], data=xn[0:128, j, 0:TB], op0=nl.multiply, operand0=gf[0:128, j:j + 1],
                               engine=_ENG2[HNG_ENG])
    return xn


def _qkva2_ovl(xn, wqkv, qn, kn, rcT, rsT, rot, kc, vc, l, TB, dmask, ident, dmask_b, at):
    """_qkva2 for TB <= 256 with the attention overlapped with the q projections (ATT_QOVL): head groups of 4, the k
    group (and v) first, then q groups; the q/k norm + rope of a group is issued one group later and the attention
    iterations of q group i two groups later, so the PE projects while the vector/scalar engines post-process and
    run the softmax. Same arithmetic as _qkv4 + _att_pipe."""
    pid = nl.program_id(0)
    SMAX = kc.shape[3]
    T = TB
    dt = xn.dtype
    QH = NQ // 2
    KH = NKV // 2
    NQK = QH + KH
    GN = 4
    NG = NQK // GN
    NTB = TB // 128
    ones_f = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    epsb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=epsb, value=EPS)
    gq = nl.ndarray((128, 1), dtype=qn.dtype, buffer=nl.sbuf)
    gk = nl.ndarray((128, 1), dtype=kn.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gq, src=qn[l])
    nisa.dma_copy(dst=gk, src=kn[l])
    gqf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    gkf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gqf, src=gq)
    nisa.tensor_copy(dst=gkf, src=gk)
    rot_sb = nl.ndarray((D, D), dtype=rot.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_sb, src=rot)
    ct = nl.ndarray((128, TB), dtype=rcT.dtype, buffer=nl.sbuf)
    st = nl.ndarray((128, TB), dtype=rsT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=ct, src=rcT[0:D, 0:TB])
    nisa.dma_copy(dst=st, src=rsT[0:D, 0:TB])
    cst = [ones_f, epsb, gqf, gkf, ct, st, rot_sb]
    qk = nl.ndarray((128, NQK, TB), dtype=dt, buffer=nl.sbuf)
    e = nl.ndarray((128, NQK, TB), dtype=dt, buffer=nl.sbuf)
    order = [NG - 1]
    for g in range(NG - 1):
        order.append(g)
    vr = None
    outs = []
    for h in range(QH):
        outs.append(at[0:128, h, 0:TB])
    ast = None
    nposted_q = 0
    for i in range(len(order) + 2):
        if i < len(order):
            g = order[i]
            _q4_proj(xn, wqkv, l, g, GN, TB, qk)
            if g == NG - 1:
                vr = _q4_v(xn, wqkv, l, TB, xn)
                for tt in range(NTB):
                    nisa.dma_copy(dst=vc.ap(pattern=[[D, 128], [SMAX * D, KH], [1, D]], offset=(l * NKV + pid * KH) * SMAX * D + tt * 128 * D),
                                  src=vr[0:128, tt, 0:KH * D])
                ast = _att_new(e, [e], [vr], 0, TB, T, dt, dmask, ident, dmask_b, outs)
        j = i - 1
        if 0 <= j < len(order):
            g = order[j]
            _q4_post(qk, e, g, GN, TB, cst)
            if g == NG - 1:
                nisa.dma_copy(dst=kc.ap(pattern=[[SMAX, 128], [D * SMAX, KH], [1, TB]], offset=(l * NKV + pid * KH) * D * SMAX), src=e[0:128, QH:NQK, 0:TB])
        # attention iterations (h-major) of the q groups whose post was issued at least one group ago
        jj = i - 2
        if 0 <= jj < len(order) and order[jj] != NG - 1:
            nposted_q += 1
            avail = nposted_q * GN * NTB
            for _k in range(avail):
                if ast[14][0] < avail:
                    _att_step(ast)
    for _k in range(_att_nsteps(ast)):
        if ast[14][0] < _att_nsteps(ast):
            _att_step(ast)
    return at


def _qkva2(xb, g1c, wqkv, qn, kn, rcT, rsT, rot, kc, vc, l):
    """_qkva for one token block xb in SBUF: returns this core's attention output at [128 d, QH, TB] in SBUF."""
    TB = xb.shape[2]
    T = TB
    dt = xb.dtype
    QH = NQ // 2
    GQ = QH // (NKV // 2)
    NTB = TB // 128
    io = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[1, 128]], offset=0, channel_multiplier=-1)
    dmask = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=dmask, data=io, op0=nl.greater, operand0=0.0, op1=nl.multiply, operand1=-30000.0)
    xn = _norm_sb(xb, g1c, TB, l)
    if ATT_QOVL and TB <= 256 and ATT_PIPE > 0 and ATT_MAX == "pe":
        dmask_b = nl.ndarray((128, 128), dtype=dt, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dmask_b, src=dmask)
        ident = _ident(dt)
        at = nl.ndarray((128, QH, TB), dtype=dt, buffer=nl.sbuf)
        return _qkva2_ovl(xn, wqkv, qn, kn, rcT, rsT, rot, kc, vc, l, TB, dmask, ident, dmask_b, at)
    if Q4 and TB >= 512:            # (TB = 256: 4.27 -> 4.28-4.43 ms per 5 layers, no gain)
        e4, v4 = _qkv4(None, g1c, wqkv, qn, kn, rcT, rsT, rot, kc, vc, None, None, None, l, xn_given=xn)
        e = e4[0]
        vr = v4[0]
    else:
        e, vr = _qkv3_blk(xn, xb, wqkv, qn, kn, rcT, rsT, rot, kc, vc, None, None, None, T, 0, TB, l)
    at = nl.ndarray((128, QH, TB), dtype=dt, buffer=nl.sbuf)
    if ATT_PIPE > 0:
        dmask_b = nl.ndarray((128, 128), dtype=dt, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dmask_b, src=dmask)
        ident = _ident(dt)
        outs = []
        for h in range(QH):
            outs.append(at[0:128, h, 0:TB])
        _att_pipe(e, [e], [vr], 0, TB, T, dt, dmask, ident, dmask_b, outs)
        return at
    for h in range(QH):
        j = h // GQ
        for qt in range(NTB):
            nk = (qt + 1) * 128
            sp = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=sp[0:128, 0:nk], stationary=e[0:128, h, qt * 128:(qt + 1) * 128], moving=e[0:128, QH + j, 0:nk])
            sc = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.sbuf)
            if nk > 128:
                nisa.tensor_scalar(dst=sc[0:128, 0:nk - 128], data=sp[0:128, 0:nk - 128], op0=nl.multiply, operand0=1.0 / (D ** 0.5))
            nisa.scalar_tensor_tensor(dst=sc[0:128, nk - 128:nk], data=sp[0:128, nk - 128:nk],
                                      op0=nl.multiply, operand0=1.0 / (D ** 0.5), op1=nl.add, operand1=dmask)
            mx = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=mx, op=nl.maximum, data=sc[0:128, 0:nk], axis=1, negate=True)
            ex = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.sbuf)
            sm = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=ex[0:128, 0:nk], op=nl.exp, data=sc[0:128, 0:nk], bias=mx, scale=1.0,
                                   reduce_op=nl.add, reduce_res=sm)
            rs = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rs, data=sm)
            pb = nl.ndarray((128, T), dtype=dt, buffer=nl.sbuf)
            nisa.activation(dst=pb[0:128, 0:nk], op=nl.copy, data=ex[0:128, 0:nk], scale=rs)
            pt_ps = nl.ndarray((128, NTB, 128), dtype=dt, buffer=nl.psum)
            for kt in range(qt + 1):
                nisa.nc_transpose(dst=pt_ps[0:128, kt, 0:128], data=pb[0:128, kt * 128:(kt + 1) * 128], engine=nisa.engine.tensor)
            pt = nl.ndarray((128, NTB, 128), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pt[0:128, 0:qt + 1, 0:128], src=pt_ps[0:128, 0:qt + 1, 0:128])
            op_ = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
            for kt in range(qt + 1):
                nisa.nc_matmul(dst=op_, stationary=vr[0:128, kt, j * D:(j + 1) * D], moving=pt[0:128, kt, 0:128])
            nisa.tensor_copy(dst=at[0:128, h, qt * 128:(qt + 1) * 128], src=op_)
    return at


def _post4(xb, at, wo, g2c, wg, wu, wd, TB, l):
    """_post3_blk on SBUF: xb [128, KT, TB] residual block, at [128, QH, TB] this core's attention output.
    Returns the layer output block [128, KT, TB] (all features, identical on both cores)."""
    dt = xb.dtype
    KT = H // 128
    QH = NQ // 2
    IC = I // 2
    NI = IC // 128
    GU = gn_of(TB)
    GN = GU if PO_GN == 0 else PO_GN   # o-proj output tiles per group (weight DMA / swap)
    GD = DN_GN
    KG = 16
    pid = nl.program_id(0)
    ones_f = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    # ---- o-proj partial over this core's heads, swapped per group; h = x + bf16(o0 + o1)
    hT = nl.ndarray((128, KT, TB), dtype=dt, buffer=nl.sbuf)
    pend = []
    s1 = None
    for g in range(KT // GN):
        w = nl.ndarray((128, QH, GN * 128), dtype=wo.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w, src=wo.ap(pattern=[[H, 128], [128 * H, QH], [1, GN * 128]], offset=l * H * H + pid * QH * D * H + g * GN * 128), dge_mode=DGE)
        acc = []
        for _v in range(GN):
            acc.append(nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum))
        for u in range(GN):
            for k in range(QH):
                nisa.nc_matmul(dst=acc[u], stationary=w[0:128, k, u * 128:(u + 1) * 128], moving=at[0:128, k, 0:TB])
        stg = nl.ndarray((128, GN, TB), dtype=nl.float32, buffer=nl.sbuf)
        for u in range(GN):
            nisa.tensor_copy(dst=stg[0:128, u, 0:TB], src=acc[u])
        pend.append((g, stg, _swap(stg, g)))
        if P4_LAG >= 0 and len(pend) > P4_LAG:
            # consume the swap of an earlier group now: h tiles and their squares accumulate while the PE runs on
            gg = pend[0][0]
            _fin(pend[0:1], hT, xb, GN, TB)
            s1 = _ssq_acc(s1 if gg > 0 else None, hT, gg * GN, (gg + 1) * GN, TB)
            pend = pend[1:]
    if P4_LAG >= 0:
        for pe_ in pend:
            gg = pe_[0]
            _fin([pe_], hT, xb, GN, TB)
            s1 = _ssq_acc(s1 if gg > 0 else None, hT, gg * GN, (gg + 1) * GN, TB)
        rstd = _ssq_fin(s1, ones_f, TB, KT)
    else:
        _fin(pend, hT, xb, GN, TB)
        rstd = _ssq_rstd(hT, ones_f, TB, KT)
    gc = nl.ndarray((128, KT), dtype=g2c.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gc, src=g2c[l])
    gcf = nl.ndarray((128, KT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gcf, src=gc)
    if True:
        hn = nl.ndarray((128, KT, TB), dtype=dt, buffer=nl.sbuf)
        for j in range(KT):
            nisa.tensor_tensor(dst=hn[0:128, j, 0:TB], data1=hT[0:128, j, 0:TB], data2=rstd, op=nl.multiply, engine=_ENG2[HN_ENG])
            if HNG_ENG == "act":
                nisa.activation(dst=hn[0:128, j, 0:TB], op=nl.copy, data=hn[0:128, j, 0:TB], scale=gcf[0:128, j:j + 1])
            else:
                nisa.tensor_scalar(dst=hn[0:128, j, 0:TB], data=hn[0:128, j, 0:TB], op0=nl.multiply, operand0=gcf[0:128, j:j + 1],
                                   engine=_ENG2[HNG_ENG])
    # ---- gate/up -> act = silu(g) * u
    act = nl.ndarray((128, NI, TB), dtype=dt, buffer=nl.sbuf)
    for g in range(NI // GU):
        c0 = pid * IC + g * GU * 128
        gu = []
        for wi in range(2):
            wmat = wg if wi == 0 else wu
            acc = []
            for _v in range(GU):
                acc.append(nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum))
            for kg in range(KT // KG):
                wt = nl.ndarray((128, KG, GU * 128), dtype=wmat.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=wt, src=wmat.ap(pattern=[[I, 128], [128 * I, KG], [1, GU * 128]], offset=l * H * I + kg * KG * 128 * I + c0), dge_mode=DGE)
                for u in range(GU):
                    for k in range(KG):
                        nisa.nc_matmul(dst=acc[u], stationary=wt[0:128, k, u * 128:(u + 1) * 128], moving=hn[0:128, kg * KG + k, 0:TB])
            r = nl.ndarray((128, GU, TB), dtype=dt, buffer=nl.sbuf)
            for u in range(GU):
                nisa.tensor_copy(dst=r[0:128, u, 0:TB], src=acc[u])
            gu.append(r)
        nisa.activation(dst=gu[0], op=nl.silu, data=gu[0])
        nisa.tensor_tensor(dst=act[0:128, g * GU:(g + 1) * GU, 0:TB], data1=gu[0], data2=gu[1], op=nl.multiply)
    # ---- down partial per group of GD output tiles, swapped; out = h + bf16(y0 + y1)
    xo = nl.ndarray((128, KT, TB), dtype=dt, buffer=nl.sbuf)
    pend = []
    for g in range(KT // GD):
        acc = []
        for _v in range(GD):
            acc.append(nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum))
        for kg in range(NI // KG):
            wdt = nl.ndarray((128, KG, GD * 128), dtype=wd.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=wdt, src=wd.ap(pattern=[[H, 128], [128 * H, KG], [1, GD * 128]], offset=l * I * H + (pid * IC + kg * KG * 128) * H + g * GD * 128), dge_mode=DGE)
            for u in range(GD):
                for k in range(KG):
                    nisa.nc_matmul(dst=acc[u], stationary=wdt[0:128, k, u * 128:(u + 1) * 128], moving=act[0:128, kg * KG + k, 0:TB])
        stg = nl.ndarray((128, GD, TB), dtype=nl.float32, buffer=nl.sbuf)
        for u in range(GD):
            nisa.tensor_copy(dst=stg[0:128, u, 0:TB], src=acc[u])
        pend.append((g, stg, _swap(stg, g)))
    _fin(pend, xo, hT, GD, TB)
    return xo


@nki.jit
def pre_layers2(X, g1c, wqkv, qn, kn, rcT, rsT, rot, kc, vc, wo, g2c, wg, wu, wd):
    """pre_layers for one token block (T <= 512) with the residual stream resident in SBUF (see _post4)."""
    NB, P_, KT, TB = X.shape
    NL = wqkv.shape[0]
    pid = nl.program_id(0)
    KH2 = KT // 2
    xb = nl.ndarray((128, KT, TB), dtype=X.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xb, src=X[0])
    for l in range(NL):
        at = _qkva2(xb, g1c, wqkv, qn, kn, rcT, rsT, rot, kc, vc, l)
        xb = _post4(xb, at, wo, g2c, wg, wu, wd, TB, l)
    Xo = nl.ndarray((NB, 128, KT, TB), dtype=X.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=Xo[0, 0:128, pid * KH2:(pid + 1) * KH2, 0:TB], src=xb[0:128, pid * KH2:(pid + 1) * KH2, 0:TB])
    return Xo
