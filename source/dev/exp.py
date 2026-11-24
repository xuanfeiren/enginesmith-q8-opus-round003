"""dev only: experimental kernels."""
import os as _os
import nki
import nki.language as nl
import nki.isa as nisa
from qwen3 import *
from build_neffs import neff_specs as _ns

@nki.jit
def attn_only(x, g1, wqkv, qn, kn, cos, sin, kc, vc, mask, posf, posu, wo, g2, wg, wu, wd, sctx):
    S = sctx.shape[1]
    attn, ko, vo = attention_block_tkg(
        X=x, X_hidden_dim_actual=H, rmsnorm_X_enabled=True, rmsnorm_X_eps=EPS, rmsnorm_X_gamma=g1,
        W_qkv=wqkv, bias_qkv=None, quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None, input_dequant_scale_qkv=None,
        rmsnorm_QK_pre_rope_enabled=True, rmsnorm_QK_pre_rope_eps=EPS,
        rmsnorm_QK_pre_rope_W_Q=qn, rmsnorm_QK_pre_rope_W_K=kn,
        cos=cos, sin=sin, rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False, rmsnorm_QK_post_rope_eps=EPS,
        rmsnorm_QK_post_rope_W_Q=None, rmsnorm_QK_post_rope_W_K=None,
        K_cache_transposed=False, active_blocks_table=None, K_cache=kc, V_cache=vc,
        attention_mask=mask, sink=None, update_cache=True, kv_cache_update_idx=posu,
        W_out=wo, bias_out=None, quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None, input_dequant_scale_out=None,
        transposed_out=False, out_in_sb=False, pos_ids=posf, S_ctx=S)
    return attn

@nki.jit
def mlp_only(x, g1, wqkv, qn, kn, cos, sin, kc, vc, mask, posf, posu, wo, g2, wg, wu, wd, sctx):
    res = mlp(hidden_tensor=x, gate_proj_weights_tensor=wg, up_proj_weights_tensor=wu,
              down_proj_weights_tensor=wd, normalization_weights_tensor=g2,
              fused_add_tensor=x, store_fused_add_result=True,
              activation_fn=ActFnType.SiLU, normalization_type=NormType.RMS_NORM, eps=EPS)
    return res[0]


DGE = {'hw': nisa.dge_mode.hwdge, 'sw': nisa.dge_mode.swdge, 'none': nisa.dge_mode.none, 'un': nisa.dge_mode.unknown}[_os.environ.get('KDGE', 'hw')] if True else None
DEBUG = bool(_os.environ.get('KDEBUG'))


def _row_to_col(dst_sb, row, n_tiles, one):
    """row [1, n_tiles*128] (SBUF) -> dst_sb [128, n_tiles] (SBUF), exact (PE matmul with 1.0)."""
    pt = nl.ndarray((128, n_tiles), dtype=nl.float32, buffer=nl.psum)
    for t in range(n_tiles):
        nisa.nc_matmul(dst=pt[0:128, t:t + 1], stationary=row[0:1, t * 128:(t + 1) * 128], moving=one[0:1, 0:1])
    nisa.tensor_copy(dst=dst_sb, src=pt)


@nki.jit
def mlp_dec(x, attn, g2, wg, wu, wd):
    """out [1,1,H] = h + down(silu(gate(n)) * up(n)), h = x + attn, n = rmsnorm(h) * g2. Weights [in, out]."""
    nprog = nl.num_programs()
    pid = nl.program_id(0) if nprog > 1 else 0
    KT = H // 128
    IC = I // nprog          # this core's intermediate columns
    GW = 2048                # gate/up columns per group (4 PSUM banks each)
    NG = IC // GW
    one = nl.ndarray((1, 1), dtype=x.dtype, buffer=nl.sbuf)
    nisa.memset(dst=one, value=1.0)
    xr = nl.ndarray((1, H), dtype=x.dtype, buffer=nl.sbuf)
    ar = nl.ndarray((1, H), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xr, src=x.reshape((1, H)))
    nisa.dma_copy(dst=ar, src=attn.reshape((1, H)))
    h = nl.ndarray((1, H), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h, data1=xr, data2=ar, op=nl.add)
    # rmsnorm
    sq = nl.ndarray((1, H), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=sq, op=nl.square, data=h, reduce_op=nl.add, reduce_res=ss)
    epsb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=epsb, value=EPS)
    rstd = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=ss, bias=epsb, scale=1.0 / H)
    hn = nl.ndarray((1, H), dtype=x.dtype, buffer=nl.sbuf)
    nisa.activation(dst=hn, op=nl.copy, data=h, scale=rstd)
    gr = nl.ndarray((1, H), dtype=g2.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gr, src=g2)
    nisa.tensor_tensor(dst=hn, data1=hn, data2=gr, op=nl.multiply)
    hT = nl.ndarray((128, KT), dtype=x.dtype, buffer=nl.sbuf)
    _row_to_col(hT, hn, KT, one)
    # gate / up over this core's columns, groups of GW
    pT = nl.ndarray((128, IC // 128), dtype=x.dtype, buffer=nl.sbuf)
    for g in range(NG):
        c0 = pid * IC + g * GW
        gp = []
        up = []
        for j in range(GW // 512):
            gp.append(nl.ndarray((1, 512), dtype=nl.float32, buffer=nl.psum))
            up.append(nl.ndarray((1, 512), dtype=nl.float32, buffer=nl.psum))
        for kt in range(KT):
            wgt = nl.ndarray((128, GW), dtype=wg.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=wgt, src=wg[kt * 128:(kt + 1) * 128, c0:c0 + GW], dge_mode=DGE)
            wut = nl.ndarray((128, GW), dtype=wu.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=wut, src=wu[kt * 128:(kt + 1) * 128, c0:c0 + GW], dge_mode=DGE)
            for j in range(GW // 512):
                nisa.nc_matmul(dst=gp[j], stationary=hT[0:128, kt:kt + 1], moving=wgt[0:128, j * 512:(j + 1) * 512])
                nisa.nc_matmul(dst=up[j], stationary=hT[0:128, kt:kt + 1], moving=wut[0:128, j * 512:(j + 1) * 512])
        gs = nl.ndarray((1, GW), dtype=x.dtype, buffer=nl.sbuf)
        us = nl.ndarray((1, GW), dtype=x.dtype, buffer=nl.sbuf)
        for j in range(GW // 512):
            nisa.tensor_copy(dst=gs[0:1, j * 512:(j + 1) * 512], src=gp[j])
            nisa.tensor_copy(dst=us[0:1, j * 512:(j + 1) * 512], src=up[j])
        nisa.activation(dst=gs, op=nl.silu, data=gs)
        nisa.tensor_tensor(dst=gs, data1=gs, data2=us, op=nl.multiply)
        _row_to_col(pT[0:128, g * (GW // 128):(g + 1) * (GW // 128)], gs, GW // 128, one)
    # down: K = this core's IC rows, N = H in 2 passes of 2048
    KD = IC // 128
    part = nl.ndarray((1, H), dtype=nl.float32, buffer=nl.sbuf)
    for ps in range(H // 2048):
        dp = []
        for j in range(4):
            dp.append(nl.ndarray((1, 512), dtype=nl.float32, buffer=nl.psum))
        for kt in range(KD):
            r0 = pid * IC + kt * 128
            wdt = nl.ndarray((128, 2048), dtype=wd.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=wdt, src=wd[r0:r0 + 128, ps * 2048:(ps + 1) * 2048], dge_mode=DGE)
            for j in range(4):
                nisa.nc_matmul(dst=dp[j], stationary=pT[0:128, kt:kt + 1], moving=wdt[0:128, j * 512:(j + 1) * 512])
        for j in range(4):
            nisa.tensor_copy(dst=part[0:1, ps * 2048 + j * 512:ps * 2048 + (j + 1) * 512], src=dp[j])
    out = nl.ndarray((1, 1, H), dtype=x.dtype, buffer=nl.shared_hbm)
    tot = nl.ndarray((1, H), dtype=nl.float32, buffer=nl.sbuf)
    if nprog > 1:
        oth = nl.ndarray((1, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.sendrecv(src=part, dst=oth, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0)
        nisa.tensor_tensor(dst=tot, data1=part, data2=oth, op=nl.add)
    else:
        nisa.tensor_copy(dst=tot, src=part)
    d = nl.ndarray((1, H), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=d, src=tot)
    o = nl.ndarray((1, H), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=o, data1=h, data2=d, op=nl.add)
    HH = H // nprog
    nisa.dma_copy(dst=out.reshape((1, H))[0:1, pid * HH:(pid + 1) * HH], src=o[0:1, pid * HH:(pid + 1) * HH])
    return out


bf, f32 = "bfloat16", "float32"
SPECS = {"mlp_dec": ("mlp_dec", [("x", (1, 1, H), bf), ("attn", (1, 1, H), bf), ("g2", (1, H), bf),
                                 ("wg", (H, I), bf), ("wu", (H, I), bf), ("wd", (I, H), bf)])}


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
    nisa.nc_matmul(dst=tp, stationary=ones_f, moving=ps)
    epsb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=epsb, value=EPS)
    rstd = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=tp, bias=epsb, scale=1.0 / (n * 128))
    nisa.activation(dst=dst, op=nl.copy, data=col, scale=rstd)
    nisa.tensor_tensor(dst=dst, data1=dst, data2=gcol, op=nl.multiply)


@nki.jit
def mlp_dec2(x, attn, g2c, wg, wu, wd):
    """As mlp_dec, weight-stationary GEMVs in column layout. g2c [128, H/128]: gamma in column layout."""
    nprog = nl.num_programs()
    pid = nl.program_id(0) if nprog > 1 else 0
    KT = H // 128
    IC = I // nprog
    NTI = IC // 128
    ones_f = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    xr = nl.ndarray((KT, 128), dtype=x.dtype, buffer=nl.sbuf)
    ar = nl.ndarray((KT, 128), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xr, src=x.reshape((KT, 128)))
    nisa.dma_copy(dst=ar, src=attn.reshape((KT, 128)))
    nisa.tensor_tensor(dst=xr, data1=xr, data2=ar, op=nl.add)
    h = nl.ndarray((128, KT), dtype=x.dtype, buffer=nl.sbuf)
    _rows_to_col(h, xr, KT)
    gc = nl.ndarray((128, KT), dtype=g2c.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gc, src=g2c)
    hn = nl.ndarray((128, KT), dtype=x.dtype, buffer=nl.sbuf)
    _col_rmsnorm(hn, h, gc, ones_f, KT)
    c0 = pid * IC
    gps = nl.ndarray((128, NTI), dtype=nl.float32, buffer=nl.psum)
    ups = nl.ndarray((128, NTI), dtype=nl.float32, buffer=nl.psum)
    for kt in range(KT):
        wgt = nl.ndarray((128, IC), dtype=wg.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wgt, src=wg[kt * 128:(kt + 1) * 128, c0:c0 + IC], dge_mode=DGE)
        for nt in range(NTI):
            nisa.nc_matmul(dst=gps[0:128, nt:nt + 1], stationary=wgt[0:128, nt * 128:(nt + 1) * 128],
                           moving=hn[0:128, kt:kt + 1])
        wut = nl.ndarray((128, IC), dtype=wu.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wut, src=wu[kt * 128:(kt + 1) * 128, c0:c0 + IC], dge_mode=DGE)
        for nt in range(NTI):
            nisa.nc_matmul(dst=ups[0:128, nt:nt + 1], stationary=wut[0:128, nt * 128:(nt + 1) * 128],
                           moving=hn[0:128, kt:kt + 1])
    gs = nl.ndarray((128, NTI), dtype=x.dtype, buffer=nl.sbuf)
    us = nl.ndarray((128, NTI), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gs, src=gps)
    nisa.tensor_copy(dst=us, src=ups)
    nisa.activation(dst=gs, op=nl.silu, data=gs)
    nisa.tensor_tensor(dst=gs, data1=gs, data2=us, op=nl.multiply)
    dps = nl.ndarray((128, KT), dtype=nl.float32, buffer=nl.psum)
    for kt in range(NTI):
        r0 = c0 + kt * 128
        wdt = nl.ndarray((128, H), dtype=wd.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wdt, src=wd[r0:r0 + 128, 0:H], dge_mode=DGE)
        for nt in range(KT):
            nisa.nc_matmul(dst=dps[0:128, nt:nt + 1], stationary=wdt[0:128, nt * 128:(nt + 1) * 128],
                           moving=gs[0:128, kt:kt + 1])
    part = nl.ndarray((128, KT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=part, src=dps)
    tot = nl.ndarray((128, KT), dtype=nl.float32, buffer=nl.sbuf)
    if nprog > 1:
        oth = nl.ndarray((128, KT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.sendrecv(src=part, dst=oth, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                      dma_engine=nisa.dma_engine.gpsimd_dma)
        nisa.tensor_tensor(dst=tot, data1=part, data2=oth, op=nl.add)
    else:
        nisa.tensor_copy(dst=tot, src=part)
    d = nl.ndarray((128, KT), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=d, src=tot)
    nisa.tensor_tensor(dst=d, data1=h, data2=d, op=nl.add)
    orow_p = nl.ndarray((KT, 128), dtype=x.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=orow_p, data=d, engine=nisa.engine.tensor)
    orow = nl.ndarray((KT, 128), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=orow, src=orow_p)
    out = nl.ndarray((1, 1, H), dtype=x.dtype, buffer=nl.shared_hbm)
    KH = KT // nprog
    nisa.dma_copy(dst=out.reshape((KT, 128))[pid * KH:(pid + 1) * KH, :], src=orow[pid * KH:(pid + 1) * KH, :])
    if DEBUG:
        dh = nl.ndarray((128, KT), dtype=x.dtype, buffer=nl.shared_hbm)
        dhn = nl.ndarray((128, KT), dtype=x.dtype, buffer=nl.shared_hbm)
        dg = nl.ndarray((nprog, 128, NTI), dtype=x.dtype, buffer=nl.shared_hbm)
        dp = nl.ndarray((nprog, 128, KT), dtype=nl.float32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=dh, src=h)
        nisa.dma_copy(dst=dhn, src=hn)
        nisa.dma_copy(dst=dg[pid], src=gs)
        nisa.dma_copy(dst=dp[pid], src=part)
        return out, dh, dhn, dg, dp
    return out



# ======================================================================================= fused decode layer (dev)
from qwen3 import _rows_to_col as r2c, _col_rmsnorm as crms, _sync


def _mlp_col(h, g2c, wg, wu, wd, ones_f, pid):
    """MLP of the column h [128, KT] bf16 -> returns d + h (column, bf16)."""
    KT = H // 128
    IC = I // 2
    NTI = IC // 128
    gc = nl.ndarray((128, KT), dtype=g2c.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gc, src=g2c)
    hn = nl.ndarray((128, KT), dtype=h.dtype, buffer=nl.sbuf)
    crms(hn, h, gc, ones_f, KT)
    c0 = pid * IC
    gps = nl.ndarray((128, NTI), dtype=nl.float32, buffer=nl.psum)
    ups = nl.ndarray((128, NTI), dtype=nl.float32, buffer=nl.psum)
    for kt in range(KT):
        wgt = nl.ndarray((128, IC), dtype=wg.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wgt, src=wg[kt * 128:(kt + 1) * 128, c0:c0 + IC], dge_mode=DGE)
        for nt in range(NTI):
            nisa.nc_matmul(dst=gps[0:128, nt:nt + 1], stationary=wgt[0:128, nt * 128:(nt + 1) * 128],
                           moving=hn[0:128, kt:kt + 1])
        wut = nl.ndarray((128, IC), dtype=wu.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wut, src=wu[kt * 128:(kt + 1) * 128, c0:c0 + IC], dge_mode=DGE)
        for nt in range(NTI):
            nisa.nc_matmul(dst=ups[0:128, nt:nt + 1], stationary=wut[0:128, nt * 128:(nt + 1) * 128],
                           moving=hn[0:128, kt:kt + 1])
    gs = nl.ndarray((128, NTI), dtype=h.dtype, buffer=nl.sbuf)
    us = nl.ndarray((128, NTI), dtype=h.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gs, src=gps)
    nisa.tensor_copy(dst=us, src=ups)
    nisa.activation(dst=gs, op=nl.silu, data=gs)
    nisa.tensor_tensor(dst=gs, data1=gs, data2=us, op=nl.multiply)
    dps = nl.ndarray((128, KT), dtype=nl.float32, buffer=nl.psum)
    for kt in range(NTI):
        r0 = c0 + kt * 128
        wdt = nl.ndarray((128, H), dtype=wd.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wdt, src=wd[r0:r0 + 128, 0:H], dge_mode=DGE)
        for nt in range(KT):
            nisa.nc_matmul(dst=dps[0:128, nt:nt + 1], stationary=wdt[0:128, nt * 128:(nt + 1) * 128],
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


@nki.jit
def layer_dec2(x, g1c, wqkv, qn, kn, rcos, rsin, rot, kc, vc, pos, posf, wo, g2c, wg, wu, wd, sctx):
    """One decoder layer for one token, everything on chip between the input and output rows.
    x [1,1,H] row; g1c/g2c [128, H/128] gammas in column layout; qn/kn [D, 1]; rcos/rsin [SMAX, D] (halves
    repeated); rot [D, D] bf16 rotate-half matrix (rot[k, m] = -1 if k = m + 64, +1 if k = m - 64);
    kc [NKV, D, SMAX] (K cache, transposed); vc [NKV, SMAX, D]; pos int32 [1, 1], posf f32 [1, 1];
    sctx [1, S] carries the attention bucket S. Core c: q heads 16c.., kv heads 4c.."""
    S = sctx.shape[1]
    SMAX = kc.shape[2]
    pid = nl.program_id(0)
    KT = H // 128
    QH = NQ // 2        # q heads per core
    KH = NKV // 2       # kv heads per core
    GQ = QH // KH       # q heads per kv head
    NC_ = QH + 2 * KH   # qkv column tiles per core
    ones_f = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    # ---- input, rmsnorm
    xr = nl.ndarray((KT, 128), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xr, src=x.reshape((KT, 128)))
    xc = nl.ndarray((128, KT), dtype=x.dtype, buffer=nl.sbuf)
    r2c(xc, xr, KT)
    gc = nl.ndarray((128, KT), dtype=g1c.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gc, src=g1c)
    xn = nl.ndarray((128, KT), dtype=x.dtype, buffer=nl.sbuf)
    crms(xn, xc, gc, ones_f, KT)
    # ---- qkv (weight-stationary): column tiles 0..QH-1 q, QH..QH+KH-1 k, then v
    qkv_ps = nl.ndarray((128, NC_), dtype=nl.float32, buffer=nl.psum)
    q0 = pid * QH * D
    k0 = NQ * D + pid * KH * D
    v0 = (NQ + NKV) * D + pid * KH * D
    for kt in range(KT):
        wq = nl.ndarray((128, QH * D), dtype=wqkv.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wq, src=wqkv[kt * 128:(kt + 1) * 128, q0:q0 + QH * D], dge_mode=DGE)
        wk = nl.ndarray((128, KH * D), dtype=wqkv.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wk, src=wqkv[kt * 128:(kt + 1) * 128, k0:k0 + KH * D], dge_mode=DGE)
        wv = nl.ndarray((128, KH * D), dtype=wqkv.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wv, src=wqkv[kt * 128:(kt + 1) * 128, v0:v0 + KH * D], dge_mode=DGE)
        for t in range(QH):
            nisa.nc_matmul(dst=qkv_ps[0:128, t:t + 1], stationary=wq[0:128, t * D:(t + 1) * D], moving=xn[0:128, kt:kt + 1])
        for t in range(KH):
            nisa.nc_matmul(dst=qkv_ps[0:128, QH + t:QH + t + 1], stationary=wk[0:128, t * D:(t + 1) * D], moving=xn[0:128, kt:kt + 1])
        for t in range(KH):
            nisa.nc_matmul(dst=qkv_ps[0:128, QH + KH + t:QH + KH + t + 1], stationary=wv[0:128, t * D:(t + 1) * D], moving=xn[0:128, kt:kt + 1])
    qkv = nl.ndarray((128, NC_), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qkv, src=qkv_ps)
    # ---- q/k rmsnorm per head (over the partitions)
    NQK = QH + KH
    sq = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=qkv[0:128, 0:NQK])
    ss = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ss, stationary=ones_f, moving=sq)
    epsb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=epsb, value=EPS)
    rstd = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=ss, bias=epsb, scale=1.0 / D)
    nqk = nl.ndarray((128, NQK), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=nqk, data1=qkv[0:128, 0:NQK], data2=rstd, op=nl.multiply)
    gq = nl.ndarray((128, 1), dtype=qn.dtype, buffer=nl.sbuf)
    gk = nl.ndarray((128, 1), dtype=kn.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gq, src=qn)
    nisa.dma_copy(dst=gk, src=kn)
    gqf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    gkf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gqf, src=gq)
    nisa.tensor_copy(dst=gkf, src=gk)
    nisa.tensor_scalar(dst=nqk[0:128, 0:QH], data=nqk[0:128, 0:QH], op0=nl.multiply, operand0=gqf)
    nisa.tensor_scalar(dst=nqk[0:128, QH:NQK], data=nqk[0:128, QH:NQK], op0=nl.multiply, operand0=gkf)
    # ---- rope: e = bf16(n * cos) + bf16(rotate_half(n) * sin)
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
    rh = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=rh, stationary=rot_sb, moving=nqk)
    ea = nl.ndarray((128, NQK), dtype=x.dtype, buffer=nl.sbuf)
    eb = nl.ndarray((128, NQK), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ea, data=nqk, op0=nl.multiply, operand0=csf)
    nisa.tensor_scalar(dst=eb, data=rh, op0=nl.multiply, operand0=snf)
    qk = nl.ndarray((128, NQK), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=qk, data1=ea, data2=eb, op=nl.add)
    # ---- cache write at pos: K column per head into kc[h, :, pos]; V as rows into vc[h, pos, :]
    nisa.dma_copy(dst=kc.ap(pattern=[[SMAX, D], [D * SMAX, KH]], offset=pid * KH * D * SMAX,
                            scalar_offset=pos_sb, indirect_dim=2), src=qk[0:128, QH:NQK])
    vr_ps = nl.ndarray((KH, 128), dtype=x.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=vr_ps, data=qkv[0:128, NQK:NC_], engine=nisa.engine.tensor)
    vr = nl.ndarray((KH, 128), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vr, src=vr_ps)
    nisa.dma_copy(dst=vc.ap(pattern=[[SMAX * D, KH], [1, D]], offset=pid * KH * SMAX * D,
                            scalar_offset=pos_sb, indirect_dim=1), src=vr)
    # ---- attention over the bucket: scores [QH, S] for all local q heads (q of other groups zeroed per kv head)
    qm = nl.ndarray((128, KH, QH), dtype=x.dtype, buffer=nl.sbuf)
    nisa.memset(dst=qm, value=0.0)
    for j in range(KH):
        nisa.tensor_copy(dst=qm[0:128, j, j * GQ:(j + 1) * GQ], src=qk[0:128, j * GQ:(j + 1) * GQ])
    kts = []
    for j in range(KH):
        kt_sb = nl.ndarray((128, S), dtype=kc.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=kt_sb, src=kc[pid * KH + j, 0:D, 0:S])
        kts.append(kt_sb)
    sc = nl.ndarray((QH, S), dtype=nl.float32, buffer=nl.sbuf)
    posq = nl.ndarray((QH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=posq, src=posf.ap(pattern=[[0, QH], [1, 1]], offset=0))
    io = nl.ndarray((QH, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[1, S]], offset=0, channel_multiplier=0)
    msk = nl.ndarray((QH, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=msk, data=io, op0=nl.greater, operand0=posq, op1=nl.multiply, operand1=-30000.0)
    for c in range(S // 512):
        sp = nl.ndarray((QH, 512), dtype=nl.float32, buffer=nl.psum)
        for j in range(KH):
            nisa.nc_matmul(dst=sp, stationary=qm[0:128, j, 0:QH], moving=kts[j][0:128, c * 512:(c + 1) * 512])
        nisa.scalar_tensor_tensor(dst=sc[0:QH, c * 512:(c + 1) * 512], data=sp, op0=nl.multiply,
                                  operand0=1.0 / (D ** 0.5), op1=nl.add, operand1=msk[0:QH, c * 512:(c + 1) * 512])
    mx = nl.ndarray((QH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mx, op=nl.maximum, data=sc, axis=1, negate=True)
    ex = nl.ndarray((QH, S), dtype=nl.float32, buffer=nl.sbuf)
    sm = nl.ndarray((QH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=ex, op=nl.exp, data=sc, bias=mx, scale=1.0, reduce_op=nl.add, reduce_res=sm)
    rs = nl.ndarray((QH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rs, data=sm)
    pb = nl.ndarray((QH, S), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=pb, data=ex, op0=nl.multiply, operand0=rs)
    NT = S // 128
    pT_ps = nl.ndarray((128, NT, QH), dtype=x.dtype, buffer=nl.psum)
    for t in range(NT):
        nisa.nc_transpose(dst=pT_ps[0:128, t, 0:QH], data=pb[0:QH, t * 128:(t + 1) * 128], engine=nisa.engine.tensor)
    pT = nl.ndarray((128, NT, QH), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pT, src=pT_ps)
    ac_ps = nl.ndarray((128, QH), dtype=x.dtype, buffer=nl.psum)
    for j in range(KH):
        vt = nl.ndarray((128, NT, D), dtype=vc.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=vt, src=vc.ap(pattern=[[D, 128], [128 * D, NT], [1, D]], offset=(pid * KH + j) * SMAX * D))
        o_ps = nl.ndarray((GQ, D), dtype=nl.float32, buffer=nl.psum)
        for t in range(NT):
            nisa.nc_matmul(dst=o_ps, stationary=pT[0:128, t, j * GQ:(j + 1) * GQ], moving=vt[0:128, t, 0:D])
        o_sb = nl.ndarray((GQ, D), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_sb, src=o_ps)
        nisa.nc_transpose(dst=ac_ps[0:128, j * GQ:(j + 1) * GQ], data=o_sb, engine=nisa.engine.tensor)
    ac = nl.ndarray((128, QH), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ac, src=ac_ps)
    # ---- o-proj: K = this core's heads (rows pid*QH*D ..), N = H; partial sums exchanged
    op_ps = nl.ndarray((128, KT), dtype=nl.float32, buffer=nl.psum)
    for kt in range(QH):
        r0 = pid * QH * D + kt * 128
        wot = nl.ndarray((128, H), dtype=wo.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wot, src=wo[r0:r0 + 128, 0:H], dge_mode=DGE)
        for nt in range(KT):
            nisa.nc_matmul(dst=op_ps[0:128, nt:nt + 1], stationary=wot[0:128, nt * 128:(nt + 1) * 128],
                           moving=ac[0:128, kt:kt + 1])
    o = _xsum(op_ps, KT, x.dtype, pid, 0)
    h = nl.ndarray((128, KT), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h, data1=xc, data2=o, op=nl.add)
    # ---- mlp, output row
    y = _mlp_col(h, g2c, wg, wu, wd, ones_f, pid)
    yr_ps = nl.ndarray((KT, 128), dtype=x.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=yr_ps, data=y, engine=nisa.engine.tensor)
    yr = nl.ndarray((KT, 128), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=yr, src=yr_ps)
    out = nl.ndarray((1, 1, H), dtype=x.dtype, buffer=nl.shared_hbm)
    KHH = KT // 2
    nisa.dma_copy(dst=out.reshape((KT, 128))[pid * KHH:(pid + 1) * KHH, :], src=yr[pid * KHH:(pid + 1) * KHH, :])
    if DEBUG:
        dq = nl.ndarray((2, 128, NC_), dtype=x.dtype, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=dq[pid, 0:128, 0:NQK], src=qk)
        nisa.dma_copy(dst=dq[pid, 0:128, NQK:NC_], src=qkv[0:128, NQK:NC_])
        da = nl.ndarray((2, 128, QH), dtype=x.dtype, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=da[pid], src=ac)
        dh = nl.ndarray((128, KT), dtype=x.dtype, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=dh, src=h)
        return out, dq, da, dh
    return out


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
def pre_cache2(k0, k1, v0, v1, kc, vc):
    _pre_cache_body(k0, k1, v0, v1, kc, vc)
    d = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=d, value=0.0)
    out = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=d)
    return out


@nki.jit
def pre_cache2_t(k0, k1, v0, v1, kc, vc):
    """test wrapper: pre_cache2's body into fresh output caches (zeroed first), returned."""
    kco = nl.ndarray(kc.shape, dtype=kc.dtype, buffer=nl.shared_hbm)
    vco = nl.ndarray(vc.shape, dtype=vc.dtype, buffer=nl.shared_hbm)
    pid = nl.program_id(0)
    z = nl.ndarray((128, kc.shape[2]), dtype=kc.dtype, buffer=nl.sbuf)
    nisa.memset(dst=z, value=0.0)
    for h in range(NKV):
        nisa.dma_copy(dst=kco[h], src=z)
        for t in range(vc.shape[1] // 128):
            nisa.dma_copy(dst=vco[h, t * 128:(t + 1) * 128, :], src=z[0:128, 0:128])
    _sync(kco)
    _sync(vco)
    _pre_cache_body(k0, k1, v0, v1, kco, vco)
    return kco, vco


# ======================================================================================= small-T prefill (dev)
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
    pid = nl.program_id(0)
    KT = H // 128
    QH = NQ // 2
    IC = I // 2
    NI = IC // 128
    GN = 4                      # output tiles per weight DMA (512 columns)
    ones_f = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    # ---- o-proj partial over this core's heads: oT[n, t] = sum_k wo[k, n] a[k, t]
    at = nl.ndarray((128, QH, T), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=at, src=a.ap(pattern=[[T, 128], [D * T, QH], [1, T]], offset=pid * QH * D * T))
    op = nl.ndarray((128, KT, T), dtype=nl.float32, buffer=nl.sbuf)
    for g in range(KT // GN):
        w = nl.ndarray((128, QH, GN * 128), dtype=wo.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w, src=wo.ap(pattern=[[H, 128], [128 * H, QH], [1, GN * 128]], offset=pid * QH * D * H + g * GN * 128), dge_mode=DGE)
        for u in range(GN):
            ps = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
            for k in range(QH):
                nisa.nc_matmul(dst=ps, stationary=w[0:128, k, u * 128:(u + 1) * 128], moving=at[0:128, k, 0:T])
            nisa.tensor_copy(dst=op[0:128, g * GN + u, 0:T], src=ps)
    o = _xsum_hbm(op, KT, T, pid, 0, KT)
    ob = nl.ndarray((128, KT, T), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ob, src=o)
    # ---- h = x + o (transposed), rmsnorm over features
    hT = nl.ndarray((128, KT, T), dtype=x.dtype, buffer=nl.sbuf)
    _xpose_rows_in(hT, x.reshape((T, H)), T, KT)
    nisa.tensor_tensor(dst=hT, data1=hT, data2=ob, op=nl.add)
    sq = nl.ndarray((128, KT, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=hT)
    s1 = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=s1, op=nl.add, data=sq.ap(pattern=[[KT * T, 128], [1, T], [T, KT]], offset=0), axis=2)
    ssp = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ssp, stationary=ones_f, moving=s1)
    epsb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=epsb, value=EPS)
    rstd = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=ssp, bias=epsb, scale=1.0 / H)
    gc = nl.ndarray((128, KT), dtype=g2c.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gc, src=g2c)
    gcf = nl.ndarray((128, KT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gcf, src=gc)
    hn = nl.ndarray((128, KT, T), dtype=x.dtype, buffer=nl.sbuf)
    for j in range(KT):
        nisa.tensor_tensor(dst=hn[0:128, j, 0:T], data1=hT[0:128, j, 0:T], data2=rstd, op=nl.multiply)
        nisa.tensor_scalar(dst=hn[0:128, j, 0:T], data=hn[0:128, j, 0:T], op0=nl.multiply, operand0=gcf[0:128, j:j + 1])
    # ---- gate/up over this core's intermediate tiles, act = silu(g) * u
    act = nl.ndarray((128, NI, T), dtype=x.dtype, buffer=nl.sbuf)
    for g in range(NI // GN):
        c0 = pid * IC + g * GN * 128
        wgt = nl.ndarray((128, KT, GN * 128), dtype=wg.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wgt, src=wg.ap(pattern=[[I, 128], [128 * I, KT], [1, GN * 128]], offset=c0), dge_mode=DGE)
        wut = nl.ndarray((128, KT, GN * 128), dtype=wu.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wut, src=wu.ap(pattern=[[I, 128], [128 * I, KT], [1, GN * 128]], offset=c0), dge_mode=DGE)
        for u in range(GN):
            gp = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
            for k in range(KT):
                nisa.nc_matmul(dst=gp, stationary=wgt[0:128, k, u * 128:(u + 1) * 128], moving=hn[0:128, k, 0:T])
            upp = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
            for k in range(KT):
                nisa.nc_matmul(dst=upp, stationary=wut[0:128, k, u * 128:(u + 1) * 128], moving=hn[0:128, k, 0:T])
            gs = nl.ndarray((128, T), dtype=x.dtype, buffer=nl.sbuf)
            us = nl.ndarray((128, T), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gs, src=gp)
            nisa.tensor_copy(dst=us, src=upp)
            nisa.activation(dst=gs, op=nl.silu, data=gs)
            nisa.tensor_tensor(dst=act[0:128, g * GN + u, 0:T], data1=gs, data2=us, op=nl.multiply)
    # ---- down partial over this core's intermediate rows
    yp = nl.ndarray((128, KT, T), dtype=nl.float32, buffer=nl.sbuf)
    for g in range(KT // GN):
        wdt = nl.ndarray((128, NI, GN * 128), dtype=wd.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wdt, src=wd.ap(pattern=[[H, 128], [128 * H, NI], [1, GN * 128]], offset=pid * IC * H + g * GN * 128), dge_mode=DGE)
        for u in range(GN):
            ps = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
            for k in range(NI):
                nisa.nc_matmul(dst=ps, stationary=wdt[0:128, k, u * 128:(u + 1) * 128], moving=act[0:128, k, 0:T])
            nisa.tensor_copy(dst=yp[0:128, g * GN + u, 0:T], src=ps)
    # ---- this core finishes feature tiles pid*KT/2..: out = h + y, back to rows
    KH2 = KT // 2
    y = _xsum_hbm(yp, KT, T, pid, pid * KH2, (pid + 1) * KH2)
    yb = nl.ndarray((128, KH2, T), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=yb, src=y)
    nisa.tensor_tensor(dst=yb, data1=yb, data2=hT[0:128, pid * KH2:(pid + 1) * KH2, 0:T], op=nl.add)
    out = nl.ndarray((1, T, H), dtype=x.dtype, buffer=nl.shared_hbm)
    o2 = out.reshape((T, H))
    for tt in range(T // 128):
        tp = nl.ndarray((128, KH2 * 128), dtype=x.dtype, buffer=nl.psum)
        for j in range(KH2):
            nisa.nc_transpose(dst=tp[0:128, j * 128:(j + 1) * 128], data=yb[0:128, j, tt * 128:(tt + 1) * 128],
                              engine=nisa.engine.tensor)
        r = nl.ndarray((128, KH2 * 128), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=r, src=tp)
        nisa.dma_copy(dst=o2[tt * 128:(tt + 1) * 128, pid * KH2 * 128:(pid + 1) * KH2 * 128], src=r)
    return out


def _rms_T(hn, hT, g, ones_f, T, n):
    """hn [128, n, T] = bf16(hT * rsqrt(mean over features + eps)) * gamma; hT [128, n, T] bf16 (feature j*128+p
    at [p, j, t]); g [128, n] gamma in column layout (SBUF, bf16)."""
    sq = nl.ndarray((128, n, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=hT)
    s1 = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=s1, op=nl.add, data=sq.ap(pattern=[[n * T, 128], [1, T], [T, n]], offset=0), axis=2)
    ssp = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ssp, stationary=ones_f, moving=s1)
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
    SMAX = kc.shape[2]
    pid = nl.program_id(0)
    KT = H // 128
    QH = NQ // 2
    KH = NKV // 2
    NQK = QH + KH
    ones_f = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    xT = nl.ndarray((128, KT, T), dtype=x.dtype, buffer=nl.sbuf)
    _xpose_rows_in(xT, x.reshape((T, H)), T, KT)
    gc = nl.ndarray((128, KT), dtype=g1c.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gc, src=g1c)
    xn = nl.ndarray((128, KT, T), dtype=x.dtype, buffer=nl.sbuf)
    _rms_T(xn, xT, gc, ones_f, T, KT)
    QKVW = (NQ + 2 * NKV) * D
    # ---- q and k heads (transposed outputs [d, T])
    qk = nl.ndarray((128, NQK, T), dtype=x.dtype, buffer=nl.sbuf)
    for g in range(NQK // 4):
        c0 = pid * QH * D + g * 512 if g < QH // 4 else NQ * D + pid * KH * D
        w = nl.ndarray((128, KT, 512), dtype=wqkv.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w, src=wqkv.ap(pattern=[[QKVW, 128], [128 * QKVW, KT], [1, 512]], offset=c0), dge_mode=DGE)
        for u in range(4):
            ps = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
            for k in range(KT):
                nisa.nc_matmul(dst=ps, stationary=w[0:128, k, u * 128:(u + 1) * 128], moving=xn[0:128, k, 0:T])
            nisa.tensor_copy(dst=qk[0:128, g * 4 + u, 0:T], src=ps)
    # ---- v rows [t, (head, d)]
    wv = nl.ndarray((128, KT, KH * D), dtype=wqkv.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=wv, src=wqkv.ap(pattern=[[QKVW, 128], [128 * QKVW, KT], [1, KH * D]], offset=(NQ + NKV) * D + pid * KH * D), dge_mode=DGE)
    vr = nl.ndarray((128, T // 128, KH * D), dtype=x.dtype, buffer=nl.sbuf)
    for tt in range(T // 128):
        ps = nl.ndarray((128, KH * D), dtype=nl.float32, buffer=nl.psum)
        for k in range(KT):
            nisa.nc_matmul(dst=ps, stationary=xn[0:128, k, tt * 128:(tt + 1) * 128], moving=wv[0:128, k, 0:KH * D])
        nisa.tensor_copy(dst=vr[0:128, tt, 0:KH * D], src=ps)
    # ---- q/k rmsnorm over d (partitions), gamma
    sq = nl.ndarray((128, NQK, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=qk)
    rstd = nl.ndarray((128, NQK, T), dtype=nl.float32, buffer=nl.sbuf)
    epsb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=epsb, value=EPS)
    for j in range(NQK):
        ssp = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ssp, stationary=ones_f, moving=sq[0:128, j, 0:T])
        nisa.activation(dst=rstd[0:128, j, 0:T], op=nl.rsqrt, data=ssp, bias=epsb, scale=1.0 / D)
    nqk = nl.ndarray((128, NQK, T), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=nqk, data1=qk, data2=rstd, op=nl.multiply)
    gq = nl.ndarray((128, 1), dtype=qn.dtype, buffer=nl.sbuf)
    gk = nl.ndarray((128, 1), dtype=kn.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gq, src=qn)
    nisa.dma_copy(dst=gk, src=kn)
    gqf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    gkf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gqf, src=gq)
    nisa.tensor_copy(dst=gkf, src=gk)
    nisa.tensor_scalar(dst=nqk[0:128, 0:QH, 0:T], data=nqk[0:128, 0:QH, 0:T], op0=nl.multiply, operand0=gqf)
    nisa.tensor_scalar(dst=nqk[0:128, QH:NQK, 0:T], data=nqk[0:128, QH:NQK, 0:T], op0=nl.multiply, operand0=gkf)
    # ---- rope: e = bf16(n * cos) + bf16(rotate_half(n) * sin), cos/sin [d, t]
    ct = nl.ndarray((128, T), dtype=rcT.dtype, buffer=nl.sbuf)
    st = nl.ndarray((128, T), dtype=rsT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=ct, src=rcT[0:D, 0:T])
    nisa.dma_copy(dst=st, src=rsT[0:D, 0:T])
    rot_sb = nl.ndarray((D, D), dtype=rot.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_sb, src=rot)
    e = nl.ndarray((128, NQK, T), dtype=x.dtype, buffer=nl.sbuf)
    for j in range(NQK):
        rp = nl.ndarray((128, T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=rp, stationary=rot_sb, moving=nqk[0:128, j, 0:T])
        ea = nl.ndarray((128, T), dtype=x.dtype, buffer=nl.sbuf)
        eb = nl.ndarray((128, T), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ea, data1=nqk[0:128, j, 0:T], data2=ct, op=nl.multiply)
        nisa.tensor_tensor(dst=eb, data1=rp, data2=st, op=nl.multiply)
        nisa.tensor_tensor(dst=e[0:128, j, 0:T], data1=ea, data2=eb, op=nl.add)
    # ---- outputs and cache writes
    qo = nl.ndarray((NQ, D, T), dtype=x.dtype, buffer=nl.shared_hbm)
    ko = nl.ndarray((NKV, D, T), dtype=x.dtype, buffer=nl.shared_hbm)
    vo = nl.ndarray((NKV, T, D), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=qo.ap(pattern=[[T, 128], [D * T, QH], [1, T]], offset=pid * QH * D * T), src=e[0:128, 0:QH, 0:T])
    nisa.dma_copy(dst=ko.ap(pattern=[[T, 128], [D * T, KH], [1, T]], offset=pid * KH * D * T), src=e[0:128, QH:NQK, 0:T])
    nisa.dma_copy(dst=kc.ap(pattern=[[SMAX, 128], [D * SMAX, KH], [1, T]], offset=pid * KH * D * SMAX), src=e[0:128, QH:NQK, 0:T])
    for tt in range(T // 128):
        nisa.dma_copy(dst=vo.ap(pattern=[[D, 128], [T * D, KH], [1, D]], offset=pid * KH * T * D + tt * 128 * D),
                      src=vr[0:128, tt, 0:KH * D])
        nisa.dma_copy(dst=vc.ap(pattern=[[D, 128], [SMAX * D, KH], [1, D]], offset=pid * KH * SMAX * D + tt * 128 * D),
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

from qwen3 import pre_attn2 as _pa2
SPECS["pre_attn2_4096"] = ("pre_attn2", [("q", (NQ, D, 4096), bf), ("k", (NKV, D, 4096), bf), ("v", (NKV, 4096, D), bf)])
SPECS["pre_attn2_1024"] = ("pre_attn2", [("q", (NQ, D, 1024), bf), ("k", (NKV, D, 1024), bf), ("v", (NKV, 1024, D), bf)])
pre_attn2 = _pa2

@nki.jit
def pre_attn1(q, k, v):
    a = attention_cte(q=q, k=k, v=v, scale=1.0 / (D ** 0.5), causal_mask=True, tp_q=False, tp_k=False, tp_out=True)
    return a
SPECS["pre_attn1_4096"] = ("pre_attn1", [("q", (NQ, D, 4096), bf), ("k", (NKV, D, 4096), bf), ("v", (NKV, 4096, D), bf)])
SPECS["pre_attn1_1024"] = ("pre_attn1", [("q", (NQ, D, 1024), bf), ("k", (NKV, D, 1024), bf), ("v", (NKV, 1024, D), bf)])
