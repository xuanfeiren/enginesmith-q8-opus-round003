# Qwen3-8B engine for Trainium2 (one logical NeuronCore): notes

## Status

Session 003 (final build): dev-eval 004 (public prompts, 2 rounds): **score 1.5701** (prefill 1.7395, decode 1.4172);
prefill 256 / 1024 / 4096 2.10 / 2.11 / 1.19, decode 1.37 / 1.48 / 1.40, 1 mismatching step per round of 1300, worst
rank 2 (round 002's formal result: 1.5049; this session's dev-evals: 001 1.5262 after the prefill changes, 002 1.5374
with the decode buckets, 003 disturbed by a 405 ms outlier, 004 with GU_FUSE + the stdin spin). Absolute (drive.py,
core 34): prefill 256 / 1024 / 4096 32.6 / 114.9 / 511.2 ms (round 002: 33.2 / 123.5 / 526.0), decode 23.35 / 23.6 /
24.3 ms per token (round 002: 23.6 / 24.0 / 24.7). Every change is bitwise identical to round 002 (same.py, 40 cases
x 3 request shapes, after each step and for the final binary, self-consistent). A clean checkout builds in ~2.3 min
(32 jobs). What changed and what was rejected: "Session 003" below.

Session 002: dev-eval (public prompts, 1 round): score 1.5075 (prefill 1.6509, decode 1.3765);
prefill 256 / 1024 / 4096 1.99 / 1.96 / 1.15, decode 1.31 / 1.44 / 1.38, 1 mismatching step of 1300 (round 001's
formal result: 1.4708). The one change: prefill T <= 512 on `pre_layers2` (below), prefill 256 37.7 -> 33.2 ms
(paired A/B, 6 rounds: 1.135x), every other case unchanged; tokens bitwise identical to round 001
(brief/skill/scripts/same.py, 40 cases x 3 request shapes, lengths 1..4097). The decode mean moved 1.36 -> 1.38
with the decode code unchanged: that is the reference's run-to-run spread; the expected gain is the prefill 256 case
(~+12%), about +2% on the score.

Session 001 history (dev-eval, public prompts, 1 round each, vs the NxD reference):

| version | score | prefill 256 / 1024 / 4096 | decode 256 / 1024 / 4096 | mismatching steps |
|---|---|---|---|---|
| seed (round 000, formal 5 rounds) | 1.0354 | 0.89 / 1.29 / 0.85 | 1.03 / 1.13 / 1.08 | 1 |
| v1 own decode MLP | 1.0799 | 0.89 / 1.29 / 0.85 | 1.13 / 1.23 / 1.17 | 1 |
| v3 own decode layer, 12 layers per NEFF | 1.1747 | 0.91 / 1.33 / 0.87 | 1.30 / 1.41 / 1.36 | 1 |
| v4 own prefill kernels T <= 512 | 1.2605 | 1.40 / 1.32 / 0.87 | 1.29 / 1.42 / 1.35 | 3 of 1300 |
| v5 own prefill kernels, every bucket | 1.3365 | 1.50 / 1.58 / 0.99 | 1.29 / 1.41 / 1.35 | 2 of 1300 |
| v6 X layout, 512-token blocks | 1.3791 | 1.47 / 1.76 / 1.10 | 1.28 / 1.40 / 1.35 | 3 of 1300 |
| v8 last layer on the decode kernel, qkv3 pipelined | 1.4003 | 1.48 / 1.83 / 1.15 | 1.28 / 1.40 / 1.35 | 2 of 1300 |
| v9 attention inside the qkv kernel (T <= 512) | 1.4568* | 1.75 / 1.84 / 1.15 | 1.31 / 1.44 / 1.36 | 1 of 1300 |
| v10 5 prefill layers per NEFF (T <= 512) | 1.4384 | 1.77 / 1.83 / 1.15 | 1.27 / 1.39 / 1.35 | 1 of 1300 |
| v12 wqkv columns grouped per core (decode QKV one slab) | 1.4500 | 1.79 / 1.83 / 1.15 | 1.31 / 1.40 / 1.35 | 1 of 1300 |
| v13 T=1024 on the multi-layer kernel (2 blocks, own attention) | **1.4708** | 1.76 / 1.96 / 1.15 | 1.32 / 1.42 / 1.36 | 1 of 1300 |
| **v14 decode bucket 1536 (final, 3 rounds)** | **1.4762** | 1.78 / 1.97 / 1.15 | 1.31 / 1.44 / 1.36 | 1 per round of 1300 |

(*the reference ran ~8% slower than usual in that run.) Run-to-run noise of the decode ratio ~1%; logical core 35 decodes
~0.8% slower than core 34 (A/B with swapped cores), so compare builds with swapped-core A/B pairs (dev: ab.py with
wrapper scripts that set NEURON_RT_VISIBLE_CORES=34 / 35).

Absolute (v14, own driver dev/drive.py, experiment core 34): prefill 100 / 256 / 500 / 1024 / 4096 tokens
30 / 37.4 / 64 / 123.5 / 526 ms (reference 256 / 1024 / 4096: 66 / 243 / 606); decode 23.4-23.7 (256) to 24.5-24.7
(4096) ms per token (reference ~31-34). All mismatches are rank-2 near-ties (worst_rank 2), far inside the gates (130 allowed).

## Layout

- `kernels/qwen3.py`: the engine's NKI kernels (one source; sizes come from input shapes).
- `kernels/nkilib/`: vendored nkilib (Apache-2.0, THIRD_PARTY.md). Still used: `attention_cte` (prefill attention),
  `cascaded_max` (argmax). The other vendored kernels (qkv/mlp/output_projection CTE, attention_block_tkg) are no
  longer called (kept for reference; the old prefill kernels `pre_qkv`, `pre_mlp`, ... are still in qwen3.py but not
  built).
- `kernels/build_neffs.py`: the NEFF list (`neff_specs`) and the build. Each NEFF is compiled in a fresh Python process
  through NKI's documented standalone route (`NKI_ARTIFACTS_DIR` + calling the kernel with numpy zeros, which also
  runs it once on `NEURON_RT_VISIBLE_CORES` = ENGINESMITH_NEURON_CORE + 2; if that run cannot claim the core, the NEFF
  is already written and used). No private nki entry points. Cached in `target/neffcache`. 49 NEFFs, ~2 min (32 jobs).
  Refuses to run with a `Q8_*` variable set (dev-only kernel switches, not in the cache key).
- `build.sh`: NEFFs -> `.incbin` -> `target/engine` (links libnrt only).
- `src/engine.cpp`: host: checkpoint mmap, weights transposed to [in, out] (one allocation per weight kind for all 36
  layers, sliced per layer and per 12-layer group), LM head swizzled, column-layout gammas, RoPE tables, rotate-half
  matrix; tensor sets built once; every call warmed up at hello (~40 s).
- `dev/`: development tools (not built): `drive.py` (timing + token dump on the experiment cores), `kc.py` (compile
  one kernel), `t_*.py` (numpy checks of kernels with HF rounding points), `pe*.py`/`dmab.py`/`gemv.py` (PE and DMA
  micro-benchmarks), `exp.py` (experiments). Session 003: `var.sh TAG MODULE KERNEL SPEC [Q8_X=v ..]` (compile one
  kc.py spec with switch overrides on core 35, profile it on core 34 under the GPU lock, print the device time),
  `prof.py` specs (`pl_T`, `pl2_T`, `dec_S`, `post3_4096`, `qkv3_4096`, ...), `lines.py` (engine busy per source line
  in a window), `dips.py` (DMA bytes per 5 us bin, low bins with the PE lines active), `pegemv.py` (PE-only GEMV).

## Forward pass

Decode step (5 executions per token, queued, one wait): `embed_dec` -> 3 x `layer_dec_S` (12 layers each) ->
`lm_head` -> `argmax`. Buckets S 384 / 512 / 1024 / 1152 / 1536 / 2048 / 3072 / 4224 / 4608 (first S > position;
multiples of 128, key chunks of 512 with a shorter last one; 384 / 1152 / 4224 sit just above the timed positions
P + 63). QKV weight slabs of 2 k-tiles per DMA and down slabs of 2 rows (session 003, DQ_KG / DD_KG).
wqkv columns are grouped per core on the host: [core c: q heads 16c.. | k heads 4c.. | v heads 4c..], so a decode
layer reads its QKV slab as one 6 KB piece per row (decode +1.5-2.8% at 256).
`layer_dec` (`_layer`, `_mlp_col`): everything in column layout (vector of n*128 as [128, n], v[t*128+p] at [p, t]);
GEMVs weight-stationary (128x128 tile of W [in, out] stationary, activation column moving, output already in column
layout). Core c: q heads 16c..16c+15, kv heads 4c..4c+3, gate/up columns c*6144.., the matching halves of o-proj's
and down's K; two fp32 exchanges per layer (`sendrecv`, gpsimd_dma, [128, 32]). Attention: K cache transposed
[NKV, D, SMAX], V cache [NKV, SMAX, D]; new K/V written at `pos` (dynamic DMA); scores for all 16 local heads in one
[16, S] tile (q of the other kv groups zeroed per kv head); mask iota > pos; fp32 softmax; bf16 P; PE transposes.
Rounding points as HF (bf16 after each op HF rounds).

Session 003 schedule changes (bitwise identical): on-chip attention (`_qkva`, `_qkva2`) software-pipelined over
(head, q-tile) iterations in three stages (`_att_a` scores + mask on the PE + one scale/min pass, `_att_b` exp +
normalization, `_att_c` P^T + PV), stage A of iteration i issued with B of i-3 and C of i-6 (`_att_pipe`); the QKV
of T >= 512 blocks in head groups of 2 with the q/k norm + rope of group u issued after the projections of group u+2
(`_qkv4`); post-attention for T >= 1024 with SBUF -> SBUF partial-sum swaps instead of the HBM buffer + barriers
(`_post5_blk`). Details and numbers: "Session 003".

Prefill for T <= 512 (session 002): `pre_layers2` (5 layers per NEFF, one token block). Same phases and arithmetic as
`pre_layers` below, but the residual block and the attention output stay in SBUF for the whole NEFF (X is read once
at the start and each core writes its half of the features at the end), and the o-proj and down partial sums are
swapped between the cores per output group with `nisa.sendrecv` (SBUF -> SBUF, default DMA engine) right after the
group's PSUM drain, consumed after the group loop (`_post4`, `_fin`). Down uses groups of 4 output tiles (`DN_GN`).

Prefill for bucket T in {128, 256, 512, 1024, 2048, 4096, 4608} (prompt padded with token 0; padded positions are only
written to the cache beyond P, decode masks them): the residual stream lives in HBM in the "X layout"
X[block, p, j, t] = x[block*TB + t, j*128 + p] (TB = min(T, 512)), i.e. [feature, token], contiguous per block.
`embed_x` (gather + PE transposes) -> layers 0..34: for T <= 1024 `pre_layers` (5 layers per NEFF; per layer `_qkva`:
rmsnorm, q/k/v, q/k norm, rope, cache writes and causal attention on chip with full-row fp32 softmax, K/V of all blocks
kept in SBUF; then the `pre_post3` body), for T >= 2048 per layer `pre_qkv3` (q, k as [d, T], v as [T, d]),
`pre_attn2` (2 x nkilib attention_cte, tp_q=False, tp_k=False, tp_out=True), `pre_post3` (o-proj + residual + rmsnorm +
MLP + residual) -> for the last layer only `pre_qkv3` (cache), then `x_row`
(the row of token P-1 of the last layer's input) -> `layer_dec1_S` (the decode kernel, one layer, at pos P-1) ->
`lm_head` -> `argmax`. Prefill kernels split weights, not tokens, between the cores (heads / intermediate columns);
tokens are the moving operand (up to 512 columns); fp32 partial sums are exchanged through a shared-HBM buffer +
`core_barrier` (session 003: `_post5_blk`, sendrecv swaps; only the layer end keeps a barrier). Weight slabs: [128, 16 k-tiles, GN*128 columns], GN = 8 (2 KB pieces) for TB <= 256, 4 for TB = 512
(PSUM: GN accumulators of [128, TB] fp32).

## Measured (neuron-explorer device time unless noted)

- Decode layer: nkilib attention_block_tkg + mlp 0.80 ms -> own layer 0.667 ms (1 layer per NEFF) -> 0.632 ms
  (12 per NEFF, MBU 0.86). Whole token 30.2 -> 23.9 ms. Inside one layer the MLP phase streams at ~670 GB/s; the
  QKV/attention part (first ~220 us) at ~520 GB/s (k/v weight slabs are 1 KB pieces; DMA dips during attention).
- Layers per decode NEFF: 1 / 2 / 4 per NEFF 665 / 654 / 641 us per layer; token 26.0 (1) / 24.3 (4) / 24.0 (12) /
  23.8 (36) ms. 36 per NEFF compiles in 6 min and made prefill 256 jump 73 -> 190-210 ms (not understood): 12 used.
- Prefill per layer at T=256: qkv3 0.19-0.26, attention 0.18, post3 0.76-0.82 ms; at T=4096: qkv3 2.43 (before the
  pipelining), attention 2.63, post3 10.1 ms. post3 at 4096 is at ~93% of the PE peak (5120 x T moving columns per core
  per layer / 2.2 columns per ns = 9.5 ms).
- PE: ~2.2 moving columns per ns per physical core for 128x128 bf16 stationaries (~72 TFLOPS/core), the same for 128,
  256, 512 columns and for 1 or 8 rotating stationaries (dev/pe.py, dev/pe2.py). LDWEIGHTS ~100 ns, hidden when
  pipelined.
- DMA of weight slabs [128 p, k-tiles, W] from a [4096, 12288] bf16 tensor, 100 MB (dev/dmab.py): W=512 (1 KB pieces)
  231 us (435 GB/s), 1024: 194 (518), 2048: 183 (550), 6144: 179 (562). Splitting a DMA into 2-4: slower.
- The reference's 4096 context-encoding NEFF: 508 ms device time (its request takes 606), HBM read 86 GB, PE active
  69%, throttle limit 0.93; it uses no NKI kernels (compiler-generated matmuls).
- nkilib attention_cte at 4096 (2 calls of 16 heads or 1 of 32: same) 2.56 ms; tp_q=False/tp_k=False layouts 2.57 vs
  2.97 ms for the transposed ones; all engines 88-94% active, throttle limit 0.75: hard to beat without less work.

## Findings and pitfalls

- nkilib CTE kernels split tokens between the two cores, so each core streams all weights: at T=256 pre_mlp read
  207 MB per 100 MB chunk. Splitting weights instead halved prefill MLP time at small T.
- `nc_matmul(accumulate=(kt > 0))` on interleaved [128, 1] PSUM accumulators silently dropped terms; leave the flag
  to the compiler (default inference was correct everywhere).
- NKI kernels: no list comprehensions, no `os.environ`, no method calls like `int.bit_length()`; Python `if` on
  `nl.program_id(0)` works (programs are traced per core).
- `.ap(..., scalar_offset=r, indirect_dim=d)` scales r by dim d's stride of the full tensor.
- x stationary with 512-wide moving weights (the LM head's form) is bound by the PE's moving rate: decode MLP 541 us vs
  515 us weight-stationary.
- In software pipelining, PE work of the next block's preparation (e.g. 32 accumulating fp32 ones-matmuls for a
  sum of squares) lands in front of the current block's projections in the PE's in-order queue; doing the sum over
  feature tiles on the vector engine and one ones-matmul fixed it (qkv3 at 1024: 0.631 -> 0.584 -> 0.552 ms).
  `is_stationary_onezero=True` on the fp32 ones-matmuls: no measurable change.
- The cross-core race of nkilib outputs (seed notes): `_sync` (core_barrier) after nkilib outputs our code reads.
- One loaded model instance per tensor set (rebinding a NEFF to another tensor set cost 2.7 ms per call).
- `cascaded_max` returns the higher index on an exact fp32 tie (protocol: lowest); not fixed (rare with fp32 logits).

- Later measurements: the per-token host gap in decode is ~0.05 ms (not worth moving the token loop on device).
  Own on-chip attention costs ~5 us per 1000 keys per (head, 128 queries): wins at T <= 1024 (fixed costs of
  attention_cte: 0.18 ms at 256, 0.47 at 1024; qkva at 1024 0.92 ms vs qkv3 + attn2 1.02) but would lose at 4096.
  Splitting _qkv3_blk into two functions (projection / post-processing) made pre_layers 2% slower (4.90 -> 5.00 ms)
  with identical instructions in principle; keep it one function. Gate/up groups of 12 tiles at TB=256 (3 KB pieces,
  12 PSUM accumulators): 0.76 -> 0.92 ms, slower. Pitfall caught by the token check: pre_layers first looped only
  over block 0 (fast and wrong at T=1024) - test every bucket class against the single-layer kernels.
- Prefill 4096 per layer: qkv3 2.06, attention 2.59, post3 10.1 ms (PE ~93% of peak); the reference's whole 4096
  NEFF takes 508 ms of device time, ours ~520: close to the practical limit.

## Session 002: measurements and rejected attempts

Tools added (dev/): `prof.py` (kc.py specs: decode NEFFs with 1/2/4 layers, `pl_T` / `pl2_T` prefill), `cap.sh`
(neuron-explorer capture + summary, JSON=1 for the per-instruction json), `tl.py` (HBM bytes and engine busy per time
bin, with the busiest PE source line), `gap.py` (instructions in a time window grouped by engine/line), `pesig.py`
(PE instructions that signal semaphores and the Sync engine's DMA triggers with what they wait on). Experiment core
35 (`NEURON_RT_VISIBLE_CORES=35 python dev/kc.py prof pre_layers2 pl2_256 out.neff`, then `XCORE=35 dev/cap.sh`).

Where the time went in `pre_layers` at T=256 (5-layer NEFF 4.97 ms, ~1.0 ms per layer; PE floor ~0.69 ms per layer):
three PE-idle stretches per layer, all latency chains through HBM: the end of the layer (down partials -> ybuf ->
barrier -> read -> residual -> Xo -> barrier -> X read -> rmsnorm: ~90 us), o-proj -> rmsnorm (~65 us), and the
attention (~80 us at low PE use). The resident variant (5-layer NEFF, T=256, device time):

| variant | ms |
|---|---|
| pre_layers (round 001) | 4.97 |
| resident, sendrecv per group, consumed right away | 4.81 (DVE blocks on the swap, PSUM drains stall) |
| same, HBM buffer + core_barrier per group | 5.12 |
| same, HBM + barrier triggered from GpSimd (swdge) | 5.04 |
| sendrecv, consumer 2 groups later / all after the loop | 4.56 / 4.52 |
| GpSimd `sendrecv(dma_engine=gpsimd_dma)` per 1 KB tile | 6.14 |
| all after the loop, weight slabs KG=8 | 4.48 |
| + down in groups of 4 output tiles (DN_GN) | **4.36** (kept) |
| down groups of 2 | 4.62 |
| HBM write per group, one barrier at the end | 4.65-4.82 |
| `_fin` adds on GpSimd | 4.53 |
| down weight DMAs triggered from the ACT queue | 4.57 |
| fence: next layer's QKV weight DMAs after the last down group | 4.49-4.50 (no gain) |
| QKV in head groups (k first) so attention overlaps the q projections | 4.44 (worse) |
| attention stage-major over 2 / 4 (head, q-tile) pairs | 4.39 / 4.41 (no gain) |
| attention: raw scores in PSUM, 1/sqrt(d) inside the exp (no scale pass) | 4.44 (worse); at T=1024 17.59 vs 17.07 |
| softmax normalization on DVE instead of ACT (T=1024) | 17.08 vs 17.07 (neutral) |
| rotate-half by two SBUF->SBUF partition-swap DMAs instead of the PE (exact) | 4.57 (T=256), qkv3 at 4096 2.19 vs 2.05 ms |
| gate/up in groups of 4 output tiles (instead of 8) | 4.63 (worse) |
| QKV projections in groups of 4 heads (instead of 8) | 4.38 (neutral) |

Program order did not move the scheduler: T=1024 with all blocks' QKV before the attention, or block b+1's
projections issued before block b's attention: 17.05 / 17.10 ms (base 17.07); pre_qkv3 at 4096 with block b+1's
projections issued before block b's q/k norm and rope: 2.22 vs 2.05 ms.

Other buckets: T=128 4.10 -> 3.69 ms, T=512 8.55 -> 8.48, T=1024 (two blocks, the K/V of both in SBUF) 17.07 -> 17.59:
1024 keeps `pre_layers`.

Compiler behaviour found on the way (neuron-explorer json): (1) the scheduler reorders matmuls across loops: with
8 accumulators per down group it ran each accumulator over all 6 of the group's weight slabs (u-outer), so every
slab of the group stayed pinned until the group's end and the weight DMA could not run ahead (Sync DMA triggers wait
on PE semaphores, `S[5] (Tensor)>=N`); smaller groups fixed it. (2) Weight DMA triggers are all on the Sync engine,
cache/activation DMAs on GpSimd (swdge). (3) In decode the weight pool is ~120 KB per partition
(0x1a000-0x38000); DMA idles when the pool is full of o-proj slabs during attention (~10 us per layer) and at the
layer start (~10-20 us); the MLP phase streams at ~330-340 GB/s per physical core. Decode per layer in the 12-layer
NEFF is ~594 us against ~573 us at 335 GB/s: decode is within ~4% of the streaming rate the MLP phase reaches.

prefill qkv3 at 4096 (2.05 ms, PE 88% active): per 512-token block ~150 us of projections at full PE, then
~100 us of q/k norm + rope that is vector-engine bound and not overlapped with the next block's projections.
p1024 (pre_layers, 3.41 ms per layer): QKV 2 x 200, attention ~450 us (ACT ~110%, DVE ~80%, PE ~100 us of work),
o-proj 2 x 150, gate/up 2 x 650, down 2 x 350, three ~50 us gaps.

## Session 003: changes, measurements, rejected attempts

Kernel switches are module constants at the top of kernels/qwen3.py (dev only: `Q8_*` environment overrides read at
import, so `dev/var.sh TAG MODULE KERNEL SPEC Q8_X=..` compiles + profiles a variant on the experiment cores; the
build never sets them). Device times below: neuron-explorer, 5-layer NEFFs unless noted, random weights; run-to-run
spread of one profile ~+-0.5%.

Kept (all bitwise identical to round 002):

| change | where | device time |
|---|---|---|
| attention stages software-pipelined in program order (`_att_pipe`: stage A of iteration i, B of i-3, C of i-6; ATT_PIPE=3) | `_qkva`, `_qkva2` | pre_layers 1024: 17.07 -> 16.76 (lag 2) / 16.73 (3, 4); pre_layers2 256: 4.35 -> 4.27-4.30 |
| softmax normalization on the vector engine (ATT_NORM=vec) | same | 1024: 17.07 -> 16.91 alone |
| diagonal causal mask added on the PE (identity x bf16 mask accumulated into the scores' PSUM; -29952 instead of -30000 where masked: exp -> 0 either way), then ONE vector pass `tensor_scalar_reduce` (sc = -(s/sqrt(d)), min) instead of ACT scale + DVE stt + DVE max; exp(-1 * sc + min) | `_att_a` (ATT_MAX=pe) | 1024: 16.73 -> 16.70 (with the pipeline; alone no gain: the stages were latency-bound before the pipeline) |
| `_post5_blk`: post3 with the o-proj partials swapped per output group by nisa.sendrecv (SBUF -> SBUF) and the down partials per pair of groups (g owned by core 0, g + 4 by core 1, computed in the order 0,4,1,5,..; each core sends the partial of the other's group), consumed POST5_LAG=3 groups later; no HBM buffer, no core_barrier inside the layer | pre_layers (T=1024), pre_post3 (T>=2048) | pre_layers 1024: 16.68 -> 16.14 (lag 0 16.46, 1 16.33, 2 16.35, 3 16.14, 4 16.30, 6 16.61, all-after 16.15); pre_post3 4096: 10.10 -> 9.79 per layer (PE floor ~9.5) |
| `_qkv4`: QKV projections in head groups of gn_of(TB)/2 heads; the q/k rmsnorm + rope of group u is issued after the projections of group u+2 (crossing into the next token block), k/v cache writes per group | pre_qkv3 (T>=2048), `_qkva` (1024), `_qkva2` for TB=512 | pre_qkv3 4096: 2.056 -> 1.976 (GN 2 lag 1 2.018, lag 2 1.976, lag 3 1.997, lag 5 2.026; GN 4 lag 1 2.064, lag 4 2.13); pre_layers 1024: 16.14 -> 15.92; pre_layers2 512: 8.34 -> 8.05; 256: no gain (4.28-4.43), not used there |
| decode QKV weight slabs of 2 k-tiles per DMA (12 KB pieces per partition row pair) and down slabs of 2 rows (DQ_KG = DD_KG = 2) | `_layer`, `_mlp_col` | 12-layer NEFF at S=512: 7.42 -> 7.383 (QKV) -> 7.374 (+down); QKV 4 k-tiles 7.390; o-proj 2 rows 7.403 (not used) |
| decode MLP: gate and up slabs of a k-tile DMAed into ONE buffer [128, 2, 6144] and accumulated into ONE PSUM tile [128, 2, 48] with the matmuls alternating gate/up per output tile (GU_FUSE): one accumulation chain, so the scheduler no longer runs gate kt..kt+4 then up kt..kt+4 (which pinned the up slabs and stalled the weight DMA) | `_mlp_col` | 12-layer NEFF S=512, 3 profiles each: 7.374 / 7.363 / 7.358 -> 7.308 / 7.221 / 7.266 (-1.3%); DMA average over 4 layers 316 -> 322 GB/s per core |
| host: stdin read non-blocking with a 10 s spin after each request before blocking (`read_line`; ENGINE_NOSPIN=1 disables) | src/engine.cpp | A/B vs the same NEFFs, both core orders, 3 rounds: prefill 256 / 1024 / 4096 ~+0.2 / +1.1 / +0.25%, decode within noise |
| decode attention buckets 384 / 1152 / 4224 added (just above P + 63 for P = 256 / 1024 / 4096; key chunks of 512 with a shorter last one) | layer_dec, build_neffs DEC_S | A/B v3 vs v2, both core orders, 3 rounds each: decode 256 / 1024 / 4096 ~1.000 / 1.007 / 1.003 |

End to end (drive.py, core 34): prefill 256 / 1024 / 4096 33.2 / 123.5 / 526.0 -> 32.7 / 115.6 / 513 ms; decode
23.4-24.5 ms per token unchanged (but see the buckets).

Rejected or neutral (measured):
- Decode weight ring (`layer_dec2`, deleted): all weight slabs of the 12-layer step through one explicit ring of
  NBUF 12 KB buffers in consumption order (QKV, o-proj, gate/up, down, next layer), DMAs issued NBUF-1 fills ahead in
  program order: 7.42 -> 7.47 ms (NBUF 9), 7.42 (12). Sync and Scalar HWDGE queues alternating: 7.51. Decode is at
  the DMA ceiling: every one of the 16 DMA engines of a core moves 12 KB descriptors back to back (460 ns each =
  26.7 GB/s minimum, many slower under contention), ~330 GB/s per physical core sustained. The remaining ~5% are
  dips at phase ends caused by the scheduler's matmul order (it runs gate kt..kt+4 then up kt..kt+4, so the ring
  buffers of the up slabs stay pinned), not fixable from program order.
- PE-only GEMV rate (dev/pegemv.py, weights resident in SBUF): weight-stationary 36 ns per 128x128 tile (910 GB/s
  per core): the PE is not the decode limit.
- One extra tiny NEFF in the decode chain (4 extra argmax executions per step): +20-40 us per execution including
  its ~26 us of device time, so NEFF boundaries cost little; fusing the LM head/argmax into the last layer NEFF was
  not pursued (estimated < 0.2%).
- T=1024: block b-1's post issued between block b's QKV and attention (pre_layers_o, deleted): 17.07 -> 18.79 (no
  overlap: each engine runs its queue in program order, and the post's working set left the weight pool too small:
  gate/up streamed at a steady 84 GB/s).
- Softmax normalization on GpSimd: 17.07 -> 20.81 (GpSimd ~4x slower than DVE on fp32 -> bf16).
- `_post4` (T=256): o-proj swaps consumed 0 / 1 / 2 groups later (with the sums of squares accumulated per group):
  4.36 / 4.34 / 4.35 vs 4.27-4.30 after the loop; after-loop order fin+squares per group: 4.34.
- GpSimd DMA (`dma_engine=gpsimd_dma`) for the prefill swaps: refused by the compiler (at most 1 KB per partition).
- Batched rmsnorm for TB=256 (`_rms_fast`: squares in 4 instructions, one strided tensor_reduce over the 32 tiles,
  broadcast multiplies): 4.48 (the strided reduce runs at 1.7 ns per element; the vector engine is ~1 element/ns
  per partition with small per-instruction cost, so batching does not shorten the chains).
- Engine assignment at T=256 (gamma multiply on ACT, _fin alternating DVE/GpSimd, squares' adds on GpSimd,
  attention copies on ACT): 4.27-4.36, no gain.
- T=1024: the next layer's block-0 rmsnorm issued inside the last block's post (before down / at its start / after
  2 gate/up groups / its sums of squares spread over the down pairs, with a core_barrier after block 0):
  15.92 -> 16.03 / 17.57 / 16.19 / 16.68 (code removed).
- T=1024: block 0's attention spread over block 1's QKV head-group units (ATT_OVL=1, `_att_step`): the attention does
  overlap the projections (PE ~210% busy there) but the layer time is unchanged (15.917 vs 15.92); off.
- Decode: the next layer's first 4 / 8 QKV slabs prefetched among the down slabs: 7.387 -> 7.43 / 8.09 (removed).
- pre_qkv3 4096: two / four head groups per weight DMA (larger pieces): 1.976 -> 2.19 / 2.69 (removed); x * rstd of
  `_qkv3_norm` on GpSimd for odd tiles 3.09, its adds on GpSimd 2.15, gamma on ACT 1.977; projection PSUM drains
  forced to ACT / DVE 1.960 / 1.952 (all neutral or worse). Remaining PE gaps there (~260 us of 1.96 ms) are the
  in-order PE waiting for the lagged q/k post (`ssq` matmuls) and for weight slabs.
- Decode weight ring again, with GU_FUSE (`layer_dec2`, deleted again): NBUF 9 / 12 / 6 x 12 KB: 7.38 / 7.44 / 7.34,
  5 x 24 KB 7.30, vs 7.26 without the ring. Decode: the first 1 / 2 / 3 fused gate/up slabs queued before the
  o-proj slabs (to stream during the attention, GU_PF): 7.27 / 7.35 / 7.33 vs 7.26 (off). With GU_FUSE: o-proj 2
  rows per DMA 7.24-7.29 (noise level), QKV 1 or 4 k-tiles per DMA 7.25-7.33 (kept 2).
- T=256: attention of q group i overlapped with the projections of later groups (k group first, ATT_QOVL=1,
  `_qkva2_ovl`): the QKV + attention phase shortens 180 -> 150 us but the o-proj then waits ~20 us for its last
  32 KB slab (pool of 2): 4.30 -> 4.35; with o-proj groups of 4 / 2 tiles (PO_GN): 4.33 / 4.33; PO_GN=4 alone 4.38.
- dev-eval 003 (GU_FUSE build, 2 rounds): round 1 of the candidate had a 404.9 ms prefill at 256 (normally 32.8)
  and +6% / +7% on prefill / decode 4096; round 2 was normal (score of round 2 alone ~1.544, decode 256 / 1024
  1466 / 1486 ms vs 1507-1527 / 1518-1525 in dev-eval 002). Three fresh drive.py processes right after: first
  prefill 32.6-33.0 ms every time; taken as an external disturbance (other runs share the machine).
- Decode down slabs of 3 / 4 / 6 rows with GU_FUSE: 7.247 / 7.357 / 7.245 vs 7.305 (2 rows): noise level, kept 2.
  Remaining decode loss vs a steady 340 GB/s per core (GU_FUSE profile, 12 layers 7.26 ms): ~576 us, of it gate/up
  173 (DMA-bound at ~325 GB/s), down 179 (three ~3 us dips per layer, the Sync queue waiting for the 16-deep
  DMA slot semaphores), attention 91, QKV 73, o-proj 42.
- Noise: decode A/B pairs of the same two builds differed by up to 3% between runs (other agents load the machine),
  and the "core 35 is 0.8% slower" bias flipped sign; average both core orders and treat <1% decode deltas as noise.
- attention_cte at 4096 (not changed): ACT 93% (2304 exp instructions of 512 elements, each followed by a 281 ns
  ACTIVATION_READ_ACCUMULATOR = 648 us), DVE 93%, PE 88%, GpSimd 69%, throttle limit 0.76: every engine is busy.

## Next steps (by expected value)

1. Prefill 256 (largest relative headroom, ~854 us per layer against a ~685 us PE floor): the three serial chains per
   layer (QKV -> q/k norm/rope -> attention -> o-proj: ~80 us; o-proj -> swap -> h -> squares -> rstd -> hn: ~45 us;
   down -> swap -> next rmsnorm: ~45 us) are vector/scalar latency chains. Batching instructions, other engines and
   lagged consumption did not shorten them (Session 003 list). Untested: attention of head group i overlapped with
   the q projections of group i+1 with the pipelined stages (session 002 tried it without pipelining: worse).
2. Prefill 4096 attention (attention_cte, 2.575 ms per layer, 18% of the layer): all engines saturated and
   throttled; fewer ACT instructions (2048-element exps, one accumulator read per large tile) would change the sum
   order (not bitwise) and DVE is equally busy. Own attention (pipelined stages) estimated no faster.
3. Prefill 1024: ~3.19 ms per layer against a 2.84 ms PE floor: attention of block 1 (~150 us, ACT/DVE bound),
   o-proj -> MLP gaps (~50 us per block), layer transition (~90 us).
4. Decode is within ~5% of the DMA ceiling (dips at phase ends from the scheduler's matmul order); NEFF boundaries
   cost little (20-40 us per extra tiny NEFF).

## Next steps of session 002 (kept for reference; 1 and 2 were done in session 003: `_post5_blk`, `_qkv4`)

1. p1024: the attention (~450 us per layer, ~350 of it beyond its PE work) and three ~50 us gaps per layer; the
   resident structure lost at two blocks (SBUF pressure?), so a hybrid (X in HBM, sendrecv swaps) is untested.
2. qkv3 at 4096: overlap block b's q/k norm + rope (DVE-bound, ~100 us) with block b+1's projections (~150 us of PE);
   0.6-0.85 ms per layer (4-6% of p4096) if it overlapped.
3. Decode: ~20 us per layer of DMA idle (o-proj slabs fill the weight pool during attention; layer start).
4. Session 001's list: LM head + exact argmax folded into the last decode NEFF (~0.1 ms/token); prefill 4096 flash
   attention faster than attention_cte (2.6 ms/layer, PE floor ~1.4).
