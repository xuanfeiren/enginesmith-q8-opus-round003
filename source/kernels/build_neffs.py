"""Build time only: trace the NKI kernels of qwen3.py for every bucket and compile them to NEFFs (no device needed).

usage: python build_neffs.py OUT_DIR CACHE_DIR [JOBS]

Each NEFF is compiled in its own Python process through NKI's documented standalone route (see compile_one).

Each NEFF is cached in CACHE_DIR under a key of (kernel sources, name, argument shapes, toolchain versions), so a
rebuild recompiles only what changed. OUT_DIR receives <name>.neff for every entry of neff_specs().
"""
import ast
import hashlib
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

# model and bucket constants (the C++ host has the same ones in src/model.h)
H, NQ, NKV, D, I, V = 4096, 32, 8, 128, 12288, 151936
SMAX = 4608                      # KV cache positions (>= 4097 + 64)
VC = 149                         # LM head chunks of 512 columns per core: 2 * 149 * 512 = 152576 >= V
PRE_T = [128, 256, 512, 1024, 2048, 4096, 4608]   # prefill buckets
PRE_SMALL = PRE_T                                 # buckets on the own prefill kernels (all of them now)
DEC_S = [384, 512, 1024, 1152, 1536, 2048, 3072, 4224, 4608]   # decode attention buckets (multiples of 128;
#   384 / 1152 / 4224 just above the timed decode positions P .. P + 63 of P = 256 / 1024 / 4096)
LNC = 2
NLD = 12                          # decode layers per NEFF (src/engine.cpp NLD)
NLP = 5                           # prefill layers per NEFF for T <= 512 (src/engine.cpp NLP); 7 x 5 = layers 0..34
def nch(T):
    """prefill MLP chunks for bucket T (src/engine.cpp nch())"""
    return 3 if T <= 512 else 6


def neff_specs():
    """name -> (kernel name, [(arg name, shape, dtype)]). Arg names must match the kernel's parameters."""
    bf, f32, i32, u32, u8 = "bfloat16", "float32", "int32", "uint32", "uint8"
    specs = {}
    specs["embed_dec"] = ("embed_dec", [("tok", (1, 1), i32), ("pos", (1, 1), i32), ("emb", (V, H), bf),
                                        ("cos_tab", (SMAX, D // 2), bf), ("sin_tab", (SMAX, D // 2), bf)])
    for S in DEC_S:
        specs[f"layer_dec_{S}"] = ("layer_dec", [
            ("x", (1, 1, H), bf), ("g1c", (NLD, 128, H // 128), bf), ("wqkv", (NLD, H, (NQ + 2 * NKV) * D), bf),
            ("qn", (NLD, D, 1), bf), ("kn", (NLD, D, 1), bf), ("rcos", (SMAX, D), bf), ("rsin", (SMAX, D), bf),
            ("rot", (D, D), bf), ("kc", (NLD, NKV, D, SMAX), bf), ("vc", (NLD, NKV, SMAX, D), bf), ("pos", (1, 1), i32),
            ("posf", (1, 1), f32), ("wo", (NLD, NQ * D, H), bf), ("g2c", (NLD, 128, H // 128), bf),
            ("wg", (NLD, H, I), bf), ("wu", (NLD, H, I), bf), ("wd", (NLD, I, H), bf), ("sctx", (1, S), u8)])
    for S in DEC_S:   # one-layer variant: the last layer of a prefill, for the last prompt token only
        specs[f"layer_dec1_{S}"] = ("layer_dec", [
            ("x", (1, 1, H), bf), ("g1c", (1, 128, H // 128), bf), ("wqkv", (1, H, (NQ + 2 * NKV) * D), bf),
            ("qn", (1, D, 1), bf), ("kn", (1, D, 1), bf), ("rcos", (SMAX, D), bf), ("rsin", (SMAX, D), bf),
            ("rot", (D, D), bf), ("kc", (1, NKV, D, SMAX), bf), ("vc", (1, NKV, SMAX, D), bf), ("pos", (1, 1), i32),
            ("posf", (1, 1), f32), ("wo", (1, NQ * D, H), bf), ("g2c", (1, 128, H // 128), bf),
            ("wg", (1, H, I), bf), ("wu", (1, H, I), bf), ("wd", (1, I, H), bf), ("sctx", (1, S), u8)])
    specs["lm_head"] = ("lm_head", [("x", (1, 1, H), bf), ("gf", (1, H), bf), ("wlm", (2, VC, 128, H // 128, 512), bf)])
    specs["argmax"] = ("argmax", [("input_tensor", (1, V), f32)])
    for T in PRE_T:
        TB = min(T, 512)
        XS = (T // TB, 128, H // 128, TB)
        specs[f"embed_x_{T}"] = ("embed_x", [("toks", (T, 1), i32), ("emb", (V, H), bf)])
        specs[f"x_row_{T}"] = ("x_row", [("X", XS, bf), ("off", (1, 1), i32)])
        def stk(n):
            return [("g1c", (n, 128, H // 128), bf), ("wqkv", (n, H, (NQ + 2 * NKV) * D), bf), ("qn", (n, D, 1), bf),
                    ("kn", (n, D, 1), bf), ("rcT", (D, SMAX), bf), ("rsT", (D, SMAX), bf), ("rot", (D, D), bf),
                    ("kc", (n, NKV, D, SMAX), bf), ("vc", (n, NKV, SMAX, D), bf)]
        post = [("wo", (1, NQ * D, H), bf), ("g2c", (1, 128, H // 128), bf), ("wg", (1, H, I), bf), ("wu", (1, H, I), bf),
                ("wd", (1, I, H), bf)]
        specs[f"pre_qkv3_{T}"] = ("pre_qkv3", [("X", XS, bf)] + stk(1))
        if T <= 1024:  # several layers per NEFF, attention inside (src/engine.cpp: qkva_T)
            # T <= 512 (one token block): the SBUF-resident variant (pre_layers2); 1024: pre_layers (faster there)
            specs[f"pre_layers_{T}"] = ("pre_layers2" if T <= 512 else "pre_layers", [("X", XS, bf)] + stk(NLP) +
                                        [(n, (NLP,) + sh[1:], dt) for n, sh, dt in post])
        else:
            specs[f"pre_attn2_{T}"] = ("pre_attn2", [("q", (NQ, D, T), bf), ("k", (NKV, D, T), bf), ("v", (NKV, T, D), bf)])
            specs[f"pre_post3_{T}"] = ("pre_post3", [("X", XS, bf), ("a", (NQ, D, T), bf)] + post)
    return specs
    # (earlier prefill kernels, no longer built)
    for T in PRE_T:
        specs[f"embed_pre_{T}"] = ("embed_pre", [("toks", (T, 1), i32), ("emb", (V, H), bf)])
        if T in PRE_SMALL:
            continue
        specs[f"pre_qkv_{T}"] = ("pre_qkv", [("x", (1, T, H), bf), ("g1", (1, H), bf),
                                             ("wqkv", (H, (NQ + 2 * NKV) * D), bf), ("qn", (1, D), bf),
                                             ("kn", (1, D), bf), ("cos", (1, T, D), bf), ("sin", (1, T, D), bf)])
        qs, ks = (NQ // 2, 1, T, D), (NKV // 2, 1, T, D)
        specs[f"pre_attn_{T}"] = ("pre_attn", [("q0", qs, bf), ("q1", qs, bf), ("k0", ks, bf), ("k1", ks, bf),
                                               ("v0", ks, bf), ("v1", ks, bf)])
        specs[f"pre_cache_{T}"] = ("pre_cache", [("k0", ks, bf), ("k1", ks, bf), ("v0", ks, bf), ("v1", ks, bf),
                                                 ("kc", (NKV, D, SMAX), bf), ("vc", (NKV, SMAX, D), bf)])
        specs[f"pre_oproj_{T}"] = ("pre_oproj", [("x", (1, T, H), bf), ("a0", (NQ // 2, D, T), bf),
                                                 ("a1", (NQ // 2, D, T), bf), ("wo", (NQ * D, H), bf)])
        for c in range(nch(T)):
            specs[f"pre_mlp{c}_{T}"] = ("pre_mlp", [("h", (1, T, H), bf), ("acc", (1, T, H), bf), ("g2", (1, H), bf),
                                                    ("wg", (H, I), bf), ("wu", (H, I), bf), ("wd", (I, H), bf),
                                                    ("chunk", (nch(T), c + 1), u8)])
    return specs


def source_digest():
    h = hashlib.sha256()
    for root, dirs, files in sorted(os.walk(HERE)):
        dirs.sort()
        for f in sorted(files):
            if f.endswith(".py") and f != "build_neffs.py":
                p = os.path.join(root, f)
                h.update(os.path.relpath(p, HERE).encode())
                h.update(open(p, "rb").read())
    import nki
    h.update(nki.__version__.encode())
    try:
        import importlib.metadata as md
        h.update(md.version("neuronx-cc").encode())
    except Exception:
        pass
    return h.hexdigest()


def compile_one(name, kname, args, out_path, work):
    """Compiles one NEFF in a fresh Python process through NKI's documented route: NKI_ARTIFACTS_DIR is set before
    nki is imported, and calling the @nki.jit kernel with numpy arrays compiles it to <dir>/kernel.neff and then runs
    it once on a NeuronCore (NEURON_RT_VISIBLE_CORES). The build only needs the NEFF: when that run cannot claim a
    core (busy, or another build process holds it), the compiled NEFF is still there and is used."""
    t0 = time.time()
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    env = dict(os.environ, NKI_ARTIFACTS_DIR=work, NEURON_PLATFORM_TARGET_OVERRIDE="trn2")
    log = os.path.join(work, "..", f"{name}.log")
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", kname, repr(args)], env=env,
                           stdout=f, stderr=subprocess.STDOUT, cwd=work)
    neff = os.path.join(work, "kernel.neff")
    if not os.path.exists(neff):
        raise RuntimeError(f"no NEFF (exit {r.returncode}); log {log}:\n" + open(log).read()[-3000:])
    shutil.copy(neff, out_path + ".tmp")
    os.replace(out_path + ".tmp", out_path)
    shutil.rmtree(work, ignore_errors=True)
    return name, time.time() - t0


def one(kname, args):
    """--one mode (inside the fresh process): call the kernel with zero inputs."""
    import numpy as np
    import ml_dtypes
    import qwen3
    dt = {"bfloat16": ml_dtypes.bfloat16}
    kw = {n: np.zeros(shape, dtype=dt.get(dtype, dtype)) for n, shape, dtype in args}
    try:
        getattr(qwen3, kname)[LNC](**kw)
        print("compiled and ran once")
    except Exception as e:  # the run (not the compile) failed if kernel.neff exists; compile_one checks
        print(f"kernel call raised {type(e).__name__}: {str(e)[-2000:]}")


def main():
    stray = [k for k in os.environ if k.startswith("Q8_")]
    if stray:   # dev-only kernel switches (kernels/qwen3.py _sw) are not part of the NEFF cache key
        sys.exit(f"refusing to build with development switches set: {stray}")
    out_dir, cache_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    only = os.environ.get("NEFF_ONLY")  # comma-separated subset, for development
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    digest = source_digest()
    specs = neff_specs()
    todo = []
    for name, (kname, args) in specs.items():
        if only and name not in only.split(","):
            continue
        key = hashlib.sha256(f"{digest}|{name}|{kname}|{args}|lnc{LNC}|nki-standalone".encode()).hexdigest()[:32]
        cached = os.path.join(cache_dir, f"{name}-{key}.neff")
        if not os.path.exists(cached):
            todo.append((name, kname, args, cached))
    print(f"{len(specs)} NEFFs, {len(todo)} to compile", flush=True)
    failed = []
    if todo:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(compile_one, n, k, a, c, os.path.join(cache_dir, "work", n)): n for n, k, a, c in todo}
            for fu in as_completed(futs):
                try:
                    n, dt = fu.result()
                    print(f"  compiled {n} in {dt:.0f}s", flush=True)
                except Exception as e:
                    failed.append(futs[fu])
                    print(f"  FAILED {futs[fu]}: {str(e)[-3000:]}", flush=True)
    if failed:
        sys.exit(f"NEFF compilation failed: {failed}")
    for name, (kname, args) in specs.items():
        if only and name not in only.split(","):
            continue
        key = hashlib.sha256(f"{digest}|{name}|{kname}|{args}|lnc{LNC}|nki-standalone".encode()).hexdigest()[:32]
        shutil.copy(os.path.join(cache_dir, f"{name}-{key}.neff"), os.path.join(out_dir, f"{name}.neff"))
    # drop stale cache entries
    keep = {f"{n}-" for n in specs}
    print("NEFFs ready", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(sys.argv[2], ast.literal_eval(sys.argv[3]))
    else:
        main()
