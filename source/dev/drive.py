"""dev only: start an engine on the experiment cores and time split requests.
usage: drive.py ENGINE [--lengths 256,1024,4096] [--reps 2] [--prof]"""
import argparse, os, sys, json, time, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../brief/skill/scripts"))
import engine_client as ec
ap = argparse.ArgumentParser()
ap.add_argument("engine")
ap.add_argument("--lengths", default="256,1024,4096")
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--decode", type=int, default=64)
ap.add_argument("--offset", type=int, default=1000)
ap.add_argument("--prof", action="store_true")
ap.add_argument("--out")
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
wrapper = tempfile.mktemp(prefix="wrap-", suffix=".sh")
with open(wrapper, "w") as f:
    f.write(f"#!/bin/bash\nexport NEURON_RT_VISIBLE_CORES={os.environ.get('XCORE','34')}-{int(os.environ.get('XCORE','34'))+1}\n"
            + ("export ENGINE_PROF=1\n" if a.prof else "") + ("export ENGINE_STEPT=1\n" if os.environ.get("STEPT") else "") + f"exec {os.path.abspath(a.engine)} \"$@\"\n")
os.chmod(wrapper, 0o755)
launch = ec.Launch("/home/ubuntu/models/Qwen3-8B", 151936) if hasattr(ec, "Launch") else None
res = {}
with ec.gpu_lock(ec.default_lock_path() if hasattr(ec, "default_lock_path") else "/tmp/engine-gpu.lock"):
    e = ec.Engine(wrapper, launch, "eng")
    print(f"hello {e.hello_s:.1f}s", flush=True)
    try:
        for L in [int(x) for x in a.lengths.split(",")]:
            for r in range(a.reps):
                p = ec.nonce_prompt(L, a.offset + 7919 * r + L, 151936)
                toks, pm, dm = e.split(p, a.decode)
                res.setdefault(L, []).append(toks)
                print(f"L={L} rep={r} prefill {pm:.1f} ms decode {dm:.1f} ms ({dm/a.decode:.2f}/tok)", flush=True)
    finally:
        e.close()
        if a.prof or True:
            print("stderr tail:", e.stderr_path)
os.unlink(wrapper)
if a.out:
    json.dump(res, open(a.out, "w"))
