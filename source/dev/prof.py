"""dev only: specs for profiling variants (NL layers per NEFF) of the decode layer kernel. Use with kc.py:
  kc.py prof layer_dec dec_2_512 /tmp/q8y/x.neff"""
from qwen3 import *
import build_neffs as B
SPECS = {}
for NL in (1, 2, 4):
    for S in B.DEC_S:
        name, args = B.neff_specs()[f"layer_dec_{S}"]
        args = [(n, ((NL,) + sh[1:]) if sh[0] == B.NLD and len(sh) > 1 else sh, dt) for n, sh, dt in args]
        SPECS[f"dec_{NL}_{S}"] = (name, args)
for T in (128, 256, 512, 1024):
    name, args = B.neff_specs()[f"pre_layers_{T}"]
    SPECS[f"pl2_{T}"] = ("pre_layers2", args)
    SPECS[f"pl_{T}"] = (name, args)
for S in B.DEC_S:
    name, args = B.neff_specs()[f"layer_dec_{S}"]
    SPECS[f"dec_{S}"] = (name, args)
SPECS["post3_4096"] = B.neff_specs()["pre_post3_4096"]
SPECS["post3_2048"] = B.neff_specs()["pre_post3_2048"]
SPECS["qkv3_4096"] = B.neff_specs()["pre_qkv3_4096"]
