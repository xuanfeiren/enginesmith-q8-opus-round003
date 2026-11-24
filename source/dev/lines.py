"""dev: engine busy time per source line (one core) over a time window: lines.py ntff.json [T0 T1]"""
import json, sys, collections
d = json.load(open(sys.argv[1]))
a = float(sys.argv[2]) if len(sys.argv) > 2 else 0; b = float(sys.argv[3]) if len(sys.argv) > 3 else 1e12
ins = [i for i in d['instruction'] if i.get('fully_qualified_subgraph') == '/sg00']
dm = [x for x in d['dma'] if x.get('subgraph') == 'sg00']
t0 = min(x['timestamp'] for x in dm)
c = collections.Counter(); n = collections.Counter()
for i in ins:
    t = (i['timestamp'] - t0) / 1e3
    if a <= t < b:
        k = (i['subgroup'], i.get('nki_source_location', '?').split(':')[-1], i['opcode'])
        c[k] += i['duration'] / 1e3; n[k] += 1
tot = collections.Counter()
for k, v in c.items(): tot[k[0]] += v
print({k: round(v) for k, v in tot.items()})
for k, v in c.most_common(40):
    print(f"{k[0]:7s} L{k[1]:5s} {k[2]:26s} n={n[k]:6d} {v:9.1f} us  {v/n[k]*1e3:7.0f} ns/instr")
