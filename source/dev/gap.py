"""dev only: list instructions (engine, source line, opcode) in a time window of a neuron-explorer json, grouped.
usage: gap.py ntff.json T0_US T1_US"""
import json, sys, collections
d = json.load(open(sys.argv[1])); a, b = float(sys.argv[2]), float(sys.argv[3])
ins = [i for i in d['instruction'] if i.get('fully_qualified_subgraph') == '/sg00']
dm = [x for x in d['dma'] if x.get('subgraph') == 'sg00']
t0 = min(x['timestamp'] for x in dm)
g = collections.OrderedDict()
for i in sorted(ins, key=lambda i: i['timestamp']):
    t = (i['timestamp'] - t0) / 1e3
    if a <= t < b:
        k = (i['subgroup'], i.get('nki_source_location', '?').split(':')[-1], i['opcode'])
        if k not in g: g[k] = [t, t, 0, 0.0]
        g[k][1] = t + i['duration'] / 1e3; g[k][2] += 1; g[k][3] += i['duration'] / 1e3
for k, v in g.items():
    print(f"{v[0]:8.1f}-{v[1]:8.1f} {k[0]:7s} L{k[1]:5s} {k[2]:22s} n={v[2]:4d} busy={v[3]:.1f}us")
