"""dev: DMA bytes per 5 us bin over a window, low bins with the PE lines active: dips.py ntff.json T0 T1"""
import json, collections, sys
d = json.load(open(sys.argv[1])); a, b = float(sys.argv[2]), float(sys.argv[3])
dm = [x for x in d['dma'] if x.get('subgraph') == 'sg00']
t0 = min(x['timestamp'] for x in dm)
B = 5.0
bins = collections.Counter()
for x in dm:
    bins[int((x['timestamp'] - t0) / 1e3 / B)] += x['transfer_size']
pel = collections.defaultdict(collections.Counter)
for i in d['instruction']:
    if i['fully_qualified_subgraph'] == '/sg00' and i['subgroup'] == 'Tensor' and i['opcode'] == 'MATMUL':
        pel[int((i['timestamp'] - t0) / 1e3 / B)][i.get('nki_source_location', '?').split(':')[-1]] += 1
tot = 0
for k in range(int(a / B), int(b / B)):
    tot += bins[k]
    g = bins[k] / B / 1e3
    if g < 250: print(k * B, round(g), dict(pel[k]))
print("bytes", tot / 1e6, "MB avg", tot / (b - a) / 1e3, "GB/s")
