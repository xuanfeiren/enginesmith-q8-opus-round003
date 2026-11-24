"""dev only: timeline of a neuron-explorer json (ntff.json): HBM bytes per bin by variable, PE busy per bin.
usage: tl.py ntff.json [BIN_US]"""
import json, sys, collections
d = json.load(open(sys.argv[1]))
B = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
dm = [x for x in d['dma'] if x.get('subgraph') == 'sg00']
t0 = min(x['timestamp'] for x in dm)
bins = collections.defaultdict(lambda: collections.Counter())
for x in dm:
    b = int((x['timestamp'] - t0) / 1e3 / B)
    bins[b][x.get('variable', '?')] += x['transfer_size']
pe = collections.Counter(); pel = collections.defaultdict(collections.Counter); act = collections.Counter(); vec = collections.Counter(); pool = collections.Counter()
for i in d['instruction']:
    if i.get('fully_qualified_subgraph') != '/sg00':
        continue
    b = int((i['timestamp'] - t0) / 1e3 / B)
    c = {'Tensor': pe, 'Scalar': act, 'Vector': vec, 'GpSimd': pool}.get(i['subgroup'])
    if c is not None:
        c[b] += i['duration']
    if i['subgroup'] == 'Tensor':
        pel[b][i.get('nki_source_location', '?').split(':')[-1]] += i['duration']
for b in sorted(set(bins) | set(pe)):
    tot = sum(bins[b].values())
    top = ", ".join(f"{k}:{v/1e6:.2f}" for k, v in bins[b].most_common(3))
    print(f"{b*B:7.0f} {tot/B/1e3:6.0f} GB/s  pe {pe[b]/B/10:4.0f}% act {act[b]/B/10:4.0f}% vec {vec[b]/B/10:4.0f}% pool {pool[b]/B/10:4.0f}%  {top}  L{'/'.join(k for k, _ in pel[b].most_common(2))}")
