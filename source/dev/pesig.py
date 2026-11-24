"""dev only: PE instructions that signal a semaphore, and Sync DMA triggers, in a window. usage: pesig.py json T0 T1 [LINE]"""
import json, sys, re
d = json.load(open(sys.argv[1])); a, b = float(sys.argv[2]), float(sys.argv[3])
dm = [x for x in d['dma'] if x.get('subgraph') == 'sg00']
t0 = min(x['timestamp'] for x in dm)
ins = [i for i in d['instruction'] if i.get('fully_qualified_subgraph') == '/sg00']
for i in sorted(ins, key=lambda i: i['timestamp']):
    t = (i['timestamp'] - t0) / 1e3
    if not (a < t < b): continue
    if i['subgroup'] == 'Tensor' and 'Tensor)++' in i['operands']:
        m = re.search(r'dst=(0x\w+)', i['operands'])
        print(f"{t:8.1f} PE  {i['opcode']:8s} L{i.get('nki_source_location','?').split(':')[-1]} acc={re.search(r'acc_flags=(\d)', i['operands']).group(1) if 'acc_flags' in i['operands'] else '-'} dst={m.group(1) if m else ''}")
    if i['subgroup'] == 'Sync' and i['opcode'] == 'DMA_DIRECT2D':
        print(f"{t:8.1f} DMA {i['duration']/1e3:5.1f} L{i.get('nki_source_location','?').split(':')[-1]} {i['operands'].split(' semaphore')[0]}")
