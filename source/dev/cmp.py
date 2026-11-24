import json, sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
for L in a:
    for i,(x,y) in enumerate(zip(a[L],b[L])):
        d=[k for k in range(len(x)) if x[k]!=y[k]]
        print(L,i,'identical' if not d else f'first diff at {d[0]} of {len(x)} ({len(d)} differ)')
