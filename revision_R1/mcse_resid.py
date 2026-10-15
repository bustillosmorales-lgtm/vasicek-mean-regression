import re,sys,math,json
M=5000
num=r'-?\.?\d[\d.]*|-\.\d+'
def isnum(t):
    try: float(t); return True
    except: return False
out={}
for part in ('I','II','III'):
    # argument: the folder with the archived SAS output of Parts I to III, which
    # accompanies the submission (resultados_parte-I.txt, -II.txt, -III.txt)
    L=open(sys.argv[1]+f'/resultados_parte-{part}.txt',encoding='latin-1').read().splitlines()
    first=[];second=[]
    inb=False
    for l in L:
        tk=l.split()
        if 'rmse_mean' in l or (part=='III' and 'variance' in tk and 'skewness' in tk and 'kurtosis' in tk and 'n' in tk) or (part=='II' and 'rmse_' in l):
            inb=True; continue
        if not tk: continue
        if not all(isnum(t) for t in tk) and 'cox_snell' not in tk and 'quantile' not in tk:
            if not any(h in l for h in ('variance','skewness','bias_')): inb=False
            continue
        if ('cox_snell' in tk or 'quantile' in tk):
            first.append(tk)
        elif inb and len(tk) in (7,8) and all(isnum(t) for t in tk):
            v=[float(t) for t in tk]
            if part=='I' and len(v)==8: second.append(v[1:])
            elif part=='III' and len(v)==8: second.append(v[:7]+[v[7]])
            elif part=='II' and len(v)==7: second.append(v)
    print(part,len(first),len(second))
    res={}
    for a,b in zip(first,second):
        r='CS' if 'cox_snell' in a else 'Q'
        nums=[float(t) for t in a if isnum(t)]
        # last 5 numbers: emp_mean var skew kurt bias_mean
        emp=nums[-5:]
        bias_mean=emp[4]; bv,bs,bk,rm,rv,rs,rk=b[:7]
        sds=[math.sqrt(max(rm**2-bias_mean**2,0)),math.sqrt(max(rv**2-bv**2,0)),
             math.sqrt(max(rs**2-bs**2,0)),math.sqrt(max(rk**2-bk**2,0))]
        se=[s/math.sqrt(M) for s in sds]
        n=int(b[7]) if part=='III' else int(nums[-6])
        res.setdefault(r,[]).append((n,se))
    for r,v in res.items():
        mx=[max(x[1][k] for x in v) for k in range(4)]
        mx200=[max(x[1][k] for x in v if x[0]==200) for k in range(4)]
        out[f'{part}-{r}']=dict(max=mx,max_n200=mx200,cells=len(v))
        print(part,r,len(v),'max SE mean/var/skew/EK',['%.4f'%m for m in mx],' n=200:',['%.4f'%m for m in mx200])
json.dump(out,open('mcse_resid.json','w'),indent=1)
