import math, json, sys
M = 5000
BS = chr(92)
# argument: the LaTeX source of the manuscript as submitted, whose tables of
# bias and RMSE (Parts I to III) are read
tex = open(sys.argv[1], encoding='utf-8').read()


def block(label):
    i = tex.index(BS + 'label{%s}' % label)
    ends = [x for x in (tex.find(BS + 'end{longtable}', i), tex.find(BS + 'end{tabular}', i)) if x > 0]
    j = min(ends)
    rows = []
    for l in tex[i:j].splitlines():
        if '&' not in l or 'multicolumn' in l:
            continue
        c = [x.strip().rstrip(BS).strip() for x in l.split('&')]
        nums = []
        for x in c:
            x = x.replace('$', '').replace(BS + 'phantom{-}', '').strip()
            try:
                nums.append(float(x))
            except ValueError:
                nums.append(None)
        if sum(v is not None for v in nums) >= 6:
            rows.append(nums)
    return rows


out = {}
for lab in ('tab:app_p1_mle', 'tab:app_p2_mle_pos', 'tab:app_p2_mle_neg', 'tab:app_p3_mle'):
    R = block(lab)
    triples = []
    for r in R:
        v = r[2:]
        for k in range(0, len(v) - 2, 3):
            b, rm, cp = v[k], v[k + 1], v[k + 2]
            triples.append((k // 3, r[1], b, rm, cp))
    print(lab, len(R), 'rows')
    for p in sorted(set(t[0] for t in triples)):
        T = [t for t in triples if t[0] == p]
        seb = max(math.sqrt(max(rm * rm - b * b, 0) * M / (M - 1)) / math.sqrt(M) for _, _, b, rm, cp in T)
        sermse = max(rm / math.sqrt(2 * M) for _, _, b, rm, cp in T)
        secp = max(math.sqrt(cp * (1 - cp) / M) for *_, cp in T)
        zmin = min((cp - 0.95) / math.sqrt(0.95 * 0.05 / M) for *_, cp in T)
        zb = max(abs(b) / (math.sqrt(max(rm * rm - b * b, 1e-30)) / math.sqrt(M)) for _, _, b, rm, cp in T)
        print('  param', p, 'maxSE bias %.4f  RMSE %.4f  CP %.4f  min z(CP) %.1f  max|bias|/SE %.1f' % (seb, sermse, secp, zmin, zb))
        out['%s:%d' % (lab, p)] = dict(se_bias=seb, se_rmse=sermse, se_cp=secp, zmin_cp=zmin)
json.dump(out, open('mcse_est.json', 'w'), indent=1)
