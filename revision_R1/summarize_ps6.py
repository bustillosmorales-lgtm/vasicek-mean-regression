"""Summary of the six-model portfolio study (portfolio_six.py shards) next to the
two-model result of portfolio_mechanism.py, as LaTeX rows for Table 14."""
import json, glob, collections
import numpy as np
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'))

CFG = [(m, N) for m in (40, 100) for N in ('500', '2000', '10000', 'inf')]
out = {}
for m, N in CFG:
    rows = [r for f in glob.glob('ps6_%d_%s_*.json' % (m, N)) for r in json.load(open(f))]
    R = len(rows)
    js = sorted(r['j'] for r in rows)
    assert js == list(range(R)), (m, N, 'missing samples')
    win = collections.Counter(r['rank'][0] for r in rows)
    vr = np.array([r['rank'].index('Vasicek') + 1 for r in rows])
    f = win['Vasicek'] / R
    beats = np.mean([r['aic']['Vasicek'] < r['aic']['Beta'] for r in rows])
    two = json.load(open('portfolio_mechanism_%d_%s.json' % (m, N if N != 'inf' else 'inf')))
    two = list(two.values())[0]
    runner = max((k for k in win if k != 'Vasicek'), key=lambda k: win[k])
    out['%d|%s' % (m, N)] = dict(R=R, first=f, se=float(np.sqrt(f * (1 - f) / R)), mean_rank=float(vr.mean()),
                                 se_rank=float(vr.std(ddof=1) / np.sqrt(R)), beats_beta_300=float(beats),
                                 beats_beta_1000=two['frac_vas'], se_1000=two['se'], wins=dict(win),
                                 runner=runner, runner_frac=win[runner] / R)
json.dump(out, open('portfolio_six_summary.json', 'w'), indent=1)
for k, v in out.items():
    m, N = k.split('|')
    Nl = {'500': '500', '2000': '2{,}000', '10000': '10{,}000', 'inf': '$\\infty$'}[N]
    print('%s & %s & %.3f (%.3f) & %.3f (%.3f) & %.2f (%.2f) & %s (%.2f) \\\\' % (
        m, Nl, v['beats_beta_1000'], v['se_1000'], v['first'], v['se'], v['mean_rank'], v['se_rank'],
        v['runner'], v['runner_frac']))
    print('   %% R=%d wins=%s beats_beta_on_these=%.3f' % (v['R'], v['wins'], v['beats_beta_300']))
