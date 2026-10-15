"""Applies the six-model comparison to the OECD data (as a check against the
article) and to the bounded-response data sets distributed with the R package
betareg (Cribari-Neto and Zeileis, 2010; Grun, Kosmidis and Zeileis, 2012),
each with the mean structure used in the betareg documentation and a constant
dispersion parameter, so that all six models carry the same number of parameters."""
import csv, json, sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import six_models as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', '')   # filled by get_data.py


def read(name):
    with open(DATA + name + '.csv', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def col(rows, c):
    return np.array([float(r[c]) for r in rows])


def dummies(rows, c):
    lev = sorted(set(r[c] for r in rows))
    return np.column_stack([[1.0 if r[c] == l else 0.0 for r in rows] for l in lev[1:]]), lev


sets = {}

r = list(csv.DictReader(open(os.path.join(HERE, '..', 'oecd', 'oecd_bli_2017.csv'), encoding='utf-8')))
y = col(r, 'educ')
X = np.column_stack([np.ones(len(y)), col(r, 'homicide'), col(r, 'dwellings'), col(r, 'labour_insec')])
sets['OECD educational attainment'] = (y, X, 'educ ~ HR + DWBF + LMI')

r = read('GasolineYield')
D, _ = dummies(r, 'batch')
sets['GasolineYield'] = (col(r, 'yield'), np.column_stack([np.ones(len(r)), D, col(r, 'temp')]), 'yield ~ batch + temp')

r = read('FoodExpenditure')
sets['FoodExpenditure'] = (col(r, 'food') / col(r, 'income'),
                           np.column_stack([np.ones(len(r)), col(r, 'income'), col(r, 'persons')]),
                           'food/income ~ income + persons')

r = read('ReadingSkills')
dys = np.array([1.0 if x['dyslexia'] == 'yes' else -1.0 for x in r])
iq = col(r, 'iq')
sets['ReadingSkills'] = (col(r, 'accuracy'), np.column_stack([np.ones(len(r)), dys, iq, dys * iq]),
                         'accuracy ~ dyslexia * iq')

r = read('StressAnxiety')
sets['StressAnxiety'] = (col(r, 'anxiety'), np.column_stack([np.ones(len(r)), col(r, 'stress')]), 'anxiety ~ stress')

r = read('MockJurors')
ver = np.array([1.0 if x['verdict'] == 'three-option' else -1.0 for x in r])
con = np.array([1.0 if x['conflict'] == 'yes' else -1.0 for x in r])
sets['MockJurors'] = (col(r, 'confidence'), np.column_stack([np.ones(len(r)), ver, con, ver * con]),
                      'confidence ~ verdict * conflict')

r = read('WeatherTask')
pr = np.array([1.0 if x['priming'] == 'two-fold' else 0.0 for x in r])
el = np.array([1.0 if x['eliciting'] == 'precise' else 0.0 for x in r])
sets['WeatherTask'] = (col(r, 'agreement'), np.column_stack([np.ones(len(r)), pr, el]), 'agreement ~ priming + eliciting')

r = read('ImpreciseTask')
tk = dummies(r, 'task')[0][:, 0]
df = col(r, 'difference')
sets['ImpreciseTask'] = (col(r, 'location'), np.column_stack([np.ones(len(r)), tk, df, tk * df]),
                         'location ~ task * difference')

r = read('CarTask')
tk = dummies(r, 'task')[0][:, 0]
sets['CarTask'] = (col(r, 'probability'), np.column_stack([np.ones(len(r)), tk, col(r, 'NFCCscale')]),
                   'probability ~ task + NFCCscale')

only = sys.argv[1:] or list(sets)
out = {}
for nm in only:
    y, X, f = sets[nm]
    assert np.all((y > 0) & (y < 1)), nm
    res = S.compare(y, X)
    res['formula'] = f
    res['y_range'] = [float(y.min()), float(y.max())]
    out[nm] = res
    v = res['vuong']
    print('%-28s n=%3d k=%2d  rank: %s' % (nm, res['n'], res['k'], ' > '.join(res['rank'])))
    print('    Vasicek AIC pos %d, weight %.4f, dAIC to best %.3f, theta %.3f; Vuong vs beta z=%+.2f p=%.4f' % (
        res['rank'].index('Vasicek') + 1, res['weight']['Vasicek'],
        res['aic']['Vasicek'] - min(res['aic'].values()), res['theta_vas'], *v['Beta']), flush=True)
json.dump(out, open('datasets_%s.json' % ('all' if not sys.argv[1:] else '_'.join(only)), 'w'), indent=1)
