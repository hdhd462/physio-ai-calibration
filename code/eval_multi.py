# -*- coding: utf-8 -*-
"""Multi-outcome calibration evaluation utilities (K classes).

Conventions match the manuscript:
- ECE = occupancy-weighted 10 equal-width bins
- calibration slope/intercept from logistic regression on logit(p)
- Platt: logit(p') = a*logit(p) + b
- Beta: logit(p') = a*ln(p) + b*ln(1-p) + c (beta calibration form)
- Repairs evaluated with 10-fold cross-fitting (fit on 9 folds, predict held-out fold)
"""
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

EPS = 1e-6

def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))

def ece(y, p, bins=10):
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return float(e)

def cal_slope_intercept(y, p):
    lp = logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
    lr.fit(lp, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])

def metrics(y, p):
    y = np.asarray(y, int); p = np.asarray(p, float)
    a = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    slope, itc = cal_slope_intercept(y, p)
    return {
        'n': int(len(y)), 'events': int(y.sum()), 'prev': float(y.mean()),
        'auc': float(a), 'brier': float(brier_score_loss(y, p)),
        'ece': ece(y, p), 'slope': slope, 'intercept': itc,
    }

def fit_platt(y, p):
    lp = logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
    lr.fit(lp, y)
    a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
    def apply(q):
        return 1 / (1 + np.exp(-(a * logit(q) + b)))
    return apply, (a, b)

def fit_beta(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    X = np.column_stack([np.log(p), np.log(1 - p)])
    lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
    lr.fit(X, y)
    a, b = lr.coef_[0]; c = float(lr.intercept_[0])
    def apply(q):
        q = np.clip(q, EPS, 1 - EPS)
        z = a * np.log(q) + b * np.log(1 - q) + c
        return 1 / (1 + np.exp(-z))
    return apply, (float(a), float(b), c)

def crossfit_repair(y, p, method='platt', folds=10, seed=42):
    """10-fold cross-fitted recalibration: fit on 9/10, predict held-out 1/10."""
    y = np.asarray(y, int); p = np.asarray(p, float)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(y))
    parts = np.array_split(order, folds)
    out = np.empty(len(y))
    for i in range(folds):
        te = parts[i]
        tr = np.concatenate([parts[j] for j in range(folds) if j != i])
        if len(np.unique(y[tr])) < 2:
            out[te] = p[te]
            continue
        fn, _ = (fit_platt if method == 'platt' else fit_beta)(y[tr], p[tr])
        out[te] = fn(p[te])
    return out

def eval_all(y, p, label=''):
    m_raw = metrics(y, p)
    p_platt = crossfit_repair(y, p, 'platt')
    p_beta = crossfit_repair(y, p, 'beta')
    m_platt = metrics(y, p_platt)
    m_beta = metrics(y, p_beta)
    print(f'[{label}] n={m_raw["n"]} ev={m_raw["events"]} prev={m_raw["prev"]*100:.2f}%')
    for name, m in [('raw', m_raw), ('platt', m_platt), ('beta', m_beta)]:
        print(f'  {name:6s} AUC={m["auc"]:.4f} Brier={m["brier"]:.4f} ECE={m["ece"]:.4f} '
              f'slope={m["slope"]:.3f} int={m["intercept"]:.3f}')
    return {'raw': m_raw, 'platt': m_platt, 'beta': m_beta}, p_platt, p_beta
