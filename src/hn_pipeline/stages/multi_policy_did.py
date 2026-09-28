from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.special import expit
from scipy.stats import chi2, norm, t as student_t
POLICIES = ['CETP', 'LCCP', 'ECERFP', 'ZWC', 'CEP', 'CEPDU', 'CACP', 'SCP', 'LECCDC', 'NGFRIPZ']
OUTCOMES = ['ECI', 'CPI', 'O3', 'PM10', 'JointImprovement']
PERIODS = {'year0': (0,), 'years1_2': (1, 2), 'years3_5': (3, 4, 5)}
PERIOD_CONTRASTS = [('years1_2', 'year0'), ('years3_5', 'year0'), ('years3_5', 'years1_2')]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = PROJECT_ROOT / 'data' / 'processed'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'multi_policy_did'
POLICY_CLUSTERS = {p: 'Code' for p in POLICIES}
RAW_COVARIATES = ['PGDP', 'POPD', 'GIA_GDP', 'Builtup', 'Coal_Share', 'Electricity_Share']
COVARIATES = ['log_PGDP', 'log_POPD', 'GIA_GDP', 'Builtup', 'Coal_Share', 'Electricity_Share', 'log_ECI', 'log_CPI']

def save_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')

def read_csv(path):
    for encoding in ['utf-8-sig', 'gb18030']:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError(f'Cannot decode CSV: {path}')

def bh(pvalues):
    p = np.asarray(pvalues, dtype=float)
    result = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if not len(idx):
        return result
    order = idx[np.argsort(p[idx])]
    adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
    result[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return result

def add_fdr(df, group_keys, pcol='p_value'):
    if df.empty or pcol not in df:
        return df
    df['q_BH'] = np.nan
    for _, ix in df.groupby(group_keys, dropna=False).groups.items():
        df.loc[ix, 'q_BH'] = bh(df.loc[ix, pcol])
    return df

def full_rank_design(raw, names):
    raw = np.asarray(raw, float)
    if not np.isfinite(raw).all():
        raise ValueError('nonfinite_design')
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    keep = scale > 1e-10
    z = (raw[:, keep] - mean[keep]) / scale[keep]
    kept_names = np.asarray(names, object)[keep].tolist()
    if z.shape[1]:
        _, rr, piv = qr(z, mode='economic', pivoting=True)
        rank = int(np.sum(np.abs(np.diag(rr)) > max(z.shape) * 1e-10))
        sel = sorted(piv[:rank])
        z = z[:, sel]
        kept_names = [kept_names[j] for j in sel]
    return (np.column_stack([np.ones(len(raw)), z]), ['Intercept'] + kept_names)

def fit_logit(x, treatment, weights=None):
    n, k = x.shape
    d = np.asarray(treatment, float)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    w = w / w.mean()
    share = np.average(d, weights=w)
    if not 0 < share < 1:
        raise ValueError('propensity_requires_both_treated_and_control')
    beta = np.zeros(k)
    beta[0] = np.log(share / (1 - share))
    for _ in range(200):
        eta = np.clip(x @ beta, -700, 700)
        ps = expit(eta)
        score = x.T @ (w * (d - ps)) / n
        info = x.T * (w * ps * (1 - ps)) @ x / n
        if not np.isfinite(info).all() or np.linalg.cond(info) > 1000000000000.0:
            raise ValueError('propensity_separation_or_ill_conditioned')
        if np.max(np.abs(score)) < 1e-08:
            return (beta, ps, info)
        step = np.linalg.solve(info, score)
        old = np.mean(w * (d * eta - np.logaddexp(0, eta)))
        length = 1.0
        accepted = False
        while length > 1e-12:
            new_beta = beta + length * step
            new_eta = np.clip(x @ new_beta, -700, 700)
            new = np.mean(w * (d * new_eta - np.logaddexp(0, new_eta)))
            if np.isfinite(new) and new >= old - 1e-12:
                beta = new_beta
                accepted = True
                break
            length *= 0.5
        if not accepted:
            raise ValueError('propensity_line_search_failed')
    raise ValueError('propensity_not_converged')

def fit_logit_ridge(x, treatment, weights=None, ridge_lambda=0.0001):
    n, k = x.shape
    d = np.asarray(treatment, float)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    w = w / w.mean()
    share = np.average(d, weights=w)
    if not 0 < share < 1:
        raise ValueError('propensity_requires_both_treated_and_control')
    penalty = np.eye(k)
    penalty[0, 0] = 0.0
    base_lambda = max(float(ridge_lambda), 1e-12)
    last_error = None
    for lam in [base_lambda, base_lambda * 10, base_lambda * 100, base_lambda * 1000, base_lambda * 10000]:
        beta = np.zeros(k)
        beta[0] = np.log(share / (1 - share))
        try:
            for _ in range(300):
                eta = np.clip(x @ beta, -700, 700)
                ps = expit(eta)
                score = x.T @ (w * (d - ps)) / n - lam * (penalty @ beta)
                info = x.T * (w * ps * (1 - ps)) @ x / n + lam * penalty
                step = np.linalg.solve(info, score)
                if np.max(np.abs(score)) < 1e-08:
                    return (beta, ps, info, lam)
                length = 1.0
                old = np.mean(w * (d * eta - np.logaddexp(0, eta))) - 0.5 * lam * beta @ penalty @ beta
                accepted = False
                while length > 1e-12:
                    nb = beta + length * step
                    ne = np.clip(x @ nb, -700, 700)
                    obj = np.mean(w * (d * ne - np.logaddexp(0, ne))) - 0.5 * lam * nb @ penalty @ nb
                    if np.isfinite(obj) and obj >= old - 1e-12:
                        beta = nb
                        accepted = True
                        break
                    length *= 0.5
                if not accepted:
                    break
            eta = np.clip(x @ beta, -700, 700)
            ps = expit(eta)
            info = x.T * (w * ps * (1 - ps)) @ x / n + lam * penalty
            if np.isfinite(ps).all() and np.isfinite(info).all():
                return (beta, ps, info, lam)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
            last_error = exc
    raise ValueError(f'ridge_propensity_failed:{last_error}')

def _dr_from_nuisance(x, d, y, w, beta, p, info, ridge_outcome=0.0, ps_clip=1e-06, method='standard_DR'):
    n = len(d)
    control = 1 - d
    if d.sum() < 1 or control.sum() < 1:
        raise ValueError('requires_treated_and_control')
    gram = x.T * (w * control) @ x / n
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    gram_fit = gram + float(ridge_outcome) * penalty
    if ridge_outcome <= 0:
        if np.linalg.matrix_rank(gram) < x.shape[1] or np.linalg.cond(gram) > 1000000000000.0:
            raise ValueError('outcome_regression_rank_deficient')
    gamma = np.linalg.solve(gram_fit, x.T @ (w * control * y) / n)
    residual = y - x @ gamma
    p_safe = np.clip(p, ps_clip, 1 - ps_clip)
    odds = p_safe / (1 - p_safe)
    denom_t = np.sum(w * d)
    denom_c = np.sum(w * control * odds)
    if denom_t <= 0 or denom_c <= 0 or (not np.isfinite(denom_c)):
        raise ValueError('invalid_DR_normalization')
    a = w * d / denom_t
    c = w * control * odds / denom_c
    mt, mc = (a @ residual, c @ residual)
    att = mt - mc
    direct = d / d.mean() * (residual - mt)
    direct -= control * odds / np.mean(control * odds) * (residual - mc)
    derivative_beta = -(c * (residual - mc)) @ x
    derivative_gamma = -(a - c) @ x
    if_beta = x * (d - p)[:, None] @ np.linalg.pinv(info)
    if_gamma = x * (control * residual)[:, None] @ np.linalg.pinv(gram_fit)
    influence = direct + if_beta @ derivative_beta + if_gamma @ derivative_gamma
    influence -= influence.mean()
    return dict(att=float(att), influence=influence, ps=p, ps_used=np.asarray(p_safe, float), control_weights=c, beta=beta, gamma=gamma, estimation_method=method, ps_clipped_n=int(np.sum((p < ps_clip) | (p > 1 - ps_clip))), ps_extreme_n=int(np.sum((p < 0.01) | (p > 0.99))), ps_extreme_any=bool(np.any((p < 0.01) | (p > 0.99))))

def unadjusted_change_did(treatment, delta):
    d = np.asarray(treatment, float)
    y = np.asarray(delta, float)
    control = 1 - d
    if d.sum() < 1 or control.sum() < 1:
        raise ValueError('requires_treated_and_control')
    mt = float(y[d == 1].mean())
    mc = float(y[d == 0].mean())
    att = mt - mc
    influence = d / d.mean() * (y - mt) - control / control.mean() * (y - mc)
    influence -= influence.mean()
    p = np.full(len(d), d.mean(), dtype=float)
    cw = control / control.sum()
    return dict(att=float(att), influence=influence, ps=p, ps_used=p.copy(), control_weights=cw, beta=np.array([np.nan]), gamma=np.array([np.nan]), estimation_method='unadjusted_change_DID_fallback', ps_clipped_n=0, ps_extreme_n=0, ps_extreme_any=False)

def dr_panel(x, treatment, delta, weights=None, ps_clip=1e-06, ridge_lambda=0.0001):
    n = len(treatment)
    d = np.asarray(treatment, float)
    y = np.asarray(delta, float)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    w = w / w.mean()
    primary_error = ''
    stabilized_error = ''
    try:
        beta, p, info = fit_logit(x, d, w)
        fit = _dr_from_nuisance(x, d, y, w, beta, p, info, ridge_outcome=0.0, ps_clip=ps_clip, method='standard_DR_no_trimming')
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        primary_error = str(exc)
        try:
            beta, p, info, used_lambda = fit_logit_ridge(x, d, w, ridge_lambda=ridge_lambda)
            fit = _dr_from_nuisance(x, d, y, w, beta, p, info, ridge_outcome=max(used_lambda, ridge_lambda), ps_clip=ps_clip, method='ridge_stabilized_DR_fallback')
            fit['ridge_lambda_used'] = float(used_lambda)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc2:
            stabilized_error = str(exc2)
            fit = unadjusted_change_did(d, y)
    fit['primary_fit_error'] = primary_error
    fit['stabilized_fit_error'] = stabilized_error
    if weights is not None:
        return fit['att']
    return fit

@dataclass
class Cell:
    row: dict
    cluster_score: np.ndarray
    treatment_mass_score: np.ndarray
    treated_clusters: set
    control_clusters: set
    treated_units: set
    control_units: set

class Panel:

    def __init__(self, args):
        p, m = (read_csv(args.policy_file), read_csv(args.model_file))
        for label, frame in [('policy', p), ('model', m)]:
            frame.columns = frame.columns.str.strip()
            if not {'Code', 'Year'}.issubset(frame):
                raise ValueError(f'{label}: missing identifying columns')
            if frame[['Code', 'Year']].isna().any().any():
                raise ValueError(f'{label}: missing IDs')
            if frame.duplicated(['Code', 'Year']).any():
                raise ValueError(f'{label}: duplicate county-year keys')
        needed = POLICIES + ['Code', 'Year']
        if not set(needed).issubset(p) or not set(RAW_COVARIATES + OUTCOMES[:4]).issubset(m):
            raise ValueError('Missing policy, outcome, or covariate columns')
        keys = p[['Code', 'Year']].merge(m[['Code', 'Year']], how='outer', indicator=True)
        if not keys['_merge'].eq('both').all():
            raise ValueError('Policy/model county-year keys differ; harmonize explicitly first')
        overlap = [c for c in POLICIES if c in m]
        if overlap:
            raise ValueError(f'Model CSV already has policy columns: {overlap}; avoid ambiguous sources')
        m = m[['Code', 'Year'] + RAW_COVARIATES + OUTCOMES[:4]].copy()
        data = m.merge(p[['Code', 'Year'] + POLICIES], on=['Code', 'Year'], validate='one_to_one').sort_values(['Code', 'Year']).reset_index(drop=True)
        self.years = np.sort(data.Year.unique()).astype(int)
        if not np.array_equal(self.years, np.arange(self.years[0], self.years[-1] + 1)):
            raise ValueError('Calendar years are not consecutive')
        self.ids = np.sort(data.Code.unique())
        if len(data) != len(self.ids) * len(self.years):
            raise ValueError('This implementation requires a balanced panel; missing rows must be audited')
        self.n = len(self.ids)
        for col in POLICIES:
            if not data[col].isin([0, 1]).all():
                raise ValueError(f'{col}: missing or nonbinary policy values')
        self.xpolicy = np.stack([self.pivot(data, p) for p in POLICIES], axis=2).astype(int)
        if (np.diff(self.xpolicy, axis=1) < 0).any():
            raise ValueError('Policy reversal found: absorbing-treatment DID is not appropriate')
        self.g = np.where(self.xpolicy.any(axis=1), self.years[0] + self.xpolicy.argmax(axis=1), np.inf)
        self.audit = {'unit': 'Code', 'inference_cluster': 'Code', 'cross_county_independence': 'user_specified_working_assumption_not_verified', 'coverage_fixed_effects': 'Year_only_no_county_FE', 'rows': len(data), 'counties': self.n, 'years': self.years.tolist(), 'nonpositive_outcomes': {}, 'state_definition': args.state_definition}
        for col in OUTCOMES[:4]:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            bad = ~np.isfinite(data[col]) | (data[col] <= 0)
            self.audit['nonpositive_outcomes'][col] = int(bad.sum())
            data[f'log_{col}'] = np.log(data[col].where(~bad))
            data[f'Y_{col}'] = -data[f'log_{col}']
        for col in ['PGDP', 'POPD']:
            data[f'log_{col}'] = np.log(data[col].where(data[col] > 0))
        if args.state_definition == 'annual':
            dc = data.groupby('Code')['log_ECI'].diff()
            dp = data.groupby('Code')['log_CPI'].diff()
        else:
            if 2013 not in self.years:
                raise ValueError('cumulative2013 requires year 2013')
            baseline = data[data.Year.eq(2013)].set_index('Code')
            dc = data.log_ECI - data.Code.map(baseline.log_ECI)
            dp = data.log_CPI - data.Code.map(baseline.log_CPI)
            dc = dc.where(data.Year.ne(2013))
            dp = dp.where(data.Year.ne(2013))
        data['Y_JointImprovement'] = ((dc < -args.state_epsilon) & (dp < -args.state_epsilon)).astype(float)
        data.loc[dc.isna() | dp.isna(), 'Y_JointImprovement'] = np.nan
        self.data = data
        self.outcomes = {o: self.pivot(data, f'Y_{o}') for o in OUTCOMES}
        self.cov = np.stack([self.pivot(data, c) for c in COVARIATES], axis=2)
        self.audit['state_available_by_year'] = data.groupby('Year').Y_JointImprovement.count().to_dict()
        self.audit['outcome_missing'] = {o: int(np.isnan(y).sum()) for o, y in self.outcomes.items()}
        self.cluster_levels = {}
        self.cluster_codes = {}
        for col in ['Code']:
            values = self.ids
            levels, codes = np.unique(values, return_inverse=True)
            self.cluster_levels[col], self.cluster_codes[col] = (levels, codes)

    def pivot(self, frame, col):
        return frame.pivot(index='Code', columns='Year', values=col).loc[self.ids, self.years].to_numpy(float)

    def year_index(self, year):
        return int(year - self.years[0])

def inference(att, score, multipliers, treated_clusters, control_clusters, args, fixed_cluster_count=None):
    active = np.abs(score) > 1e-14
    k = int(active.sum()) if fixed_cluster_count is None else int(fixed_cluster_count)
    factor = np.sqrt(k / (k - 1)) if k > 1 else 1.0
    centered = score - score.mean()
    se = float(np.sqrt(np.sum(centered ** 2)) * factor)
    boot = multipliers @ centered * factor
    calculable = bool(len(treated_clusters) >= 1 and len(control_clusters) >= 1 and (k > 1) and np.isfinite(se) and (se > 1e-14))
    preferred = bool(len(treated_clusters) >= args.min_inference_clusters and len(control_clusters) >= args.min_inference_clusters and calculable)
    result = {'se': se, 'n_treated_clusters': len(treated_clusters), 'n_control_clusters': len(control_clusters), 'inference_ok': calculable, 'preferred_inference_support': preferred, 'inference_note': 'Code_cluster_multiplier_independent_counties_assumption' if preferred else 'exploratory_few_clusters_but_CI_p_reported' if calculable else 'degenerate_variance_or_single_active_cluster', 'ci_low': np.nan, 'ci_high': np.nan, 'p_value': np.nan}
    if calculable:
        scaled = np.abs(boot / se)
        crit = max(1.0, float(np.quantile(scaled[np.isfinite(scaled)], 0.95)))
        result.update(ci_low=att - crit * se, ci_high=att + crit * se, p_value=(1 + np.sum(np.abs(boot) >= abs(att))) / (len(boot) + 1))
    return (result, boot)

def aggregate(cells, multipliers, args, fixed_cluster_count=None):
    groups = {g: [c for c in cells if c.row['g'] == g] for g in sorted({c.row['g'] for c in cells})}
    group_values = []
    for cohort in groups.values():
        theta = np.mean([c.row['att'] for c in cohort])
        score = np.mean([c.cluster_score for c in cohort], axis=0)
        mass = np.mean([c.row['n_treated'] for c in cohort])
        mass_score = np.mean([c.treatment_mass_score for c in cohort], axis=0)
        group_values.append((theta, score, mass, mass_score))
    total = sum((v[2] for v in group_values))
    att = sum((v[0] * v[2] for v in group_values)) / total
    score = sum((v[2] / total * v[1] + (v[0] - att) / total * v[3] for v in group_values))
    tr = set().union(*(c.treated_clusters for c in cells))
    co = set().union(*(c.control_clusters for c in cells))
    inf, boot = inference(att, score, multipliers, tr, co, args, fixed_cluster_count)
    row = {'att': att, **inf, 'n_cohorts': len(groups), 'n_cells': len(cells), 'n_treated_counties_unique': len(set().union(*(c.treated_units for c in cells))), 'n_control_counties_unique': len(set().union(*(c.control_units for c in cells))), 'min_cell_n_treated': min((c.row['n_treated'] for c in cells)), 'max_cell_n_treated': max((c.row['n_treated'] for c in cells)), 'min_cell_n_control': min((c.row['n_control'] for c in cells)), 'max_cell_n_control': max((c.row['n_control'] for c in cells)), 'weighting': 'equal_post_years_within_g_then_mean_retained_cohort_size'}
    return (row, score, boot)

def add_interpretation(row, outcome):
    if outcome == 'JointImprovement':
        row['effect_unit'] = 'probability; times 100 = percentage points'
        row['effect_readable'] = 100 * row['att']
        row['readable_unit'] = 'percentage_points'
    else:
        row['effect_unit'] = 'negative_log_level'
        row['effect_readable'] = 100 * (1 - np.exp(-row['att']))
        row['readable_unit'] = 'log_scale_implied_percent_reduction_not_mean_level_ATT'
    return row

def lag_contrast(later, earlier, multipliers, treated_clusters, control_clusters, args):
    lr, ls, lb = later
    er, es, eb = earlier
    k = len(treated_clusters | control_clusters)
    factor = k / (k - 1) if k > 1 else 1.0
    covariance = float((ls - ls.mean()) @ (es - es.mean()) * factor)
    difference = lr['att'] - er['att']
    inf, draws = inference(difference, ls - es, multipliers, treated_clusters, control_clusters, args, k)
    p_greater = (1 + np.sum(draws >= difference)) / (len(draws) + 1) if inf['inference_ok'] else np.nan
    return {'att_later': lr['att'], 'att_earlier': er['att'], 'difference': difference, **inf, 'p_greater': p_greater, 'variance_later': lr['se'] ** 2, 'variance_earlier': er['se'] ** 2, 'covariance_earlier_later': covariance, 'variance_difference': inf['se'] ** 2, 'bootstrap_covariance_earlier_later': float(np.cov(eb, lb, ddof=1)[0, 1])}
import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from types import SimpleNamespace
core = SimpleNamespace(POLICIES=POLICIES, OUTCOMES=OUTCOMES, PERIODS=PERIODS, PERIOD_CONTRASTS=PERIOD_CONTRASTS, DEFAULT_DIR=DEFAULT_DIR, COVARIATES=COVARIATES, Panel=Panel, Cell=Cell, save_csv=save_csv, read_csv=read_csv, bh=bh, add_fdr=add_fdr, full_rank_design=full_rank_design, fit_logit=fit_logit, fit_logit_ridge=fit_logit_ridge, dr_panel=dr_panel, inference=inference, aggregate=aggregate, add_interpretation=add_interpretation, lag_contrast=lag_contrast, __file__=__file__)
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
POLICIES = core.POLICIES
OUTCOMES = core.OUTCOMES
PERIODS = core.PERIODS
CONTRASTS = core.PERIOD_CONTRASTS

def names(bits):
    return '+'.join((p for j, p in enumerate(POLICIES) if int(bits) & 1 << j)) or 'NONE'

@dataclass
class Design:
    id: str
    kind: str
    before_n: int
    after_n: int
    event: np.ndarray
    before_bits: int | None = None
    after_bits: int | None = None

    def cluster(self, args):
        return 'Code'

    def info(self):
        return {'design_id': self.id, 'design_type': self.kind, 'from_count': self.before_n, 'to_count': self.after_n, 'added_count': self.after_n - self.before_n, 'transition': f'{self.before_n}->{self.after_n}', 'before_policies': names(self.before_bits) if self.before_bits is not None else 'MIXED_BASELINE_PORTFOLIOS', 'after_policies': names(self.after_bits) if self.after_bits is not None else 'MIXED_ADDITIONS', 'added_policies': names(self.after_bits ^ self.before_bits) if self.before_bits is not None else 'MIXED_ADDITIONS', 'effect_type': 'one_policy_increment' if self.after_n - self.before_n == 1 else 'simultaneous_added_package', 'superadditivity_test': False}

def make_designs(panel):
    counts = panel.xpolicy.sum(axis=2)
    bits = (panel.xpolicy * (2 ** np.arange(len(POLICIES)))[None, None, :]).sum(axis=2)
    panel.counts = counts
    panel.bits = bits
    changed = np.zeros_like(counts, dtype=bool)
    changed[:, 1:] = counts[:, 1:] > counts[:, :-1]
    before = np.roll(bits, 1, axis=1)
    before[:, 0] = -1
    before_n = np.roll(counts, 1, axis=1)
    before_n[:, 0] = -1
    designs = []
    pairs = sorted(set(zip(before_n[changed].tolist(), counts[changed].tolist())))
    for k, l in pairs:
        event = changed & (before_n == k) & (counts == l)
        designs.append(Design(f'count_{k}_to_{l}', 'count_single_add' if l - k == 1 else 'count_package_jump', k, l, event))
    recipes = sorted(set(zip(before[changed].tolist(), bits[changed].tolist())))
    for b, a in recipes:
        k = int(b).bit_count()
        l = int(a).bit_count()
        if l < 2:
            continue
        event = changed & (before == b) & (bits == a)
        designs.append(Design(f'recipe_{b}_to_{a}', 'recipe_single_add' if l - k == 1 else 'recipe_package_jump', k, l, event, int(b), int(a)))
    for design in designs:
        if (design.event.sum(axis=1) > 1).any():
            raise ValueError('Repeated transition requires nonabsorbing/multiple-event design')
    return designs

def inventory(panel, design, args):
    tr = design.event.any(axis=1)
    years = panel.years[np.where(design.event)[1]]
    nt = int(tr.sum())
    allowed = not (design.kind.startswith('recipe') and nt < args.min_recipe_treated)
    return {**design.info(), 'raw_treated_counties': nt, 'cohorts': ';'.join(map(str, sorted(set(years)))), 'raw_n_cohorts': len(set(years)), 'cluster': design.cluster(args), 'screened_for_estimation': allowed, 'screen_reason': '' if allowed else 'below_prespecified_recipe_raw_count_threshold'}

def annotate_small_samples(frame, inventories):
    if frame.empty or 'design_id' not in frame:
        return frame
    frame = frame.copy()
    counts = {r['design_id']: r['raw_treated_counties'] for r in inventories}
    frame['raw_treated_counties'] = frame.design_id.map(counts)
    frame['raw_treated_lt20'] = frame.raw_treated_counties.lt(20)
    support = next((c for c in ['n_treated_counties_unique', 'n_treated', 'n_treated_clusters'] if c in frame), None)
    effective = pd.to_numeric(frame[support], errors='coerce') if support else pd.Series(np.nan, index=frame.index)
    frame['effective_treated_lt20'] = effective.lt(20).where(effective.notna()).astype('boolean')
    minimum = pd.to_numeric(frame['min_cell_n_treated'], errors='coerce') if 'min_cell_n_treated' in frame else pd.Series(np.nan, index=frame.index)
    frame['some_cell_treated_lt20'] = minimum.lt(20).where(minimum.notna()).astype('boolean')

    def note(row):
        notes = []
        if row['raw_treated_lt20']:
            notes.append(f"Only {int(row['raw_treated_counties'])} counties originally experienced this policy change, fewer than 20.")
        if pd.notna(row['effective_treated_lt20']) and row['effective_treated_lt20']:
            notes.append(f'Only {int(row[support])} counties originally experienced this policy change, fewer than 20.')
        if pd.notna(row['some_cell_treated_lt20']) and row['some_cell_treated_lt20']:
            notes.append(f"Some group-time cells contain fewer than 20 treated counties; the minimum is {int(row['min_cell_n_treated'])} counties.")
        if notes:
            notes.append('Exploratory small-sample result; it may be sensitive to individual counties and random variation and should not be treated as a robust general conclusion.')
        return '；'.join(notes)
    frame['small_sample_note'] = frame.apply(note, axis=1)
    return frame

def comparison_support(panel, args, design, b, tr, co):
    tr = tr.copy()
    co = co.copy()
    if args.matching_rule == 'strict':
        history = np.where(panel.g <= b, panel.g, 9999).astype(int)
        _, strata = np.unique(history, axis=0, return_inverse=True)
        control_counts = np.bincount(strata, weights=co, minlength=int(strata.max()) + 1)
        treatment_counts = np.bincount(strata, weights=tr, minlength=len(control_counts))
        supported = (control_counts >= args.min_stratum_controls) & (treatment_counts > 0)
        tr &= supported[strata]
        co &= supported[strata]
    else:
        strata = None
    return (tr, co, strata)

def adjustment_design(panel, args, bi, ix, strata):
    raw = panel.cov[ix, bi, :]
    raw_names = list(core.COVARIATES)
    if args.matching_rule == 'strict':
        h = pd.get_dummies(pd.Series(strata[ix], name='history').astype(str), prefix='history', drop_first=True, dtype=float)
        return (np.column_stack([raw, h.to_numpy()]), raw_names + h.columns.tolist())
    if args.background_adjustment == 'status-duration':
        status = panel.xpolicy[ix, bi, :].astype(float)
        duration = panel.xpolicy[ix, :bi + 1, :].sum(axis=1).astype(float)
        initial = panel.xpolicy[ix, 0, :].astype(float)
        raw = np.column_stack([raw, status, duration, initial])
        raw_names += [f'baseline_{p}' for p in POLICIES] + [f'observed_duration_{p}' for p in POLICIES] + [f'initially_active_{p}' for p in POLICIES]
    return (raw, raw_names)

def build_cell(panel, args, design, outcome, min_pre, g, t, cluster, horizon=None):
    b = g - args.anticipation - 1
    end = t if horizon is None else g + horizon
    info = {**design.info(), 'outcome': outcome, 'min_pre': min_pre, 'g': int(g), 'year': int(t), 'event_time': int(t - g), 'baseline': int(b), 'cluster': cluster, 'analysis': 'ordinary' if horizon is None else 'lag_balanced', 'common_horizon': horizon, 'matching_rule': args.matching_rule, 'background_adjustment': args.background_adjustment if args.matching_rule == 'relaxed' else 'exact_history_FE', 'status': 'skipped', 'reason': ''}
    if b - min_pre + 1 < panel.years[0] or end > panel.years[-1]:
        return (None, {**info, 'reason': 'insufficient_pre_or_full_post_calendar'}, [])
    gi, bi, ti, ei = [panel.year_index(y) for y in (g, b, t, end)]
    y = panel.outcomes[outcome]
    valid = np.isfinite(y[:, bi - min_pre + 1:bi + 1]).all(axis=1)
    valid &= np.isfinite(y[:, ti]) & np.isfinite(panel.cov[:, bi, :]).all(axis=1)
    if horizon is not None:
        valid &= np.isfinite(y[:, gi:ei + 1]).all(axis=1)
    tr = design.event[:, gi].copy()
    tr &= panel.bits[:, bi] == panel.bits[:, gi - 1]
    tr &= valid
    if end >= g:
        tr &= panel.bits[:, ei] == panel.bits[:, gi]
    co = (panel.counts[:, bi] == design.before_n) & valid & ~design.event[:, gi]
    if design.before_bits is not None:
        co &= panel.bits[:, bi] == design.before_bits
    cutoff = max(b, end) + args.anticipation
    future = np.where(panel.g > b, panel.g, np.inf).min(axis=1)
    co &= future > cutoff
    info['n_treated_before_history_match'] = int(tr.sum())
    info['n_control_before_history_match'] = int(co.sum())
    if not tr.any():
        return (None, {**info, 'reason': 'no_treated_with_outcomes_and_unchanged_target_portfolio'}, [])
    tr, co, strata = comparison_support(panel, args, design, b, tr, co)
    info['n_treated_after_history_match'] = int(tr.sum())
    info['n_control_after_history_match'] = int(co.sum())
    if tr.sum() < args.min_treated or co.sum() < args.min_controls:
        reason = 'insufficient_sample_after_exact_background_history_match' if args.matching_rule == 'strict' else 'insufficient_sample_under_relaxed_baseline_comparison'
        return (None, {**info, 'reason': reason}, [])
    ix = np.flatnonzero(tr | co)
    d = tr[ix].astype(float)
    raw, raw_names = adjustment_design(panel, args, bi, ix, strata)
    delta = y[ix, ti] - y[ix, bi]
    x, _ = core.full_rank_design(raw, raw_names)
    info['trimmed_treated'] = 0
    info['trimmed_control'] = 0
    info['propensity_trimming_applied'] = False
    try:
        fit = core.dr_panel(x, d, delta, ps_clip=args.ps_clip, ridge_lambda=args.ridge_lambda)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return (None, {**info, 'reason': f'all_estimators_failed:{exc}'}, [])
    codes = panel.cluster_codes[cluster][ix]
    ncl = len(panel.cluster_levels[cluster])
    nt = int(d.sum())
    nc = len(d) - nt
    score = np.bincount(codes, weights=fit['influence'] / len(ix), minlength=ncl)
    mass = np.bincount(codes, weights=d, minlength=ncl).astype(float) - nt / ncl
    trcl = set(codes[d == 1])
    cocl = set(codes[d == 0])
    info.update(status='estimated', att=fit['att'], n_treated=nt, n_control=nc, n_treated_clusters=len(trcl), n_control_clusters=len(cocl), sample_hash=hashlib.sha256(panel.ids[ix].tobytes()).hexdigest()[:16], control_ess=float(1 / np.sum(fit['control_weights'] ** 2)), n_design_columns=x.shape[1], estimation_method=fit.get('estimation_method', ''), primary_fit_error=fit.get('primary_fit_error', ''), stabilized_fit_error=fit.get('stabilized_fit_error', ''), ridge_lambda_used=fit.get('ridge_lambda_used', np.nan), ps_extreme_any=fit.get('ps_extreme_any', False), ps_extreme_n=fit.get('ps_extreme_n', 0), ps_clipped_n=fit.get('ps_clipped_n', 0), ps_treated_min=float(fit['ps'][d == 1].min()), ps_treated_max=float(fit['ps'][d == 1].max()), ps_control_min=float(fit['ps'][d == 0].min()), ps_control_max=float(fit['ps'][d == 0].max()))
    balances = []
    for j, name in enumerate(raw_names):
        v = raw[:, j]
        sd = np.sqrt((np.var(v[d == 1]) + np.var(v[d == 0])) / 2)
        balances.append({'design_id': design.id, 'outcome': outcome, 'min_pre': min_pre, 'g': g, 'year': t, 'analysis': info['analysis'], 'common_horizon': horizon, 'covariate': name, 'smd_before': (v[d == 1].mean() - v[d == 0].mean()) / sd if sd > 1e-12 else np.nan, 'smd_after': (v[d == 1].mean() - fit['control_weights'] @ v) / sd if sd > 1e-12 else np.nan})
    cell = core.Cell(info, score, mass, trcl, cocl, set(panel.ids[ix[d == 1]]), set(panel.ids[ix[d == 0]]))
    return (cell, info, balances)

def unavailable(common, reason):
    return {**common, 'status': 'not_estimable', 'reason': reason, 'att': np.nan, 'se': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'p_value': np.nan, 'inference_ok': False}

def aggregate(cells, mult, args, fixed=None):
    result = core.aggregate(cells, mult, args, fixed)
    result[0]['cohorts'] = ';'.join(map(str, sorted({c.row['g'] for c in cells})))
    return result

def lag_outputs(panel, args, design, outcome, min_pre, cohorts, cluster, mult, common):
    cache = {}
    audit = []
    periods = []
    contrasts = []
    for h in (0, 2, 5):
        accepted = []
        for g in cohorts:
            if g - args.anticipation - min_pre < panel.years[0] or g + h > panel.years[-1]:
                continue
            group = []
            for e in range(h + 1):
                c, row, _ = build_cell(panel, args, design, outcome, min_pre, g, g + e, cluster, h)
                audit.append(row.copy())
                if c is None:
                    break
                group.append(c)
            if len(group) != h + 1:
                audit.append({**common, 'g': g, 'analysis': 'lag_balanced', 'common_horizon': h, 'status': 'skipped_cohort', 'reason': 'not_all_horizon_cells_supported'})
                continue
            if any((c.treated_units != group[0].treated_units or c.control_units != group[0].control_units for c in group)):
                raise RuntimeError('Common lag sample changed across periods')
            accepted.extend(group)
        cache[h] = accepted

    def summary(period, h):
        cells = [c for c in cache[h] if c.row['event_time'] in PERIODS[period]]
        if not cells:
            return None
        tr = set().union(*(c.treated_clusters for c in cells))
        co = set().union(*(c.control_clusters for c in cells))
        result = aggregate(cells, mult, args, len(tr | co))
        keys = [(c.row['g'], c.row['sample_hash']) for c in cells if c.row['event_time'] == min(PERIODS[period])]
        result[0]['support_hash'] = hashlib.sha256(repr(keys).encode()).hexdigest()[:16]
        return result
    for period, events in PERIODS.items():
        h = max(events)
        s = summary(period, h)
        row = {**common, 'period': period, 'support_end_event': h, 'support_scope': 'period_specific_complete_window', 'pretrend_note': 'common_sample_pretrend_not_retested'}
        periods.append(core.add_interpretation({**row, **s[0], 'status': 'estimated'}, outcome) if s else unavailable(row, 'no_complete_supported_cohort'))
    for later, earlier in CONTRASTS:
        h = max(PERIODS[later] + PERIODS[earlier])
        l = summary(later, h)
        e = summary(earlier, h)
        row = {**common, 'contrast': later + '_minus_' + earlier, 'support_end_event': h, 'later_period': later, 'earlier_period': earlier, 'support_scope': 'common_cohorts_units_and_weights', 'pretrend_note': 'common_sample_pretrend_not_retested'}
        if l is None or e is None:
            contrasts.append({**unavailable(row, 'no_common_complete_supported_cohort'), 'difference': np.nan, 'p_greater': np.nan})
            continue
        if l[0]['support_hash'] != e[0]['support_hash']:
            raise RuntimeError('Lag contrast support mismatch')
        tr = set().union(*(c.treated_clusters for c in cache[h]))
        co = set().union(*(c.control_clusters for c in cache[h]))
        stats = core.lag_contrast(l, e, mult, tr, co, args)
        row.update(stats, status='estimated', cohorts=l[0]['cohorts'], support_hash=l[0]['support_hash'], n_treated_counties_unique=l[0]['n_treated_counties_unique'], n_control_counties_unique=l[0]['n_control_counties_unique'], n_cohorts=l[0]['n_cohorts'], min_cell_n_treated=l[0]['min_cell_n_treated'])
        if outcome == 'JointImprovement':
            row['difference_percentage_points'] = 100 * row['difference']
        contrasts.append(row)
    return (periods, contrasts, audit)

def run_did(panel, args, designs, inventories, out):
    rng = np.random.default_rng(args.seed)
    multiplier = {col: rng.choice([-1.0, 1.0], size=(args.bootstrap, len(levels))) for col, levels in panel.cluster_levels.items()}
    rows = []
    overall = []
    events = []
    pretests = []
    audit = []
    balance = []
    periods = []
    contrasts = []
    catalog = {r['design_id']: r for r in inventories}
    for index, design in enumerate(designs, 1):
        eligible = catalog[design.id]['screened_for_estimation']
        cluster = design.cluster(args)
        mult = multiplier[cluster]
        cohorts = sorted(set(panel.years[np.where(design.event)[1]].tolist()))
        if eligible:
            pass
        for minimum in args.min_pre:
            for outcome in args.outcomes:
                common = {**design.info(), 'outcome': outcome, 'min_pre': minimum, 'cluster': cluster, 'matching_rule': args.matching_rule, 'background_adjustment': args.background_adjustment if args.matching_rule == 'relaxed' else 'exact_history_FE'}
                if not eligible:
                    reason = catalog[design.id]['screen_reason']
                    overall.append(unavailable(common, reason))
                    audit.append({**common, 'status': 'skipped_design', 'reason': reason})
                    continue
                cells = []
                for g in cohorts:
                    b = g - args.anticipation - 1
                    if b - minimum + 1 < panel.years[0]:
                        audit.append({**common, 'g': g, 'status': 'skipped_cohort', 'reason': 'insufficient_pre_calendar'})
                        continue
                    for t in panel.years:
                        if t == b or b < t < g:
                            continue
                        c, row, bal = build_cell(panel, args, design, outcome, minimum, g, int(t), cluster)
                        audit.append(row.copy())
                        balance.extend(bal)
                        if c is not None:
                            inf, _ = core.inference(row['att'], c.cluster_score, mult, c.treated_clusters, c.control_clusters, args)
                            c.row.update(inf)
                            rows.append(core.add_interpretation(c.row.copy(), outcome))
                            cells.append(c)
                post = [c for c in cells if c.row['event_time'] >= 0]
                if post:
                    s, _, _ = aggregate(post, mult, args)
                    overall.append(core.add_interpretation({**common, **s, 'status': 'estimated', 'reason': ''}, outcome))
                else:
                    overall.append(unavailable(common, 'no_supported_post_cells'))
                dynamic = []
                for event in sorted({c.row['event_time'] for c in cells}):
                    s, score, boot = aggregate([c for c in cells if c.row['event_time'] == event], mult, args)
                    dynamic.append(({**common, **s, 'event_time': event}, score, boot))
                usable = [r for r in dynamic if r[0]['inference_ok']]
                critical = float(np.quantile(np.max(np.column_stack([np.abs(r[2] / r[0]['se']) for r in usable]), axis=1), 0.95)) if usable else np.nan
                for row, _, _ in dynamic:
                    row['simultaneous_ci_low'] = row['att'] - critical * row['se'] if row['inference_ok'] else np.nan
                    row['simultaneous_ci_high'] = row['att'] + critical * row['se'] if row['inference_ok'] else np.nan
                    events.append(core.add_interpretation(row, outcome))
                pre = [r for r in dynamic if r[0]['event_time'] < -args.anticipation - 1 and r[0]['inference_ok']]
                if pre:
                    est = np.array([r[0]['att'] for r in pre])
                    draws = np.column_stack([r[2] for r in pre])
                    vcov = np.atleast_2d(np.cov(draws, rowvar=False))
                    inv = np.linalg.pinv(vcov)
                    w = float(est @ inv @ est)
                    sim = np.einsum('bi,ij,bj->b', draws, inv, draws)
                    rank = np.linalg.matrix_rank(vcov)
                    pretests.append({**common, 'n_pre_coefficients': len(pre), 'wald': w, 'p_value': (1 + (sim >= w).sum()) / (len(sim) + 1), 'status': 'available' if rank == len(pre) else 'rank_deficient_partial_test'})
                else:
                    pretests.append({**common, 'n_pre_coefficients': 0, 'p_value': np.nan, 'status': 'not_testable'})
                if not args.skip_lag:
                    p, c, a = lag_outputs(panel, args, design, outcome, minimum, cohorts, cluster, mult, common)
                    periods.extend(p)
                    contrasts.extend(c)
                    audit.extend(a)
    tables = {'stack_did_overall': pd.DataFrame(overall), 'stack_did_group_time': pd.DataFrame(rows), 'stack_did_event_study': pd.DataFrame(events), 'stack_did_pretrend_tests': pd.DataFrame(pretests), 'stack_sample_audit': pd.DataFrame(audit), 'stack_covariate_balance': pd.DataFrame(balance), 'stack_period_effects': pd.DataFrame(periods), 'stack_period_contrasts': pd.DataFrame(contrasts)}
    for name in ['stack_did_overall', 'stack_period_effects', 'stack_period_contrasts']:
        df = tables[name]
        if not df.empty:
            core.add_fdr(df, ['design_id', 'min_pre'])
            df['q_BH_search_family'] = np.nan
            for _, ix in df.groupby(['design_type', 'min_pre']).groups.items():
                df.loc[ix, 'q_BH_search_family'] = core.bh(df.loc[ix, 'p_value'])
    ct = tables['stack_period_contrasts']
    if not ct.empty:
        ct['q_BH_greater'] = np.nan
        for _, ix in ct.groupby(['design_id', 'min_pre']).groups.items():
            ct.loc[ix, 'q_BH_greater'] = core.bh(ct.loc[ix, 'p_greater'])
        ct['q_BH_greater_search_family'] = np.nan
        for _, ix in ct.groupby(['design_type', 'min_pre']).groups.items():
            ct.loc[ix, 'q_BH_greater_search_family'] = core.bh(ct.loc[ix, 'p_greater'])
    df = tables['stack_did_overall']
    pt = tables['stack_did_pretrend_tests']
    if not df.empty and (not pt.empty):
        pt = pt[['design_id', 'min_pre', 'outcome', 'status', 'p_value']].rename(columns={'status': 'pretrend_status', 'p_value': 'pretrend_p_value'})
        df = df.merge(pt, on=['design_id', 'min_pre', 'outcome'], how='left', validate='one_to_one')
        df['interpretation_note'] = np.where(df.pretrend_p_value < 0.05, 'pretrend_rejected_no_causal_claim', np.where(df.pretrend_p_value.isna(), 'pretrend_not_testable', 'nonrejection_is_not_proof'))
        tables['stack_did_overall'] = df
    for name, table in tables.items():
        if name.startswith('stack_period') and args.skip_lag:
            continue
        table = annotate_small_samples(table, inventories)
        tables[name] = table
        core.save_csv(table, out / f'{name}.csv')
    ad = tables['stack_sample_audit']
    if not ad.empty:
        diagnostic = ad.groupby(['design_id', 'outcome', 'min_pre', 'status', 'reason'], dropna=False).size().reset_index(name='audit_record_count')
        core.save_csv(diagnostic, out / 'stack_failure_reason_counts.csv')
    return tables

def coverage(panel, args, out):
    data = panel.data.copy()
    data['policy_count'] = data[POLICIES].sum(axis=1).astype(int)
    covs = ['log_PGDP', 'log_POPD', 'GIA_GDP', 'Builtup', 'Coal_Share', 'Electricity_Share']
    summary = []
    descriptive = []
    diagnostics = []
    covariance = []
    max_count = int(data.policy_count.max())
    for outcome in args.outcomes:
        valid = np.isfinite(data[[f'Y_{outcome}'] + covs].to_numpy(float)).all(axis=1)
        q = data.loc[valid].copy()
        if not q.policy_count.eq(0).any():
            raise ValueError('No zero-policy observations for requested coverage reference')
        for year, group in q.groupby('Year'):
            for k in range(max_count + 1):
                s = group[group.policy_count.eq(k)]
                descriptive.append({'outcome': outcome, 'year': int(year), 'policy_count': k, 'n_counties': int(s.Code.nunique()), 'n_rows': len(s), 'mean_raw': s[outcome].mean() if outcome != 'JointImprovement' else s.Y_JointImprovement.mean(), 'mean_analysis_outcome': s[f'Y_{outcome}'].mean()})
        present = sorted(q.policy_count.unique().tolist())
        dummy = pd.DataFrame({f'count_{k}': q.policy_count.eq(k).astype(float) for k in present if k > 0}, index=q.index)
        fe = pd.get_dummies(q[['Year']].astype(str), drop_first=True, dtype=float)
        z = q[covs]
        z = (z - z.mean()) / z.std(ddof=0).replace(0, 1)
        frame = pd.concat([dummy, z, fe], axis=1)
        frame.insert(0, 'Intercept', 1.0)
        x = frame.to_numpy(float)
        y = q[f'Y_{outcome}'].to_numpy(float)
        _, sv, vh = np.linalg.svd(x, full_matrices=False)
        rank = int((sv > max(x.shape) * sv[0] * np.finfo(float).eps).sum())
        estimable = np.ones(x.shape[1], bool) if rank == x.shape[1] else np.linalg.norm(vh[rank:], axis=0) < 1e-08
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        resid = y - x @ beta
        codes, levels = pd.factorize(q[args.coverage_cluster], sort=True)
        nc = len(levels)
        meat = np.zeros((nc, x.shape[1]))
        np.add.at(meat, codes, x * resid[:, None])
        bread = np.linalg.pinv(x.T @ x)
        factor = nc / (nc - 1) * (len(q) - 1) / (len(q) - rank)
        vcov = bread @ meat.T @ meat @ bread * factor
        se = np.sqrt(np.maximum(0, np.diag(vcov)))
        nref = int(q.loc[q.policy_count.eq(0), args.coverage_cluster].nunique())
        for k in range(max_count + 1):
            group = q[q.policy_count.eq(k)]
            row = {'outcome': outcome, 'policy_count': k, 'reference_count': 0, 'n_category_rows': len(group), 'n_category_counties': int(group.Code.nunique()), 'n_category_clusters': int(group[args.coverage_cluster].nunique()), 'n_reference_clusters': nref, 'cluster': args.coverage_cluster, 'n_model_rows': len(q), 'interpretation': 'conditional_coverage_count_association_not_causal_effect', 'association': np.nan, 'se': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'p_value': np.nan, 'inference_ok': False}
            if k == 0:
                row.update(status='reference', association=0.0, ci_low=0.0, ci_high=0.0)
            elif k not in present:
                row['status'] = 'not_observed'
            else:
                j = frame.columns.get_loc(f'count_{k}')
                ok = bool(estimable[j] and se[j] > 1e-14 and (nc > 1))
                preferred = bool(ok and min(row['n_category_clusters'], nref) >= args.min_inference_clusters)
                row.update(status='estimated' if estimable[j] else 'not_identifiable_collinearity', association=float(beta[j]) if estimable[j] else np.nan, se=float(se[j]) if estimable[j] else np.nan, inference_ok=ok, preferred_inference_support=preferred, inference_note='preferred_cluster_support' if preferred else 'exploratory_few_clusters_but_CI_p_reported' if ok else 'variance_not_identifiable')
                if ok:
                    crit = student_t.ppf(0.975, nc - 1)
                    row.update(ci_low=float(beta[j] - crit * se[j]), ci_high=float(beta[j] + crit * se[j]), p_value=float(2 * student_t.sf(abs(beta[j] / se[j]), nc - 1)))
            if outcome == 'JointImprovement':
                row['association_percentage_points'] = 100 * row['association']
            summary.append(row)
        for k in present:
            for l in present:
                if k == 0 or l == 0:
                    v = 0.0
                else:
                    j = frame.columns.get_loc(f'count_{k}')
                    jj = frame.columns.get_loc(f'count_{l}')
                    v = float(vcov[j, jj]) if estimable[j] and estimable[jj] else np.nan
                covariance.append({'outcome': outcome, 'count_a': k, 'count_b': l, 'covariance': v})
        if outcome == 'JointImprovement':
            fitted = x @ beta
            diagnostics.append({'outcome': outcome, 'lpm_fraction_outside_0_1': float(np.mean((fitted < 0) | (fitted > 1)))})
    result = pd.DataFrame(summary)
    result['q_BH_all_counts_outcomes'] = core.bh(result.p_value)
    core.save_csv(result, out / 'coverage_count_associations.csv')
    core.save_csv(descriptive, out / 'coverage_count_descriptive.csv')
    core.save_csv(covariance, out / 'coverage_count_covariance.csv')
    core.save_csv(diagnostics, out / 'coverage_count_diagnostics.csv')
    return result

def plots(tables, cover, args, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    folder = out / 'figures'
    folder.mkdir(exist_ok=True)
    if cover is not None:
        fig, axs = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
        for ax, outcome in zip(axs, OUTCOMES):
            q = cover[cover.outcome.eq(outcome)]
            scale = 100 if outcome == 'JointImprovement' else 1
            ax.axvline(0, color='gray', lw=0.8)
            ax.scatter(q.association * scale, q.policy_count, s=20)
            ax.hlines(q.policy_count, q.ci_low * scale, q.ci_high * scale)
            ax.set_title(outcome)
            ax.set_xlabel('Percentage points' if outcome == 'JointImprovement' else 'Negative log association')
        axs[0].set_ylabel('Number of covered policies (0 = reference)')
        fig.suptitle('Policy coverage count: conditional associations, NOT causal effects')
        fig.text(0.5, 0.01, '0 = reference contrast; non-reference points without CI have insufficient inference support.', ha='center', fontsize=9)
        fig.tight_layout(rect=[0, 0.04, 1, 0.97])
        fig.savefig(folder / 'coverage_count_associations.png', dpi=220)
        plt.close(fig)
    df = tables.get('stack_did_overall', pd.DataFrame())
    if not df.empty:
        count = df[df.design_type.str.startswith('count')]
        for minimum, q in count.groupby('min_pre'):
            labels = q[['design_id', 'transition', 'added_count']].drop_duplicates().set_index('design_id')
            order = labels.index.tolist()
            pos = np.arange(len(order))
            fig, axs = plt.subplots(1, 5, figsize=(18, 7), sharey=True)
            for ax, outcome in zip(axs, OUTCOMES):
                s = q[q.outcome.eq(outcome)].set_index('design_id').reindex(order)
                scale = 100 if outcome == 'JointImprovement' else 1
                ax.axvline(0, color='gray', lw=0.8)
                ax.scatter(s.att * scale, pos, s=20)
                ax.hlines(pos, s.ci_low * scale, s.ci_high * scale)
                ax.set_yticks(pos, [labels.loc[i, 'transition'] + (' package' if labels.loc[i, 'added_count'] > 1 else '') for i in order])
                ax.set_title(outcome)
                ax.set_xlabel('Percentage points' if outcome == 'JointImprovement' else 'Incremental ATT')
            axs[0].invert_yaxis()
            fig.suptitle(f'Policy additions DID: minimum {minimum} pre-outcome year(s)')
            fig.tight_layout()
            fig.savefig(folder / f'stack_overall_pre{minimum}.png', dpi=220)
            plt.close(fig)
    event = tables.get('stack_did_event_study', pd.DataFrame())
    if not event.empty:
        if not args.plot_recipes:
            event = event[event.design_type.str.startswith('count')]
        for (design, minimum), q in event.groupby(['design_id', 'min_pre']):
            fig, axs = plt.subplots(2, 3, figsize=(13, 7))
            for ax, outcome in zip(axs.ravel(), OUTCOMES):
                sub = q[q.outcome.eq(outcome)].sort_values('event_time')
                ax.axhline(0, color='gray', lw=0.8)
                ax.axvline(0, color='gray', ls=':', lw=0.8)
                if not sub.empty:
                    scale = 100 if outcome == 'JointImprovement' else 1
                    grid = sub.set_index('event_time').reindex(range(int(sub.event_time.min()), int(sub.event_time.max()) + 1))
                    ax.plot(grid.index, grid.att * scale, 'o-', ms=3)
                    ax.fill_between(grid.index, grid.simultaneous_ci_low.to_numpy() * scale, grid.simultaneous_ci_high.to_numpy() * scale, alpha=0.15)
                    ax.vlines(sub.event_time, sub.ci_low * scale, sub.ci_high * scale, lw=0.8)
                    ax.scatter([-args.anticipation - 1], [0], facecolors='white', edgecolors='gray', s=25, zorder=4)
                else:
                    ax.text(0.5, 0.5, 'Not estimable', transform=ax.transAxes, ha='center')
                ax.set_title(outcome)
                ax.set_xlabel('Years since policy addition')
            axs.ravel()[-1].axis('off')
            axs.ravel()[-1].text(0, 0.9, 'Bars: pointwise CI\nShade: simultaneous band\nHollow dot: baseline\nMissing CI: insufficient clusters\nSamples can vary across years', va='top')
            fig.suptitle(f'{design}, minimum pre = {minimum}')
            fig.tight_layout()
            fig.savefig(folder / f'{design}_pre{minimum}.png', dpi=180)
            plt.close(fig)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy-file', type=Path, default=core.DEFAULT_DIR / 'policy_conditions.csv')
    p.add_argument('--model-file', type=Path, default=core.DEFAULT_DIR / 'Model_Data_Total.csv')
    p.add_argument('--out', type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--matching-rule', choices=['relaxed', 'strict'], default='relaxed', help='Relaxed: baseline count for count designs; baseline portfolio for recipes; no exact entry-year matching')
    p.add_argument('--background-adjustment', choices=['status-duration', 'none'], default='status-duration', help='Relaxed mode regression adjustment for baseline policy indicators, observed durations and initial coverage; none retains original 8 covariates only')
    p.add_argument('--min-pre', type=int, nargs='+', choices=[1, 2, 3], default=[1, 2, 3])
    p.add_argument('--outcomes', nargs='+', choices=OUTCOMES, default=OUTCOMES)
    p.add_argument('--bootstrap', type=int, default=999)
    p.add_argument('--seed', type=int, default=20260916)
    p.add_argument('--anticipation', type=int, choices=[0, 1], default=0)
    p.add_argument('--state-definition', choices=['annual', 'cumulative2013'], default='annual')
    p.add_argument('--state-epsilon', type=float, default=0.0)
    p.add_argument('--ps-trim', type=float, default=0.01, help='Legacy diagnostic threshold only; no observations are trimmed in maximum-retention mode')
    p.add_argument('--ps-clip', type=float, default=1e-06, help='Numerical clipping used only inside propensity odds; observations are never dropped')
    p.add_argument('--ridge-lambda', type=float, default=0.0001, help='Initial ridge penalty for stabilized DR fallback')
    p.add_argument('--min-treated', type=int, default=1, help='Minimum treated counties per group-time cell')
    p.add_argument('--min-controls', type=int, default=1, help='Minimum control counties per group-time cell; default 1 for maximum retention')
    p.add_argument('--min-stratum-controls', type=int, default=5, help='Only used with --matching-rule strict')
    p.add_argument('--min-inference-clusters', type=int, default=5)
    p.add_argument('--min-recipe-treated', type=int, default=1, help='Raw county screen for specific combinations; values below 20 are flagged, not automatically excluded')
    p.add_argument('--cluster-override', choices=['Code'])
    p.add_argument('--coverage-cluster', choices=['Code'], default='Code')
    p.add_argument('--design-ids', nargs='+', help='Run listed design IDs from stack_design_inventory.csv')
    p.add_argument('--counts-only', action='store_true')
    p.add_argument('--skip-did', action='store_true')
    p.add_argument('--skip-coverage', action='store_true')
    p.add_argument('--skip-lag', action='store_true')
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--plot-recipes', action='store_true')
    args = p.parse_args(argv)
    if args.bootstrap < 99 or not 0 <= args.ps_trim < 0.25:
        p.error('bootstrap >=99 and 0 <= ps-trim < .25 required')
    if not 0 < args.ps_clip < 0.1 or args.ridge_lambda <= 0:
        p.error('Require 0 < ps-clip < 0.1 and ridge-lambda > 0')
    if min(args.min_treated, args.min_controls, args.min_stratum_controls, args.min_recipe_treated) < 1 or args.state_epsilon < 0:
        p.error('Invalid sample/state threshold')
    if args.min_inference_clusters < 1:
        p.error('min-inference-clusters must be >=1')
    args.min_pre = list(dict.fromkeys(args.min_pre))
    args.outcomes = list(dict.fromkeys(args.outcomes))
    out = args.out / f'run_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['inference_assumption'] = 'Independent counties; arbitrary serial dependence within Code'
    config['coverage_model'] = 'Pooled OLS/LPM; year FE only; no county FE; noncausal'
    config['comparison_rule'] = 'count: same baseline count; recipe: same baseline portfolio; entry dates need not match' if args.matching_rule == 'relaxed' else 'exact baseline portfolio and observed adoption history'
    config['duration_definition'] = 'Observed exposed years through baseline; initial coverage is left-censored and separately indicated'
    config['small_sample_policy'] = 'Raw or effective treated counties below 20 are flagged; treated/control point-estimation floors default to 1'
    config['propensity_policy'] = 'No propensity trimming; extreme scores retained. Numerical odds use ps-clip only.'
    config['fallback_policy'] = 'standard DR -> ridge-stabilized DR -> unadjusted change DID'
    config['identification_caution'] = 'Relaxation requires conditional parallel trends across differing histories; regression adjustment does not guarantee identification. Stable post-event portfolios select the supported sample.'
    config['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config['input_sha256'] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in [args.policy_file, args.model_file]}
    config['engine_sha256'] = hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()
    (out / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    panel = core.Panel(args)
    designs = make_designs(panel)
    inv = [inventory(panel, d, args) for d in designs]
    core.save_csv(annotate_small_samples(pd.DataFrame(inv), inv), out / 'stack_design_inventory.csv')
    (out / 'input_audit.json').write_text(json.dumps(panel.audit, ensure_ascii=False, indent=2), encoding='utf-8')
    membership = panel.data[['Code', 'Year'] + POLICIES + [f'Y_{o}' for o in OUTCOMES]].copy()
    membership['policy_count'] = membership[POLICIES].sum(axis=1)
    membership['portfolio_bits'] = (membership[POLICIES].to_numpy() * 2 ** np.arange(10)).sum(axis=1)
    membership['portfolio'] = membership.portfolio_bits.map(names)
    core.save_csv(membership, out / 'county_policy_portfolio_panel.csv')
    if args.counts_only:
        designs = [d for d in designs if d.kind.startswith('count')]
    if args.design_ids:
        requested = set(args.design_ids)
        found = {d.id for d in designs}
        if requested - found:
            p.error(f'Unknown/incompatible design IDs: {requested - found}')
        designs = [d for d in designs if d.id in requested]
    tables = {} if args.skip_did else run_did(panel, args, designs, inv, out)
    assoc = None if args.skip_coverage else coverage(panel, args, out)
    if not args.no_plots:
        plots(tables, assoc, args, out)
    summary = {'output_directory': str(out), 'design_inventory_count': len(inv), 'selected_DID_designs': len(designs), 'DID_summary_rows': len(tables.get('stack_did_overall', [])), 'period_rows': len(tables.get('stack_period_effects', [])), 'period_contrast_rows': len(tables.get('stack_period_contrasts', [])), 'coverage_rows': len(assoc) if assoc is not None else 0, 'matching_rule': args.matching_rule, 'background_adjustment': config['background_adjustment'], 'notes': ['Recipe DID compares adding policies versus retaining the same baseline portfolio; count DID compares increasing versus retaining the same baseline count.', 'Under relaxed matching, policy entry years need not match; count designs also allow different baseline policy identities.', 'Under relaxed status-duration adjustment, baseline policy status, observed duration and initial coverage are regression covariates, not exact matching constraints.', 'Initially covered policy durations are left-censored; residual history confounding remains possible.', 'Simultaneous multi-policy entries are separate package effects.', 'Count-transition effects average observed policy recipes; they do not prove superadditivity.', 'Coverage count regressions are conditional associations versus zero-policy coverage.', 'All initially covered counties remain in coverage analysis.', 'Failure records are not zero effects. Few county clusters are flagged as exploratory.', 'Raw recipe, treated-cell and control-cell floors default to 1; samples below 20 counties carry exploratory warnings.', 'Extreme propensity scores do not trigger observation deletion. Estimator order is standard DR, ridge-stabilized DR, then unadjusted change DID fallback.', 'Code-only inference assumes cross-county independence; not empirically established.', 'Coverage now uses year FE only; coefficients can change from the earlier specification.']}
    (out / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    pass
    return out
if __name__ == '__main__':
    main()
