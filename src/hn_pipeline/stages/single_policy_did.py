from __future__ import annotations
import argparse
import csv
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'single_policy_did'
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

def fit_logit(x, treatment, weights=None, diagnostics=None):
    n, k = x.shape
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    w = w / w.mean()
    beta = np.zeros(k)
    share = np.average(treatment, weights=w)
    beta[0] = np.log(share / (1 - share))
    for _ in range(150):
        eta = x @ beta
        ps = expit(eta)
        score = x.T @ (w * (treatment - ps)) / n
        info = x.T * (w * ps * (1 - ps)) @ x / n
        if diagnostics is not None:
            diagnostics.update(ps=ps.copy(), eta=eta.copy(), iterations=_ + 1, information_condition=float(np.linalg.cond(info)), converged=False)
        if np.linalg.cond(info) > 1000000000000.0:
            raise ValueError('propensity_separation_or_ill_conditioned')
        if np.max(np.abs(score)) < 1e-08:
            if diagnostics is not None:
                diagnostics['converged'] = True
            return (beta, ps, info)
        step = np.linalg.solve(info, score)
        old = np.mean(w * (treatment * eta - np.logaddexp(0, eta)))
        length = 1.0
        while length > 1e-10:
            new_beta = beta + length * step
            new_eta = x @ new_beta
            new = np.mean(w * (treatment * new_eta - np.logaddexp(0, new_eta)))
            if new >= old - 1e-12:
                break
            length *= 0.5
        if length <= 1e-10:
            raise ValueError('propensity_line_search_failed')
        beta = new_beta
    raise ValueError('propensity_not_converged')

def dr_panel(x, treatment, delta, weights=None, diagnostics=None):
    n = len(treatment)
    d = np.asarray(treatment, float)
    y = np.asarray(delta, float)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    w = w / w.mean()
    beta, p, info = fit_logit(x, d, w, diagnostics)
    if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError('propensity_at_numeric_boundary_no_finite_odds')
    control = 1 - d
    gram = x.T * (w * control) @ x / n
    if np.linalg.matrix_rank(gram) < x.shape[1] or np.linalg.cond(gram) > 1000000000000.0:
        raise ValueError('outcome_regression_rank_deficient')
    gamma = np.linalg.solve(gram, x.T @ (w * control * y) / n)
    residual = y - x @ gamma
    odds = p / (1 - p)
    a = w * d / np.sum(w * d)
    c = w * control * odds / np.sum(w * control * odds)
    mt, mc = (a @ residual, c @ residual)
    att = mt - mc
    if weights is not None:
        return att
    direct = d / d.mean() * (residual - mt)
    direct -= control * odds / np.mean(control * odds) * (residual - mc)
    derivative_beta = -(c * (residual - mc)) @ x
    derivative_gamma = -(a - c) @ x
    if_beta = x * (d - p)[:, None] @ np.linalg.inv(info)
    if_gamma = x * (control * residual)[:, None] @ np.linalg.inv(gram)
    influence = direct + if_beta @ derivative_beta + if_gamma @ derivative_gamma
    influence -= influence.mean()
    if not np.isfinite(att) or not np.isfinite(influence).all():
        raise ValueError('nonfinite_ATT_or_influence')
    return dict(att=float(att), influence=influence, ps=p, control_weights=c, beta=beta, gamma=gamma)

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

def policy_history(panel, k, baseline):
    other = [j for j in range(len(POLICIES)) if j != k]
    history = np.where(panel.g[:, other] <= baseline, panel.g[:, other], 9999).astype(int)
    return (other, history)

def record_propensity(panel, row, ix, d, diag, stage, failed=False):
    if 'ps' not in diag:
        return
    ps, eta = (diag['ps'], diag['eta'])
    extreme = (ps < 1e-05) | (ps > 1 - 1e-05) | (np.abs(eta) > 30)
    row.update({stage + '_ps_min': float(ps.min()), stage + '_ps_max': float(ps.max()), stage + '_max_abs_linear_predictor': float(np.max(np.abs(eta))), stage + '_extreme_count': int(extreme.sum()), stage + '_information_condition': diag['information_condition'], stage + '_converged': diag['converged']})
    writer = getattr(panel, 'ps_writer', None)
    if writer is not None:
        for j in np.flatnonzero(np.ones(len(ps), bool) if failed else extreme):
            writer.writerow({**{k: row.get(k, '') for k in ['policy', 'outcome', 'min_pre', 'g', 'year', 'analysis', 'common_horizon']}, 'stage': stage, 'Code': panel.ids[ix[j]], 'treated': int(d[j]), 'propensity_last_iterate': ps[j], 'linear_predictor': eta[j], 'fit_converged': diag['converged'], 'fit_failed': failed, 'extreme_prediction': bool(extreme[j]), 'failure_reason': row.get('reason', '') if failed else ''})

def suppress_extreme_inference(stats, extreme):
    if extreme:
        stats.update(inference_ok=False, inference_note='extreme_propensity_point_estimate_only', ci_low=np.nan, ci_high=np.nan, p_value=np.nan)
    return stats

def annotate_samples(frame, panel):
    if frame.empty or 'policy' not in frame:
        return frame
    frame = frame.copy()
    raw = {p: int(np.sum(np.isfinite(panel.g[:, j]) & (panel.g[:, j] > panel.years[0]))) for j, p in enumerate(POLICIES)}
    frame['raw_newly_treated_counties'] = frame.policy.map(raw)
    support = next((k for k in ['n_treated_counties_unique', 'n_treated', 'n_treated_clusters'] if k in frame), None)

    def note(r):
        notes = []
        if r.raw_newly_treated_counties < 20:
            notes.append(f'Only {int(r.raw_newly_treated_counties)} counties.')
        if support and pd.notna(r[support]) and (r[support] < 20):
            notes.append(f'Only {int(r[support])} counties.')
        if pd.notna(r.get('min_cell_n_treated')) and r['min_cell_n_treated'] < 20:
            notes.append(f"Some group-time cells contain fewer than 20 counties; the minimum is {int(r['min_cell_n_treated'])} counties.")
        return '；'.join(notes + ['Exploratory small-sample result; it may be sensitive to individual counties and random variation.']) if notes else ''
    frame['small_sample_note'] = frame.apply(note, axis=1)
    frame['small_sample_warning'] = frame.small_sample_note.ne('')
    return frame

def build_cell(panel, args, policy, outcome, min_pre, g, t, cluster_col, common_horizon=None):
    k = POLICIES.index(policy)
    b = g - args.anticipation - 1
    row = dict(policy=policy, outcome=outcome, min_pre=min_pre, g=int(g), year=int(t), event_time=int(t - g), baseline=int(b), cluster=cluster_col, status='skipped', reason='', sample_mode=args.other_policy_mode, analysis='ordinary' if common_horizon is None else 'lag_balanced', common_horizon=common_horizon)
    y = panel.outcomes[outcome]
    if b not in panel.years or b - min_pre + 1 < panel.years[0]:
        row['reason'] = 'insufficient_pre_calendar'
        return (None, row, [])
    bi, ti = (panel.year_index(b), panel.year_index(t))
    valid = np.isfinite(y[:, bi - min_pre + 1:bi + 1]).all(axis=1)
    valid &= np.isfinite(y[:, ti]) & np.isfinite(panel.cov[:, bi, :]).all(axis=1)
    end_year = t if common_horizon is None else g + common_horizon
    if common_horizon is not None:
        if end_year not in panel.years:
            row['reason'] = 'incomplete_lag_calendar_window'
            return (None, row, [])
        valid &= np.isfinite(y[:, panel.year_index(g):panel.year_index(end_year) + 1]).all(axis=1)
    tr = panel.g[:, k] == g
    if args.control_group == 'never':
        co = np.isinf(panel.g[:, k])
    else:
        co = panel.g[:, k] > max(b, end_year) + args.anticipation
    co &= ~tr
    tr &= valid
    co &= valid
    row['n_treated_before_policy_screen'] = int(tr.sum())
    row['n_control_before_policy_screen'] = int(co.sum())
    if not tr.any():
        row['reason'] = 'no_treated_with_required_pre_outcomes_and_covariates'
        return (None, row, [])
    other, history = policy_history(panel, k, b)
    if args.other_policy_mode == 'clean':
        lo, hi = (min(bi, ti), max(bi, ti, panel.year_index(end_year)))
        stable = (np.diff(panel.xpolicy[:, lo:hi + 1, other], axis=1) == 0).all(axis=(1, 2))
        tr &= stable
        co &= stable
    row['n_treated_after_policy_screen'] = int(tr.sum())
    row['n_control_after_policy_screen'] = int(co.sum())
    if tr.sum() < args.min_treated or co.sum() < args.min_controls:
        row['reason'] = 'insufficient_treated_or_controls_after_other_policy_stability_screen'
        return (None, row, [])
    ix = np.flatnonzero(tr | co)
    d = tr[ix].astype(float)
    status = panel.xpolicy[ix, bi, :][:, other].astype(float)
    duration = panel.xpolicy[ix, :bi + 1, :].sum(axis=1)[:, other].astype(float)
    initial = panel.xpolicy[ix, 0, :][:, other].astype(float)
    raw = np.column_stack([panel.cov[ix, bi, :], status, duration, initial])
    names = COVARIATES + [f'baseline_{POLICIES[j]}' for j in other] + [f'observed_duration_{POLICIES[j]}' for j in other] + [f'initially_active_{POLICIES[j]}' for j in other]
    delta = y[ix, ti] - y[ix, bi]
    x, xnames = full_rank_design(raw, names)
    diag = {}
    stage = 'initial'
    initial_ix = ix.copy()
    initial_d = d.copy()
    row['n_design_columns'] = x.shape[1]
    try:
        _, ps, _ = fit_logit(x, d, diagnostics=diag)
        record_propensity(panel, row, ix, d, diag, stage)
        retain = (ps >= args.ps_trim) & (ps <= 1 - args.ps_trim)
        row['n_trimmed_treated'] = int(np.sum((d == 1) & ~retain))
        row['n_trimmed_control'] = int(np.sum((d == 0) & ~retain))
        if not retain.all():
            ix, d, delta, raw = (ix[retain], d[retain], delta[retain], raw[retain])
            if d.sum() < args.min_treated or (1 - d).sum() < args.min_controls:
                raise ValueError('insufficient_sample_after_ps_trim')
            x, xnames = full_rank_design(raw, names)
        diag = {}
        stage = 'final'
        estimate = dr_panel(x, d, delta, diagnostics=diag)
        record_propensity(panel, row, ix, d, diag, stage)
        row['extreme_propensity'] = bool((estimate['ps'] < 1e-05).any() or (estimate['ps'] > 1 - 1e-05).any() or np.max(np.abs(diag['eta'])) > 30)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        row['reason'] = str(exc)
        record_propensity(panel, row, initial_ix if stage == 'initial' else ix, initial_d if stage == 'initial' else d, diag, stage, failed=True)
        return (None, row, [])
    codes = panel.cluster_codes[cluster_col][ix]
    nt = int(d.sum())
    nc = len(d) - nt
    tr_clusters = set(codes[d == 1])
    co_clusters = set(codes[d == 0])
    ncluster = len(panel.cluster_levels[cluster_col])
    score = np.bincount(codes, weights=estimate['influence'] / len(ix), minlength=ncluster)
    mass = np.bincount(codes, weights=d, minlength=ncluster).astype(float)
    mass -= nt / ncluster
    ess = 1 / np.sum(estimate['control_weights'] ** 2)
    row.update(status='estimated', reason='', att=estimate['att'], n_treated=nt, n_control=nc, sample_hash=hashlib.sha256(panel.ids[ix].tobytes()).hexdigest()[:16], n_treated_clusters=len(tr_clusters), n_control_clusters=len(co_clusters), n_design_columns=x.shape[1], control_ess=float(ess), ps_treated_min=float(estimate['ps'][d == 1].min()), ps_treated_max=float(estimate['ps'][d == 1].max()), ps_control_min=float(estimate['ps'][d == 0].min()), ps_control_max=float(estimate['ps'][d == 0].max()))
    balance = []
    for j, name in enumerate(names):
        val = raw[:, j]
        tv = val[d == 1]
        cv = val[d == 0]
        sd = np.sqrt((np.var(tv) + np.var(cv)) / 2)
        if sd < 1e-12:
            before = after = np.nan
        else:
            before = (tv.mean() - cv.mean()) / sd
            after = (tv.mean() - estimate['control_weights'] @ val) / sd
        balance.append({**{v: row[v] for v in ['policy', 'outcome', 'min_pre', 'g', 'year']}, 'covariate': name, 'smd_before': before, 'smd_after': after})
    return (Cell(row, score, mass, tr_clusters, co_clusters, set(panel.ids[ix[d == 1]]), set(panel.ids[ix[d == 0]])), row, balance)

def inference(att, score, multipliers, treated_clusters, control_clusters, args, fixed_cluster_count=None):
    active = np.abs(score) > 1e-14
    k = int(active.sum()) if fixed_cluster_count is None else fixed_cluster_count
    factor = np.sqrt(k / (k - 1)) if k > 1 else 1.0
    centered = score - score.mean()
    se = float(np.sqrt(np.sum(centered ** 2)) * factor)
    boot = multipliers @ centered * factor
    reliable = len(treated_clusters) >= args.min_inference_clusters and len(control_clusters) >= args.min_inference_clusters and (k > 1) and (se > 1e-14)
    result = {'se': se, 'n_treated_clusters': len(treated_clusters), 'n_control_clusters': len(control_clusters), 'inference_ok': reliable, 'inference_note': 'Code_cluster_multiplier_independent_counties_assumption' if reliable else 'few_clusters_or_degenerate_variance', 'ci_low': np.nan, 'ci_high': np.nan, 'p_value': np.nan}
    if reliable:
        crit = max(1.0, float(np.quantile(np.abs(boot / se), 0.95)))
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
    extreme = any((c.row.get('extreme_propensity', False) for c in cells))
    suppress_extreme_inference(inf, extreme)
    row = {'att': att, **inf, 'n_cohorts': len(groups), 'n_cells': len(cells), 'extreme_propensity': extreme, 'n_treated_counties_unique': len(set().union(*(c.treated_units for c in cells))), 'n_control_counties_unique': len(set().union(*(c.control_units for c in cells))), 'min_cell_n_treated': min((c.row['n_treated'] for c in cells)), 'max_cell_n_treated': max((c.row['n_treated'] for c in cells)), 'min_cell_n_control': min((c.row['n_control'] for c in cells)), 'max_cell_n_control': max((c.row['n_control'] for c in cells)), 'weighting': 'equal_post_years_within_g_then_mean_retained_cohort_size'}
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
    extreme = bool(lr.get('extreme_propensity', False) or er.get('extreme_propensity', False))
    suppress_extreme_inference(inf, extreme)
    p_greater = (1 + np.sum(draws >= difference)) / (len(draws) + 1) if inf['inference_ok'] else np.nan
    return {'att_later': lr['att'], 'att_earlier': er['att'], 'difference': difference, **inf, 'extreme_propensity': extreme, 'p_greater': p_greater, 'variance_later': lr['se'] ** 2, 'variance_earlier': er['se'] ** 2, 'covariance_earlier_later': covariance, 'variance_difference': inf['se'] ** 2, 'bootstrap_covariance_earlier_later': float(np.cov(eb, lb, ddof=1)[0, 1])}

def run_lag_periods(panel, args, policy, outcome, min_pre, cohorts, cluster, multipliers):
    common = {'policy': policy, 'outcome': outcome, 'min_pre': min_pre, 'cluster': cluster, 'sample_mode': 'clean_fixed_horizon', 'pretrend_note': 'common_horizon_sample_pretrend_not_retested'}
    horizon_cells = {}
    audit = []
    for horizon in (0, 2, 5):
        accepted = []
        for g in cohorts:
            base = g - args.anticipation - 1
            if base - min_pre + 1 < panel.years[0] or g + horizon > panel.years[-1]:
                audit.append({**common, 'analysis': 'lag_balanced', 'common_horizon': horizon, 'g': g, 'status': 'skipped_cohort', 'reason': 'insufficient_pre_or_full_lag_calendar'})
                continue
            cohort_cells = []
            for event in range(horizon + 1):
                cell, row, _ = build_cell(panel, args, policy, outcome, min_pre, g, g + event, cluster, horizon)
                audit.append(row.copy())
                if cell is None:
                    break
                cohort_cells.append(cell)
            if len(cohort_cells) != horizon + 1:
                audit.append({**common, 'analysis': 'lag_balanced', 'common_horizon': horizon, 'g': g, 'status': 'skipped_cohort', 'reason': 'not_all_lag_cells_estimable'})
                continue
            first = cohort_cells[0]
            if any((c.treated_units != first.treated_units or c.control_units != first.control_units for c in cohort_cells)):
                raise RuntimeError('Lag comparison did not retain identical units within cohort')
            accepted.extend(cohort_cells)
        horizon_cells[horizon] = accepted

    def summarize(period, horizon):
        cells = [c for c in horizon_cells[horizon] if c.row['event_time'] in PERIODS[period]]
        if not cells:
            return None
        tr = set().union(*(c.treated_clusters for c in cells))
        co = set().union(*(c.control_clusters for c in cells))
        agg, score, boot = aggregate(cells, multipliers, args, len(tr | co))
        agg['cohorts'] = ';'.join((str(g) for g in sorted({c.row['g'] for c in cells})))
        support = [(c.row['g'], c.row['sample_hash']) for c in cells if c.row['event_time'] == min(PERIODS[period])]
        agg['support_hash'] = hashlib.sha256(repr(support).encode()).hexdigest()[:16]
        return (agg, score, boot)
    periods = []
    contrasts = []
    for name, events in PERIODS.items():
        horizon = max(events)
        result = summarize(name, horizon)
        row = {**common, 'period': name, 'event_start': min(events), 'event_end': horizon, 'support_end_event': horizon, 'support_scope': 'period_specific_full_window'}
        if result is None:
            row.update(status='not_estimable', reason='no_complete_supported_cohort', att=np.nan, se=np.nan, ci_low=np.nan, ci_high=np.nan, p_value=np.nan, inference_ok=False)
        else:
            row.update(result[0], status='estimated', reason='')
            add_interpretation(row, outcome)
        periods.append(row)
    for later, earlier in PERIOD_CONTRASTS:
        horizon = max(PERIODS[later] + PERIODS[earlier])
        l = summarize(later, horizon)
        e = summarize(earlier, horizon)
        row = {**common, 'contrast': later + '_minus_' + earlier, 'later_period': later, 'earlier_period': earlier, 'support_end_event': horizon, 'support_scope': 'pairwise_common_cohorts_units_and_weights', 'alternative_greater': 'later_effect_greater_than_earlier_effect'}
        if l is None or e is None:
            row.update(status='not_estimable', reason='no_common_complete_supported_cohort', difference=np.nan, p_value=np.nan, p_greater=np.nan, ci_low=np.nan, ci_high=np.nan, se=np.nan, inference_ok=False)
        else:
            if l[0]['support_hash'] != e[0]['support_hash']:
                raise RuntimeError('Lag contrast support mismatch')
            cells = horizon_cells[horizon]
            tr = set().union(*(c.treated_clusters for c in cells))
            co = set().union(*(c.control_clusters for c in cells))
            row.update(lag_contrast(l, e, multipliers, tr, co, args), status='estimated', reason='', cohorts=l[0]['cohorts'], support_hash=l[0]['support_hash'], n_cohorts=l[0]['n_cohorts'], n_treated_counties_unique=l[0]['n_treated_counties_unique'], n_control_counties_unique=l[0]['n_control_counties_unique'])
            if outcome == 'JointImprovement':
                row['difference_percentage_points'] = 100 * row['difference']
            else:
                row['difference_unit'] = 'difference_of_negative_log_ATT_not_percentage_point_difference'
        contrasts.append(row)
    return (periods, contrasts, audit)

def run_did(panel, args, out):
    panel.ps_stream = (out / 'did_propensity_extreme_and_failed_counties.csv').open('w', encoding='utf-8-sig', newline='')
    panel.ps_writer = csv.DictWriter(panel.ps_stream, fieldnames=['policy', 'outcome', 'min_pre', 'g', 'year', 'analysis', 'common_horizon', 'stage', 'Code', 'treated', 'propensity_last_iterate', 'linear_predictor', 'fit_converged', 'fit_failed', 'extreme_prediction', 'failure_reason'])
    panel.ps_writer.writeheader()
    rng = np.random.default_rng(args.seed)
    multipliers = {c: rng.choice([-1.0, 1.0], size=(args.bootstrap, len(levels))) for c, levels in panel.cluster_levels.items()}
    all_rows = []
    audits = []
    balances = []
    overall = []
    dynamic = []
    pretests = []
    cov_checks = []
    period_rows = []
    period_contrasts = []
    for min_pre in args.min_pre:
        for policy in args.policies:
            cluster = POLICY_CLUSTERS[policy]
            if args.cluster_override:
                cluster = args.cluster_override
            pass
            gvec = panel.g[:, POLICIES.index(policy)]
            cohorts = [int(g) for g in np.unique(gvec) if np.isfinite(g) and g > panel.years[0]]
            outcome_results = {}
            for outcome in args.outcomes:
                cells = []
                if np.any(gvec == panel.years[0]):
                    audits.append(dict(policy=policy, outcome=outcome, min_pre=min_pre, g=int(panel.years[0]), status='skipped_cohort', reason='already_treated_at_start_no_observed_pre_policy_period', n_initially_treated=int(np.sum(gvec == panel.years[0]))))
                for g in cohorts:
                    b = g - args.anticipation - 1
                    if b - min_pre + 1 < panel.years[0]:
                        audits.append(dict(policy=policy, outcome=outcome, min_pre=min_pre, g=g, status='skipped_cohort', reason='insufficient_pre_calendar'))
                        continue
                    for year in panel.years:
                        if year == b or b < year < g:
                            continue
                        cell, row, balance = build_cell(panel, args, policy, outcome, min_pre, g, int(year), cluster)
                        audits.append(row.copy())
                        balances.extend(balance)
                        if cell is not None:
                            stats, _ = inference(row['att'], cell.cluster_score, multipliers[cluster], cell.treated_clusters, cell.control_clusters, args)
                            suppress_extreme_inference(stats, cell.row.get('extreme_propensity', False))
                            cell.row.update(stats)
                            all_rows.append(add_interpretation(cell.row.copy(), outcome))
                            cells.append(cell)
                post = [c for c in cells if c.row['year'] >= c.row['g']]
                common = {'policy': policy, 'outcome': outcome, 'min_pre': min_pre, 'cluster': cluster, 'sample_mode': args.other_policy_mode}
                if post:
                    agg, score, boot = aggregate(post, multipliers[cluster], args)
                    overall.append(add_interpretation({**common, **agg, 'status': 'estimated'}, outcome))
                    outcome_results[outcome] = (agg, score, boot, post)
                else:
                    overall.append({**common, 'status': 'not_estimable', 'att': np.nan, 'p_value': np.nan, 'se': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'inference_ok': False, 'inference_note': 'no_supported_post_cells'})
                dyn_here = []
                for event in sorted({c.row['event_time'] for c in cells}):
                    cs = [c for c in cells if c.row['event_time'] == event]
                    agg, score, boot = aggregate(cs, multipliers[cluster], args)
                    dyn_here.append(({**common, 'event_time': event, **agg}, score, boot))
                usable = [x for x in dyn_here if x[0]['inference_ok']]
                if usable:
                    max_stat = np.max(np.column_stack([np.abs(x[2] / x[0]['se']) for x in usable]), axis=1)
                    simultaneous = float(np.quantile(max_stat, 0.95))
                else:
                    simultaneous = np.nan
                for row, score, boot in dyn_here:
                    row['simultaneous_ci_low'] = row['att'] - simultaneous * row['se'] if row['inference_ok'] else np.nan
                    row['simultaneous_ci_high'] = row['att'] + simultaneous * row['se'] if row['inference_ok'] else np.nan
                    dynamic.append(add_interpretation(row, outcome))
                pre = [x for x in dyn_here if x[0]['event_time'] < -args.anticipation - 1 and x[0]['inference_ok']]
                if pre:
                    est = np.array([x[0]['att'] for x in pre])
                    draws = np.column_stack([x[2] for x in pre])
                    cov = np.cov(draws, rowvar=False, ddof=1)
                    cov = np.atleast_2d(cov)
                    rank = np.linalg.matrix_rank(cov)
                    inverse = np.linalg.pinv(cov)
                    statistic = float(est @ inverse @ est)
                    bs = np.einsum('bi,ij,bj->b', draws, inverse, draws)
                    pretests.append({**common, 'n_pre_coefficients': len(est), 'covariance_rank': rank, 'wald': statistic, 'p_value': (1 + (bs >= statistic).sum()) / (len(bs) + 1), 'status': 'available' if rank == len(est) else 'rank_deficient_partial_test'})
                else:
                    pretests.append({**common, 'n_pre_coefficients': 0, 'p_value': np.nan, 'status': 'not_testable_no_supported_pre_coefficients'})
                periods, contrasts, lag_audit = run_lag_periods(panel, args, policy, outcome, min_pre, cohorts, cluster, multipliers[cluster])
                period_rows.extend(periods)
                period_contrasts.extend(contrasts)
                audits.extend(lag_audit)
            if 'ECI' in outcome_results and 'CPI' in outcome_results:
                a, sa, ba, ca = outcome_results['ECI']
                b, sb, bb, cb = outcome_results['CPI']
                signature = lambda cc: [(c.row['g'], c.row['year'], c.row['sample_hash']) for c in cc]
                if signature(ca) == signature(cb):
                    tr = set().union(*(c.treated_clusters for c in ca + cb))
                    co = set().union(*(c.control_clusters for c in ca + cb))
                    diff = a['att'] - b['att']
                    inf, _ = inference(diff, sa - sb, multipliers[cluster], tr, co, args)
                    suppress_extreme_inference(inf, a.get('extreme_propensity', False) or b.get('extreme_propensity', False))
                    cov_checks.append({'policy': policy, 'min_pre': min_pre, 'contrast': 'ECI_minus_CPI_log_ATT', 'att': diff, **inf, 'bootstrap_covariance': float(np.cov(ba, bb)[0, 1])})
    tables = {'did_group_time': pd.DataFrame(all_rows), 'did_overall': pd.DataFrame(overall), 'did_event_study': pd.DataFrame(dynamic), 'did_pretrend_tests': pd.DataFrame(pretests), 'did_sample_audit': pd.DataFrame(audits), 'did_covariate_balance': pd.DataFrame(balances), 'did_endpoint_contrast': pd.DataFrame(cov_checks), 'did_period_effects': pd.DataFrame(period_rows), 'did_period_contrasts': pd.DataFrame(period_contrasts)}
    tables['did_overall'] = add_fdr(tables['did_overall'], ['policy', 'min_pre'])
    tables['did_period_effects'] = add_fdr(tables['did_period_effects'], ['policy', 'min_pre'])
    tables['did_period_contrasts'] = add_fdr(tables['did_period_contrasts'], ['policy', 'min_pre'])
    contrast_table = tables['did_period_contrasts']
    contrast_table['q_BH_greater'] = np.nan
    if not contrast_table.empty:
        for _, ix in contrast_table.groupby(['policy', 'min_pre']).groups.items():
            contrast_table.loc[ix, 'q_BH_greater'] = bh(contrast_table.loc[ix, 'p_greater'])
    if not tables['did_overall'].empty:
        pt = tables['did_pretrend_tests'][['policy', 'outcome', 'min_pre', 'p_value', 'status']].rename(columns={'p_value': 'pretrend_p_value', 'status': 'pretrend_status'})
        tables['did_overall'] = tables['did_overall'].merge(pt, on=['policy', 'outcome', 'min_pre'], how='left')
        tables['did_overall']['interpretation_note'] = np.where(tables['did_overall'].pretrend_p_value < 0.05, 'pretrend_rejected_do_not_claim_causal_effect', np.where(tables['did_overall'].pretrend_p_value.isna(), 'pretrend_not_testable_or_no_estimate', 'pretrend_not_rejected_is_not_proof_of_parallel_trends'))
    for name, table in tables.items():
        table = annotate_samples(table, panel)
        tables[name] = table
        save_csv(table, out / f'{name}.csv')
    keys = ['policy', 'outcome', 'min_pre']
    wide = tables['did_overall'].copy()
    for period in PERIODS:
        pf = tables['did_period_effects']
        cols = ['att', 'se', 'ci_low', 'ci_high', 'p_value', 'status', 'inference_ok', 'inference_note', 'extreme_propensity', 'small_sample_note', 'n_treated_counties_unique']
        cols = [c for c in cols if c in pf]
        part = pf.loc[pf.period.eq(period), keys + cols].rename(columns={c: period + '_' + c for c in cols})
        wide = wide.merge(part, on=keys, how='left', validate='one_to_one')
    ad = tables['did_sample_audit']
    post_ad = ad.loc[ad.get('analysis', pd.Series('', index=ad.index)).eq('ordinary') & (pd.to_numeric(ad.get('event_time', pd.Series(np.nan, index=ad.index)), errors='coerce') >= 0)]
    if not post_ad.empty:
        failure = post_ad.loc[post_ad.status.ne('estimated')].groupby(keys)['reason'].agg(lambda s: ';'.join(sorted(set(s.dropna())))).reset_index(name='post_failure_reasons')
        wide = wide.merge(failure, on=keys, how='left', validate='one_to_one')
    save_csv(wide, out / 'did_summary_separate_columns.csv')
    dyn = tables['did_event_study']
    if not dyn.empty:
        save_csv(dyn.loc[dyn.event_time < -args.anticipation - 1], out / 'did_pre_policy_trends.csv')
        save_csv(dyn.loc[dyn.event_time >= 0], out / 'did_post_policy_trends.csv')
    else:
        empty = pd.DataFrame(columns=keys + ['event_time', 'att', 'ci_low', 'ci_high', 'p_value', 'inference_ok'])
        save_csv(empty, out / 'did_pre_policy_trends.csv')
        save_csv(empty, out / 'did_post_policy_trends.csv')
    diagnostic_cols = keys + ['g', 'year', 'event_time', 'analysis', 'common_horizon', 'status', 'reason', 'n_treated_before_policy_screen', 'n_control_before_policy_screen', 'n_treated_after_policy_screen', 'n_control_after_policy_screen']
    diagnostic_cols += [c for c in ad if c.startswith(('initial_', 'final_')) or c in ['extreme_propensity', 'n_design_columns']]
    save_csv(ad[[c for c in diagnostic_cols if c in ad]], out / 'did_propensity_diagnostics.csv')
    panel.ps_stream.close()
    return tables

def run_coverage(panel, args, out):
    results = []
    descriptive = []
    failures = []
    d = panel.data.copy()
    coverage_cov = ['log_PGDP', 'log_POPD', 'GIA_GDP', 'Builtup', 'Coal_Share', 'Electricity_Share']
    for outcome in args.outcomes:
        valid = np.isfinite(d[[f'Y_{outcome}'] + coverage_cov].to_numpy(float)).all(axis=1)
        q = d.loc[valid].copy()
        for policy in POLICIES:
            for year, group in q.groupby('Year'):
                for covered in [0, 1]:
                    sub = group[group[policy].eq(covered)]
                    descriptive.append(dict(policy=policy, outcome=outcome, year=int(year), covered=covered, n_counties=int(sub.Code.nunique()), n_rows=len(sub), mean_analysis_outcome=sub[f'Y_{outcome}'].mean(), mean_raw_outcome=sub[outcome].mean() if outcome != 'JointImprovement' else sub.Y_JointImprovement.mean()))
        fe = pd.get_dummies(q[['Year']].astype(str), drop_first=True, dtype=float)
        controls = q[coverage_cov].copy()
        controls = (controls - controls.mean()) / controls.std(ddof=0).replace(0, 1)
        frame = pd.concat([q[POLICIES].astype(float), controls, fe], axis=1)
        frame.insert(0, 'Intercept', 1.0)
        x = frame.to_numpy(float)
        y = q[f'Y_{outcome}'].to_numpy(float)
        _, s, vh = np.linalg.svd(x, full_matrices=False)
        rank = int(np.sum(s > max(x.shape) * s[0] * np.finfo(float).eps))
        null = vh[rank:]
        estimable = np.ones(x.shape[1], bool) if rank == x.shape[1] else np.linalg.norm(null, axis=0) < 1e-08
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = y - x @ coef
        cluster_codes, cluster_names = pd.factorize(q[args.coverage_cluster], sort=True)
        ncl = len(cluster_names)
        meat = np.zeros((ncl, x.shape[1]))
        np.add.at(meat, cluster_codes, x * residual[:, None])
        bread = np.linalg.pinv(x.T @ x)
        correction = ncl / (ncl - 1) * (len(y) - 1) / (len(y) - rank) if ncl > 1 and len(y) > rank else np.nan
        vcov = bread @ meat.T @ meat @ bread * correction
        se = np.sqrt(np.maximum(0, np.diag(vcov)))
        for policy in POLICIES:
            j = frame.columns.get_loc(policy)
            exposed = q.groupby(args.coverage_cluster)[policy].max()
            nt = int((exposed == 1).sum())
            nc = int(q.loc[q[policy].eq(0), args.coverage_cluster].nunique())
            infer = bool(estimable[j] and se[j] > 1e-14 and (ncl >= args.min_inference_clusters) and (nt >= args.min_inference_clusters) and (nc >= args.min_inference_clusters))
            row = {'policy': policy, 'outcome': outcome, 'association': float(coef[j]) if estimable[j] else np.nan, 'se': float(se[j]) if estimable[j] else np.nan, 'n_rows': len(q), 'n_counties': int(q.Code.nunique()), 'cluster': args.coverage_cluster, 'n_clusters': ncl, 'n_exposed_clusters': nt, 'n_unexposed_row_clusters': nc, 'inference_ok': infer, 'status': 'estimated' if estimable[j] else 'not_identifiable_collinearity', 'model': 'pooled_OLS_or_LPM_10_policies_plus_contemporaneous_covariates_year_FE_only', 'interpretation': 'conditional_association_not_causal_effect', 'p_value': np.nan, 'ci_low': np.nan, 'ci_high': np.nan}
            if infer:
                critical = student_t.ppf(0.975, ncl - 1)
                row.update(p_value=float(2 * student_t.sf(abs(coef[j] / se[j]), ncl - 1)), ci_low=float(coef[j] - critical * se[j]), ci_high=float(coef[j] + critical * se[j]))
            if outcome == 'JointImprovement':
                row['association_percentage_points'] = 100 * row['association']
            results.append(row)
        if outcome == 'JointImprovement':
            pred = x @ coef
            failures.append({'outcome': outcome, 'lpm_prediction_outside_0_1_fraction': float(np.mean((pred < 0) | (pred > 1))), 'note': 'LPM coefficient is a conditional probability difference, not logistic odds'})
    result = add_fdr(pd.DataFrame(results), ['policy'])
    save_csv(result, out / 'coverage_conditional_associations.csv')
    save_csv(descriptive, out / 'coverage_descriptive_by_year.csv')
    save_csv(failures, out / 'coverage_diagnostics.csv')
    return result

def plot_results(tables, coverage, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 220})
    figdir = out / 'figures'
    figdir.mkdir(exist_ok=True)
    event = tables.get('did_event_study', pd.DataFrame())
    if not event.empty:
        for (policy, minimum), q in event.groupby(['policy', 'min_pre']):
            fig, axs = plt.subplots(2, 3, figsize=(13, 7))
            for ax, outcome in zip(axs.ravel(), OUTCOMES):
                sub = q[q.outcome.eq(outcome)].sort_values('event_time')
                scale = 100 if outcome == 'JointImprovement' else 1
                ax.axhline(0, color='gray', lw=0.8)
                ax.axvline(0, color='gray', ls=':', lw=0.8)
                if not sub.empty:
                    grid = sub.set_index('event_time').reindex(range(int(sub.event_time.min()), int(sub.event_time.max()) + 1))
                    ax.plot(grid.index, grid.att * scale, 'o-', color='#2166ac', ms=3)
                    ax.fill_between(grid.index.to_numpy(), grid.simultaneous_ci_low.to_numpy() * scale, grid.simultaneous_ci_high.to_numpy() * scale, color='#2166ac', alpha=0.18)
                    baselines = tables['did_group_time'].query('policy == @policy and outcome == @outcome and min_pre == @minimum')
                    if not baselines.empty:
                        reference = int((baselines.baseline - baselines.g).iloc[0])
                        ax.scatter([reference], [0], s=28, facecolors='white', edgecolors='gray', zorder=4)
                    good = sub.ci_low.notna()
                    ax.vlines(sub.loc[good, 'event_time'], sub.loc[good, 'ci_low'] * scale, sub.loc[good, 'ci_high'] * scale, color='#2166ac', lw=1)
                else:
                    ax.text(0.5, 0.5, 'Not estimable', ha='center', transform=ax.transAxes)
                ax.set_title(outcome)
                ax.set_xlabel('Years since implementation')
                ax.set_ylabel('Percentage points' if outcome == 'JointImprovement' else 'ATT: negative log level')
            axs.ravel()[-1].axis('off')
            axs.ravel()[-1].text(0, 0.9, 'Dots: ATT (may be exploratory)\nHollow dot: reference period (= 0)\nBars: pointwise 95% CI\nShade: simultaneous 95% band\nNo CI: extreme propensity or\ninsufficient inference support\nCohort composition may vary', va='top')
            fig.suptitle(f'{policy}: minimum {minimum} pre-outcome year(s)')
            fig.tight_layout()
            fig.savefig(figdir / f'DID_{policy}_pre{minimum}.png')
            plt.close(fig)
    sources = []
    if not tables.get('did_overall', pd.DataFrame()).empty:
        for minimum, q in tables['did_overall'].groupby('min_pre'):
            sources.append((f'DID_overall_pre{minimum}', q, 'att'))
    if coverage is not None and (not coverage.empty):
        sources.append(('Coverage_association_NONCAUSAL', coverage, 'association'))
    for name, q, col in sources:
        fig, axs = plt.subplots(1, 5, figsize=(18, 6), sharey=True)
        for ax, outcome in zip(axs, OUTCOMES):
            sub = q[q.outcome.eq(outcome)].set_index('policy').reindex(POLICIES)
            scale = 100 if outcome == 'JointImprovement' else 1
            pos = np.arange(len(POLICIES))
            ax.axvline(0, color='gray', lw=0.8)
            ax.scatter(sub[col] * scale, pos, color='#2166ac', s=24)
            ax.hlines(pos, sub.ci_low * scale, sub.ci_high * scale, color='#2166ac')
            ax.set_title(outcome)
            ax.set_yticks(pos, POLICIES)
            ax.set_xlabel('Percentage points' if outcome == 'JointImprovement' else 'Negative log scale')
        axs[0].invert_yaxis()
        fig.suptitle(name + ' (missing CI = insufficient inference support)')
        fig.tight_layout()
        fig.savefig(figdir / f'{name}.png')
        plt.close(fig)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-file', type=Path, default=DEFAULT_DIR / 'policy_conditions.csv')
    parser.add_argument('--model-file', type=Path, default=DEFAULT_DIR / 'Model_Data_Total.csv')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--min-pre', type=int, nargs='+', default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument('--policies', nargs='+', choices=POLICIES, default=POLICIES)
    parser.add_argument('--outcomes', nargs='+', choices=OUTCOMES, default=OUTCOMES)
    parser.add_argument('--bootstrap', type=int, default=999)
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--anticipation', type=int, default=0, choices=[0, 1])
    parser.add_argument('--control-group', choices=['notyet', 'never'], default='notyet')
    parser.set_defaults(other_policy_mode='clean')
    parser.add_argument('--state-definition', choices=['annual', 'cumulative2013'], default='annual')
    parser.add_argument('--state-epsilon', type=float, default=0.0)
    parser.add_argument('--ps-trim', type=float, default=0.0, help='Default keeps finite extreme predictions for exploratory ATT; --ps-trim .01 restores overlap trimming')
    parser.add_argument('--min-treated', type=int, default=1)
    parser.add_argument('--min-controls', type=int, default=20)
    parser.add_argument('--min-stratum-controls', type=int, default=5, help='Legacy compatibility only; no exact history strata in this version')
    parser.add_argument('--min-inference-clusters', type=int, default=5)
    parser.add_argument('--cluster-override', choices=['Code'])
    parser.add_argument('--coverage-cluster', choices=['Code'], default='Code')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--skip-did', action='store_true')
    parser.add_argument('--skip-coverage', action='store_true')
    args = parser.parse_args(argv)
    if args.bootstrap < 99 or not 0 <= args.ps_trim < 0.25:
        parser.error('Use at least 99 multiplier draws and 0 <= ps-trim < .25')
    if args.state_epsilon < 0 or min(args.min_treated, args.min_controls, args.min_stratum_controls) < 1:
        parser.error('Invalid threshold')
    if args.min_inference_clusters < 5:
        parser.error('Inference requires at least 5 clusters per arm; singleton ATT is exploratory only')
    out = args.out / f'run_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['policy_cluster_defaults'] = POLICY_CLUSTERS
    config['inference_assumption'] = 'Independent counties; arbitrary serial dependence within Code'
    config['coverage_model'] = 'Pooled OLS/LPM; year FE only; no county FE; noncausal'
    config['matching_rule'] = 'No exact entry-year or baseline-portfolio matching; other-policy stability within comparison window retained'
    config['background_adjustment'] = 'Other-policy baseline status, observed exposed years, initial coverage; all at pre-treatment baseline'
    config['extreme_prediction_policy'] = 'Finite numerically identifiable ATT retained; extreme-propensity inference suppressed; singular/nonconverged models remain unavailable with last-iterate diagnostics'
    config['small_sample_policy'] = 'Treated minimum 1, controls 20; below 20 treated flagged; inference minimum 5 per arm is not a guarantee of reliability'
    config['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config['input_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.policy_file, args.model_file]}
    (out / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    panel = Panel(args)
    (out / 'input_audit.json').write_text(json.dumps(panel.audit, ensure_ascii=False, indent=2), encoding='utf-8')
    cohorts = []
    for j, policy in enumerate(POLICIES):
        for g in np.unique(panel.g[:, j]):
            mask = panel.g[:, j] == g
            cohorts.append({'policy': policy, 'first_observed_treated': int(g) if np.isfinite(g) else 'never', 'already_treated_at_start': bool(g == panel.years[0]), 'n_counties': int(mask.sum()), 'pre_calendar_years': int(g - panel.years[0]) if np.isfinite(g) else np.nan, 'post_calendar_years_including_g': int(panel.years[-1] - g + 1) if np.isfinite(g) else np.nan})
    save_csv(cohorts, out / 'policy_cohorts.csv')
    save_csv(panel.data[['Code', 'Year'] + [f'Y_{o}' for o in OUTCOMES]], out / 'analysis_outcomes.csv')
    try:
        tables = {} if args.skip_did else run_did(panel, args, out)
    finally:
        if hasattr(panel, 'ps_stream'):
            panel.ps_stream.close()
    coverage = None if args.skip_coverage else run_coverage(panel, args, out)
    if not args.no_plots:
        plot_results(tables, coverage, out)
    summary = {'output_directory': str(out), 'DID_summary_rows': len(tables.get('did_overall', [])), 'DID_period_rows': len(tables.get('did_period_effects', [])), 'DID_period_contrast_rows': len(tables.get('did_period_contrasts', [])), 'coverage_rows': len(coverage) if coverage is not None else 0, 'notes': ['No automatic switching to TWFE or a different estimator on failure.', 'Relaxed background regression adjustment replaces exact entry-year matching.', 'Default ps-trim 0 retains numerically estimable extreme ATT, but does not validate its inference.', 'Extreme ATT and failed-fit last-iterate diagnostics are separately flagged; unavailable inference is not fabricated.', 'Non-estimable policies and cells remain in audit/summary files.', 'Coverage estimates are associations, including initially treated counties.', 'Code-only inference assumes cross-county independence; not empirically established.', 'Coverage now uses year FE only; coefficients can change from the earlier specification.', 'Annual joint improvement is undefined in first observed year.', 'Minimum pre-period is a gate; all supported pre/post years are examined.', 'Calendar support, propensity overlap, and pretrend evidence are separate checks.']}
    (out / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    pass
    return out
if __name__ == '__main__':
    main()
