from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.special import expit
from scipy import stats
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
POLICY_FILE = BASE_DIR / 'policy_conditions.csv'
MODEL_FILE = BASE_DIR / 'model_data_total.csv'
OUTPUT_ROOT = OUTPUT_DIR / 'policy_boundaries'
POLICIES = ['CETP', 'LCCP', 'ECERFP', 'ZWC', 'CEP', 'CEPDU', 'CACP', 'SCP', 'LECCDC', 'NGFRIPZ']
MECHANISMS = ['PGDP', 'POPD', 'VAF_GDP', 'GIA_GDP', 'TSS_GDP', 'Mean_Light', 'Builtup', 'Road_Density', 'PIU_PIR_Ratio', 'Fiscal_Revenue_GDP', 'Energy_Intensity', 'Coal_Share', 'Electricity_Share', 'Fossil_Energy_Share', 'Energy_Diversity_Index_Norm', 'Industrial_Coal_Share', 'TPSP_Petroleum_Share', 'Forest', 'Grassland', 'Shrub', 'Water', 'Wetland', 'Annual NDVI', 'Summer NDVI', 'AMEVT', 'LAI_HV', 'LAI_LV', 'BG_AI', 'BG_Divsion', 'EEI_GDP', 'EES_GEF', 'GPAI', 'GIPAI', 'GIPGS']
ENV_OUTCOMES = ['ECI', 'CPI', 'O3', 'PM10']
RAW_COVARIATES = ['PGDP', 'POPD', 'GIA_GDP', 'Builtup', 'Coal_Share', 'Electricity_Share']
COVARIATES = ['log_PGDP', 'log_POPD', 'GIA_GDP', 'Builtup', 'Coal_Share', 'Electricity_Share', 'log_ECI', 'log_CPI']
MIN_PRE = [1, 2, 3]
MAX_POST = 5
PRE_LAGS = 3
BOOTSTRAP = 999
SEED = 2026
MIN_TREATED = 1
MIN_CONTROL = 1
MIN_INFERENCE_PER_ARM = 4
SMALL_SAMPLE_WARNING = 20
LAGS = [1, 2]
PS_CLIP = 1e-06
PS_EXTREME_CUTOFF = 0.01
LOGIT_COND_LIMIT = 1000000000000.0
OLS_COND_LIMIT = 1000000000000.0
RIDGE_GRID = (1e-08, 1e-06, 0.0001, 0.01, 0.1, 1.0)
PERIODS = {'year0': (0, 0), 'years1_2': (1, 2), 'years3_5': (3, 5)}

def read_csv(path):
    for enc in ('utf-8-sig', 'gb18030'):
        try:
            return pd.read_csv(path, encoding=enc, dtype={'Code': 'string'})
        except UnicodeDecodeError:
            pass
    raise ValueError('Cannot decode CSV: ' + str(path))

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('Unsupported JSON value: ' + type(value).__name__)

def bh(values):
    p = np.asarray(values, float)
    q = np.full(p.shape, np.nan)
    ix = np.flatnonzero(np.isfinite(p))
    if len(ix):
        order = ix[np.argsort(p[ix])]
        q[order] = np.minimum(1.0, np.minimum.accumulate((p[order] * len(ix) / np.arange(1, len(ix) + 1))[::-1])[::-1])
    return q

def add_q(frame, keys, pcol='p_value', qcol='q_BH'):
    frame = frame.copy()
    if frame.empty or pcol not in frame:
        return frame
    frame[qcol] = np.nan
    if keys:
        for ix in frame.groupby(keys, dropna=False, sort=False).groups.values():
            frame.loc[ix, qcol] = bh(frame.loc[ix, pcol])
    else:
        frame[qcol] = bh(frame[pcol])
    return frame

def write_csv(rows, path):
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df.empty and len(df.columns) == 0:
        df = pd.DataFrame(columns=['status', 'reason'])
    df.to_csv(path, index=False, encoding='utf-8-sig')
    return df

def load_panel(args):
    p, d = (read_csv(args.policy_file), read_csv(args.model_file))
    for name, df in [('policy', p), ('model', d)]:
        df.columns = df.columns.str.strip()
        if not {'Code', 'Year'}.issubset(df):
            raise ValueError(name + ': missing Code/Year')
        df['Code'] = df['Code'].str.strip().str.replace('\\.0+$', '', regex=True)
        if df.Code.isna().any() or df.Code.eq('').any():
            raise ValueError(name + ': invalid Code')
        yy = pd.to_numeric(df.Year, errors='raise')
        if not np.isfinite(yy).all() or not yy.eq(np.floor(yy)).all():
            raise ValueError(name + ': noninteger Year')
        df['Year'] = yy.astype(int)
        if df.duplicated(['Code', 'Year']).any():
            raise ValueError(name + ': duplicate Code-Year keys')
    if not set(POLICIES).issubset(p):
        raise ValueError('Missing required policy columns')
    if not set(RAW_COVARIATES + ENV_OUTCOMES).issubset(d):
        raise ValueError('Missing core covariate/environment columns')
    for wanted in args.mechanisms:
        if wanted not in d:
            candidates = [wanted.replace(' ', '_')]
            if wanted == 'BG_Divsion':
                candidates.append('BG_Division')
            found = [x for x in candidates if x in d and x != wanted]
            if len(found) == 1:
                d = d.rename(columns={found[0]: wanted})
    missing_columns = [x for x in args.mechanisms if x not in d]
    for col in missing_columns:
        d[col] = np.nan
    cols = list(dict.fromkeys(RAW_COVARIATES + ENV_OUTCOMES + args.mechanisms))
    keys = p[['Code', 'Year']].merge(d[['Code', 'Year']], how='outer', indicator=True)
    if not keys['_merge'].eq('both').all():
        raise ValueError('Input Code-Year keys do not agree; reconcile before estimation')
    df = d[['Code', 'Year'] + cols].merge(p[['Code', 'Year'] + POLICIES], on=['Code', 'Year'], validate='one_to_one')
    df = df.sort_values(['Code', 'Year']).reset_index(drop=True)
    for col in cols + POLICIES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[~np.isfinite(df[col]), col] = np.nan
    if not df[POLICIES].isin([0, 1]).all().all():
        raise ValueError('Policies must be complete 0/1 indicators')
    ids, years = (np.sort(df.Code.unique()), np.sort(df.Year.unique()))
    if len(df) != len(ids) * len(years) or not np.array_equal(years, np.arange(years[0], years[-1] + 1)):
        raise ValueError('Consecutive balanced Code-Year panel required')

    def pivot(col):
        return df.pivot(index='Code', columns='Year', values=col).loc[ids, years].to_numpy(float)
    policies = np.stack([pivot(x) for x in POLICIES], axis=2)
    if (np.diff(policies, axis=1) < 0).any():
        raise ValueError('Policy reversals require a different design')
    for name in ['PGDP', 'POPD', 'ECI', 'CPI']:
        df['log_' + name] = np.log(df[name].where(df[name] > 0))
    cov = np.stack([pivot(x) for x in COVARIATES], axis=2)
    mechanisms = {m: pivot(m) for m in args.mechanisms}
    outcomes = {e: -np.log(pivot(e), where=pivot(e) > 0, out=np.full((len(ids), len(years)), np.nan)) for e in ENV_OUTCOMES}
    audit = []
    for m, y in mechanisms.items():
        means = np.nanmean(y, axis=1, keepdims=True) if np.isfinite(y).any() else np.zeros((len(ids), 1))
        audit.append({'mechanism': m, 'missing_column': m in missing_columns, 'n_nonmissing': int(np.isfinite(y).sum()), 'n_missing': int((~np.isfinite(y)).sum()), 'n_zero': int((y == 0).sum()), 'n_negative': int((y < 0).sum()), 'first_year_SD': float(np.nanstd(y[:, 0], ddof=1)) if np.isfinite(y[:, 0]).sum() > 1 else np.nan, 'within_county_SD': float(np.nanstd(y - means)) if np.isfinite(y).any() else np.nan, 'transform': 'raw', 'positive_ATT_means': 'indicator_increase_not_automatically_improvement'})
    return {'df': df, 'ids': ids, 'years': years, 'policies': policies, 'count': policies.sum(axis=2).astype(int), 'bits': (policies * 2 ** np.arange(len(POLICIES))).sum(axis=2).astype(int), 'cov': cov, 'mechanisms': mechanisms, 'outcomes': outcomes, 'audit': pd.DataFrame(audit)}

def impute_covariates(raw: np.ndarray, names: list[str]):
    raw = np.asarray(raw, dtype=float)
    columns = []
    output_names = []
    diagnostics = []
    for j, name in enumerate(names):
        values = raw[:, j].copy()
        missing = ~np.isfinite(values)
        observed = values[~missing]
        if observed.size == 0:
            diagnostics.append({'covariate': name, 'n_missing': int(missing.sum()), 'imputation': 'dropped_all_missing_in_risk_set'})
            continue
        median = float(np.median(observed))
        values[missing] = median
        columns.append(values)
        output_names.append(name)
        diagnostics.append({'covariate': name, 'n_missing': int(missing.sum()), 'imputation': 'median_within_group_time_risk_set' if missing.any() else 'none', 'imputation_value': median if missing.any() else np.nan})
        if missing.any():
            columns.append(missing.astype(float))
            output_names.append(f'missing_{name}')
    if not columns:
        return (np.empty((len(raw), 0), dtype=float), [], diagnostics)
    return (np.column_stack(columns), output_names, diagnostics)

def full_rank_design(raw: np.ndarray, names: list[str]):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[0] == 0:
        raise ValueError('empty_design')
    if raw.shape[1] == 0:
        return (np.ones((len(raw), 1), dtype=float), ['Intercept'])
    if not np.isfinite(raw).all():
        raise ValueError('nonfinite_design_after_imputation')
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    keep = scale > 1e-10
    z = (raw[:, keep] - mean[keep]) / scale[keep]
    kept_names = np.asarray(names, dtype=object)[keep].tolist()
    if z.shape[1]:
        _, rr, piv = qr(z, mode='economic', pivoting=True)
        diag = np.abs(np.diag(rr))
        tolerance = max(z.shape) * np.finfo(float).eps * (diag.max() if diag.size else 1.0)
        rank = int(np.sum(diag > tolerance))
        selected = sorted(piv[:rank])
        z = z[:, selected]
        kept_names = [kept_names[j] for j in selected]
    return (np.column_stack([np.ones(len(raw)), z]), ['Intercept'] + kept_names)

def fit_logit_unpenalized(x: np.ndarray, treatment: np.ndarray):
    n, k = x.shape
    d = np.asarray(treatment, dtype=float)
    share = d.mean()
    if not 0 < share < 1:
        raise ValueError('propensity_requires_both_groups')
    beta = np.zeros(k, dtype=float)
    beta[0] = np.log(share / (1.0 - share))
    for _ in range(150):
        eta = x @ beta
        p = expit(eta)
        score = x.T @ (d - p) / n
        info = x.T * (p * (1.0 - p)) @ x / n
        if not np.isfinite(info).all() or np.linalg.cond(info) > LOGIT_COND_LIMIT:
            raise ValueError('propensity_separation_or_ill_conditioned')
        if np.max(np.abs(score)) < 1e-08:
            if np.max(np.abs(eta)) > 30:
                raise ValueError('propensity_extreme_linear_predictor')
            return (beta, p, info, 0.0)
        step = np.linalg.solve(info, score)
        old = np.mean(d * eta - np.logaddexp(0.0, eta))
        length = 1.0
        accepted = False
        while length > 1e-10:
            trial = beta + length * step
            trial_eta = x @ trial
            objective = np.mean(d * trial_eta - np.logaddexp(0.0, trial_eta))
            if objective >= old - 1e-12:
                beta = trial
                accepted = True
                break
            length *= 0.5
        if not accepted:
            raise ValueError('propensity_line_search_failed')
    raise ValueError('propensity_not_converged')

def fit_logit_ridge(x: np.ndarray, treatment: np.ndarray):
    n, k = x.shape
    d = np.asarray(treatment, dtype=float)
    share = d.mean()
    if not 0 < share < 1:
        raise ValueError('propensity_requires_both_groups')
    penalty = np.eye(k)
    penalty[0, 0] = 0.0
    last_error = None
    for lam in RIDGE_GRID:
        beta = np.zeros(k, dtype=float)
        beta[0] = np.log(share / (1.0 - share))
        try:
            for _ in range(250):
                eta = x @ beta
                p = expit(eta)
                score = x.T @ (d - p) / n - lam * (penalty @ beta)
                info = x.T * (p * (1.0 - p)) @ x / n + lam * penalty
                if not np.isfinite(info).all():
                    raise ValueError('ridge_propensity_nonfinite_information')
                if np.max(np.abs(score)) < 1e-08:
                    return (beta, p, info, float(lam))
                step = np.linalg.solve(info, score)
                old = np.mean(d * eta - np.logaddexp(0.0, eta)) - 0.5 * lam * float(beta @ penalty @ beta)
                length = 1.0
                accepted = False
                while length > 1e-10:
                    trial = beta + length * step
                    trial_eta = x @ trial
                    objective = np.mean(d * trial_eta - np.logaddexp(0.0, trial_eta)) - 0.5 * lam * float(trial @ penalty @ trial)
                    if objective >= old - 1e-12:
                        beta = trial
                        accepted = True
                        break
                    length *= 0.5
                if not accepted:
                    raise ValueError('ridge_propensity_line_search_failed')
            raise ValueError('ridge_propensity_not_converged')
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            last_error = str(exc)
            continue
    raise ValueError('ridge_propensity_failed:' + str(last_error))

def fit_propensity(x: np.ndarray, treatment: np.ndarray):
    try:
        beta, p, info, lam = fit_logit_unpenalized(x, treatment)
        if p.min() < PS_CLIP or p.max() > 1.0 - PS_CLIP:
            raise ValueError('propensity_extreme_probability_use_ridge_stabilization')
        method = 'unpenalized_logit'
        primary_error = ''
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        primary_error = str(exc)
        beta, p, info, lam = fit_logit_ridge(x, treatment)
        method = 'ridge_logit_fallback'
    return {'beta': beta, 'ps': p, 'information': info, 'ridge_lambda': lam, 'method': method, 'primary_error': primary_error}

def fit_control_outcome_regression(x: np.ndarray, treatment: np.ndarray, y: np.ndarray):
    n, k = x.shape
    d = np.asarray(treatment, dtype=float)
    control = 1.0 - d
    penalty = np.eye(k)
    penalty[0, 0] = 0.0
    gram0 = x.T * control @ x / n
    rhs = x.T @ (control[:, None] * y) / n
    if np.linalg.matrix_rank(gram0) == k and np.isfinite(gram0).all() and (np.linalg.cond(gram0) <= OLS_COND_LIMIT):
        gamma = np.linalg.solve(gram0, rhs)
        return {'gamma': gamma, 'gram': gram0, 'ridge_lambda': 0.0, 'method': 'control_OLS', 'primary_error': ''}
    primary_error = 'outcome_regression_rank_deficient_or_ill_conditioned'
    last_error = None
    for lam in RIDGE_GRID:
        gram = gram0 + lam * penalty
        try:
            gamma = np.linalg.solve(gram, rhs)
            if np.isfinite(gamma).all():
                return {'gamma': gamma, 'gram': gram, 'ridge_lambda': float(lam), 'method': 'control_ridge_fallback', 'primary_error': primary_error}
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            last_error = str(exc)
    raise ValueError('outcome_ridge_failed:' + str(last_error))

@dataclass
class Design:
    id: str
    kind: str
    event: np.ndarray
    focal: int = -1
    before: int = -1
    after: int = -1

def make_designs(panel, families):
    p, count = (panel['policies'], panel['count'])
    designs = []
    if 'single' in families:
        for j, name in enumerate(POLICIES):
            event = np.zeros(count.shape, bool)
            event[:, 1:] = np.diff(p[:, :, j], axis=1) == 1
            designs.append(Design(name, 'single', event, j))
    if 'count' in families:
        change = count[:, 1:] > count[:, :-1]
        pairs = sorted(set(zip(count[:, :-1][change], count[:, 1:][change])))
        for before, after in pairs:
            event = np.zeros(count.shape, bool)
            event[:, 1:] = (count[:, :-1] == before) & (count[:, 1:] == after)
            designs.append(Design(f'count_{before}_to_{after}', 'count', event, before=int(before), after=int(after)))
    for des in designs:
        if (des.event.sum(axis=1) > 1).any():
            raise ValueError('Repeated transition unsupported: ' + des.id)
    return designs

def policy_masks(panel, design, g, t, min_pre, rule, horizon=None):
    p, bits = (panel['policies'], panel['bits'])
    post = t >= g
    b = g - 1 if post else t - 1
    if b < min_pre - 1:
        return None
    end = g + horizon if horizon is not None else max(t, g - 1)
    if end >= len(panel['years']):
        return None
    tr = design.event[:, g].copy()
    if design.kind == 'single':
        co = p[:, max(g, end), design.focal] == 0
        other = [j for j in range(len(POLICIES)) if j != design.focal]
        if rule == 'stable':
            stable = np.all(p[:, b, other] == p[:, end, other], axis=1)
            tr &= stable
            co &= stable
    else:
        co = (panel['count'][:, g - 1] == design.before) & (bits[:, b] == bits[:, end]) & ~design.event[:, g]
        if post:
            tr &= bits[:, end] == bits[:, g]
            tr &= bits[:, b] == bits[:, g - 1]
        else:
            tr &= bits[:, b] == bits[:, g - 1]
    return (b, end, tr, co)

def dr_batch(x, d, y, fit_mode='stabilized'):
    n = len(d)
    if d.sum() < MIN_TREATED or (1 - d).sum() < MIN_CONTROL:
        raise ValueError('insufficient_treated_or_controls')
    if fit_mode == 'ordinary':
        beta, p, info, lam = fit_logit_unpenalized(x, d)
        propensity = {'beta': beta, 'ps': p, 'information': info, 'ridge_lambda': lam, 'method': 'unpenalized_logit', 'primary_error': ''}
    else:
        propensity = fit_propensity(x, d)
    out = fit_control_outcome_regression(x, d, y)
    if fit_mode == 'ordinary' and out['ridge_lambda'] > 0:
        raise ValueError('control_OLS_rank_deficient_or_ill_conditioned')
    p = propensity['ps']
    safe = np.clip(p, PS_CLIP, 1 - PS_CLIP)
    if fit_mode == 'ordinary' and np.any(p != safe):
        raise ValueError('ordinary_mode_numerically_unusable_propensity')
    c = (1 - d) * safe / (1 - safe)
    if not np.isfinite(c).all() or c.sum() <= 1e-20:
        raise ValueError('invalid_control_weight_mass')
    c /= c.sum()
    a = d / d.sum()
    r = y - x @ out['gamma']
    mt, mc = (a @ r, c @ r)
    att = mt - mc
    influence = (n * a)[:, None] * (r - mt) - (n * c)[:, None] * (r - mc)
    active_derivative = (p > PS_CLIP) & (p < 1 - PS_CLIP)
    db = -x.T @ ((c * active_derivative)[:, None] * (r - mc))
    dg = -(a - c) @ x
    ib = x * (d - p)[:, None] @ np.linalg.pinv(propensity['information'])
    influence += ib @ db
    influence += ((1 - d) * (x @ np.linalg.pinv(out['gram']) @ dg))[:, None] * r
    influence -= influence.mean(axis=0)
    if not np.isfinite(att).all() or not np.isfinite(influence).all():
        raise ValueError('nonfinite_estimate_or_influence')
    return {'att': att, 'IF': influence, 'p': p, 'safe_p': safe, 'weights': c, 'propensity': propensity, 'outcome_fit': out}

def empty_result(reason):
    return {'status': 'not_estimable', 'reason': reason, 'att': np.nan, 'se': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'p_value': np.nan, 'inference_ok': False, 'n_treated': 0, 'n_control': 0}

def infer(att, score, multiplier, tr, co):
    active = len(tr | co)
    factor = active / (active - 1) if active > 1 else np.nan
    variance = float(score @ score) * factor
    ans = {'se': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'p_value': np.nan, 'inference_ok': False, 'inference_reason': '', 'score_variance': variance}
    if min(len(tr), len(co)) < MIN_INFERENCE_PER_ARM:
        ans['inference_reason'] = 'fewer_than_minimum_Code_clusters_in_one_arm'
        return ans
    if not np.isfinite(variance) or variance <= 1e-24:
        ans['inference_reason'] = 'degenerate_or_nonfinite_variance'
        return ans
    se = np.sqrt(variance)
    draws = multiplier @ score * np.sqrt(factor)
    critical = float(np.quantile(np.abs(draws / se), 0.95))
    ans.update(se=se, ci_low=att - critical * se, ci_high=att + critical * se, p_value=float((1 + (np.abs(draws) >= abs(att)).sum()) / (len(draws) + 1)), inference_ok=True)
    return ans

def aggregate(cells, multiplier, cache):
    if not cells:
        return (empty_result('no_valid_group_time_cells'), None, set(), set())
    signature = tuple((id(c) for c in cells))
    if signature in cache:
        return cache[signature]
    groups = {}
    for c in cells:
        groups.setdefault(c['g'], []).append(c)
    parts = list(groups.values())
    w = np.array([np.mean([c['nt'] for c in part]) for part in parts], float)
    w /= w.sum()
    att = float(sum((wi * np.mean([c['att'] for c in part]) for wi, part in zip(w, parts))))
    score = sum((wi * np.mean([c['score'] for c in part], axis=0) for wi, part in zip(w, parts)))
    tr = set().union(*(c['treated'] for c in cells))
    co = set().union(*(c['control'] for c in cells))
    row = {'status': 'estimated', 'reason': '', 'att': att, **infer(att, score, multiplier, tr, co), 'n_treated': len(tr), 'n_control': len(co), 'n_cohorts': len(parts), 'n_group_time_cells': len(cells), 'min_cell_treated': min((c['nt'] for c in cells)), 'min_cell_control': min((c['nc'] for c in cells)), 'min_control_ESS': min((c['ess'] for c in cells)), 'ridge_cell_count': sum((c['ridge'] for c in cells)), 'quality_flags': ';'.join(sorted(set((f for c in cells for f in c['flags']))))}
    cache[signature] = (row, score, tr, co)
    return cache[signature]

def pretrend_test(pre, multiplier, cache):
    events = sorted(set((c['event'] for c in pre)))
    agg = [aggregate([c for c in pre if c['event'] == e], multiplier, cache) for e in events]
    usable = [a for a in agg if a[0]['inference_ok']]
    row = {'pretrend_p': np.nan, 'pretrend_status': 'unavailable', 'n_inferable_pre_events': len(usable)}
    if len(usable) < 2:
        row['pretrend_reason'] = 'fewer_than_two_inferable_adjacent_year_placebos'
        return row
    score = np.column_stack([a[1] for a in usable])
    theta = np.array([a[0]['att'] for a in usable])
    active = set().union(*(a[2] | a[3] for a in usable))
    factor = len(active) / (len(active) - 1)
    cov = score.T @ score * factor
    if np.linalg.matrix_rank(cov) != len(theta):
        row['pretrend_reason'] = 'singular_joint_covariance'
        return row
    inv = np.linalg.inv(cov)
    stat = float(theta @ inv @ theta)
    draws = multiplier @ score * np.sqrt(factor)
    simulated = np.einsum('bi,ij,bj->b', draws, inv, draws)
    row.update(pretrend_p=float((1 + (simulated >= stat).sum()) / (len(draws) + 1)), pretrend_status='estimated', pretrend_reason='', pretrend_statistic=stat)
    return row
STRATIFIERS = ['Coal_Share', 'Energy_Intensity', 'GIA_GDP', 'PGDP', 'ECI', 'CPI', 'Forest', 'Grassland', 'Water', 'Wetland', 'Annual NDVI', 'Builtup']
REGION_PREFIXES = {'East': ['11', '12', '13', '31', '32', '33', '35', '37', '44', '46'], 'Central': ['14', '34', '36', '41', '42', '43'], 'West': ['15', '45', '50', '51', '52', '53', '54', '61', '62', '63', '64', '65'], 'Northeast': ['21', '22', '23']}
from itertools import combinations

def split_median(values):
    values = np.asarray(values, float)
    good = np.isfinite(values)
    cut = float(np.median(values[good])) if good.any() else np.nan
    labels = np.full(len(values), '', dtype=object)
    labels[good & (values <= cut)] = 'Low'
    labels[good & (values > cut)] = 'High'
    return (labels, cut)

def build_strata(panel, args):
    year = args.baseline_year if args.baseline_year is not None else int(panel['years'][0])
    if year not in panel['years']:
        raise ValueError('baseline-year must exist in panel')
    baseline = panel['df'].loc[panel['df'].Year.eq(year)].set_index('Code').reindex(panel['ids'])
    strata, inventory, assignments = ({}, [], [])
    for feature in args.stratifiers:
        values = pd.to_numeric(baseline[feature], errors='coerce').to_numpy(float)
        labels, cut = split_median(values)
        strata[feature] = (labels, ['Low', 'High'])
        for label in ['Low', 'High']:
            inventory.append({'stratifier': feature, 'group': label, 'baseline_year': year, 'threshold': cut, 'n_counties': int((labels == label).sum()), 'n_missing': int((labels == '').sum()), 'rule': 'Low<=baseline_median;High>baseline_median;ties_not_split'})
        assignments.extend(({'Code': code, 'stratifier': feature, 'group': label, 'baseline_year': year, 'baseline_value': value} for code, label, value in zip(panel['ids'], labels, values)))
    if args.region_mode != 'none':
        if args.region_file is not None:
            reg = read_csv(args.region_file)
            if not {'Code', 'Region'}.issubset(reg):
                raise ValueError('region-file must have Code,Region')
            reg['Code'] = reg.Code.str.strip().str.replace('\\.0+$', '', regex=True)
            if reg.Code.isna().any() or reg.Code.eq('').any() or reg.Code.duplicated().any():
                raise ValueError('region-file contains missing/duplicate Code')
            rr = reg.set_index('Code').Region.reindex(panel['ids'])
            labels = rr.fillna('').astype(str).str.strip().to_numpy()
            groups = sorted(set(labels) - {''})
            if len(groups) < 2:
                raise ValueError('region-file needs at least two matched regions')
            region_source = 'user_Code_Region_file'
        else:
            lookup = {prefix: name for name, prefixes in REGION_PREFIXES.items() for prefix in prefixes}
            labels = np.array([lookup.get(str(code)[:2], '') if len(str(code)) == 6 else '' for code in panel['ids']], object)
            groups, region_source = (list(REGION_PREFIXES), 'four_macro_regions_from_six_digit_Code')
        strata['Region'] = (labels, groups)
        for label in groups:
            inventory.append({'stratifier': 'Region', 'group': label, 'baseline_year': year, 'threshold': np.nan, 'n_counties': int((labels == label).sum()), 'n_missing': int((labels == '').sum()), 'rule': region_source})
        assignments.extend(({'Code': code, 'stratifier': 'Region', 'group': label, 'baseline_year': year, 'baseline_value': ''} for code, label in zip(panel['ids'], labels)))
    return (strata, inventory, assignments, year)

def fit_boundary_cell(panel, design, labels, group, g, t, mp, args, audits, raw_cells):
    masks = policy_masks(panel, design, g, t, mp, args.background_rule)
    if masks is None:
        return {}
    b, end, tr, co = masks
    member = labels == group
    tr, co = (tr & member, co & member)
    needed = list(range(b - mp + 1, b + 1)) + [t]
    common_valid = np.ones(len(labels), bool)
    for y in panel['outcomes'].values():
        common_valid &= np.isfinite(y[:, needed]).all(axis=1)
    ys = dict(panel['outcomes'])
    ys.update({m: panel['mechanisms'][m] for m in args.mechanism_outcomes})
    banks = {}
    for v, y in ys.items():
        valid = common_valid & np.isfinite(y[:, needed]).all(axis=1)
        tv, cv = (tr & valid, co & valid)
        key = (np.packbits(tv).tobytes(), np.packbits(cv).tobytes())
        banks.setdefault(key, {'tr': tv, 'co': cv, 'variables': []})['variables'].append(v)
    result = {}
    for block in banks.values():
        tv, cv, variables = (block['tr'], block['co'], block['variables'])
        nt, nc = (int(tv.sum()), int(cv.sum()))
        record = {'cohort': int(panel['years'][g]), 'year': int(panel['years'][t]), 'event_time': t - g, 'group': group, 'variables': ';'.join(variables), 'baseline': int(panel['years'][b]), 'n_raw_treated': int((design.event[:, g] & member).sum()), 'n_treated': nt, 'n_control': nc}
        if nt < MIN_TREATED or nc < MIN_CONTROL:
            audits.append({**record, 'status': 'not_estimable', 'reason': 'no_treated_or_control_in_subgroup'})
            continue
        ix = np.flatnonzero(tv | cv)
        treatment = tv[ix].astype(float)
        raw = panel['cov'][ix, b, :]
        names = list(COVARIATES)
        js = [j for j in range(len(POLICIES)) if j != design.focal]
        raw = np.column_stack([raw, panel['policies'][ix, b][:, js]])
        names += ['baseline_' + POLICIES[j] for j in js]
        try:
            filled, names, _ = impute_covariates(raw, names)
            x, names = full_rank_design(filled, names)
            delta = np.column_stack([ys[v][ix, t] - ys[v][ix, b] for v in variables])
            fit = dr_batch(x, treatment, delta, args.fit_mode)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            audits.append({**record, 'status': 'not_estimable', 'reason': str(exc)})
            continue
        p, w = (fit['p'], fit['weights'])
        ess = float(1 / (w @ w))
        flags = []
        if min(nt, nc) < SMALL_SAMPLE_WARNING:
            flags.append('group_count_below_20')
        if ess < SMALL_SAMPLE_WARNING:
            flags.append('weighted_control_ESS_below_20')
        extreme = (p < PS_EXTREME_CUTOFF) | (p > 1 - PS_EXTREME_CUTOFF)
        if extreme.any():
            flags.append('extreme_PS_retained')
        clipped = p != fit['safe_p']
        if clipped.any():
            flags.append('PS_numerical_clipping')
        ridge = fit['propensity']['ridge_lambda'] > 0 or fit['outcome_fit']['ridge_lambda'] > 0
        if ridge:
            flags.append('ridge_stabilized_estimate')
        record.update(status='estimated', reason='', control_ESS=ess, quality_flags=';'.join(flags), ps_extreme_n=int(extreme.sum()), ps_clipped_n=int(clipped.sum()), max_control_weight=float(w.max()), ps_treated_min=float(p[treatment == 1].min()), ps_treated_max=float(p[treatment == 1].max()), ps_control_min=float(p[treatment == 0].min()), ps_control_max=float(p[treatment == 0].max()), propensity_method=fit['propensity']['method'], propensity_ridge_lambda=fit['propensity']['ridge_lambda'], outcome_method=fit['outcome_fit']['method'], outcome_ridge_lambda=fit['outcome_fit']['ridge_lambda'], n_covariate_columns=x.shape[1])
        audits.append(record)
        for j, v in enumerate(variables):
            score = np.zeros(len(labels))
            score[ix] = fit['IF'][:, j] / len(ix)
            result[v] = {'att': float(fit['att'][j]), 'score': score, 'treated': set(np.flatnonzero(tv)), 'control': set(np.flatnonzero(cv)), 'nt': nt, 'nc': nc, 'g': g, 't': t, 'event': t - g, 'flags': flags, 'ess': ess, 'ridge': ridge}
            raw_cells.append({**record, 'variable': v, 'att': result[v]['att']})
    return result

def scaled(row, v, difference=False):
    row = {**row, 'variable': v, 'role': 'environment' if v in ENV_OUTCOMES else 'mechanism'}
    if v in ENV_OUTCOMES:
        row['unit'] = 'difference_in_negative_log_ATT' if difference else 'negative_log_outcome'
        row['positive_means'] = 'group_B_greater_improvement' if difference else 'environment_improvement'
        if not difference:
            for col in ['att', 'ci_low', 'ci_high']:
                with np.errstate(over='ignore', invalid='ignore'):
                    row[col + '_improvement_percent'] = -100 * np.expm1(-row.get(col, np.nan))
    else:
        row['unit'] = 'original_indicator_units'
        row['positive_means'] = 'greater_indicator_increase_in_group_B' if difference else 'indicator_increase'
    return row

def weighted_component(cells, weights, multiplier):
    score = sum((w * c['score'] for w, c in zip(weights, cells)))
    att = float(sum((w * c['att'] for w, c in zip(weights, cells))))
    tr = set().union(*(c['treated'] for c in cells))
    co = set().union(*(c['control'] for c in cells))
    row = {'att': att, **infer(att, score, multiplier, tr, co), 'n_treated': len(tr), 'n_control': len(co)}
    return (row, score, tr, co)

def matched_difference(cells_a, cells_b, multiplier):
    a = {(c['g'], c['t']): c for c in cells_a}
    b = {(c['g'], c['t']): c for c in cells_b}
    common = sorted(set(a) & set(b))
    if not common:
        return empty_result('no_common_group_time_cells')
    cohorts = sorted(set((g for g, t in common)))
    sizes = {g: np.mean([a[k]['nt'] + b[k]['nt'] for k in common if k[0] == g]) for g in cohorts}
    total = sum(sizes.values())
    weights = np.array([sizes[g] / total / sum((k[0] == g for k in common)) for g, t in common])
    aa = weighted_component([a[k] for k in common], weights, multiplier)
    bb = weighted_component([b[k] for k in common], weights, multiplier)
    na, nb = (len(aa[2] | aa[3]), len(bb[2] | bb[3]))
    sa = aa[1] * np.sqrt(na / (na - 1))
    sb = bb[1] * np.sqrt(nb / (nb - 1))
    score = sb - sa
    diff, variance = (bb[0]['att'] - aa[0]['att'], float(score @ score))
    covariance = float(sa @ sb)
    flags = sorted(set((f for k in common for c in (a[k], b[k]) for f in c['flags'])))
    row = {'status': 'estimated', 'reason': '', 'att': diff, 'se': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'p_value': np.nan, 'inference_ok': False, 'inference_reason': '', 'score_variance': variance, 'group_A_ATT': aa[0]['att'], 'group_B_ATT': bb[0]['att'], 'group_A_variance': float(sa @ sa), 'group_B_variance': float(sb @ sb), 'covariance_A_B': covariance, 'n_common_cells': len(common), 'n_common_cohorts': len(cohorts), 'n_A_treated': len(aa[2]), 'n_A_control': len(aa[3]), 'n_B_treated': len(bb[2]), 'n_B_control': len(bb[3]), 'quality_flags': ';'.join(flags), 'common_cells': ';'.join((str(g) + ':' + str(t) for g, t in common)), 'weight_rule': 'same_cohorts_equal_years_pooled_treated_cohort_weights'}
    if min(len(aa[2]), len(aa[3]), len(bb[2]), len(bb[3])) < MIN_INFERENCE_PER_ARM:
        row['inference_reason'] = 'fewer_than_minimum_Code_clusters_in_one_of_four_arms'
    elif not np.isfinite(variance) or variance <= 1e-24:
        row['inference_reason'] = 'degenerate_or_nonfinite_variance'
    else:
        draws = multiplier @ score
        se = np.sqrt(variance)
        critical = float(np.quantile(np.abs(draws / se), 0.95))
        row.update(se=se, ci_low=diff - critical * se, ci_high=diff + critical * se, p_value=float((1 + (np.abs(draws) >= abs(diff)).sum()) / (len(draws) + 1)), inference_ok=True)
    return row

def scopes(cells):
    yield ('overall', 'post_all', [c for c in cells if c['event'] >= 0])
    for name, (lo, hi) in PERIODS.items():
        yield ('period', name, [c for c in cells if lo <= c['event'] <= hi])
    for ev in sorted(set((c['event'] for c in cells))):
        yield ('dynamic', str(ev), [c for c in cells if c['event'] == ev])

def analyze_boundary(panel, design, feature, labels, groups, mp, args, multiplier, baseline_year):
    bank = {g: {} for g in groups}
    audit, gt, effect_rows, difference_rows = ([], [], [], [])
    for g in np.flatnonzero(design.event.any(axis=0)):
        if g < mp or panel['years'][g] <= baseline_year:
            continue
        times = list(range(max(1, g - args.pre_lags), g)) + list(range(g, min(len(panel['years']), g + args.max_post + 1)))
        for t in times:
            for group in groups:
                fitted = fit_boundary_cell(panel, design, labels, group, g, t, mp, args, audit, gt)
                for v, c in fitted.items():
                    bank[group].setdefault(v, []).append(c)
    common = {'design': design.id, 'design_kind': design.kind, 'stratifier': feature, 'min_pre': mp, 'grouping_year': baseline_year, 'background_rule': args.background_rule, 'from_count': design.before if design.kind == 'count' else np.nan, 'to_count': design.after if design.kind == 'count' else np.nan}
    variables = ENV_OUTCOMES + args.mechanism_outcomes
    for group in groups:
        cache = {}
        for v in variables:
            cells = bank[group].get(v, [])
            pre = pretrend_test([c for c in cells if c['event'] < 0], multiplier, cache)
            for scope, period, cc in scopes(cells):
                row, _, _, _ = aggregate(cc, multiplier, cache)
                if not cc:
                    if not np.any(labels == group):
                        why = 'empty_baseline_subgroup_or_missing_stratifier'
                    elif not design.event.any():
                        why = 'no_observed_new_policy_entry'
                    elif not np.any((panel['years'] > baseline_year) & (np.arange(len(panel['years'])) >= mp) & design.event.any(axis=0)):
                        why = 'no_eligible_event_after_grouping_year_and_required_preperiod'
                    else:
                        why = 'no_valid_subgroup_cells_for_scope;see_diagnostics_fit_audit'
                    row = {**row, 'reason': why}
                effect_rows.append(scaled({**common, 'group': group, 'scope': scope, 'period': period, **row, **(pre if scope == 'overall' else {})}, v))
    for ga, gb in combinations(groups, 2):
        for v in variables:
            ac, bc = (bank[ga].get(v, []), bank[gb].get(v, []))
            aa = {(s, p): c for s, p, c in scopes(ac)}
            bb = {(s, p): c for s, p, c in scopes(bc)}
            keys = sorted(set(aa) | set(bb))
            for scope, period in keys:
                diff = matched_difference(aa.get((scope, period), []), bb.get((scope, period), []), multiplier)
                difference_rows.append(scaled({**common, 'group_A': ga, 'group_B': gb, 'difference': gb + ' minus ' + ga, 'scope': scope, 'period': period, **diff}, v, True))
    return (effect_rows, difference_rows, [{**common, **r} for r in audit], [{**common, **r} for r in gt])

def response_type(eci, cpi, margin):

    def state(r):
        if not r.get('inference_ok', False) or not np.isfinite(r.get('q_BH', np.nan)):
            return 'uncertain'
        if r['q_BH'] < 0.05 and r['ci_low'] > 0:
            return 'benefit'
        if r['q_BH'] < 0.05 and r['ci_high'] < 0:
            return 'harm'
        return 'uncertain'
    ec, cp = (state(eci), state(cpi))
    if ec == cp == 'benefit':
        return ('Dual-end improvement response', 'dual_benefit')
    if {ec, cp} == {'benefit', 'harm'}:
        return ('Potential trade-off response', 'potential_tradeoff')
    if ec == 'benefit' or cp == 'benefit':
        return ('Single-end improvement evidence_other end uncertain', 'single_benefit_other_uncertain')
    if ec == 'harm' or cp == 'harm':
        return ('Adverse response evidence', 'adverse_response')
    if margin is not None:
        low, high = (-np.log1p(margin / 100), -np.log1p(-margin / 100))
        if all((r.get('inference_ok', False) and r['ci_low'] > low and (r['ci_high'] < high) for r in [eci, cpi])):
            return ('Low-magnitude response_interval within preset bounds', 'low_magnitude_within_prespecified_margin')
    return ('Insufficient evidence', 'insufficient_evidence')

def boundary_summary(effects, differences, margin):
    overall = effects.loc[effects.scope.eq('overall') & effects.role.eq('environment')]
    output = []
    for key, frame in overall.groupby(['design', 'design_kind', 'stratifier', 'group', 'min_pre'], sort=False):
        row = dict(zip(['design', 'design_kind', 'stratifier', 'group', 'min_pre'], key))
        records = frame.set_index('variable').to_dict('index')
        for v, r in records.items():
            for col in ['att', 'ci_low', 'ci_high', 'p_value', 'q_BH', 'att_improvement_percent', 'n_treated', 'n_control', 'pretrend_p', 'pretrend_q_BH', 'status', 'reason', 'inference_reason', 'quality_flags']:
                row[v + '_' + col] = r.get(col, np.nan)
        row['response_type'], row['response_code'] = response_type(records['ECI'], records['CPI'], margin)
        row['classification_level'] = 'policy_by_baseline_subgroup_not_individual_county'
        row['equivalence_margin_percent'] = margin
        pre = [r.get('pretrend_p', np.nan) for r in records.values()]
        row['pretrend_warning'] = 'some_pretrends_rejected' if any((np.isfinite(p) and p < 0.05 for p in pre)) else 'some_pretrends_unavailable' if any((not np.isfinite(p) for p in pre)) else 'not_rejected_is_not_proof'
        related = differences.loc[differences.scope.eq('overall') & differences.design.eq(key[0]) & differences.stratifier.eq(key[2]) & differences.min_pre.eq(key[4]) & (differences.group_A.eq(key[3]) | differences.group_B.eq(key[3]))]
        for v in ENV_OUTCOMES:
            rr = related.loc[related.variable.eq(v) & related.q_BH.lt(0.05)]
            row[v + '_significant_group_differences'] = ';'.join(rr.difference)
        row['interpretation'] = 'conditional_subgroup_response;not_causal_effect_of_stratifier'
        output.append(row)
    return pd.DataFrame(output)

def parse_boundary_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy-file', type=Path, default=POLICY_FILE)
    p.add_argument('--model-file', type=Path, default=MODEL_FILE)
    p.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    p.add_argument('--families', nargs='+', choices=['single', 'count'], default=['single', 'count'])
    p.add_argument('--designs', nargs='+')
    p.add_argument('--stratifiers', nargs='+', default=STRATIFIERS)
    p.add_argument('--baseline-year', type=int, default=None)
    p.add_argument('--region-mode', choices=['code', 'none'], default='code')
    p.add_argument('--region-file', type=Path, default=None)
    p.add_argument('--mechanism-outcomes', nargs='*', default=[], help='Optional raw mechanism outcomes, or ALL for all 34')
    p.add_argument('--min-pre', nargs='+', type=int, choices=[1, 2, 3], default=MIN_PRE)
    p.add_argument('--max-post', type=int, default=MAX_POST)
    p.add_argument('--pre-lags', type=int, default=PRE_LAGS)
    p.add_argument('--bootstrap', type=int, default=BOOTSTRAP)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--background-rule', choices=['stable', 'allow_changes'], default='stable')
    p.add_argument('--fit-mode', choices=['ordinary', 'stabilized'], default='stabilized')
    p.add_argument('--equivalence-margin-percent', type=float, default=None, help='Optional prespecified negligible percent effect, 0<x<100; not a significance tuning parameter')
    return p.parse_args()

def main():
    args = parse_boundary_args()
    if args.bootstrap < 99 or args.max_post < 0 or args.pre_lags < 0:
        raise ValueError('bootstrap>=99; horizons>=0 required')
    if args.equivalence_margin_percent is not None and (not 0 < args.equivalence_margin_percent < 100):
        raise ValueError('equivalence margin percent must be in (0,100)')
    if args.region_file is not None and args.region_mode == 'none':
        raise ValueError('region-file cannot be combined with region-mode none')
    args.stratifiers = list(dict.fromkeys(args.stratifiers))
    args.min_pre = list(dict.fromkeys(args.min_pre))
    args.mechanism_outcomes = list(MECHANISMS) if args.mechanism_outcomes == ['ALL'] else list(dict.fromkeys(args.mechanism_outcomes))
    if not set(args.mechanism_outcomes).issubset(MECHANISMS):
        raise ValueError('Unknown mechanism outcome')
    if not set(args.stratifiers).issubset(set(MECHANISMS + ENV_OUTCOMES)):
        raise ValueError('Unknown stratifier; use original model column spelling')
    args.mechanisms = list(dict.fromkeys([x for x in args.stratifiers if x not in ENV_OUTCOMES] + args.mechanism_outcomes))
    started = time.time()
    panel = load_panel(args)
    strata, inventory, assignments, year = build_strata(panel, args)
    designs = make_designs(panel, args.families)
    if args.designs:
        if not set(args.designs).issubset({d.id for d in designs}):
            raise ValueError('Requested design not observed')
        designs = [d for d in designs if d.id in args.designs]
    out = args.output_root / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    diagnostics = out / 'diagnostics'
    diagnostics.mkdir(parents=True, exist_ok=False)
    write_csv(inventory, diagnostics / 'group_definitions.csv')
    write_csv(assignments, diagnostics / 'county_group_membership.csv')
    write_csv(panel['audit'], diagnostics / 'variable_audit.csv')
    write_csv([{'design': d.id, 'kind': d.kind, 'raw_treated_counties': int(d.event.any(axis=1).sum())} for d in designs], diagnostics / 'design_inventory.csv')
    multiplier = np.random.default_rng(args.seed).choice([-1.0, 1.0], (args.bootstrap, len(panel['ids'])))
    effects, diffs = ([], [])
    total = len(designs) * len(strata) * len(args.min_pre)
    n = 0
    first_audit, first_gt = (True, True)
    for design in designs:
        for feature, (labels, groups) in strata.items():
            for mp in args.min_pre:
                n += 1
                pass
                ee, dd, au, gt = analyze_boundary(panel, design, feature, labels, groups, mp, args, multiplier, year)
                effects.extend(ee)
                diffs.extend(dd)
                for rows, filename, first in [(au, 'fit_audit.csv', first_audit), (gt, 'group_time.csv', first_gt)]:
                    if not rows:
                        continue
                    basecols = ['design', 'design_kind', 'stratifier', 'min_pre', 'grouping_year', 'background_rule', 'from_count', 'to_count']
                    cols = basecols + ['cohort', 'year', 'event_time', 'group', 'variables', 'baseline', 'n_raw_treated', 'n_treated', 'n_control', 'status', 'reason', 'control_ESS', 'quality_flags', 'ps_extreme_n', 'ps_clipped_n', 'max_control_weight', 'ps_treated_min', 'ps_treated_max', 'ps_control_min', 'ps_control_max', 'propensity_method', 'propensity_ridge_lambda', 'outcome_method', 'outcome_ridge_lambda', 'n_covariate_columns']
                    if filename == 'group_time.csv':
                        cols += ['variable', 'att']
                    pd.DataFrame(rows).reindex(columns=cols).to_csv(diagnostics / filename, index=False, mode='w' if first else 'a', header=first, encoding='utf-8-sig' if first else 'utf-8')
                    if filename == 'fit_audit.csv':
                        first_audit = False
                    else:
                        first_gt = False
    effects, diffs = (pd.DataFrame(effects), pd.DataFrame(diffs))
    effects = add_q(effects, ['min_pre', 'role', 'scope'])
    diffs = add_q(diffs, ['min_pre', 'role', 'scope'])
    overall = effects.scope.eq('overall')
    effects.loc[overall, 'pretrend_q_BH'] = effects.loc[overall].groupby(['min_pre', 'role'], dropna=False)['pretrend_p'].transform(lambda s: bh(s.to_numpy()))
    write_csv(effects, out / 'group_effects.csv')
    write_csv(diffs, out / 'group_differences.csv')
    summary = boundary_summary(effects, diffs, args.equivalence_margin_percent)
    write_csv(summary, out / 'boundary_summary.csv')
    for filename, first in [('fit_audit.csv', first_audit), ('group_time.csv', first_gt)]:
        if first:
            write_csv([], diagnostics / filename)
    manifest = {'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'county_count': len(panel['ids']), 'years': panel['years'].tolist(), 'grouping_year': year, 'design_count': len(designs), 'stratifier_count': len(strata), 'grouping': 'fixed baseline calendar values;median pooled across counties;ties not split', 'inference': 'Code-cluster shared multipliers;cross-county independence assumed', 'difference': 'direct B-A on common (g,t) support and identical pooled treated cohort weights', 'interpretation_limits': ['Do not infer heterogeneity from one significant and one nonsignificant subgroup.', 'Subgroup effects do not identify an individual county effect or the causal effect of its baseline feature.', 'Baseline can be after older background policies, but must precede the focal new entry.', 'Available-period summaries are not balanced between periods. No temporal contrast is inferred from them.', 'Post-policy variables never define groups; missing baseline strata are not imputed.', 'No trimming;ridge/clipping retain estimates but do not restore overlap or guarantee DR consistency.', 'Minimum 4 clusters per arm is only a reporting floor, not a reliability guarantee.', 'Pointwise confidence intervals are not multiple-comparison-adjusted;BH q values are reported.', 'Nonrejection is not equivalence. Low response requires a user-prespecified negligible-effect bound.', 'Mechanism subgroup effects are not causal mediation effects.'], 'sha256': {str(path): sha256(path) for path in [args.policy_file, args.model_file, Path(__file__)] + ([args.region_file] if args.region_file is not None else [])}, 'elapsed_seconds': round(time.time() - started, 1)}
    (out / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    pass
if __name__ == '__main__':
    main()
