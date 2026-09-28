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
OUT_DIR = PROJECT_ROOT / 'outputs'
POLICY_FILE = BASE_DIR / 'policy_conditions.csv'
MODEL_FILE = BASE_DIR / 'model_data_total.csv'
OUTPUT_ROOT = OUT_DIR / 'policy_mechanisms'
POLICIES = ['CETP', 'LCCP', 'ECERFP', 'ZWC', 'CEP', 'CEPDU', 'CACP', 'SCP', 'LECCDC', 'NGFRIPZ']
MECHANISMS = ['PGDP', 'POPD', 'VAF_GDP', 'GIA_GDP', 'TSS_GDP', 'Mean_Light', 'Builtup', 'Road_Density', 'PIU_PIR_Ratio', 'Fiscal_Revenue_GDP', 'Energy_Intensity', 'Coal_Share', 'Electricity_Share', 'Fossil_Energy_Share', 'Energy_Diversity_Index_Norm', 'Industrial_Coal_Share', 'TPSP_Petroleum_Share', 'Forest', 'Grassland', 'Shrub', 'Water', 'Wetland', 'Annual NDVI', 'Summer NDVI', 'AMEVT', 'LAI_HV', 'LAI_LV', 'BG_AI', 'BG_Divsion', 'EEI_GDP', 'EES_GEF', 'GPAI', 'GIPAI', 'GIPGS']
ENV_OUTCOMES = ['ECI', 'CPI', 'O3', 'PM10']
RAW_COVARIATES = ['PGDP', 'POPD', 'GIA_GDP', 'Builtup', 'Coal_Share', 'Electricity_Share']
COVARIATES = ['log_PGDP', 'log_POPD', 'GIA_GDP', 'Builtup', 'Coal_Share', 'Electricity_Share', 'log_ECI', 'log_CPI']
MIN_PRE = [1, 2, 3]
MAX_POST = 5
PRE_LAGS = 3
BOOTSTRAP = 999
SEED = 20260918
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

def fit_cells(panel, design, g, t, min_pre, args, phase, audit, gt, horizon=None):
    masks = policy_masks(panel, design, g, t, min_pre, args.background_rule, horizon)
    if masks is None:
        return {}
    b, end, raw_tr, raw_co = masks
    n, _ = panel['count'].shape
    times = list(range(g, end + 1)) if horizon is not None else [t]
    needed = list(range(b - min_pre + 1, b + 1)) + times
    env_valid = np.ones(n, bool)
    for y in panel['outcomes'].values():
        env_valid &= np.isfinite(y[:, needed]).all(axis=1)
    groups = {}
    for mechanism, y in panel['mechanisms'].items():
        valid = env_valid & np.isfinite(y[:, needed]).all(axis=1)
        tr, co = (raw_tr & valid, raw_co & valid)
        key = (np.packbits(tr).tobytes(), np.packbits(co).tobytes())
        groups.setdefault(key, {'tr': tr, 'co': co, 'mechanisms': []})['mechanisms'].append(mechanism)
    results = {}
    for group in groups.values():
        tr, co, members = (group['tr'], group['co'], group['mechanisms'])
        nt, nc = (int(tr.sum()), int(co.sum()))
        base = {'design': design.id, 'design_kind': design.kind, 'min_pre': min_pre, 'phase': phase, 'cohort': int(panel['years'][g]), 'year': int(panel['years'][t]), 'baseline': int(panel['years'][b]), 'common_horizon': horizon, 'mechanisms': ';'.join(members), 'n_raw_treated': int(design.event[:, g].sum()), 'n_treated_after_policy_rules': int(raw_tr.sum()), 'n_control_after_policy_rules': int(raw_co.sum()), 'n_treated': nt, 'n_control': nc, 'status': 'not_estimable', 'reason': ''}
        if nt < MIN_TREATED or nc < MIN_CONTROL:
            audit.append({**base, 'reason': 'no_treated_or_controls_after_policy_rules_and_observed_results'})
            continue
        ix = np.flatnonzero(tr | co)
        d = tr[ix].astype(float)
        raw = panel['cov'][ix, b, :]
        cov_names = list(COVARIATES)
        if args.background_adjustment == 'status':
            js = [j for j in range(len(POLICIES)) if j != design.focal]
            raw = np.column_stack([raw, panel['policies'][ix, b][:, js]])
            cov_names += ['baseline_' + POLICIES[j] for j in js]
        try:
            filled, cn, imputation = impute_covariates(raw, cov_names)
            x, design_columns = full_rank_design(filled, cn)
            variables = members + ENV_OUTCOMES
            ys = {**panel['mechanisms'], **panel['outcomes']}
            delta = np.column_stack([ys[v][ix, tt] - ys[v][ix, b] for tt in times for v in variables])
            fit = dr_batch(x, d, delta, args.fit_mode)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            audit.append({**base, 'reason': str(exc)})
            continue
        score = np.zeros((n, len(times) * len(variables)))
        score[ix] = fit['IF'] / len(ix)
        weights, ps = (fit['weights'], fit['p'])
        ess = float(1 / np.sum(weights ** 2))
        extreme = (ps < PS_EXTREME_CUTOFF) | (ps > 1 - PS_EXTREME_CUTOFF)
        flags = []
        if min(nt, nc) < SMALL_SAMPLE_WARNING:
            flags.append('group_count_below_20')
        if ess < SMALL_SAMPLE_WARNING:
            flags.append('weighted_control_ESS_below_20')
        if extreme.any():
            flags.append('extreme_PS_retained')
        ridge = fit['propensity']['ridge_lambda'] > 0 or fit['outcome_fit']['ridge_lambda'] > 0
        if ridge:
            flags.append('ridge_stabilized_estimate')
        if np.any(ps != fit['safe_p']):
            flags.append('PS_numerical_clipping')
        record = {**base, 'status': 'estimated', 'reason': '', 'control_ESS': ess, 'max_control_weight': float(weights.max()), 'ps_treated_min': float(ps[d == 1].min()), 'ps_treated_max': float(ps[d == 1].max()), 'ps_control_min': float(ps[d == 0].min()), 'ps_control_max': float(ps[d == 0].max()), 'ps_extreme_n': int(extreme.sum()), 'ps_clipped_n': int((ps != fit['safe_p']).sum()), 'propensity_method': fit['propensity']['method'], 'propensity_fallback_reason': fit['propensity']['primary_error'], 'propensity_ridge_lambda': fit['propensity']['ridge_lambda'], 'outcome_method': fit['outcome_fit']['method'], 'outcome_ridge_lambda': fit['outcome_fit']['ridge_lambda'], 'n_design_columns': x.shape[1], 'quality_flags': ';'.join(flags), 'n_missing_covariate_values': int((~np.isfinite(raw)).sum()), 'sample_hash': hashlib.sha256(';'.join(panel['ids'][ix]).encode()).hexdigest()[:16], 'treatment_hash': hashlib.sha256(np.packbits(tr).tobytes()).hexdigest()[:16]}
        audit.append(record)
        tr_ids, co_ids = (set(np.flatnonzero(tr)), set(np.flatnonzero(co)))
        for ti, tt in enumerate(times):
            shared = {}
            for vi, v in enumerate(variables):
                k = ti * len(variables) + vi
                cell = {'att': float(fit['att'][k]), 'score': score[:, k], 'treated': tr_ids, 'control': co_ids, 'nt': nt, 'nc': nc, 'g': g, 't': tt, 'event': tt - g, 'phase': phase, 'flags': flags, 'ess': ess, 'ridge': ridge}
                shared[v] = cell
                gt.append({**{key: record[key] for key in ['design', 'design_kind', 'min_pre', 'phase', 'cohort', 'baseline', 'common_horizon', 'n_treated', 'n_control', 'sample_hash', 'treatment_hash']}, 'year': int(panel['years'][tt]), 'event_time': tt - g, 'variable': v, 'role': 'mechanism' if v in members else 'paired_environment', 'matched_mechanisms': ';'.join(members), 'att': cell['att'], 'quality_flags': record['quality_flags']})
            for m in members:
                for v in [m] + ENV_OUTCOMES:
                    results.setdefault((m, v), []).append(shared[v])
    return results

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

def scale_result(row, variable, mechanism, panel):
    r = dict(row)
    r['role'] = 'mechanism' if variable == mechanism else 'paired_environment'
    r['variable'] = variable
    r['mechanism'] = mechanism
    if variable == mechanism:
        sd = panel['audit'].set_index('mechanism').loc[mechanism, 'first_year_SD']
        r['unit'] = 'original_indicator_units'
        r['positive_means'] = 'indicator_increase'
        r['reference_SD'] = sd
        for k in ['att', 'se', 'ci_low', 'ci_high']:
            r[k + '_per_reference_SD'] = r.get(k, np.nan) / sd if np.isfinite(sd) and sd > 0 else np.nan
    else:
        r['unit'] = 'negative_log_environment_outcome'
        r['positive_means'] = 'environment_improvement'
        for k in ['att', 'ci_low', 'ci_high']:
            with np.errstate(over='ignore', invalid='ignore'):
                r[k + '_improvement_percent'] = -100 * np.expm1(-r.get(k, np.nan))
    return r

def analyze_design(panel, design, min_pre, args, multiplier):
    audit, gt = ([], [])
    bank = {'available': {}, 'balanced2': {}, 'balanced5': {}}
    cohorts = np.flatnonzero(design.event.any(axis=0))
    for g in cohorts:
        if g < min_pre:
            continue
        for t in list(range(max(1, g - args.pre_lags), g)) + list(range(g, min(len(panel['years']), g + args.max_post + 1))):
            fitted = fit_cells(panel, design, g, t, min_pre, args, 'available', audit, gt)
            for k, cc in fitted.items():
                bank['available'].setdefault(k, []).extend(cc)
        if not args.skip_balanced:
            for h in [2, 5]:
                if h > args.max_post:
                    continue
                fitted = fit_cells(panel, design, g, g, min_pre, args, 'balanced' + str(h), audit, gt, horizon=h)
                for k, cc in fitted.items():
                    bank['balanced' + str(h)].setdefault(k, []).extend(cc)
    output = {k: [] for k in ['overall', 'dynamic', 'pretrends', 'periods', 'contrasts']}
    cache = {}
    for mechanism in panel['mechanisms']:
        for variable in [mechanism] + ENV_OUTCOMES:
            key = (mechanism, variable)
            common = {'design': design.id, 'design_kind': design.kind, 'min_pre': min_pre, 'background_rule': args.background_rule, 'from_count': design.before if design.kind == 'count' else np.nan, 'to_count': design.after if design.kind == 'count' else np.nan, 'n_raw_treated': int(design.event.any(axis=1).sum())}
            available = bank['available'].get(key, [])
            post = [c for c in available if c['event'] >= 0]
            pre = [c for c in available if c['event'] < 0]
            overall, _, _, _ = aggregate(post, multiplier, cache)
            if not post:
                if not design.event.any():
                    why = 'no_observed_new_policy_entry'
                elif not np.any(cohorts >= min_pre):
                    why = 'insufficient_pre_policy_calendar_for_requested_min_pre'
                elif not np.isfinite(panel['mechanisms'][mechanism]).any():
                    why = 'mechanism_entirely_missing'
                else:
                    failures = {}
                    for failure in audit:
                        if failure['phase'] == 'available' and failure['year'] >= failure['cohort'] and (mechanism in failure['mechanisms'].split(';')) and (failure['status'] != 'estimated'):
                            reason = failure['reason']
                            failures[reason] = failures.get(reason, 0) + 1
                    why = json.dumps(failures, sort_keys=True) if failures else 'no_valid_group_time_cells'
                overall = {**overall, 'reason': why}
            trend = pretrend_test(pre, multiplier, cache)
            output['overall'].append(scale_result({**common, **overall, **trend}, variable, mechanism, panel))
            output['pretrends'].append({**common, 'mechanism': mechanism, 'variable': variable, **trend})
            for event in sorted(set((c['event'] for c in available))):
                agg, _, _, _ = aggregate([c for c in available if c['event'] == event], multiplier, cache)
                output['dynamic'].append(scale_result({**common, 'event_time': event, 'phase': 'post' if event >= 0 else 'adjacent_year_placebo', **agg}, variable, mechanism, panel))
            for phase, entries in bank.items():
                if args.skip_balanced and phase != 'available':
                    continue
                cells = entries.get(key, [])
                agg_periods = {}
                for name, (lo, hi) in PERIODS.items():
                    if phase == 'balanced2' and hi > 2:
                        continue
                    chosen = [c for c in cells if lo <= c['event'] <= hi]
                    a = aggregate(chosen, multiplier, cache)
                    agg_periods[name] = a
                    output['periods'].append(scale_result({**common, 'sample_basis': phase, 'period': name, **a[0]}, variable, mechanism, panel))
                pairs = [('years1_2', 'year0')] if phase == 'balanced2' else [('years3_5', 'year0'), ('years3_5', 'years1_2')] if phase == 'balanced5' else []
                for later, earlier in pairs:
                    aa, bb = (agg_periods[later], agg_periods[earlier])
                    row = {'sample_basis': phase, 'contrast': later + ' minus ' + earlier}
                    if aa[1] is None or bb[1] is None:
                        row.update(empty_result('no_common_full_horizon_sample'))
                    else:
                        if aa[2] != bb[2] or aa[3] != bb[3]:
                            raise AssertionError('Balanced periods use different county sets')
                        diff, score = (aa[0]['att'] - bb[0]['att'], aa[1] - bb[1])
                        active = len(aa[2] | aa[3])
                        factor = active / (active - 1) if active > 1 else np.nan
                        row.update(status='estimated', reason='', att=diff, **infer(diff, score, multiplier, aa[2], aa[3]), n_treated=len(aa[2]), n_control=len(aa[3]), covariance_earlier_later=float(aa[1] @ bb[1]) * factor, earlier_att=bb[0]['att'], later_att=aa[0]['att'], quality_flags=';'.join(sorted(set(aa[0].get('quality_flags', '').split(';') + bb[0].get('quality_flags', '').split(';')))))
                    scaled = scale_result({**common, **row}, variable, mechanism, panel)
                    if variable != mechanism:
                        for col in ['att_improvement_percent', 'ci_low_improvement_percent', 'ci_high_improvement_percent']:
                            scaled.pop(col, None)
                        scaled['unit'] = 'difference_in_negative_log_effects_not_percentage_points'
                    output['contrasts'].append(scaled)
    output['fit_audit'], output['group_time'] = (audit, gt)
    return output

def demean_two_way(matrix, codes, years):
    z = np.asarray(matrix, float).copy()
    for _ in range(1000):
        old = z.copy()
        for labels in [codes, years]:
            size = np.bincount(labels)
            sums = np.zeros((len(size), z.shape[1]))
            np.add.at(sums, labels, z)
            z -= (sums / size[:, None])[labels]
        if np.max(np.abs(z - old)) < 1e-10:
            return z
    raise ValueError('two_way_demeaning_not_converged')

def lag_associations(panel, args):
    rows = []
    n, tt = panel['count'].shape
    source_codes = np.repeat(np.arange(n), tt)
    source_years = np.tile(np.arange(tt), n)
    policy_matrix = panel['policies'].reshape(n * tt, len(POLICIES))
    base_cov = panel['cov'][:, :, :6]
    for lag in args.lags:
        for m, raw_m in panel['mechanisms'].items():
            remove = RAW_COVARIATES.index(m) if m in RAW_COVARIATES else -1
            js = [j for j in range(6) if j != remove]
            lm = np.full_like(raw_m, np.nan)
            lm[:, lag:] = raw_m[:, :-lag]
            lx = np.full_like(base_cov[:, :, js], np.nan)
            lx[:, lag:, :] = base_cov[:, :-lag, js]
            controls = np.column_stack([lx.reshape(n * tt, len(js)), policy_matrix])
            for env, raw_y in panel['outcomes'].items():
                common = {'mechanism': m, 'environment': env, 'lag': lag, 'interpretation': 'conditional_association_not_causal_mediation', 'positive_coefficient_means': 'higher_mechanism_associated_with_environment_improvement'}
                y, mv = (raw_y.ravel(), lm.ravel())
                valid = np.isfinite(y) & np.isfinite(mv) & np.isfinite(controls).all(axis=1)
                if valid.sum() < 10:
                    unavailable = empty_result('insufficient_complete_lagged_observations')
                    unavailable['coefficient'] = unavailable.pop('att')
                    rows.append({**common, **unavailable})
                    continue
                codes0, years0 = (source_codes[valid], source_years[valid])
                _, codes = np.unique(codes0, return_inverse=True)
                _, years = np.unique(years0, return_inverse=True)
                ng, ny = (int(codes.max() + 1), int(years.max() + 1))
                xx = np.column_stack([mv[valid], controls[valid]])
                scales = np.std(xx, axis=0)
                scales[scales < 1e-12] = 1.0
                xx = (xx - xx.mean(axis=0)) / scales
                try:
                    residualized = demean_two_way(np.column_stack([y[valid], xx]), codes, years)
                    yy, mm, zz = (residualized[:, 0], residualized[:, 1], residualized[:, 2:])
                    variable_columns = np.std(zz, axis=0) > 1e-10
                    zz = zz[:, variable_columns]
                    if zz.shape[1]:
                        q, r, piv = qr(zz, mode='economic', pivoting=True)
                        diag = np.abs(np.diag(r))
                        rank = int(np.sum(diag > max(zz.shape) * np.finfo(float).eps * diag.max()))
                        q = q[:, :rank]
                        mr, yr = (mm - q @ (q.T @ mm), yy - q @ (q.T @ yy))
                    else:
                        rank, mr, yr = (0, mm.copy(), yy.copy())
                    denom = float(mr @ mr)
                    if denom <= 1e-12:
                        raise ValueError('no_independent_within_county_mechanism_variation')
                    beta_std = float(mr @ yr / denom)
                    errors = yr - beta_std * mr
                    score = np.bincount(codes, weights=mr * errors / denom, minlength=ng)
                    from scipy.sparse import coo_matrix
                    from scipy.sparse.csgraph import connected_components
                    edges = coo_matrix((np.ones(len(codes) * 2), (np.r_[codes, ng + years], np.r_[ng + years, codes])), shape=(ng + ny, ng + ny)).tocsr()
                    components = connected_components(edges, directed=False, return_labels=False)
                    df = len(yy) - (ng + ny - components) - rank - 1
                    if df <= 0:
                        raise ValueError('nonpositive_residual_degrees_of_freedom')
                    correction = ng / (ng - 1) * (len(yy) - 1) / df if ng > 1 else np.nan
                    se = np.sqrt(float(score @ score) * correction) / scales[0]
                    beta = beta_std / scales[0]
                    ok = ng >= MIN_INFERENCE_PER_ARM and np.isfinite(se) and (se > 0)
                    critical = stats.t.ppf(0.975, ng - 1) if ok else np.nan
                    rows.append({**common, 'status': 'estimated', 'reason': '', 'coefficient': beta, 'se': se, 'ci_low': beta - critical * se, 'ci_high': beta + critical * se, 'p_value': float(2 * stats.t.sf(abs(beta / se), ng - 1)) if ok else np.nan, 'inference_ok': ok, 'n_observations': len(yy), 'n_Code_clusters': ng, 'residual_df': df, 'unit': 'negative_log_environment_per_raw_mechanism_unit', 'controls': ';'.join([COVARIATES[j] for j in js] + POLICIES), 'fixed_effects': 'Code_and_Year', 'quality_flags': 'few_Code_clusters' if ng < 20 else ''})
                except (ValueError, np.linalg.LinAlgError) as exc:
                    unavailable = empty_result(str(exc))
                    unavailable['coefficient'] = unavailable.pop('att')
                    rows.append({**common, **unavailable})
    return add_q(pd.DataFrame(rows), ['lag'], qcol='q_BH_lag_family')

def evidence_summary(overall, dynamics, associations):
    if overall.empty:
        return pd.DataFrame()
    output = []
    keys = ['design', 'design_kind', 'min_pre', 'mechanism']
    for key, df in overall.groupby(keys, sort=False, dropna=False):
        m = df.loc[df.role.eq('mechanism')].iloc[0]
        for env in ENV_OUTCOMES:
            ey = df.loc[df.variable.eq(env)]
            if ey.empty:
                continue
            e = ey.iloc[0]
            row = dict(zip(keys, key))
            row.update(environment=env, mechanism_ATT=m['att'], mechanism_p=m['p_value'], mechanism_q=m.get('q_BH', np.nan), mechanism_pretrend_p=m.get('pretrend_p', np.nan), environment_ATT_negative_log=e['att'], environment_p=e['p_value'], environment_q=e.get('q_BH', np.nan), environment_pretrend_p=e.get('pretrend_p', np.nan), n_treated=m.get('n_treated', 0), n_control=m.get('n_control', 0), mechanism_status=m['status'], environment_status=e['status'], quality_flags=m.get('quality_flags', ''), conclusion='reported_evidence_not_automatic_mediation_claim')
            for prefix, estimate in [('mechanism', m), ('environment', e)]:
                for col in ['ci_low', 'ci_high', 'min_control_ESS', 'ridge_cell_count']:
                    row[prefix + '_' + col] = estimate.get(col, np.nan)
            if not dynamics.empty:
                subset = dynamics.loc[dynamics.design.eq(key[0]) & dynamics.min_pre.eq(key[2]) & dynamics.mechanism.eq(key[3]) & dynamics.event_time.ge(0)]
                first_events = {}
                for label, v in [('mechanism', key[3]), ('environment', env)]:
                    ss = subset.loc[subset.variable.eq(v) & subset.q_BH.lt(0.05)]
                    first = ss.event_time.min() if not ss.empty else np.nan
                    row[label + '_first_detected_post_event_q05'] = first
                    first_events[label] = first
                fm, fe = (first_events['mechanism'], first_events['environment'])
                row['detected_timing_pattern'] = 'insufficient_detected_events' if not np.isfinite([fm, fe]).all() else 'mechanism_detected_earlier' if fm < fe else 'detected_same_year' if fm == fe else 'environment_detected_earlier'
            for lag in sorted(associations.lag.unique()) if not associations.empty else LAGS:
                a = associations.loc[associations.mechanism.eq(key[-1]) & associations.environment.eq(env) & associations.lag.eq(lag)] if not associations.empty else pd.DataFrame()
                if not a.empty:
                    row[f'lag{lag}_association'] = a.iloc[0].get('coefficient', np.nan)
                    row[f'lag{lag}_p'] = a.iloc[0].get('p_value', np.nan)
                    row[f'lag{lag}_q'] = a.iloc[0].get('q_BH_lag_family', np.nan)
            row['timing_note'] = 'compare_paired_dynamic_curves;first_significance_is_not_true_onset'
            output.append(row)
    return pd.DataFrame(output)

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy-file', type=Path, default=POLICY_FILE)
    p.add_argument('--model-file', type=Path, default=MODEL_FILE)
    p.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    p.add_argument('--families', nargs='+', choices=['single', 'count'], default=['single', 'count'])
    p.add_argument('--designs', nargs='*', default=None, help='Optional exact design IDs for a test run')
    p.add_argument('--mechanisms', nargs='+', default=MECHANISMS)
    p.add_argument('--min-pre', nargs='+', type=int, choices=[1, 2, 3], default=MIN_PRE)
    p.add_argument('--max-post', type=int, default=MAX_POST)
    p.add_argument('--pre-lags', type=int, default=PRE_LAGS)
    p.add_argument('--bootstrap', type=int, default=BOOTSTRAP)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--background-rule', choices=['stable', 'allow_changes'], default='stable', help='Single-policy comparison rule; count-transition controls always keep baseline portfolio')
    p.add_argument('--background-adjustment', choices=['status', 'none'], default='status')
    p.add_argument('--fit-mode', choices=['ordinary', 'stabilized'], default='stabilized')
    p.add_argument('--skip-balanced', action='store_true')
    p.add_argument('--skip-associations', action='store_true')
    p.add_argument('--lags', nargs='+', type=int, default=LAGS)
    return p.parse_args()

def main():
    args = parse_args()
    if args.bootstrap < 99 or args.max_post < 0 or args.pre_lags < 0:
        raise ValueError('Require bootstrap>=99 and nonnegative time horizons')
    if any((l < 1 for l in args.lags)):
        raise ValueError('Lags must be positive')
    args.mechanisms = list(dict.fromkeys(args.mechanisms))
    args.min_pre = list(dict.fromkeys(args.min_pre))
    if not set(args.mechanisms).issubset(MECHANISMS):
        raise ValueError('Unknown mechanism name')
    started = time.time()
    out = args.output_root / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    panel = load_panel(args)
    designs = make_designs(panel, args.families)
    if args.designs:
        chosen = set(args.designs)
        if not chosen.issubset({d.id for d in designs}):
            raise ValueError('Unknown requested design ID')
        designs = [d for d in designs if d.id in chosen]
    write_csv(panel['audit'], out / 'variable_audit.csv')
    write_csv([{'design': d.id, 'kind': d.kind, 'from_count': d.before, 'to_count': d.after, 'raw_treated_counties': int(d.event.any(axis=1).sum()), 'cohorts': ';'.join(map(str, panel['years'][d.event.any(axis=0)]))} for d in designs], out / 'design_inventory.csv')
    n = len(panel['ids'])
    mult = np.random.default_rng(args.seed).choice([-1.0, 1.0], (args.bootstrap, n))
    all_rows = {}
    total = len(designs) * len(args.min_pre)
    counter = 0
    for design in designs:
        for mp in args.min_pre:
            counter += 1
            pass
            result = analyze_design(panel, design, mp, args, mult)
            for name, rows in result.items():
                all_rows.setdefault(name, []).extend(rows)
            checkpoint = out / 'checkpoints'
            checkpoint.mkdir(exist_ok=True)
            (checkpoint / f'{design.id}_pre{mp}.json').write_text(json.dumps({k: v for k, v in result.items() if k not in ['fit_audit', 'group_time']}, ensure_ascii=False, allow_nan=True, default=json_scalar), encoding='utf-8')
    tables = {}
    for name, rows in all_rows.items():
        frame = pd.DataFrame(rows)
        if name in ['overall', 'dynamic', 'periods', 'contrasts'] and (not frame.empty):
            family = ['min_pre', 'role']
            if name == 'periods':
                family += ['sample_basis', 'period']
            if name == 'contrasts':
                family += ['sample_basis', 'contrast']
            frame = add_q(frame, family)
        if name == 'pretrends' and (not frame.empty):
            frame = add_q(frame, ['min_pre'], pcol='pretrend_p', qcol='pretrend_q_BH')
        tables[name] = write_csv(frame, out / (name + '.csv'))
    if not args.skip_associations:
        pass
        associations = lag_associations(panel, args)
    else:
        associations = pd.DataFrame(columns=['mechanism', 'environment', 'lag'])
    write_csv(associations, out / 'lagged_mechanism_environment_associations.csv')
    evidence = evidence_summary(tables['overall'], tables['dynamic'], associations)
    write_csv(evidence, out / 'pathway_evidence_summary.csv')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest = {'configuration': config, 'county_count': n, 'years': panel['years'].tolist(), 'mechanism_scale': 'raw_values;positive_means_increase_not_necessarily_improvement', 'environment_scale': '-ln(value);positive_means_improvement', 'environment_percent': '100*(1-exp(-ATT));geometric_scale_not_mean_individual_percent', 'inference': 'shared_Code_multipliers;cross_county_independence_assumed', 'minimum_point_counts': [MIN_TREATED, MIN_CONTROL], 'minimum_inference_clusters_per_arm': MIN_INFERENCE_PER_ARM, 'aggregation': 'equal_available_years_within_cohort;mean_effective_treated_count_between_cohorts', 'period_contrasts': 'reestimated_common_full_horizon_samples;shared_covariance_retained', 'baseline_covariates': COVARIATES, 'relaxations': ['No exact policy adoption-year/history matching.', 'Count designs match baseline count, not policy identity.', 'No common-support trimming; extreme PS retained and reported.', 'Covariate missingness uses cell-specific median and missing indicators.', 'Unstable nuisance regressions use explicitly labelled ridge fallback.', 'Point estimates allowed with one treated county and one control county.'], 'principle_limits': ['No outcome/mediator imputation; zeros and signed mechanism values are preserved.', 'No preperiod/onset/control or nonfinite estimates remain unavailable with a reason.', 'Ridge and clipping do not fix absent overlap or guarantee DR consistency.', 'Inference conditions on imputation, penalty choice, and aggregation weights.', 'At least four clusters is only a reporting floor, not reliable asymptotics.', 'Code clustering does not protect against cross-county correlation.', 'Policy-stable restrictions select a subset; allow_changes mixes concurrent policies.', 'Lagged FE coefficients are associations, not causal mediation.', 'No significance-based policy/feature selection; all requested indicators reported.', 'Annual timing does not identify precise ordering within a year.', 'Variable definitions and mechanical overlap with ECI/CPI must be checked by the author.'], 'sha256': {str(p): sha256(p) for p in [args.policy_file, args.model_file, Path(__file__)]}}
    (out / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    overall = tables['overall']
    summary = {'output_directory': str(out), 'designs': len(designs), 'mechanisms': len(args.mechanisms), 'overall_rows': len(overall), 'numeric_ATT': int(overall.att.notna().sum()), 'inferable_rows': int(overall.inference_ok.eq(True).sum()), 'elapsed_seconds': round(time.time() - started, 1)}
    (out / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    pass
if __name__ == '__main__':
    main()
