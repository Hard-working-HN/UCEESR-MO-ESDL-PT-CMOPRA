from __future__ import annotations
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import qr
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors, font_manager
from matplotlib.ticker import MaxNLocator, ScalarFormatter
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = PROJECT_ROOT / 'data' / 'processed'
POLICY_FILE = BASE_DIR / 'policy_conditions.csv'
MODEL_FILE = BASE_DIR / 'model_data_total.csv'
SHP_FILE = PROJECT_ROOT / 'data' / 'spatial' / 'county_2023.shp'
SHP_CODE_FIELD = 'code'
SHP_ENCODING = None
OUTPUT_ROOT = PROJECT_ROOT / 'outputs' / 'neighbor_spillover'
POLICIES = ['CETP', 'LCCP', 'ECERFP', 'ZWC', 'CEP', 'CEPDU', 'CACP', 'SCP', 'LECCDC', 'NGFRIPZ']
OUTCOMES = ['ECI', 'CPI', 'O3', 'PM10']
RAW_COVARIATES = ['PGDP', 'POPD', 'GIA_GDP', 'Builtup', 'Coal_Share', 'Electricity_Share']
COVARIATES = ['log_PGDP', 'log_POPD', 'GIA_GDP', 'Builtup', 'Coal_Share', 'Electricity_Share', 'log_ECI', 'log_CPI']
THRESHOLDS = list(range(1, 7))
MAX_POST = 5
MIN_PRE = 1
PRE_LAGS = 3
NEIGHBOR_POLICY_RULE = 'stable'
MIN_TREATED = 1
MIN_CONTROL = 1
MIN_INFERENCE_PER_ARM = 4
SMALL_SAMPLE_WARNING = 20
BOOTSTRAP = 999
SEED = 2026
PS_CLIP = 1e-06
PS_EXTREME_CUTOFF = 0.01
LOGIT_COND_LIMIT = 1000000000000.0
OLS_COND_LIMIT = 1000000000000.0
RIDGE_GRID = (1e-08, 1e-06, 0.0001, 0.01, 0.1, 1.0)
OVERLAP_MAX_M2 = 0.1
OVERLAP_MAX_RATIO = 1e-08
OUTCOME_LABELS = {'ECI': 'ECI', 'CPI': 'CPI', 'O3': 'O₃', 'PM10': 'PM₁₀'}
COLORS = {'ECI': '#DE582B', 'CPI': '#1868B2', 'O3': '#71649D', 'PM10': '#037F77'}
FONT_FAMILY = 'Times New Roman'
FONT_SIZE = 16
AXIS_LABEL_SIZE = 24
TICK_FONT_SIZE = 16
NA_FONT_SIZE = 12
AXIS_LABEL_PAD = 16
MARKER_SIZE = 5.2
CI_LINE_WIDTH = 1.25
CAP_SIZE = 2.5
AXIS_LINE_WIDTH = 0.8
ZERO_LINE_COLOR = '#808080'
ZERO_LINE_WIDTH = 0.9
GRID_COLOR = '#E7E7E7'
SHOW_GRID = True
SHOW_NA = True
FIG_SIZE = (11.2, 5.1)
FIG_LEFT = 0.105
FIG_RIGHT = 0.99
FIG_BOTTOM = 0.14
FIG_TOP = 0.98
FIG_WSPACE = 0.28
DPI = 900
FORMATS = ['png', 'pdf', 'svg', 'tiff']
XLIMS = {}

def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ['utf-8-sig', 'gb18030']:
        try:
            return pd.read_csv(path, encoding=encoding, dtype={'Code': 'string'})
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Cannot decode CSV: {path}')

def normalize_codes(values) -> pd.Series:
    result = pd.Series(values).astype('string').str.strip().str.replace('\\.0+$', '', regex=True)
    invalid = result.isna() | result.isin(['', 'nan', 'None', '<NA>'])
    if invalid.any():
        raise ValueError('Missing or empty county Code.')
    return result.astype(str)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def bh_adjust(pvalues) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    result = np.full(values.shape, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return result
    order = finite_indices[np.argsort(values[finite_indices])]
    adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
    result[order] = np.minimum(1.0, np.minimum.accumulate(adjusted[::-1])[::-1])
    return result

def load_balanced_panel(policy_file: Path, model_file: Path):
    policy = read_csv(policy_file)
    model = read_csv(model_file)
    for label, frame in [('policy', policy), ('model', model)]:
        frame.columns = frame.columns.str.strip()
        if not {'Code', 'Year'}.issubset(frame.columns):
            raise ValueError(f'{label}: missing Code or Year.')
        frame['Code'] = normalize_codes(frame['Code']).to_numpy()
        years = pd.to_numeric(frame['Year'], errors='raise')
        if not np.isfinite(years).all() or not (years == np.floor(years)).all():
            raise ValueError(f'{label}: invalid Year.')
        frame['Year'] = years.astype(int)
        if frame.duplicated(['Code', 'Year']).any():
            raise ValueError(f'{label}: duplicate Code-Year rows.')
    if not set(POLICIES).issubset(policy.columns):
        raise ValueError('Policy file is missing one or more policy columns.')
    required_model = set(OUTCOMES + RAW_COVARIATES)
    if not required_model.issubset(model.columns):
        missing = sorted(required_model - set(model.columns))
        raise ValueError(f'Model file is missing columns: {missing}')
    policy = policy[['Code', 'Year'] + POLICIES].copy()
    model = model[['Code', 'Year'] + RAW_COVARIATES + OUTCOMES].copy()
    key_check = policy[['Code', 'Year']].merge(model[['Code', 'Year']], how='outer', indicator=True)
    if not key_check['_merge'].eq('both').all():
        raise ValueError('Policy and model files have different Code-Year keys.')
    data = model.merge(policy, on=['Code', 'Year'], validate='one_to_one').sort_values(['Code', 'Year']).reset_index(drop=True)
    for column in POLICIES + RAW_COVARIATES + OUTCOMES:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    for column in POLICIES:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f'{column}: policy must be binary and nonmissing.')
    county_ids = np.sort(data['Code'].unique())
    years = np.sort(data['Year'].unique())
    if not np.array_equal(years, np.arange(years[0], years[-1] + 1)):
        raise ValueError('Calendar years are not consecutive.')
    if len(data) != len(county_ids) * len(years):
        raise ValueError('A balanced Code-Year panel is required.')

    def pivot(column):
        return data.pivot(index='Code', columns='Year', values=column).loc[county_ids, years].to_numpy(float)
    policy_array = np.stack([pivot(column) for column in POLICIES], axis=2)
    if (np.diff(policy_array, axis=1) < 0).any():
        raise ValueError('Policy reversal found; absorbing-treatment DID is inappropriate.')
    outcome_arrays = {}
    bad_outcomes = {}
    for outcome in OUTCOMES:
        raw = pivot(outcome)
        invalid = ~np.isfinite(raw) | (raw <= 0)
        bad_outcomes[outcome] = int(invalid.sum())
        logged = np.full(raw.shape, np.nan, dtype=float)
        logged[~invalid] = -np.log(raw[~invalid])
        outcome_arrays[outcome] = logged
    for column in ['PGDP', 'POPD']:
        values = pd.to_numeric(data[column], errors='coerce')
        data[f'log_{column}'] = np.log(values.where(values > 0))
    for outcome in ['ECI', 'CPI']:
        values = pd.to_numeric(data[outcome], errors='coerce')
        data[f'log_{outcome}'] = np.log(values.where(values > 0))
    covariate_array = np.stack([pivot(column) for column in COVARIATES], axis=2)
    covariate_missing = {name: int((~np.isfinite(covariate_array[:, :, j])).sum()) for j, name in enumerate(COVARIATES)}
    return {'data': data, 'county_ids': county_ids, 'years': years, 'policies': policy_array, 'policy_counts': policy_array.sum(axis=2), 'outcomes': outcome_arrays, 'covariates': covariate_array, 'covariate_names': list(COVARIATES), 'bad_outcomes': bad_outcomes, 'covariate_missing': covariate_missing}

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
    rhs = x.T @ (control * y) / n
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

def dr_att(x: np.ndarray, treatment: np.ndarray, delta: np.ndarray):
    d = np.asarray(treatment, dtype=float)
    y = np.asarray(delta, dtype=float)
    n = len(d)
    if d.sum() < 1 or (1.0 - d).sum() < 1:
        raise ValueError('dr_requires_both_groups')
    propensity = fit_propensity(x, d)
    outcome_fit = fit_control_outcome_regression(x, d, y)
    p_raw = np.asarray(propensity['ps'], dtype=float)
    p_safe = np.clip(p_raw, PS_CLIP, 1.0 - PS_CLIP)
    odds = p_safe / (1.0 - p_safe)
    control = 1.0 - d
    gamma = outcome_fit['gamma']
    residual = y - x @ gamma
    a = d / d.sum()
    control_mass = float(np.sum(control * odds))
    if not np.isfinite(control_mass) or control_mass <= 1e-20:
        raise ValueError('nonpositive_propensity_control_weight_mass')
    c = control * odds / control_mass
    mt = float(a @ residual)
    mc = float(c @ residual)
    att = mt - mc
    direct = d / d.mean() * (residual - mt)
    direct -= control * odds / np.mean(control * odds) * (residual - mc)
    clip_derivative = (p_raw > PS_CLIP) & (p_raw < 1.0 - PS_CLIP)
    derivative_beta = -(c * (residual - mc) * clip_derivative) @ x
    derivative_gamma = -(a - c) @ x
    info_inv = np.linalg.pinv(propensity['information'])
    gram_inv = np.linalg.pinv(outcome_fit['gram'])
    if_beta = x * (d - p_raw)[:, None] @ info_inv
    if_gamma = x * (control * residual)[:, None] @ gram_inv
    influence = direct + if_beta @ derivative_beta + if_gamma @ derivative_gamma
    influence = np.asarray(influence, dtype=float)
    influence -= influence.mean()
    if not np.isfinite(att) or not np.isfinite(influence).all():
        raise ValueError('nonfinite_ATT_or_influence')
    return {'att': float(att), 'influence': influence, 'ps_raw': p_raw, 'ps_safe': p_safe, 'control_weights': c, 'propensity': propensity, 'outcome_fit': outcome_fit, 'gamma': gamma}

def load_queen_adjacency(shp_file: Path, shp_code_field: str, county_ids: np.ndarray, encoding=None):
    try:
        import geopandas as gpd
        import shapely
    except ImportError as exc:
        raise ImportError('Install geopandas and shapely>=2 before running this script.') from exc
    if int(shapely.__version__.split('.')[0]) < 2:
        raise ImportError('shapely>=2 is required.')
    if not shp_file.is_file():
        raise FileNotFoundError(f'Shapefile not found: {shp_file}')
    kwargs = {'encoding': encoding} if encoding else {}
    geo = gpd.read_file(shp_file, **kwargs)
    if shp_code_field not in geo.columns:
        raise ValueError(f'SHP missing field {shp_code_field}; available fields: {geo.columns.tolist()}')
    if geo.crs is None:
        raise ValueError('SHP CRS is missing; do not guess the coordinate system.')
    geo = geo[[shp_code_field, 'geometry']].copy()
    geo['Code'] = normalize_codes(geo[shp_code_field]).to_numpy()
    if shp_code_field != 'Code':
        geo = geo.drop(columns=shp_code_field)
    missing = sorted(set(county_ids) - set(geo['Code']))
    if missing:
        raise ValueError(f'{len(missing)} panel counties are missing from SHP; examples: {missing[:20]}')
    extra = sorted(set(geo['Code']) - set(county_ids))
    geo = geo.loc[geo['Code'].isin(county_ids)].copy()
    if geo.geometry.isna().any() or geo.geometry.is_empty.any():
        raise ValueError('Matched counties contain missing or empty geometries.')
    if not geo.geom_type.isin(['Polygon', 'MultiPolygon']).all():
        raise ValueError('County boundaries must be Polygon or MultiPolygon.')
    if not geo.is_valid.all():
        raise ValueError('Invalid county geometries; repair them explicitly first.')
    duplicate_parts = int(geo['Code'].duplicated().sum())
    geo = geo.dissolve(by='Code').loc[county_ids]
    if not geo.is_valid.all():
        raise ValueError('Invalid geometry after same-Code dissolve.')
    geometries = geo.geometry.to_numpy()
    metric_geo = geo.to_crs('EPSG:6933')
    county_areas = metric_geo.area.to_numpy()
    if not np.isfinite(county_areas).all() or (county_areas <= 0).any():
        raise ValueError('Nonpositive or nonfinite county areas.')
    pairs = shapely.STRtree(geometries).query(geometries, predicate='intersects')
    left, right = pairs[:, pairs[0] < pairs[1]]
    adjacency = np.zeros((len(county_ids), len(county_ids)), dtype=float)
    exact_touch = shapely.touches(geometries[left], geometries[right])
    for i, j in zip(left[exact_touch], right[exact_touch]):
        adjacency[i, j] = adjacency[j, i] = 1.0
    overlap_i = left[~exact_touch]
    overlap_j = right[~exact_touch]
    intersections = shapely.intersection(geometries[overlap_i], geometries[overlap_j])
    metric_intersections = gpd.GeoSeries(intersections, crs=geo.crs).to_crs('EPSG:6933')
    overlap_records = []
    rejected = []
    for i, j, intersection in zip(overlap_i, overlap_j, metric_intersections):
        area = float(intersection.area)
        ratio = area / min(county_areas[i], county_areas[j])
        accepted = bool(np.isfinite(area) and area <= OVERLAP_MAX_M2 and (ratio <= OVERLAP_MAX_RATIO))
        record = {'Code_a': str(county_ids[i]), 'Code_b': str(county_ids[j]), 'overlap_m2': area, 'fraction_of_smaller_county': ratio, 'accepted': accepted}
        overlap_records.append(record)
        if accepted:
            adjacency[i, j] = adjacency[j, i] = 1.0
        else:
            rejected.append(record)
    if rejected:
        raise ValueError(f'{len(rejected)} county overlaps exceed the declared tolerance; do not silently repair or discard them. Examples: {rejected[:5]}')
    degree = adjacency.sum(axis=1)
    audit = {'shp': str(shp_file), 'shp_code_field': shp_code_field, 'matched_counties': len(county_ids), 'extra_shp_counties': len(extra), 'extra_shp_code_examples': extra[:20], 'same_code_parts_dissolved': duplicate_parts, 'crs': str(geo.crs), 'exact_touch_pairs': int(exact_touch.sum()), 'accepted_tiny_overlap_pairs': len(overlap_records), 'isolated_counties': int((degree == 0).sum()), 'overlap_max_m2': OVERLAP_MAX_M2, 'overlap_max_ratio': OVERLAP_MAX_RATIO, 'adjacency': 'Queen: exact touches plus audited tiny overlaps; no geometry editing; sample counties only', 'overlap_records': overlap_records, 'geopandas_version': gpd.__version__, 'shapely_version': shapely.__version__}
    return (sparse.csr_matrix(adjacency), degree, audit)

def first_local_crossing(counts, threshold):
    reached = counts >= threshold
    onset = np.full(counts.shape[0], counts.shape[1], dtype=int)
    ever = reached.any(axis=1)
    onset[ever] = reached[ever].argmax(axis=1)
    return onset

def neighbor_mean_fixed(values, adjacency, degree):
    a = np.asarray(values, dtype=float)
    shape = a.shape
    flat = a.reshape(shape[0], -1)
    valid = np.isfinite(flat)
    sums = adjacency @ np.where(valid, flat, 0.0)
    observed = adjacency @ valid.astype(float)
    result = np.full(sums.shape, np.nan)
    ok = (degree[:, None] > 0) & (observed == degree[:, None])
    np.divide(sums, degree[:, None], out=result, where=ok)
    return result.reshape(shape)

def build_source_panel(panel, adjacency, degree):
    if adjacency.diagonal().any():
        raise ValueError('Self-neighbors are forbidden.')
    if (adjacency != adjacency.T).nnz:
        raise ValueError('Queen adjacency must be symmetric.')
    w = sparse.diags(np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)) @ adjacency
    neighboring_cov = neighbor_mean_fixed(panel['covariates'], adjacency, degree)
    neighboring_count = neighbor_mean_fixed(panel['policy_counts'], adjacency, degree)
    two_step = (w @ w).tocsr()
    two_step.setdiag(0.0)
    two_step.eliminate_zeros()
    other_mass = np.asarray(two_step.sum(axis=1)).ravel()
    other_count = np.zeros_like(panel['policy_counts'], dtype=float)
    np.divide(two_step @ panel['policy_counts'], other_mass[:, None], out=other_count, where=other_mass[:, None] > 0)
    cov = np.concatenate([panel['covariates'], neighboring_cov, panel['policy_counts'][:, :, None], neighboring_count[:, :, None], other_count[:, :, None]], axis=2)
    names = ['source_' + name for name in panel['covariate_names']] + ['neighbor_' + name for name in panel['covariate_names']] + ['source_policy_count', 'neighbor_policy_count', 'other_source_policy_count']
    return {**panel, 'neighbor_outcomes': {name: neighbor_mean_fixed(arr, adjacency, degree) for name, arr in panel['outcomes'].items()}, 'source_covariates': cov, 'source_covariate_names': names, 'other_sources': two_step}

def source_masks(onset, g, t, y_neighbor, policies, adjacency, degree, min_pre, neighbor_rule):
    post = t >= g
    b = g - 1 if post else t - 1
    if b < 0 or g < min_pre:
        return None
    changed = np.any(policies[:, t, :] != policies[:, b, :], axis=1)
    changed_neighbor_n = np.asarray(adjacency @ changed.astype(float)).ravel()
    neighbor_stable = changed_neighbor_n == 0
    pre_start = g - min_pre if post else max(0, b - min_pre + 1)
    if not post and b - pre_start + 1 < min_pre:
        return None
    complete = np.isfinite(y_neighbor[:, pre_start:b + 1]).all(axis=1) & np.isfinite(y_neighbor[:, t])
    eligible = (degree > 0) & complete
    if neighbor_rule == 'stable':
        eligible &= neighbor_stable
    control_cutoff = t if post else g
    potential_treated = onset == g
    potential_control = onset > control_cutoff
    return {'baseline_index': b, 'post': post, 'treated': potential_treated & eligible, 'control': potential_control & eligible, 'n_raw_treated_sources': int(potential_treated.sum()), 'n_raw_control_sources': int(potential_control.sum()), 'n_treated_excluded_missing_neighbor_outcome': int((potential_treated & ~complete).sum()), 'n_treated_excluded_neighbor_policy_change': int((potential_treated & ~neighbor_stable).sum()), 'changed_neighbors': changed_neighbor_n}

def inference(estimate, score, multipliers, treated, control):
    active = len(treated | control)
    ans = {'se_negative_log': np.nan, 'ci_low_negative_log': np.nan, 'ci_high_negative_log': np.nan, 'p_value': np.nan, 'inference_ok': False, 'cluster': 'source_Code', 'n_active_source_counties': active, 'inference_note': 'source_Code_independence_assumed_not_spatially_robust'}
    if min(len(treated), len(control)) < MIN_INFERENCE_PER_ARM:
        ans['inference_note'] += ';fewer_than_minimum_sources_in_one_arm'
        return ans
    variance = float(score @ score) * active / (active - 1)
    if not np.isfinite(variance) or variance <= 1e-20:
        ans['inference_note'] += ';degenerate_variance'
        return ans
    se = np.sqrt(variance)
    draws = multipliers @ score * np.sqrt(active / (active - 1))
    critical = float(np.quantile(np.abs(draws / se), 0.95))
    ans.update(se_negative_log=se, ci_low_negative_log=estimate - critical * se, ci_high_negative_log=estimate + critical * se, p_value=float((1 + (np.abs(draws) >= abs(estimate)).sum()) / (len(draws) + 1)), inference_ok=True)
    return ans

def source_cell(panel, adjacency, degree, onset, threshold, outcome, g, t, min_pre, neighbor_rule, multipliers):
    y = panel['neighbor_outcomes'][outcome]
    masks = source_masks(onset, g, t, y, panel['policies'], adjacency, degree, min_pre, neighbor_rule)
    if masks is None:
        return (None, None, [], [])
    b = masks['baseline_index']
    treated, control = (masks['treated'], masks['control'])
    ti, ci = (np.flatnonzero(treated), np.flatnonzero(control))
    nt, nc = (len(ti), len(ci))
    common = {'threshold': threshold, 'outcome': outcome, 'min_pre': min_pre, 'cohort': int(panel['years'][g]), 'year': int(panel['years'][t]), 'baseline': int(panel['years'][b]), 'event_time': int(t - g), 'phase': 'post' if masks['post'] else 'pre_adjacent_year_placebo', 'neighbor_policy_rule': neighbor_rule, 'n_treated_sources': nt, 'n_control_sources': nc, **{k: v for k, v in masks.items() if k.startswith('n_')}, 'status': 'not_estimable', 'reason': ''}
    if nt < MIN_TREATED or nc < MIN_CONTROL:
        common['reason'] = 'insufficient_treated_or_not_yet_crossed_source_counties'
        return (None, common, [], [])
    ix = np.flatnonzero(treated | control)
    d = treated[ix].astype(float)
    delta = y[ix, t] - y[ix, b]
    raw = panel['source_covariates'][ix, b, :]
    names = panel['source_covariate_names']
    try:
        imputed, imputed_names, imputation = impute_covariates(raw, names)
        x, design_names = full_rank_design(imputed, imputed_names)
        fit = dr_att(x, d, delta)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        common['reason'] = str(exc)
        return (None, common, [], [])
    n = len(onset)
    score = np.zeros(n)
    score[ix] = fit['influence'] / len(ix)
    ti_set, ci_set = (set(ti.tolist()), set(ci.tolist()))
    tr = np.asarray(adjacency[ti].sum(axis=0)).ravel() > 0
    cr = np.asarray(adjacency[ci].sum(axis=0)).ravel() > 0
    ps, weights = (fit['ps_raw'], fit['control_weights'])
    ps_extreme = (ps < PS_EXTREME_CUTOFF) | (ps > 1 - PS_EXTREME_CUTOFF)
    ess = float(1.0 / np.sum(weights ** 2))
    changed = np.any(panel['policies'][:, t, :] != panel['policies'][:, b, :], axis=1)
    competing_change = panel['other_sources'] @ changed.astype(float) > 0
    observed_names = set(design_names)
    warnings = []
    if min(nt, nc) < SMALL_SAMPLE_WARNING:
        warnings.append('one_source_arm_below_20')
    if ess < SMALL_SAMPLE_WARNING:
        warnings.append('weighted_control_ESS_below_20')
    if ps_extreme.any():
        warnings.append('extreme_PS_retained')
    if fit['propensity']['ridge_lambda'] or fit['outcome_fit']['ridge_lambda']:
        warnings.append('ridge_fallback_inference_conditional_on_fitted_penalties')
    if np.any(tr & cr):
        warnings.append('treated_control_neighborhoods_overlap')
    inf = inference(fit['att'], score, multipliers, ti_set, ci_set)
    row = {**common, 'status': 'estimated', 'att_negative_log': fit['att'], **inf, 'n_treated_recipient_counties': int(tr.sum()), 'n_control_recipient_counties': int(cr.sum()), 'n_shared_recipient_counties': int((tr & cr).sum()), 'mean_treated_neighbor_count': float(degree[ti].mean()), 'share_treated_sources_with_other_source_policy_changes': float(np.mean(competing_change[ti])), 'share_control_sources_with_other_source_policy_changes': float(np.mean(competing_change[ci])), 'share_treated_sources_with_recipient_policy_changes': float(np.mean(masks['changed_neighbors'][ti] > 0)), 'n_design_columns': x.shape[1], 'dropped_constant_or_dependent_columns': ';'.join((name for name in imputed_names if name not in observed_names)), 'n_covariate_missing_values_imputed': int((~np.isfinite(raw)).sum()), 'imputation_details': json.dumps(imputation, ensure_ascii=False), 'propensity_method': fit['propensity']['method'], 'propensity_ridge_lambda': fit['propensity']['ridge_lambda'], 'propensity_primary_error': fit['propensity']['primary_error'], 'outcome_regression_method': fit['outcome_fit']['method'], 'outcome_ridge_lambda': fit['outcome_fit']['ridge_lambda'], 'ps_treated_min': float(ps[d == 1].min()), 'ps_treated_max': float(ps[d == 1].max()), 'ps_control_min': float(ps[d == 0].min()), 'ps_control_max': float(ps[d == 0].max()), 'ps_extreme_n': int(ps_extreme.sum()), 'ps_numerically_clipped_n': int((ps != fit['ps_safe']).sum()), 'control_weight_ESS': ess, 'max_normalized_control_weight': float(weights.max()), 'small_sample_or_numerical_warning': ';'.join(warnings)}
    diagnostic_rows = []
    for pos in np.flatnonzero(ps_extreme | (ps != fit['ps_safe'])):
        diagnostic_rows.append({'threshold': threshold, 'outcome': outcome, 'cohort': row['cohort'], 'year': row['year'], 'phase': row['phase'], 'source_Code': panel['county_ids'][ix[pos]], 'treated': int(d[pos]), 'ps_raw': ps[pos], 'ps_for_odds': fit['ps_safe'][pos], 'normalized_control_weight': weights[pos], 'neighbor_outcome_change_negative_log': delta[pos], 'predicted_neighbor_change_negative_log': float(x[pos] @ fit['gamma'])})
    balance = []
    for j, name in enumerate(imputed_names):
        values = imputed[:, j]
        sd = np.sqrt((np.var(values[d == 1]) + np.var(values[d == 0])) / 2)
        balance.append({'threshold': threshold, 'outcome': outcome, 'cohort': row['cohort'], 'year': row['year'], 'phase': row['phase'], 'covariate': name, 'smd_before': (values[d == 1].mean() - values[d == 0].mean()) / sd if sd > 1e-12 else np.nan, 'smd_after': (values[d == 1].mean() - weights @ values) / sd if sd > 1e-12 else np.nan})
    cell = {'row': row, 'att': fit['att'], 'score': score, 'treated': ti_set, 'control': ci_set, 'n_treated': nt, 'recipients': set(np.flatnonzero(tr).tolist())}
    return (cell, row, diagnostic_rows, balance)

def aggregate_source_cells(cells, multipliers):
    groups = {}
    for cell in cells:
        groups.setdefault(cell['row']['cohort'], []).append(cell)
    parts = list(groups.values())
    mass = np.array([np.mean([c['n_treated'] for c in group]) for group in parts])
    mass /= mass.sum()
    att = sum((w * np.mean([c['att'] for c in group]) for w, group in zip(mass, parts)))
    score = sum((w * np.mean([c['score'] for c in group], axis=0) for w, group in zip(mass, parts)))
    treated = set().union(*(c['treated'] for c in cells))
    control = set().union(*(c['control'] for c in cells))
    recipients = set().union(*(c['recipients'] for c in cells))
    row = {'status': 'estimated', 'reason': '', 'att_negative_log': float(att), **inference(att, score, multipliers, treated, control), 'n_treated_sources': len(treated), 'n_control_sources': len(control), 'n_treated_recipient_counties': len(recipients), 'n_cohorts': len(groups), 'n_cells': len(cells), 'min_cell_treated_sources': min((len(c['treated']) for c in cells)), 'min_cell_control_sources': min((len(c['control']) for c in cells)), 'min_control_weight_ESS': min((c['row']['control_weight_ESS'] for c in cells)), 'cells_with_extreme_PS': sum((c['row']['ps_extreme_n'] > 0 for c in cells)), 'cells_using_ridge': sum((c['row']['propensity_ridge_lambda'] > 0 or c['row']['outcome_ridge_lambda'] > 0 for c in cells)), 'cells_with_clipped_PS': sum((c['row']['ps_numerically_clipped_n'] > 0 for c in cells)), 'cells_with_shared_recipients': sum((c['row']['n_shared_recipient_counties'] > 0 for c in cells)), 'small_sample_or_numerical_warning': ';'.join(sorted(set((item for c in cells for item in c['row']['small_sample_or_numerical_warning'].split(';') if item)))), 'aggregation': 'equal_years_within_cohort;mean_retained_source_count_cohort_weights', 'inference_limit': 'fixed_aggregation_weights;fixed_imputation_and_penalty_selection;overlapping_neighborhoods_not_corrected'}
    return (row, score, treated, control)

def pretrend_joint(pre_results, multipliers):
    base = {'pretrend_p_value': np.nan, 'pretrend_status': 'unavailable', 'n_pre_events': len(pre_results)}
    usable = [r for r in pre_results if r[0]['inference_ok']]
    if len(usable) < 2:
        base['reason'] = 'fewer_than_two_inferable_pre_events'
        return base
    vec = np.array([r[0]['att_negative_log'] for r in usable])
    score = np.column_stack([r[1] for r in usable])
    active = set().union(*(r[2] | r[3] for r in usable))
    factor = len(active) / (len(active) - 1)
    cov = score.T @ score * factor
    rank = np.linalg.matrix_rank(cov)
    if rank != len(vec):
        base['reason'] = 'rank_deficient_pretrend_covariance'
        return base
    inv = np.linalg.inv(cov)
    observed = float(vec @ inv @ vec)
    draws = multipliers @ score * np.sqrt(factor)
    stats = np.einsum('bi,ij,bj->b', draws, inv, draws)
    base.update(pretrend_status='estimated', pretrend_statistic=observed, pretrend_p_value=float((1 + (stats >= observed).sum()) / (len(stats) + 1)), reason='', note='source_Code_independence;adjacent_year_placebos')
    return base

def estimate_source_effects(panel, adjacency, degree, args):
    n = len(panel['county_ids'])
    multipliers = np.random.default_rng(args.seed).choice([-1.0, 1.0], size=(args.bootstrap, n))
    overall, group_time, dynamic, pretests, extremes, balance, onset_records = ([], [], [], [], [], [], [])
    for threshold in THRESHOLDS:
        onset = first_local_crossing(panel['policy_counts'], threshold)
        for i, code in enumerate(panel['county_ids']):
            g = onset[i]
            onset_records.append({'threshold': threshold, 'source_Code': code, 'first_local_crossing_year': int(panel['years'][g]) if g < len(panel['years']) else np.nan, 'initially_at_or_above_threshold': bool(g == 0), 'never_crossed': bool(g == len(panel['years'])), 'neighbor_count': int(degree[i]), 'baseline_local_policy_count': panel['policy_counts'][i, g - 1] if 0 < g < len(panel['years']) else np.nan, 'entry_local_policy_count': panel['policy_counts'][i, g] if g < len(panel['years']) else np.nan})
        cohorts = sorted(set(onset[(onset >= args.min_pre) & (onset < len(panel['years']))]))
        for outcome in OUTCOMES:
            pass
            cells = []
            rows_this = []
            for g in cohorts:
                times = list(range(max(1, g - args.pre_lags), g))
                times += list(range(g, min(g + args.max_post + 1, len(panel['years']))))
                for t in times:
                    cell, row, diagnostics, smd = source_cell(panel, adjacency, degree, onset, threshold, outcome, g, t, args.min_pre, args.neighbor_policy_rule, multipliers)
                    if row is not None:
                        group_time.append(row)
                        rows_this.append(row)
                    extremes.extend(diagnostics)
                    balance.extend(smd)
                    if cell is not None:
                        cells.append(cell)
            common = {'threshold': threshold, 'design': f'source_local_ge{threshold}', 'outcome': outcome, 'min_pre': args.min_pre, 'neighbor_policy_rule': args.neighbor_policy_rule, 'estimand': 'source_local_threshold_entry_effect_on_fixed_neighbor_mean_negative_log', 'negative_means': 'neighbor_improvement', 'positive_means': 'neighbor_deterioration', 'raw_newly_crossing_sources': int(((onset > 0) & (onset < len(panel['years']))).sum()), 'initially_crossed_sources_excluded': int((onset == 0).sum())}
            post = [c for c in cells if c['row']['phase'] == 'post']
            if post:
                agg, _, _, _ = aggregate_source_cells(post, multipliers)
            else:
                reasons = {}
                for row in rows_this:
                    if row['phase'] == 'post':
                        reasons[row['reason']] = reasons.get(row['reason'], 0) + 1
                agg = {'status': 'not_estimable', 'reason': json.dumps(reasons) if reasons else 'no_new_onset_with_required_preperiod', 'att_negative_log': np.nan, 'se_negative_log': np.nan, 'ci_low_negative_log': np.nan, 'ci_high_negative_log': np.nan, 'p_value': np.nan, 'inference_ok': False}
            overall.append({**common, **agg})
            pre_results = []
            for event in sorted(set((c['row']['event_time'] for c in cells))):
                subset = [c for c in cells if c['row']['event_time'] == event]
                a, score, tr, co = aggregate_source_cells(subset, multipliers)
                dynamic.append({**common, 'event_time': event, 'phase': 'post' if event >= 0 else 'pre_adjacent_year_placebo', **a})
                if event < 0:
                    pre_results.append((a, score, tr, co))
            pretests.append({**common, **pretrend_joint(pre_results, multipliers)})
    overall = pd.DataFrame(overall)
    overall['q_BH_overall'] = bh_adjust(overall['p_value'])
    pretests = pd.DataFrame(pretests)
    overall = overall.merge(pretests[['threshold', 'outcome', 'pretrend_p_value', 'pretrend_status']], on=['threshold', 'outcome'], validate='one_to_one')
    dynamic = pd.DataFrame(dynamic)
    if not dynamic.empty:
        dynamic['q_BH_dynamic'] = bh_adjust(dynamic['p_value'])
    return {'overall_ATT_negative_log': overall, 'group_time_ATT_negative_log': pd.DataFrame(group_time), 'event_study_negative_log': dynamic, 'pretrend_tests': pretests, 'extreme_propensity_source_counties': pd.DataFrame(extremes, columns=None if extremes else ['threshold', 'outcome', 'cohort', 'year', 'source_Code', 'ps_raw']), 'group_time_covariate_balance': pd.DataFrame(balance), 'source_threshold_onsets': pd.DataFrame(onset_records)}

def transform_to_original_percent(frame):
    result = frame.copy()
    att = pd.to_numeric(result.get('att_negative_log', pd.Series(np.nan, index=result.index)), errors='coerce')
    ci_low_nlog = pd.to_numeric(result.get('ci_low_negative_log', pd.Series(np.nan, index=result.index)), errors='coerce')
    ci_high_nlog = pd.to_numeric(result.get('ci_high_negative_log', pd.Series(np.nan, index=result.index)), errors='coerce')
    with np.errstate(over='ignore', invalid='ignore'):
        result['att'] = 100.0 * np.expm1(-att)
        result['ci_low'] = 100.0 * np.expm1(-ci_high_nlog)
        result['ci_high'] = 100.0 * np.expm1(-ci_low_nlog)
    result['att_scale'] = '100*(exp(-ATT)-1);geometric_scale_neighbor_original_percent_change'
    result['negative_means'] = 'neighbor_improvement'
    result['positive_means'] = 'neighbor_deterioration'
    return result

def finite(value) -> bool:
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def is_true(value) -> bool:
    return str(value).lower() == 'true'

def setup_fonts():
    path = font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY), fallback_to_default=False)
    if font_manager.FontProperties(fname=path).get_name() != FONT_FAMILY:
        raise RuntimeError('Times New Roman is unavailable.')
    plt.rcParams.update({'font.family': FONT_FAMILY, 'font.size': FONT_SIZE, 'axes.labelsize': AXIS_LABEL_SIZE, 'xtick.labelsize': TICK_FONT_SIZE, 'ytick.labelsize': TICK_FONT_SIZE, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': AXIS_LINE_WIDTH, 'axes.unicode_minus': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'text.usetex': False, 'legend.frameon': False})
    for color in COLORS.values():
        mcolors.to_rgba(color)

def valid_ci(record) -> bool:
    if record.get('status') != 'estimated':
        return False
    if not is_true(record.get('inference_ok')):
        return False
    if not all((finite(record.get(column)) for column in ['att', 'ci_low', 'ci_high'])):
        return False
    if not float(record['ci_low']) <= float(record['att']) <= float(record['ci_high']):
        raise ValueError('Transformed CI does not bracket ATT: ' + str(record))
    return True

def axis_bounds(records, manual=None):
    values = [0.0]
    for record in records:
        if record.get('status') != 'estimated' or not finite(record.get('att')):
            continue
        values.append(float(record['att']))
        if valid_ci(record):
            values.extend([float(record['ci_low']), float(record['ci_high'])])
    low, high = (min(values), max(values))
    if manual is not None:
        if manual[0] >= manual[1] or manual[0] > low or manual[1] < high:
            raise ValueError('Manual x limits clip an estimate, CI or zero.')
        return manual
    span = high - low or 0.1
    return (low - 0.12 * span, high + 0.12 * span)

def style_axis(ax, xlabel):
    ax.axvline(0, color=ZERO_LINE_COLOR, linestyle='--', linewidth=ZERO_LINE_WIDTH, zorder=0)
    if SHOW_GRID:
        ax.grid(axis='x', color=GRID_COLOR, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=7)
    ax.set_xlabel(xlabel, labelpad=AXIS_LABEL_PAD)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    formatter = ScalarFormatter(useOffset=False, useMathText=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)

def draw_record(ax, record, y, color):
    if record.get('status') != 'estimated' or not finite(record.get('att')):
        if SHOW_NA:
            ax.text(0.0, y, 'NA', va='center', ha='center', fontsize=NA_FONT_SIZE, color='#888888', zorder=4, bbox=dict(facecolor='white', edgecolor='none', pad=0.15))
        return 'NA'
    estimate = float(record['att'])
    if valid_ci(record):
        low = float(record['ci_low'])
        high = float(record['ci_high'])
        ax.errorbar(estimate, y, xerr=[[estimate - low], [high - estimate]], fmt='o', color=color, markersize=MARKER_SIZE, elinewidth=CI_LINE_WIDTH, capsize=CAP_SIZE, markeredgewidth=0.8, zorder=3)
        return 'estimate_and_95CI'
    ax.plot(estimate, y, marker='x', color=color, markersize=MARKER_SIZE + 1, markeredgewidth=1.2, linestyle='none', zorder=3)
    return 'point_only'

def plot_figure(results: pd.DataFrame, output_dir: Path):
    setup_fonts()
    fig, axes = plt.subplots(1, len(OUTCOMES), figsize=FIG_SIZE, sharey=True, squeeze=False)
    fig.subplots_adjust(left=FIG_LEFT, right=FIG_RIGHT, bottom=FIG_BOTTOM, top=FIG_TOP, wspace=FIG_WSPACE)
    plotted_rows = []
    positions = list(range(len(THRESHOLDS)))
    labels = [f'At least {threshold}' for threshold in THRESHOLDS]
    for ax, outcome in zip(axes.ravel(), OUTCOMES):
        outcome_data = results.loc[results['outcome'].eq(outcome)].copy()
        records = []
        for threshold in THRESHOLDS:
            selected = outcome_data.loc[outcome_data['threshold'].eq(threshold)]
            if len(selected) > 1:
                raise ValueError(f'Duplicate result: threshold={threshold}, outcome={outcome}')
            if len(selected) == 1:
                records.append(selected.iloc[0].to_dict())
            else:
                records.append({'threshold': threshold, 'outcome': outcome, 'status': 'missing_source_row'})
        ax.set_xlim(axis_bounds(records, XLIMS.get(outcome)))
        for y, record in zip(positions, records):
            plotted_as = draw_record(ax, record, y, COLORS[outcome])
            plotted_rows.append({**record, 'plotted_as': plotted_as})
        ax.set_yticks(positions, labels)
        ax.set_ylim(len(THRESHOLDS) - 0.45, -0.55)
        style_axis(ax, f'{OUTCOME_LABELS[outcome]} ATT (%)')
    axes[0, 0].set_ylabel('', fontsize=AXIS_LABEL_SIZE)
    fig.canvas.draw()
    from matplotlib.text import Text
    for text in fig.findobj(Text):
        if text.get_text() and text.get_fontproperties().get_name() != FONT_FAMILY:
            raise RuntimeError('Unexpected font: ' + text.get_text())
    stem = 'Fig_local_policy_count_neighbor_effect'
    for file_format in FORMATS:
        fig.savefig(output_dir / f'{stem}.{file_format}', dpi=DPI, bbox_inches='tight', pad_inches=0.045)
    plt.close(fig)
    pd.DataFrame(plotted_rows).to_csv(output_dir / f'{stem}_source_data.csv', index=False, encoding='utf-8-sig')

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-file', type=Path, default=POLICY_FILE)
    parser.add_argument('--model-file', type=Path, default=MODEL_FILE)
    parser.add_argument('--shp-file', type=Path, default=SHP_FILE)
    parser.add_argument('--shp-code', default=SHP_CODE_FIELD)
    parser.add_argument('--shp-encoding', default=SHP_ENCODING)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--min-pre', type=int, choices=[1, 2, 3], default=MIN_PRE)
    parser.add_argument('--pre-lags', type=int, default=PRE_LAGS)
    parser.add_argument('--max-post', type=int, default=MAX_POST)
    parser.add_argument('--neighbor-policy-rule', choices=['stable', 'allow_changes'], default=NEIGHBOR_POLICY_RULE)
    parser.add_argument('--bootstrap', type=int, default=BOOTSTRAP)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--dpi', type=int, default=DPI)
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg', 'tiff'], default=FORMATS)
    return parser.parse_args()

def main():
    global DPI, FORMATS
    args = parse_args()
    if args.bootstrap < 99 or args.max_post < 0 or args.pre_lags < 0 or (args.dpi < 72):
        raise ValueError('Require bootstrap>=99, nonnegative time horizons, dpi>=72.')
    DPI, FORMATS = (args.dpi, args.formats)
    for path in [args.policy_file, args.model_file, args.shp_file]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.no_plot:
        setup_fonts()
    out = args.output_root / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    pass
    raw = load_balanced_panel(args.policy_file, args.model_file)
    pass
    adjacency, degree, audit = load_queen_adjacency(args.shp_file, args.shp_code, raw['county_ids'], args.shp_encoding)
    panel = build_source_panel(raw, adjacency, degree)
    neighbors = adjacency.tocoo()
    pd.DataFrame({'source_Code': raw['county_ids'][neighbors.row], 'recipient_Code': raw['county_ids'][neighbors.col], 'weight': 1.0 / degree[neighbors.row]}).to_csv(out / 'source_recipient_weights.csv', index=False, encoding='utf-8-sig')
    pass
    outputs = estimate_source_effects(panel, adjacency, degree, args)
    outputs['overall_ATT_neighbor_percent_change'] = transform_to_original_percent(outputs['overall_ATT_negative_log'])
    outputs['event_study_neighbor_percent_change'] = transform_to_original_percent(outputs['event_study_negative_log'])
    for name, df in outputs.items():
        df.to_csv(out / (name + '.csv'), index=False, encoding='utf-8-sig')
    if not args.no_plot:
        pass
        plot_figure(outputs['overall_ATT_neighbor_percent_change'], out)
    manifest = {'treatment_unit': 'policy-source county Code, NOT receiver county', 'treatment': 'first local count >= k; initially crossed sources excluded', 'controls': 'source counties whose own count has not reached k at comparison year', 'outcome': 'equal-weight fixed-neighbor mean of negative-log outcome; no self', 'missing_outcomes': 'require every fixed recipient observed; never fill with zero', 'policy_rule': args.neighbor_policy_rule, 'stability_note': 'stable applies to RECIPIENTS, not to treated local source', 'adjustment': panel['source_covariate_names'], 'scale': '100*(exp(-ATT)-1);geometric-scale original-outcome percent change', 'negative_means': 'neighbor improvement', 'positive_means': 'neighbor deterioration', 'min_pre': args.min_pre, 'max_post': args.max_post, 'pre_lags': args.pre_lags, 'bootstrap': args.bootstrap, 'seed': args.seed, 'cluster': 'source_Code', 'min_point_estimation_sources': [MIN_TREATED, MIN_CONTROL], 'min_inference_sources_per_arm': MIN_INFERENCE_PER_ARM, 'small_sample_warning_threshold': SMALL_SAMPLE_WARNING, 'PS_clipping_for_odds': PS_CLIP, 'extreme_PS_diagnostic_cutoff': PS_EXTREME_CUTOFF, 'ridge_grid': RIDGE_GRID, 'inference_limits': ['Cross-source independence is a working assumption.', 'Shared recipients and other spatial dependence are NOT corrected by Code clustering.', 'Intervals condition on selected penalty, imputation and aggregation weights.', 'Ridge/clipped fits are stabilized exploratory estimates; no guarantee of double robustness.', 'Very few sources or extreme weights can make inference unreliable.'], 'identification_limits': ['Requires conditional parallel trends, overlap, no anticipation and adequate exposure mapping.', 'Other sources may affect recipients concurrently; baseline adjustment does not prove absence of confounding.', 'Stable-recipient restriction selects policy-stable neighborhoods, not all source counties.', "allow_changes includes recipients' own policy changes, not a pure spillover.", 'Thresholds are nested; jumps and later policy additions are included.', 'No cross-threshold marginal or superadditive interpretation.'], 'aggregation': 'equal post years within source cohort; mean retained source counts across cohorts', 'multiple_testing': 'BH over available overall P values among 28 planned tests; dynamic tests a separate family; plotted CIs pointwise', 'figure_contract': '4 outcome facets, local threshold rows, points and 95% CI; no titles/footnotes', 'font': FONT_FAMILY, 'dpi': DPI, 'formats': FORMATS, 'input_sha256': {str(p): sha256(p) for p in [args.policy_file, args.model_file, Path(__file__)] + sorted(args.shp_file.parent.glob(args.shp_file.stem + '.*')) if p.is_file()}, 'synthetic_data': False}
    (out / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'data_audit.json').write_text(json.dumps({'n_source_counties': len(raw['county_ids']), 'years': raw['years'].tolist(), 'missing_or_nonpositive_outcomes': raw['bad_outcomes'], 'spatial': audit}, ensure_ascii=False, indent=2), encoding='utf-8')
    overall = outputs['overall_ATT_negative_log']
    pass
if __name__ == '__main__':
    main()
