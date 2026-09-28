from pathlib import Path
import hashlib
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
warnings.filterwarnings('ignore')
PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORK_DIR = PROJECT_ROOT
MODULE2_DIR = PROJECT_ROOT / 'data' / 'processed' / 'module2_time_space_results'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'cpsi_spatiotemporal'
COUNTY_FILE = MODULE2_DIR / '01_county_annual_panel.csv'
CITY_FILE = MODULE2_DIR / '02_city_annual_panel.csv'
PROVINCE_FILE = MODULE2_DIR / '03_province_annual_panel.csv'
NATIONAL_FILE = MODULE2_DIR / '04_national_annual_panel.csv'
WEIGHTS_EDGE_FILE = MODULE2_DIR / '10a_spatial_weights_edge_list.csv'
MODULE2_GPKG = MODULE2_DIR / 'Spatial_maps_and_locking.gpkg'
SHAPEFILE_PATH = str(PROJECT_ROOT / 'data' / 'spatial' / 'county_2023.shp')
SHAPE_CODE_FIELD = 'code'
BASE_YEAR = 2013
START_YEAR = 2013
END_YEAR = 2022
ANALYSIS_YEARS = list(range(BASE_YEAR + 1, END_YEAR + 1))
EPSILON = 1e-12
CPI_AUDIT_TOLERANCE = 1e-08
CPI_COMPONENT_RATIO_COLUMNS = ['SO2_National_Baseline_Ratio', 'NO2_National_Baseline_Ratio', 'CO_National_Baseline_Ratio', 'PM25_National_Baseline_Ratio']
PRIMARY_ABSOLUTE_QUANTILE = 0.99
SENSITIVITY_SCHEMES = {'P975_Absolute': 0.975, 'P990_Absolute_Primary': 0.99, 'P995_Absolute': 0.995}
SPATIAL_PERMUTATIONS = 999
SPATIAL_RANDOM_SEED = 20260826
SPATIAL_FDR_ALPHA = 0.05
LOCK_MIN_CONSECUTIVE_YEARS = 3
NEIGHBOR_IMPROVEMENT_SHARE_THRESHOLD = 0.5
MAP_YEARS = [2014, 2018, 2022]
GENERATE_ALL_DIAGNOSTIC_MAPS = True
FIGURE_DPI = 450
CPSI_DEFINITIONS = {'CPSIcp': {'pollution_change': 'Pollution_Improvement', 'label': 'CPSIcp', 'role': 'Primary'}, 'SO2': {'pollution_change': 'SO2_Improvement', 'label': 'CPSI-SO2', 'role': 'Primary_diagnostic'}, 'NO2': {'pollution_change': 'NO2_Improvement', 'label': 'CPSI-NO2', 'role': 'Primary_diagnostic'}, 'CO': {'pollution_change': 'CO_Improvement', 'label': 'CPSI-CO', 'role': 'Primary_diagnostic'}, 'PM25': {'pollution_change': 'PM25_Improvement', 'label': 'CPSI-PM2.5', 'role': 'Primary_diagnostic'}, 'O3': {'pollution_change': 'O3_Improvement', 'label': 'CPSI-O3', 'role': 'Supplementary_diagnostic'}, 'PM10': {'pollution_change': 'PM10_Improvement', 'label': 'CPSI-PM10', 'role': 'Supplementary_diagnostic'}}
CPSI_ORDER = list(CPSI_DEFINITIONS)
STATE_LABELS = {3: 'Both_improved', 2: 'Carbon_only', 1: 'Pollution_only', 0: 'Neither_improved'}
STATE_COLORS = {'Both_improved': '#009E73', 'Carbon_only': '#56B4E9', 'Pollution_only': '#E69F00', 'Neither_improved': '#BDBDBD'}
LISA_COLORS = {'HH': '#B2182B', 'LL': '#2166AC', 'HL': '#EF8A62', 'LH': '#67A9CF', 'Not_significant': '#D9D9D9'}

def load_spatial_dependencies():
    try:
        import geopandas as gpd
        from libpysal.weights import W
        from libpysal.weights.spatial_lag import lag_spatial
        from esda.moran import Moran, Moran_Local, Moran_BV
        from esda.join_counts import Join_Counts
    except ImportError as exc:
        raise ImportError(f'Spatial dependencies are missing. Run:\npip install geopandas shapely libpysal esda\nOriginal error: {exc}') from exc
    return {'gpd': gpd, 'W': W, 'lag_spatial': lag_spatial, 'Moran': Moran, 'Moran_Local': Moran_Local, 'Moran_BV': Moran_BV, 'Join_Counts': Join_Counts}

def read_csv_safely(path, dtype=None):
    if not path.exists():
        raise FileNotFoundError(f'Input file not found: {path}')
    last_error = None
    for encoding in ['utf-8-sig', 'utf-8', 'gb18030']:
        try:
            return pd.read_csv(path, encoding=encoding, dtype=dtype)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f'Unable to identify CSV encoding: {path}；{last_error}')

def normalize_code_series(series):
    return series.astype('string').str.strip().str.replace('\\.0+$', '', regex=True).str.extract('(\\d{6})', expand=False).str.zfill(6)

def stable_seed(*parts):
    text = '|'.join(map(str, parts)).encode('utf-8')
    digest = hashlib.sha256(text).hexdigest()[:8]
    return (SPATIAL_RANDOM_SEED + int(digest, 16)) % (2 ** 32 - 1)

def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    valid = np.isfinite(p_values)
    if valid.sum() == 0:
        return adjusted
    valid_p = p_values[valid]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    n = len(ranked)
    ranked_adjusted = ranked * n / np.arange(1, n + 1)
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0, 1)
    restored = np.empty(n)
    restored[order] = ranked_adjusted
    adjusted[valid] = restored
    return adjusted

def apply_publication_style():
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'axes.unicode_minus': False, 'savefig.dpi': FIGURE_DPI, 'savefig.bbox': 'tight'})

def save_figure(fig, stem):
    fig.savefig(OUTPUT_DIR / f'{stem}.png', dpi=FIGURE_DPI, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'{stem}.svg', bbox_inches='tight')
    plt.close(fig)

def audit_module2_cpi_identity(panel, level):
    id_map = {'County': ['Code'], 'City': ['Province_ID', 'City_ID'], 'Province': ['Province_ID'], 'National': ['National_ID']}
    id_columns = [column for column in id_map[level] if column in panel.columns]
    if not id_columns:
        raise ValueError(f'{level}The panel lacks identifier columns required to match the 2013 baseline.')
    calculated_cpi = np.exp(np.log(panel[CPI_COMPONENT_RATIO_COLUMNS]).mean(axis=1))
    cpi_formula_difference = np.abs(calculated_cpi.to_numpy(dtype=float) - panel['Composite_Pollution_Index'].to_numpy(dtype=float))
    base = panel.loc[panel['Year'] == BASE_YEAR, [*id_columns, 'Composite_Pollution_Index']].copy()
    if base.empty:
        raise ValueError(f'{level}The panel is missing {BASE_YEAR} baseline-year CPI records.')
    if base.duplicated(id_columns).any():
        raise ValueError(f'{level}The panel has {BASE_YEAR} nonunique baseline-year CPI records.')
    base = base.rename(columns={'Composite_Pollution_Index': '_Base_CPI'})
    check = panel.merge(base, on=id_columns, how='left', validate='many_to_one')
    if check['_Base_CPI'].isna().any():
        raise ValueError(f'{level}Some panel units are missing {BASE_YEAR} baseline-year CPI values.')
    if (check['Composite_Pollution_Index'] <= 0).any() or (check['_Base_CPI'] <= 0).any():
        raise ValueError(f'{level}The panel contains nonpositive CPI values and log improvement cannot be calculated.')
    improvement_from_cpi = -np.log(check['Composite_Pollution_Index'] / check['_Base_CPI'])
    improvement_difference = np.abs(improvement_from_cpi.to_numpy(dtype=float) - check['Pollution_Improvement'].to_numpy(dtype=float))
    trend_expected = 100.0 * np.exp(-check['Pollution_Improvement'].to_numpy(dtype=float))
    trend_difference = np.abs(trend_expected - check['Composite_Pollution_Trend_Index_2013_100'].to_numpy(dtype=float))
    audit = {'Level': level, 'n_rows': int(len(panel)), 'CPI_Formula_Max_Abs_Difference': float(np.max(cpi_formula_difference)), 'Improvement_Identity_Max_Abs_Difference': float(np.max(improvement_difference)), 'Trend_Index_Max_Abs_Difference': float(np.max(trend_difference)), 'Tolerance': CPI_AUDIT_TOLERANCE}
    audit['Passed'] = int(audit['CPI_Formula_Max_Abs_Difference'] <= CPI_AUDIT_TOLERANCE and audit['Improvement_Identity_Max_Abs_Difference'] <= CPI_AUDIT_TOLERANCE and (audit['Trend_Index_Max_Abs_Difference'] <= CPI_AUDIT_TOLERANCE))
    if audit['Passed'] != 1:
        raise ValueError(f'{level}The panel failed the CPI consistency audit: {audit}')
    return audit

def load_module2_panels():
    files = {'County': COUNTY_FILE, 'City': CITY_FILE, 'Province': PROVINCE_FILE, 'National': NATIONAL_FILE}
    panels = {}
    required_changes = {'Year', 'Carbon_Improvement', 'Composite_Pollution_Index', 'Composite_Pollution_Trend_Index_2013_100', *CPI_COMPONENT_RATIO_COLUMNS, *[definition['pollution_change'] for definition in CPSI_DEFINITIONS.values()]}
    audit_rows = []
    for level, path in files.items():
        dtype = {'Code': 'string'} if level == 'County' else None
        panel = read_csv_safely(path, dtype=dtype)
        missing = required_changes - set(panel.columns)
        if missing:
            raise ValueError(f'{path.name}Missing columns: {sorted(missing)}')
        panel['Year'] = pd.to_numeric(panel['Year'], errors='coerce')
        panel = panel[panel['Year'].between(START_YEAR, END_YEAR)].copy()
        panel['Year'] = panel['Year'].astype(int)
        if level == 'County':
            panel['Code'] = normalize_code_series(panel['Code'])
        for column in required_changes - {'Year'}:
            panel[column] = pd.to_numeric(panel[column], errors='coerce')
        if panel[list(required_changes)].isna().any().any():
            counts = panel[list(required_changes)].isna().sum()
            raise ValueError(f'{path.name} has missing CPSI inputs:\n{counts[counts > 0].to_string()}')
        audit_rows.append(audit_module2_cpi_identity(panel, level))
        panels[level] = panel
    return (panels, pd.DataFrame(audit_rows))

def fit_one_scale(values, absolute_quantile=PRIMARY_ABSOLUTE_QUANTILE):
    values = pd.to_numeric(pd.Series(np.asarray(values).reshape(-1)), errors='coerce').dropna().to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError('Cannot estimate Q scaling parameters from an empty array.')
    if not 0 < absolute_quantile <= 1:
        raise ValueError('absolute_quantile must be in (0, 1].')
    absolute_cap = float(np.quantile(np.abs(values), absolute_quantile))
    if absolute_cap <= EPSILON:
        absolute_cap = float(np.max(np.abs(values)))
    if absolute_cap <= EPSILON:
        raise ValueError('The improvement measure has no variation, so Q cannot be constructed.')
    return absolute_cap

def fit_scaling_parameters(county_panel, absolute_quantile=PRIMARY_ABSOLUTE_QUANTILE, scheme='Primary'):
    analysis = county_panel[county_panel['Year'] > BASE_YEAR]
    source_columns = {'Carbon': 'Carbon_Improvement'}
    source_columns.update({pcsi_type: definition['pollution_change'] for pcsi_type, definition in CPSI_DEFINITIONS.items()})
    rows = []
    for scale_name, source_column in source_columns.items():
        raw_values = pd.to_numeric(analysis[source_column], errors='coerce').dropna().to_numpy(dtype=float)
        absolute_cap = fit_one_scale(raw_values, absolute_quantile)
        n_clipped = int(np.sum(np.abs(raw_values) > absolute_cap))
        rows.append({'Scheme': scheme, 'Scale_Name': scale_name, 'Source_Column': source_column, 'Absolute_Quantile': absolute_quantile, 'Absolute_Cap': absolute_cap, 'n_county_years': int(len(raw_values)), 'n_clipped': n_clipped, 'Clipped_Percent': float(n_clipped / len(raw_values) * 100)})
    return pd.DataFrame(rows)

def transform_to_q(values, parameter_row):
    values = np.asarray(values, dtype=float)
    absolute_cap = float(parameter_row['Absolute_Cap'])
    if absolute_cap <= EPSILON:
        raise ValueError('Absolute_Cap must be positive.')
    winsorized = np.sign(values) * np.minimum(np.abs(values), absolute_cap)
    q = winsorized / absolute_cap
    return np.clip(q, -1.0, 1.0)

def calculate_pcsi(q_c, q_p):
    q_c = np.asarray(q_c, dtype=float)
    q_p = np.asarray(q_p, dtype=float)
    h_raw = 2.0 * np.abs(q_c * q_p) / (np.abs(q_c) + np.abs(q_p) + EPSILON)
    h = np.clip(h_raw, 0.0, 1.0)
    both = (q_c > 0) & (q_p > 0)
    carbon_only = (q_c > 0) & (q_p <= 0)
    pollution_only = (q_c <= 0) & (q_p > 0)
    state_code = np.select([both, carbon_only, pollution_only], [3, 2, 1], default=0).astype(int)
    pcsi = state_code.astype(float) + h
    state = pd.Series(state_code).map(STATE_LABELS).to_numpy()
    return (pcsi, h, h_raw, state_code, state)

def id_columns_for_level(panel, level):
    candidates = {'County': ['County', 'Code', 'Province_ID', 'City_ID', 'Prefecture'], 'City': ['Province_ID', 'City_ID', 'Prefecture'], 'Province': ['Province_ID'], 'National': ['National_ID']}[level]
    return [column for column in candidates if column in panel.columns]

def build_pcsi_panel(panel, level, scaling_parameters):
    data = panel[panel['Year'] > BASE_YEAR].copy()
    carbon_parameter = scaling_parameters[scaling_parameters['Scale_Name'] == 'Carbon'].iloc[0]
    q_c = transform_to_q(data['Carbon_Improvement'], carbon_parameter)
    identifiers = id_columns_for_level(data, level)
    frames = []
    for pcsi_type, definition in CPSI_DEFINITIONS.items():
        pollution_parameter = scaling_parameters[scaling_parameters['Scale_Name'] == pcsi_type].iloc[0]
        q_p = transform_to_q(data[definition['pollution_change']], pollution_parameter)
        pcsi, h, h_raw, state_code, state = calculate_pcsi(q_c, q_p)
        frame = data[[*identifiers, 'Year']].copy()
        frame['Level'] = level
        frame['CPSI_Type'] = pcsi_type
        frame['CPSI_Label'] = definition['label']
        frame['CPSI_Role'] = definition['role']
        frame['Carbon_Change_Raw'] = data['Carbon_Improvement'].to_numpy(dtype=float)
        frame['Pollution_Change_Raw'] = data[definition['pollution_change']].to_numpy(dtype=float)
        frame['Q_C'] = q_c
        frame['Q_P'] = q_p
        frame['H_raw_before_clip'] = h_raw
        frame['H'] = h
        frame['CPSI'] = pcsi
        frame['State_Code'] = state_code
        frame['State'] = state
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)

def validate_pcsi(pcsi_panel):
    checks = [('Q_C_outside_minus1_1', ((pcsi_panel['Q_C'] < -1) | (pcsi_panel['Q_C'] > 1)).sum()), ('Q_P_outside_minus1_1', ((pcsi_panel['Q_P'] < -1) | (pcsi_panel['Q_P'] > 1)).sum()), ('H_raw_above_1', (pcsi_panel['H_raw_before_clip'] > 1 + 1e-10).sum()), ('H_outside_0_1', ((pcsi_panel['H'] < 0) | (pcsi_panel['H'] > 1)).sum()), ('CPSI_outside_0_4', ((pcsi_panel['CPSI'] < 0) | (pcsi_panel['CPSI'] > 4)).sum()), ('Missing_CPSI', pcsi_panel['CPSI'].isna().sum()), ('Q_C_sign_changed', (np.sign(pcsi_panel['Carbon_Change_Raw']) != np.sign(pcsi_panel['Q_C'])).sum()), ('Q_P_sign_changed', (np.sign(pcsi_panel['Pollution_Change_Raw']) != np.sign(pcsi_panel['Q_P'])).sum())]
    audit = pd.DataFrame(checks, columns=['Check', 'Count'])
    if audit['Count'].sum() > 0:
        raise ValueError('The CPSI range audit failed:\n' + audit.to_string(index=False))
    return audit

def summarize_pcsi_by_year(pcsi_panel):
    rows = []
    for (level, pcsi_type, year), group in pcsi_panel.groupby(['Level', 'CPSI_Type', 'Year'], dropna=False):
        state_counts = group['State'].value_counts()
        rows.append({'Level': level, 'CPSI_Type': pcsi_type, 'Year': int(year), 'n_units': int(len(group)), 'CPSI_Mean': float(group['CPSI'].mean()), 'CPSI_Median': float(group['CPSI'].median()), 'CPSI_Q25': float(group['CPSI'].quantile(0.25)), 'CPSI_Q75': float(group['CPSI'].quantile(0.75)), 'CPSI_Min': float(group['CPSI'].min()), 'CPSI_Max': float(group['CPSI'].max()), 'H_Mean': float(group['H'].mean()), 'H_Median': float(group['H'].median()), 'Q_C_Median': float(group['Q_C'].median()), 'Q_P_Median': float(group['Q_P'].median()), 'Both_Improved_Percent': float(state_counts.get('Both_improved', 0) / len(group) * 100), 'Carbon_Only_Percent': float(state_counts.get('Carbon_only', 0) / len(group) * 100), 'Pollution_Only_Percent': float(state_counts.get('Pollution_only', 0) / len(group) * 100), 'Neither_Improved_Percent': float(state_counts.get('Neither_improved', 0) / len(group) * 100)})
    return pd.DataFrame(rows)

def unit_id_columns(level):
    return {'County': ['Code'], 'City': ['Province_ID', 'City_ID'], 'Province': ['Province_ID'], 'National': []}[level]

def calculate_one_pcsi_trend(group, value_column='CPSI'):
    data = group[['Year', value_column]].dropna().sort_values('Year')
    if len(data) < 3 or data['Year'].nunique() < 3:
        return {'n_years': len(data), 'Theil_Sen_Slope_per_year': np.nan, 'Slope_CI95_Lower': np.nan, 'Slope_CI95_Upper': np.nan, 'Kendall_tau': np.nan, 'Kendall_p': np.nan}
    x = data['Year'].to_numpy(dtype=float)
    y = data[value_column].to_numpy(dtype=float)
    slope, _, lower, upper = stats.theilslopes(y, x, alpha=0.95)
    tau, p_value = stats.kendalltau(x, y)
    return {'n_years': int(len(data)), 'Theil_Sen_Slope_per_year': float(slope), 'Slope_CI95_Lower': float(lower), 'Slope_CI95_Upper': float(upper), 'Kendall_tau': float(tau), 'Kendall_p': float(p_value)}

def calculate_pcsi_trends(pcsi_panel):
    rows = []
    for level, level_data in pcsi_panel.groupby('Level'):
        id_columns = unit_id_columns(level)
        group_columns = [*id_columns, 'CPSI_Type']
        if id_columns:
            groups = level_data.groupby(group_columns, dropna=False)
        else:
            groups = level_data.groupby('CPSI_Type', dropna=False)
        for key, group in groups:
            if not isinstance(key, tuple):
                key = (key,)
            if id_columns:
                id_values = dict(zip(group_columns, key))
                pcsi_type = id_values.pop('CPSI_Type')
            else:
                id_values = {}
                pcsi_type = key[0]
            record = {'Level': level, **id_values, 'CPSI_Type': pcsi_type}
            record.update(calculate_one_pcsi_trend(group, 'CPSI'))
            rows.append(record)
    return pd.DataFrame(rows)

def summarize_pcsi_trends(trend_table):
    rows = []
    for (level, pcsi_type), group in trend_table.groupby(['Level', 'CPSI_Type']):
        slopes = group['Theil_Sen_Slope_per_year'].dropna()
        p_values = group['Kendall_p']
        rows.append({'Level': level, 'CPSI_Type': pcsi_type, 'n_units': int(len(slopes)), 'Slope_Median': float(slopes.median()), 'Slope_Q25': float(slopes.quantile(0.25)), 'Slope_Q75': float(slopes.quantile(0.75)), 'Positive_Slope_Percent': float((slopes > 0).mean() * 100), 'Negative_Slope_Percent': float((slopes < 0).mean() * 100), 'Significant_Positive_Percent': float(((group['Theil_Sen_Slope_per_year'] > 0) & (p_values < 0.05)).mean() * 100), 'Significant_Negative_Percent': float(((group['Theil_Sen_Slope_per_year'] < 0) & (p_values < 0.05)).mean() * 100)})
    return pd.DataFrame(rows)

def calculate_state_transitions(pcsi_panel):
    annual_rows = []
    pooled_rows = []
    endpoint_rows = []
    for (level, pcsi_type), data in pcsi_panel.groupby(['Level', 'CPSI_Type']):
        id_columns = unit_id_columns(level)
        if not id_columns:
            data = data.copy()
            data['_Unit'] = 'National'
            id_columns = ['_Unit']
        years = sorted(data['Year'].unique())
        all_pairs = []
        for previous_year, current_year in zip(years[:-1], years[1:]):
            previous = data[data['Year'] == previous_year][[*id_columns, 'State', 'State_Code']].rename(columns={'State': 'Previous_State', 'State_Code': 'Previous_State_Code'})
            current = data[data['Year'] == current_year][[*id_columns, 'State', 'State_Code']].rename(columns={'State': 'Current_State', 'State_Code': 'Current_State_Code'})
            merged = previous.merge(current, on=id_columns, how='inner')
            merged['Previous_Year'] = previous_year
            merged['Current_Year'] = current_year
            all_pairs.append(merged)
            counts = merged.groupby(['Previous_State', 'Current_State']).size().rename('n_units').reset_index()
            counts['Previous_State_Total'] = counts.groupby('Previous_State')['n_units'].transform('sum')
            counts['Transition_Probability'] = counts['n_units'] / counts['Previous_State_Total']
            for record in counts.to_dict('records'):
                annual_rows.append({'Level': level, 'CPSI_Type': pcsi_type, 'Previous_Year': int(previous_year), 'Current_Year': int(current_year), **record})
        if all_pairs:
            pooled = pd.concat(all_pairs, ignore_index=True)
            counts = pooled.groupby(['Previous_State', 'Current_State']).size().rename('n_transitions').reset_index()
            counts['Previous_State_Total'] = counts.groupby('Previous_State')['n_transitions'].transform('sum')
            counts['Transition_Probability'] = counts['n_transitions'] / counts['Previous_State_Total']
            for record in counts.to_dict('records'):
                pooled_rows.append({'Level': level, 'CPSI_Type': pcsi_type, **record})
        first_year, last_year = (min(years), max(years))
        first = data[data['Year'] == first_year][[*id_columns, 'State']].rename(columns={'State': 'Initial_State'})
        last = data[data['Year'] == last_year][[*id_columns, 'State']].rename(columns={'State': 'Final_State'})
        endpoints = first.merge(last, on=id_columns, how='inner')
        counts = endpoints.groupby(['Initial_State', 'Final_State']).size().rename('n_units').reset_index()
        counts['Initial_State_Total'] = counts.groupby('Initial_State')['n_units'].transform('sum')
        counts['Transition_Probability'] = counts['n_units'] / counts['Initial_State_Total']
        for record in counts.to_dict('records'):
            endpoint_rows.append({'Level': level, 'CPSI_Type': pcsi_type, 'Initial_Year': int(first_year), 'Final_Year': int(last_year), **record})
    return (pd.DataFrame(annual_rows), pd.DataFrame(pooled_rows), pd.DataFrame(endpoint_rows))

def longest_run(values, target=None):
    longest = 0
    current = 0
    previous = object()
    for value in values:
        match = value == target if target is not None else value == previous
        if target is None:
            current = current + 1 if value == previous else 1
            previous = value
        else:
            current = current + 1 if match else 0
        longest = max(longest, current)
    return longest

def calculate_county_state_persistence(county_pcsi):
    rows = []
    for (code, pcsi_type), group in county_pcsi.groupby(['Code', 'CPSI_Type']):
        group = group.sort_values('Year')
        states = group['State'].tolist()
        counts = pd.Series(states).value_counts()
        dominant = counts.index[0]
        dominant_count = int(counts.iloc[0])
        longest_any = longest_run(states)
        rows.append({'Code': code, 'CPSI_Type': pcsi_type, 'n_years': len(states), 'Dominant_State': dominant, 'Dominant_State_Years': dominant_count, 'Dominant_State_Percent': dominant_count / len(states) * 100, 'Longest_Any_State_Run': longest_any, 'Longest_Both_Improved_Run': longest_run(states, 'Both_improved'), 'Longest_Carbon_Only_Run': longest_run(states, 'Carbon_only'), 'Longest_Pollution_Only_Run': longest_run(states, 'Pollution_only'), 'Longest_Neither_Improved_Run': longest_run(states, 'Neither_improved'), 'State_Locked': int(longest_any >= LOCK_MIN_CONSECUTIVE_YEARS), 'Lock_Threshold_Years': LOCK_MIN_CONSECUTIVE_YEARS, 'Initial_State': states[0], 'Final_State': states[-1], 'State_Changed_Initial_to_Final': int(states[0] != states[-1])})
    return pd.DataFrame(rows)

def calculate_rank_persistence(county_pcsi):
    rows = []
    for pcsi_type, data in county_pcsi.groupby('CPSI_Type'):
        years = sorted(data['Year'].unique())
        for previous_year, current_year in zip(years[:-1], years[1:]):
            previous = data[data['Year'] == previous_year][['Code', 'CPSI']].rename(columns={'CPSI': 'Previous_CPSI'})
            current = data[data['Year'] == current_year][['Code', 'CPSI']].rename(columns={'CPSI': 'Current_CPSI'})
            merged = previous.merge(current, on='Code')
            rho, p_value = stats.spearmanr(merged['Previous_CPSI'], merged['Current_CPSI'])
            rows.append({'CPSI_Type': pcsi_type, 'Previous_Year': int(previous_year), 'Current_Year': int(current_year), 'n_counties': int(len(merged)), 'CPSI_Rank_Spearman': float(rho), 'p_value': float(p_value)})
    table = pd.DataFrame(rows)
    table['p_FDR_BH'] = np.nan
    for _, indices in table.groupby('CPSI_Type').groups.items():
        table.loc[indices, 'p_FDR_BH'] = benjamini_hochberg(table.loc[indices, 'p_value'].to_numpy())
    return table

def resolve_geometry_source():
    if MODULE2_GPKG.exists():
        return MODULE2_GPKG
    if SHAPEFILE_PATH is not None:
        path = Path(SHAPEFILE_PATH)
        if not path.is_absolute():
            path = WORK_DIR / path
        if not path.exists():
            raise FileNotFoundError(f'Spatial boundary file not found: {path}')
        return path
    environment_path = os.environ.get('COUNTY_SHP', '').strip()
    if environment_path:
        path = Path(environment_path)
        if path.exists():
            return path
    candidates = [path for path in WORK_DIR.rglob('*.shp') if OUTPUT_DIR not in path.parents]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise FileNotFoundError('No Module2 GeoPackage or county shapefile was found; set SHAPEFILE_PATH.')
    raise RuntimeError('Multiple shapefiles were found; set SHAPEFILE_PATH explicitly:\n' + '\n'.join(map(str, candidates)))

def load_geometry(spatial_modules):
    gpd = spatial_modules['gpd']
    source = resolve_geometry_source()
    geometry = gpd.read_file(source)
    if geometry.crs is None:
        raise ValueError('The spatial boundary file has no CRS.')
    if 'Code' in geometry.columns:
        code_column = 'Code'
    else:
        lookup = {str(column).strip().lower(): column for column in geometry.columns}
        requested = str(SHAPE_CODE_FIELD).strip().lower()
        if requested not in lookup:
            raise ValueError(f'The boundary file contains neither Code nor {SHAPE_CODE_FIELD!r}; available fields: {list(geometry.columns)}')
        code_column = lookup[requested]
    geometry['Code'] = normalize_code_series(geometry[code_column])
    geometry = geometry.dropna(subset=['Code', 'geometry'])
    if (~geometry.geometry.is_valid).any():
        try:
            geometry['geometry'] = geometry.geometry.make_valid()
        except AttributeError:
            geometry['geometry'] = geometry.geometry.buffer(0)
    geometry = geometry[['Code', 'geometry']].dissolve(by='Code', as_index=False)
    geometry.attrs['source_path'] = str(source)
    return geometry

def build_weights_from_module2(county_pcsi, geometry, spatial_modules):
    if not WEIGHTS_EDGE_FILE.exists():
        raise FileNotFoundError(f'Module2 spatial weight edge list not found: {WEIGHTS_EDGE_FILE}. Run the Module2 spatial analysis first.')
    W = spatial_modules['W']
    edges = read_csv_safely(WEIGHTS_EDGE_FILE, dtype={'Origin_Code': 'string', 'Neighbor_Code': 'string'})
    required = {'Origin_Code', 'Neighbor_Code', 'Row_Standardized_Weight'}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f'The spatial weight edge list is missing columns: {sorted(missing)}')
    edges['Origin_Code'] = normalize_code_series(edges['Origin_Code'])
    edges['Neighbor_Code'] = normalize_code_series(edges['Neighbor_Code'])
    edges['Row_Standardized_Weight'] = pd.to_numeric(edges['Row_Standardized_Weight'], errors='coerce')
    panel_codes = set(county_pcsi['Code'].astype(str).unique())
    geometry_codes = set(geometry['Code'].astype(str).unique())
    edge_codes = set(edges['Origin_Code']) | set(edges['Neighbor_Code'])
    common_codes = sorted(panel_codes & geometry_codes & edge_codes)
    if len(common_codes) < 5:
        raise ValueError('Fewer than five counties are shared by CPSI data, spatial boundaries, and weights.')
    edges = edges[edges['Origin_Code'].isin(common_codes) & edges['Neighbor_Code'].isin(common_codes)].copy()
    neighbors = {code: [] for code in common_codes}
    weights_dict = {code: [] for code in common_codes}
    for row in edges.itertuples(index=False):
        neighbors[row.Origin_Code].append(row.Neighbor_Code)
        weights_dict[row.Origin_Code].append(float(row.Row_Standardized_Weight))
    zero_neighbor = [code for code in common_codes if not neighbors[code]]
    if zero_neighbor:
        raise ValueError(f'The weight subset contains {len(zero_neighbor)} counties without neighbors; verify the Module2 output.')
    weights = W(neighbors, weights=weights_dict, id_order=common_codes, silence_warnings=True)
    weights.transform = 'R'
    audit = pd.DataFrame([{'Geometry_Source': geometry.attrs.get('source_path', 'Provided_geometry'), 'Panel_Codes': len(panel_codes), 'Geometry_Codes': len(geometry_codes), 'Weight_Codes': len(edge_codes), 'Common_Codes': len(common_codes), 'Dropped_Panel_Codes': len(panel_codes - set(common_codes)), 'n_components': int(weights.n_components), 'islands': len(weights.islands), 'mean_neighbors': float(weights.mean_neighbors)}])
    audit.to_csv(OUTPUT_DIR / '08a_spatial_matching_and_weights_audit.csv', index=False, encoding='utf-8-sig')
    geometry = geometry[geometry['Code'].isin(common_codes)].copy()
    geometry = geometry.sort_values('Code').reset_index(drop=True)
    county_pcsi = county_pcsi[county_pcsi['Code'].astype(str).isin(common_codes)].copy()
    return (county_pcsi, geometry, weights)

def align_pcsi(county_pcsi, weights, pcsi_type, year):
    data = county_pcsi[(county_pcsi['CPSI_Type'] == pcsi_type) & (county_pcsi['Year'] == year)].copy()
    data['Code'] = data['Code'].astype(str)
    data = data.set_index('Code').reindex(weights.id_order)
    required = ['CPSI', 'H', 'Q_C', 'Q_P', 'State_Code']
    if data[required].isna().any().any():
        raise ValueError(f'{pcsi_type}—{year} contains missing values after alignment with spatial weights.')
    return data

def standardize_vector(values):
    values = np.asarray(values, dtype=float)
    standard_deviation = np.std(values, ddof=0)
    if not np.isfinite(standard_deviation) or standard_deviation <= EPSILON:
        return np.full_like(values, np.nan)
    return (values - np.mean(values)) / standard_deviation

def run_local_moran(values, weights, seed, spatial_modules):
    Moran_Local = spatial_modules['Moran_Local']
    np.random.seed(seed)
    try:
        return Moran_Local(values, weights, permutations=SPATIAL_PERMUTATIONS, seed=seed)
    except TypeError:
        np.random.seed(seed)
        return Moran_Local(values, weights, permutations=SPATIAL_PERMUTATIONS)

def calculate_pcsi_spatial_statistics(county_pcsi, weights, spatial_modules):
    Moran = spatial_modules['Moran']
    Moran_BV = spatial_modules['Moran_BV']
    Join_Counts = spatial_modules['Join_Counts']
    lag_spatial = spatial_modules['lag_spatial']
    global_rows = []
    join_rows = []
    bivariate_rows = []
    local_frames = []
    quadrant_labels = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
    for pcsi_type in CPSI_ORDER:
        for year in ANALYSIS_YEARS:
            aligned = align_pcsi(county_pcsi, weights, pcsi_type, year)
            for field in ['CPSI', 'H', 'Q_C', 'Q_P', 'State_Code']:
                values = aligned[field].to_numpy(dtype=float)
                if np.std(values) <= EPSILON:
                    continue
                seed = stable_seed('global', pcsi_type, year, field)
                np.random.seed(seed)
                moran = Moran(values, weights, permutations=SPATIAL_PERMUTATIONS)
                global_rows.append({'CPSI_Type': pcsi_type, 'Year': year, 'Field': field, 'n_counties': len(values), 'Global_Moran_I': float(moran.I), 'Expected_I': float(moran.EI), 'z_permutation': float(moran.z_sim), 'p_permutation': float(moran.p_sim), 'permutations': SPATIAL_PERMUTATIONS})
            binary_both = (aligned['State'] == 'Both_improved').astype(int).to_numpy()
            if np.unique(binary_both).size > 1:
                seed = stable_seed('join', pcsi_type, year)
                np.random.seed(seed)
                join = Join_Counts(binary_both, weights, permutations=SPATIAL_PERMUTATIONS)
                join_rows.append({'CPSI_Type': pcsi_type, 'Year': year, 'n_counties': len(binary_both), 'Both_Improved_Count': int(binary_both.sum()), 'BB_Join_Count': float(join.bb), 'Expected_BB': float(join.mean_bb), 'p_BB_permutation': float(join.p_sim_bb), 'BW_Join_Count': float(join.bw), 'Expected_BW': float(join.mean_bw), 'p_BW_permutation': float(join.p_sim_bw), 'permutations': SPATIAL_PERMUTATIONS})
            q_c = aligned['Q_C'].to_numpy(dtype=float)
            q_p = aligned['Q_P'].to_numpy(dtype=float)
            if np.std(q_c) > EPSILON and np.std(q_p) > EPSILON:
                for direction, x, y in [('Local_QC_to_neighbor_QP', q_c, q_p), ('Local_QP_to_neighbor_QC', q_p, q_c)]:
                    seed = stable_seed('bivariate', pcsi_type, year, direction)
                    np.random.seed(seed)
                    moran_bv = Moran_BV(x, y, weights, permutations=SPATIAL_PERMUTATIONS)
                    bivariate_rows.append({'CPSI_Type': pcsi_type, 'Year': year, 'Direction': direction, 'n_counties': len(x), 'Bivariate_Moran_I': float(moran_bv.I), 'z_permutation': float(moran_bv.z_sim), 'p_permutation': float(moran_bv.p_sim), 'permutations': SPATIAL_PERMUTATIONS})
            pcsi_values = aligned['CPSI'].to_numpy(dtype=float)
            if np.std(pcsi_values) > EPSILON:
                seed = stable_seed('local', pcsi_type, year)
                local = run_local_moran(pcsi_values, weights, seed, spatial_modules)
                p_local = np.asarray(local.p_sim, dtype=float)
                p_fdr = benjamini_hochberg(p_local)
                z = standardize_vector(pcsi_values)
                lag_z = lag_spatial(weights, z)
                quadrants = np.asarray(local.q, dtype=int)
                frame = pd.DataFrame({'Code': weights.id_order, 'CPSI_Type': pcsi_type, 'Year': year, 'CPSI': pcsi_values, 'State': aligned['State'].to_numpy(), 'State_Code': aligned['State_Code'].to_numpy(dtype=int), 'Standardized_CPSI': z, 'Spatial_Lag_Standardized_CPSI': lag_z, 'Local_Moran_I': np.asarray(local.Is, dtype=float), 'Quadrant': quadrants, 'p_local_permutation': p_local, 'p_local_FDR_BH': p_fdr})
                frame['LISA_Cluster_Raw'] = [quadrant_labels.get(q, 'Not_significant') if p < 0.05 else 'Not_significant' for q, p in zip(quadrants, p_local)]
                frame['LISA_Cluster_FDR'] = [quadrant_labels.get(q, 'Not_significant') if p < SPATIAL_FDR_ALPHA else 'Not_significant' for q, p in zip(quadrants, p_fdr)]
                local_frames.append(frame)
    global_table = pd.DataFrame(global_rows)
    global_table['p_FDR_BH'] = np.nan
    for _, indices in global_table.groupby(['Year', 'Field']).groups.items():
        global_table.loc[indices, 'p_FDR_BH'] = benjamini_hochberg(global_table.loc[indices, 'p_permutation'].to_numpy())
    global_table['FDR_Significant_0.05'] = (global_table['p_FDR_BH'] < 0.05).astype('Int64')
    join_table = pd.DataFrame(join_rows)
    if not join_table.empty:
        join_table['p_BB_FDR_BH'] = np.nan
        for _, indices in join_table.groupby('Year').groups.items():
            join_table.loc[indices, 'p_BB_FDR_BH'] = benjamini_hochberg(join_table.loc[indices, 'p_BB_permutation'].to_numpy())
    bivariate_table = pd.DataFrame(bivariate_rows)
    bivariate_table['p_FDR_BH'] = np.nan
    for _, indices in bivariate_table.groupby(['Year', 'Direction']).groups.items():
        bivariate_table.loc[indices, 'p_FDR_BH'] = benjamini_hochberg(bivariate_table.loc[indices, 'p_permutation'].to_numpy())
    bivariate_table['FDR_Significant_0.05'] = (bivariate_table['p_FDR_BH'] < 0.05).astype('Int64')
    local_table = pd.concat(local_frames, ignore_index=True)
    return (global_table, join_table, bivariate_table, local_table)

def calculate_lisa_locking(local_table):
    county_rows = []
    transition_rows = []
    annual_rows = []
    for pcsi_type, data in local_table.groupby('CPSI_Type'):
        data = data.sort_values(['Code', 'Year'])
        for code, group in data.groupby('Code'):
            group = group.sort_values('Year')
            clusters = group['LISA_Cluster_FDR'].tolist()
            longest_hh = longest_run(clusters, 'HH')
            longest_ll = longest_run(clusters, 'LL')
            locked_hh = longest_hh >= LOCK_MIN_CONSECUTIVE_YEARS
            locked_ll = longest_ll >= LOCK_MIN_CONSECUTIVE_YEARS
            if locked_hh and locked_ll:
                lock_type = 'Both_HH_and_LL'
            elif locked_hh:
                lock_type = 'Persistent_HH'
            elif locked_ll:
                lock_type = 'Persistent_LL'
            else:
                lock_type = 'Not_locked'
            county_rows.append({'Code': code, 'CPSI_Type': pcsi_type, 'n_years': len(clusters), 'HH_Years': int(sum((x == 'HH' for x in clusters))), 'LL_Years': int(sum((x == 'LL' for x in clusters))), 'HL_Years': int(sum((x == 'HL' for x in clusters))), 'LH_Years': int(sum((x == 'LH' for x in clusters))), 'Nonsignificant_Years': int(sum((x == 'Not_significant' for x in clusters))), 'Longest_HH_Run': longest_hh, 'Longest_LL_Run': longest_ll, 'Locked_HH': int(locked_hh), 'Locked_LL': int(locked_ll), 'LISA_Lock_Type': lock_type, 'Lock_Threshold_Years': LOCK_MIN_CONSECUTIVE_YEARS})
        years = sorted(data['Year'].unique())
        for previous_year, current_year in zip(years[:-1], years[1:]):
            previous = data[data['Year'] == previous_year][['Code', 'LISA_Cluster_FDR', 'CPSI']].rename(columns={'LISA_Cluster_FDR': 'Previous_Cluster', 'CPSI': 'Previous_CPSI'})
            current = data[data['Year'] == current_year][['Code', 'LISA_Cluster_FDR', 'CPSI']].rename(columns={'LISA_Cluster_FDR': 'Current_Cluster', 'CPSI': 'Current_CPSI'})
            merged = previous.merge(current, on='Code')
            counts = merged.groupby(['Previous_Cluster', 'Current_Cluster']).size().rename('n_counties').reset_index()
            counts['Previous_Total'] = counts.groupby('Previous_Cluster')['n_counties'].transform('sum')
            counts['Transition_Probability'] = counts['n_counties'] / counts['Previous_Total']
            for record in counts.to_dict('records'):
                transition_rows.append({'CPSI_Type': pcsi_type, 'Previous_Year': int(previous_year), 'Current_Year': int(current_year), **record})
            previous_sig = merged['Previous_Cluster'].isin(['HH', 'LL'])
            exact_retained = (merged['Previous_Cluster'] == merged['Current_Cluster']) & previous_sig
            rho, p_value = stats.spearmanr(merged['Previous_CPSI'], merged['Current_CPSI'])
            annual_rows.append({'CPSI_Type': pcsi_type, 'Previous_Year': int(previous_year), 'Current_Year': int(current_year), 'n_counties': int(len(merged)), 'Rank_Spearman': float(rho), 'Rank_p': float(p_value), 'HH_LL_Exact_Retention_Percent': float(exact_retained.sum() / previous_sig.sum() * 100) if previous_sig.sum() else np.nan, 'HH_Retention_Percent': float((merged.loc[merged['Previous_Cluster'] == 'HH', 'Current_Cluster'] == 'HH').mean() * 100) if (merged['Previous_Cluster'] == 'HH').any() else np.nan, 'LL_Retention_Percent': float((merged.loc[merged['Previous_Cluster'] == 'LL', 'Current_Cluster'] == 'LL').mean() * 100) if (merged['Previous_Cluster'] == 'LL').any() else np.nan})
    county_locking = pd.DataFrame(county_rows)
    locking_summary = county_locking.groupby(['CPSI_Type', 'LISA_Lock_Type']).size().rename('n_counties').reset_index()
    locking_summary['Total_Counties'] = locking_summary.groupby('CPSI_Type')['n_counties'].transform('sum')
    locking_summary['Percent'] = locking_summary['n_counties'] / locking_summary['Total_Counties'] * 100
    return (county_locking, locking_summary, pd.DataFrame(transition_rows), pd.DataFrame(annual_rows))

def neighbor_state_from_component_shares(carbon_improved_share, pollution_improved_share):
    carbon_improved_share = np.asarray(carbon_improved_share, dtype=float)
    pollution_improved_share = np.asarray(pollution_improved_share, dtype=float)
    carbon_improved = (carbon_improved_share >= NEIGHBOR_IMPROVEMENT_SHARE_THRESHOLD).astype(int)
    pollution_improved = (pollution_improved_share >= NEIGHBOR_IMPROVEMENT_SHARE_THRESHOLD).astype(int)
    return (2 * carbon_improved + pollution_improved).astype(int)

def calculate_spatial_markov(county_pcsi, weights, spatial_modules):
    lag_spatial = spatial_modules['lag_spatial']
    rows = []
    for pcsi_type in CPSI_ORDER:
        annual = {year: align_pcsi(county_pcsi, weights, pcsi_type, year) for year in ANALYSIS_YEARS}
        for previous_year, current_year in zip(ANALYSIS_YEARS[:-1], ANALYSIS_YEARS[1:]):
            previous_frame = annual[previous_year]
            previous_codes = previous_frame['State_Code'].to_numpy(dtype=int)
            current_codes = annual[current_year]['State_Code'].to_numpy(dtype=int)
            previous_carbon_improved = (previous_frame['Q_C'].to_numpy(dtype=float) > 0).astype(float)
            previous_pollution_improved = (previous_frame['Q_P'].to_numpy(dtype=float) > 0).astype(float)
            neighbor_carbon_share = lag_spatial(weights, previous_carbon_improved)
            neighbor_pollution_share = lag_spatial(weights, previous_pollution_improved)
            neighbor_codes = neighbor_state_from_component_shares(neighbor_carbon_share, neighbor_pollution_share)
            frame = pd.DataFrame({'Previous_Local_State_Code': previous_codes, 'Current_Local_State_Code': current_codes, 'Previous_Neighbor_State_Code': neighbor_codes, 'Previous_Neighbor_Carbon_Improved_Share': neighbor_carbon_share, 'Previous_Neighbor_Pollution_Improved_Share': neighbor_pollution_share})
            counts = frame.groupby(['Previous_Neighbor_State_Code', 'Previous_Local_State_Code', 'Current_Local_State_Code']).size().rename('n_counties').reset_index()
            counts['Conditional_Total'] = counts.groupby(['Previous_Neighbor_State_Code', 'Previous_Local_State_Code'])['n_counties'].transform('sum')
            counts['Conditional_Transition_Probability'] = counts['n_counties'] / counts['Conditional_Total']
            for record in counts.to_dict('records'):
                rows.append({'CPSI_Type': pcsi_type, 'Previous_Year': previous_year, 'Current_Year': current_year, 'Previous_Neighbor_State': STATE_LABELS[int(record['Previous_Neighbor_State_Code'])], 'Previous_Local_State': STATE_LABELS[int(record['Previous_Local_State_Code'])], 'Current_Local_State': STATE_LABELS[int(record['Current_Local_State_Code'])], **record})
    table = pd.DataFrame(rows)
    pooled = table.groupby(['CPSI_Type', 'Previous_Neighbor_State', 'Previous_Local_State', 'Current_Local_State'])['n_counties'].sum().rename('n_transitions').reset_index()
    pooled['Conditional_Total'] = pooled.groupby(['CPSI_Type', 'Previous_Neighbor_State', 'Previous_Local_State'])['n_transitions'].transform('sum')
    pooled['Conditional_Transition_Probability'] = pooled['n_transitions'] / pooled['Conditional_Total']
    return (table, pooled)

def calculate_scaling_sensitivity(county_panel, primary_pcsi):
    comparison_rows = []
    parameter_tables = []
    primary = primary_pcsi[['Code', 'Year', 'CPSI_Type', 'CPSI', 'H', 'State']].rename(columns={'CPSI': 'Primary_CPSI', 'H': 'Primary_H', 'State': 'Primary_State'})
    for scheme, absolute_quantile in SENSITIVITY_SCHEMES.items():
        parameters = fit_scaling_parameters(county_panel, absolute_quantile=absolute_quantile, scheme=scheme)
        parameter_tables.append(parameters)
        sensitivity = build_pcsi_panel(county_panel, 'County', parameters)[['Code', 'Year', 'CPSI_Type', 'CPSI', 'H', 'State']]
        merged = primary.merge(sensitivity, on=['Code', 'Year', 'CPSI_Type'], how='inner', suffixes=('', '_Sensitivity'))
        for (pcsi_type, year), group in merged.groupby(['CPSI_Type', 'Year']):
            rho, _ = stats.spearmanr(group['Primary_CPSI'], group['CPSI'])
            comparison_rows.append({'Scheme': scheme, 'CPSI_Type': pcsi_type, 'Year': int(year), 'n_counties': len(group), 'Spearman_vs_Primary': float(rho), 'Mean_Absolute_CPSI_Difference': float(np.mean(np.abs(group['Primary_CPSI'] - group['CPSI']))), 'Mean_Absolute_H_Difference': float(np.mean(np.abs(group['Primary_H'] - group['H']))), 'State_Agreement_Percent': float((group['Primary_State'] == group['State']).mean() * 100)})
    return (pd.concat(parameter_tables, ignore_index=True), pd.DataFrame(comparison_rows))

def calculate_leave_one_pollutant_out(county_panel, primary_parameters, primary_pcsi):
    pollutants = ['SO2', 'NO2', 'CO', 'PM25']
    primary_main = primary_pcsi[primary_pcsi['CPSI_Type'] == 'CPSIcp'][['Code', 'Year', 'CPSI', 'H', 'State']].rename(columns={'CPSI': 'Primary_CPSI', 'H': 'Primary_H', 'State': 'Primary_State'})
    carbon_parameter = primary_parameters[primary_parameters['Scale_Name'] == 'Carbon'].iloc[0]
    analysis = county_panel[county_panel['Year'] > BASE_YEAR].copy()
    q_c = transform_to_q(analysis['Carbon_Improvement'], carbon_parameter)
    rows = []
    detail_frames = []
    for excluded in pollutants:
        included = [p for p in pollutants if p != excluded]
        change_columns = [f'{p}_Improvement' for p in included]
        alternative_change = analysis[change_columns].mean(axis=1)
        absolute_cap = fit_one_scale(alternative_change, PRIMARY_ABSOLUTE_QUANTILE)
        parameter = pd.Series({'Absolute_Quantile': PRIMARY_ABSOLUTE_QUANTILE, 'Absolute_Cap': absolute_cap})
        q_p = transform_to_q(alternative_change, parameter)
        pcsi, h, _, _, state = calculate_pcsi(q_c, q_p)
        detail = analysis[['Code', 'Year']].copy()
        detail['Excluded_Pollutant'] = excluded
        detail['Alternative_Pollution_Change'] = alternative_change
        detail['Alternative_Q_P'] = q_p
        detail['Alternative_H'] = h
        detail['Alternative_CPSI'] = pcsi
        detail['Alternative_State'] = state
        detail_frames.append(detail)
        merged = primary_main.merge(detail, on=['Code', 'Year'])
        for year, group in merged.groupby('Year'):
            rho, _ = stats.spearmanr(group['Primary_CPSI'], group['Alternative_CPSI'])
            rows.append({'Excluded_Pollutant': excluded, 'Year': int(year), 'n_counties': len(group), 'Spearman_vs_Main': float(rho), 'Mean_Absolute_CPSI_Difference': float(np.mean(np.abs(group['Primary_CPSI'] - group['Alternative_CPSI']))), 'State_Agreement_Percent': float((group['Primary_State'] == group['Alternative_State']).mean() * 100)})
    return (pd.DataFrame(rows), pd.concat(detail_frames, ignore_index=True))

def create_main_temporal_figure(county_summary):
    data = county_summary[(county_summary['Level'] == 'County') & (county_summary['CPSI_Type'] == 'CPSIcp')].sort_values('Year')
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    axes[0].plot(data['Year'], data['CPSI_Median'], color='#0072B2', marker='o', linewidth=1.8, label='Median CPSI')
    axes[0].fill_between(data['Year'].to_numpy(), data['CPSI_Q25'].to_numpy(), data['CPSI_Q75'].to_numpy(), color='#0072B2', alpha=0.15, label='IQR')
    for boundary in [1, 2, 3]:
        axes[0].axhline(boundary, color='#777777', linestyle=':', linewidth=0.7)
    axes[0].set_ylim(0, 4)
    axes[0].set_xticks(ANALYSIS_YEARS[::2])
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('CPSIcp')
    axes[0].set_title('A  CPSIcp distribution')
    axes[0].grid(axis='y', color='#D9D9D9', linewidth=0.5, alpha=0.7)
    axes[0].legend()
    categories = [('Both_Improved_Percent', 'Both improved'), ('Pollution_Only_Percent', 'Pollution only'), ('Carbon_Only_Percent', 'Carbon only'), ('Neither_Improved_Percent', 'Neither improved')]
    color_lookup = {'Both improved': STATE_COLORS['Both_improved'], 'Pollution only': STATE_COLORS['Pollution_only'], 'Carbon only': STATE_COLORS['Carbon_only'], 'Neither improved': STATE_COLORS['Neither_improved']}
    bottom = np.zeros(len(data))
    for column, label in categories:
        values = data[column].to_numpy(dtype=float)
        axes[1].bar(data['Year'], values, bottom=bottom, color=color_lookup[label], width=0.75, label=label)
        bottom += values
    axes[1].set_ylim(0, 100)
    axes[1].set_xticks(ANALYSIS_YEARS[::2])
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Share of counties (%)')
    axes[1].set_title('B  CPSI state composition')
    axes[1].grid(axis='y', color='#D9D9D9', linewidth=0.5, alpha=0.7)
    axes[1].legend(loc='lower center', ncol=2, fontsize=7)
    save_figure(fig, 'Figure_Main_CPSI_temporal_evolution')

def create_diagnostic_pcsi_heatmap(county_summary):
    data = county_summary[county_summary['Level'] == 'County']
    matrix = data.pivot(index='CPSI_Type', columns='Year', values='CPSI_Median').reindex(index=CPSI_ORDER, columns=ANALYSIS_YEARS)
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.5, 3.6), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap='cividis', vmin=0, vmax=4, aspect='auto')
    ax.set_xticks(np.arange(len(ANALYSIS_YEARS)), ANALYSIS_YEARS)
    ax.set_yticks(np.arange(len(CPSI_ORDER)), [CPSI_DEFINITIONS[x]['label'] for x in CPSI_ORDER])
    ax.set_xlabel('Year')
    ax.set_title('Median CPSI by pollutant')
    for row in range(len(CPSI_ORDER)):
        for col in range(len(ANALYSIS_YEARS)):
            value = matrix.iloc[row, col]
            if pd.notna(value):
                ax.text(col, row, f'{value:.2f}', ha='center', va='center', fontsize=6.5, color='white' if value < 1.2 or value > 3.2 else 'black')
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label('Median CPSI')
    save_figure(fig, 'Figure_S_CPSI_all_pollutants_heatmap')

def create_main_transition_heatmap(pooled_transitions):
    data = pooled_transitions[(pooled_transitions['Level'] == 'County') & (pooled_transitions['CPSI_Type'] == 'CPSIcp')]
    order = ['Neither_improved', 'Pollution_only', 'Carbon_only', 'Both_improved']
    matrix = data.pivot(index='Previous_State', columns='Current_State', values='Transition_Probability').reindex(index=order, columns=order).fillna(0)
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(4.4, 3.8), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(), cmap='Blues', vmin=0, vmax=1)
    short = ['Neither', 'Pollution', 'Carbon', 'Both']
    ax.set_xticks(np.arange(4), short)
    ax.set_yticks(np.arange(4), short)
    ax.set_xlabel('Current-year state')
    ax.set_ylabel('Previous-year state')
    ax.set_title('Pooled county CPSI transition probabilities')
    for row in range(4):
        for col in range(4):
            value = matrix.iloc[row, col]
            ax.text(col, row, f'{value:.2f}', ha='center', va='center', color='white' if value > 0.55 else 'black')
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    save_figure(fig, 'Figure_Main_CPSI_transition_matrix')

def create_global_moran_figure(global_table):
    data = global_table[global_table['Field'] == 'CPSI'].copy()
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#999999']
    for pcsi_type, color in zip(CPSI_ORDER, colors):
        group = data[data['CPSI_Type'] == pcsi_type].sort_values('Year')
        ax.plot(group['Year'], group['Global_Moran_I'], marker='o', markersize=3, linewidth=1.4, color=color, label=CPSI_DEFINITIONS[pcsi_type]['label'])
    ax.axhline(0, color='#555555', linestyle=':', linewidth=0.8)
    ax.set_xticks(ANALYSIS_YEARS)
    ax.set_xlabel('Year')
    ax.set_ylabel("Global Moran's I")
    ax.set_title('Spatial clustering of CPSI')
    ax.grid(axis='y', color='#D9D9D9', linewidth=0.5, alpha=0.7)
    ax.legend(ncol=2, fontsize=7)
    save_figure(fig, 'Figure_S_CPSI_global_moran')

def create_main_pcsi_maps(geometry, county_pcsi):
    data = county_pcsi[(county_pcsi['CPSI_Type'] == 'CPSIcp') & county_pcsi['Year'].isin(MAP_YEARS)]
    years = sorted(data['Year'].unique())
    apply_publication_style()
    fig, axes = plt.subplots(1, len(years), figsize=(4.0 * len(years), 4.0), constrained_layout=True)
    if len(years) == 1:
        axes = [axes]
    for ax, year in zip(axes, years):
        annual = data[data['Year'] == year][['Code', 'CPSI']]
        mapped = geometry.merge(annual, on='Code', how='left')
        mapped.plot(ax=ax, column='CPSI', cmap='cividis', vmin=0, vmax=4, edgecolor='white', linewidth=0.08, missing_kwds={'color': '#EEEEEE'})
        ax.set_title(str(year))
        ax.set_axis_off()
    scalar = plt.cm.ScalarMappable(norm=plt.Normalize(0, 4), cmap='cividis')
    colorbar = fig.colorbar(scalar, ax=axes, orientation='horizontal', fraction=0.035, pad=0.02, shrink=0.65)
    colorbar.set_label('CPSIcp')
    fig.suptitle('Spatiotemporal distribution of CPSIcp', fontsize=10)
    save_figure(fig, 'Figure_Main_CPSI_maps')

def create_main_state_maps(geometry, county_pcsi):
    data = county_pcsi[(county_pcsi['CPSI_Type'] == 'CPSIcp') & county_pcsi['Year'].isin(MAP_YEARS)]
    years = sorted(data['Year'].unique())
    apply_publication_style()
    fig, axes = plt.subplots(1, len(years), figsize=(4.0 * len(years), 4.0), constrained_layout=True)
    if len(years) == 1:
        axes = [axes]
    for ax, year in zip(axes, years):
        annual = data[data['Year'] == year][['Code', 'State']]
        mapped = geometry.merge(annual, on='Code', how='left')
        for state, color in STATE_COLORS.items():
            subset = mapped[mapped['State'] == state]
            if not subset.empty:
                subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.08)
        ax.set_title(str(year))
        ax.set_axis_off()
    handles = [Patch(color=color, label=state) for state, color in STATE_COLORS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=7)
    fig.suptitle('CPSIcp states', fontsize=10)
    save_figure(fig, 'Figure_Main_CPSI_state_maps')

def create_main_lisa_maps(geometry, local_table):
    data = local_table[(local_table['CPSI_Type'] == 'CPSIcp') & local_table['Year'].isin(MAP_YEARS)]
    years = sorted(data['Year'].unique())
    apply_publication_style()
    fig, axes = plt.subplots(1, len(years), figsize=(4.0 * len(years), 4.0), constrained_layout=True)
    if len(years) == 1:
        axes = [axes]
    for ax, year in zip(axes, years):
        annual = data[data['Year'] == year][['Code', 'LISA_Cluster_FDR']]
        mapped = geometry.merge(annual, on='Code', how='left')
        mapped['LISA_Cluster_FDR'] = mapped['LISA_Cluster_FDR'].fillna('Not_significant')
        for cluster, color in LISA_COLORS.items():
            subset = mapped[mapped['LISA_Cluster_FDR'] == cluster]
            if not subset.empty:
                subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.08)
        ax.set_title(str(year))
        ax.set_axis_off()
    handles = [Patch(color=color, label=cluster) for cluster, color in LISA_COLORS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=7)
    fig.suptitle('CPSIcp LISA clusters (FDR adjusted)', fontsize=10)
    save_figure(fig, 'Figure_Main_CPSI_LISA_maps')

def create_diagnostic_final_year_maps(geometry, county_pcsi):
    if not GENERATE_ALL_DIAGNOSTIC_MAPS:
        return
    types = ['SO2', 'NO2', 'CO', 'PM25', 'O3', 'PM10']
    data = county_pcsi[county_pcsi['CPSI_Type'].isin(types) & (county_pcsi['Year'] == END_YEAR)]
    apply_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.4), constrained_layout=True)
    for ax, pcsi_type in zip(axes.flat, types):
        metric = data[data['CPSI_Type'] == pcsi_type][['Code', 'CPSI']]
        mapped = geometry.merge(metric, on='Code', how='left')
        mapped.plot(ax=ax, column='CPSI', cmap='cividis', vmin=0, vmax=4, edgecolor='white', linewidth=0.06, missing_kwds={'color': '#EEEEEE'})
        ax.set_title(CPSI_DEFINITIONS[pcsi_type]['label'])
        ax.set_axis_off()
    scalar = plt.cm.ScalarMappable(norm=plt.Normalize(0, 4), cmap='cividis')
    colorbar = fig.colorbar(scalar, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.025, pad=0.02, shrink=0.65)
    colorbar.set_label(f'CPSI in {END_YEAR}')
    save_figure(fig, 'Figure_S_diagnostic_CPSI_final_year_maps')

def create_state_lock_map(geometry, state_persistence, lisa_locking):
    state_data = state_persistence[state_persistence['CPSI_Type'] == 'CPSIcp'][['Code', 'Dominant_State', 'State_Locked']]
    lisa_data = lisa_locking[lisa_locking['CPSI_Type'] == 'CPSIcp'][['Code', 'LISA_Lock_Type']]
    mapped = geometry.merge(state_data, on='Code', how='left').merge(lisa_data, on='Code', how='left')
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), constrained_layout=True)
    for state, color in STATE_COLORS.items():
        subset = mapped[(mapped['Dominant_State'] == state) & (mapped['State_Locked'] == 1)]
        if not subset.empty:
            subset.plot(ax=axes[0], color=color, edgecolor='white', linewidth=0.08)
    unlocked = mapped[mapped['State_Locked'] != 1]
    if not unlocked.empty:
        unlocked.plot(ax=axes[0], color='#EEEEEE', edgecolor='white', linewidth=0.08)
    axes[0].set_title('A  Persistent CPSI state')
    axes[0].set_axis_off()
    lisa_lock_colors = {'Persistent_HH': '#B2182B', 'Persistent_LL': '#2166AC', 'Both_HH_and_LL': '#7B3294', 'Not_locked': '#D9D9D9'}
    for lock_type, color in lisa_lock_colors.items():
        subset = mapped[mapped['LISA_Lock_Type'] == lock_type]
        if not subset.empty:
            subset.plot(ax=axes[1], color=color, edgecolor='white', linewidth=0.08)
    axes[1].set_title('B  Persistent LISA cluster')
    axes[1].set_axis_off()
    save_figure(fig, 'Figure_Main_CPSI_spatial_locking')

def export_pcsi_geopackage(geometry, county_pcsi, local_table, state_persistence, lisa_locking):
    path = OUTPUT_DIR / 'CPSI_maps_and_spatial_locking.gpkg'
    if path.exists():
        path.unlink()
    first = True
    for pcsi_type in CPSI_ORDER:
        for year in MAP_YEARS:
            pcsi_data = county_pcsi[(county_pcsi['CPSI_Type'] == pcsi_type) & (county_pcsi['Year'] == year)][['Code', 'CPSI', 'H', 'Q_C', 'Q_P', 'State', 'State_Code']]
            lisa_data = local_table[(local_table['CPSI_Type'] == pcsi_type) & (local_table['Year'] == year)][['Code', 'Local_Moran_I', 'p_local_FDR_BH', 'LISA_Cluster_FDR']]
            layer = geometry.merge(pcsi_data, on='Code', how='left').merge(lisa_data, on='Code', how='left')
            layer.to_file(path, layer=f'pcsi_{pcsi_type}_{year}'[:63], driver='GPKG', mode='w' if first else 'a')
            first = False
        persistence = state_persistence[state_persistence['CPSI_Type'] == pcsi_type]
        lisa_lock = lisa_locking[lisa_locking['CPSI_Type'] == pcsi_type]
        layer = geometry.merge(persistence, on='Code', how='left').merge(lisa_lock, on=['Code', 'CPSI_Type'], how='left')
        layer.to_file(path, layer=f'locking_{pcsi_type}'[:63], driver='GPKG', mode='w' if first else 'a')
        first = False

def write_data_manifest():
    rows = [['Figure_Main_CPSI_temporal_evolution', '03_CPSI_annual_summary_all_levels.csv', 'Real', 'PNG; SVG'], ['Figure_S_CPSI_all_pollutants_heatmap', '03_CPSI_annual_summary_all_levels.csv', 'Real', 'PNG; SVG'], ['Figure_Main_CPSI_transition_matrix', '05b_CPSI_state_transitions_pooled.csv', 'Real', 'PNG; SVG'], ['Figure_S_CPSI_global_moran', '09_global_moran_CPSI_components.csv', 'Real', 'PNG; SVG'], ['Figure_Main_CPSI_maps', '02a_county_CPSI_panel.csv', 'Real', 'PNG; SVG; GPKG'], ['Figure_Main_CPSI_state_maps', '02a_county_CPSI_panel.csv', 'Real', 'PNG; SVG; GPKG'], ['Figure_Main_CPSI_LISA_maps', '11_local_moran_CPSI.csv', 'Real', 'PNG; SVG; GPKG'], ['Figure_Main_CPSI_spatial_locking', '06_county_CPSI_state_persistence.csv; 12a_LISA_locking_county.csv', 'Real', 'PNG; SVG; GPKG'], ['Figure_S_diagnostic_CPSI_final_year_maps', '02a_county_CPSI_panel.csv', 'Real', 'PNG; SVG']]
    pd.DataFrame(rows, columns=['Figure', 'Source_Data', 'Real_or_mock', 'Outputs']).to_csv(OUTPUT_DIR / 'Figure_Data_Manifest.csv', index=False, encoding='utf-8-sig')

def write_readme():
    pass

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pass
    panels, cpi_audit = load_module2_panels()
    cpi_audit.to_csv(OUTPUT_DIR / '00_module2_CPI_consistency_audit.csv', index=False, encoding='utf-8-sig')
    pass
    primary_parameters = fit_scaling_parameters(panels['County'], absolute_quantile=PRIMARY_ABSOLUTE_QUANTILE, scheme='Primary')
    primary_parameters.to_csv(OUTPUT_DIR / '01a_CPSI_scaling_parameters.csv', index=False, encoding='utf-8-sig')
    pass
    pcsi_panels = {level: build_pcsi_panel(panel, level, primary_parameters) for level, panel in panels.items()}
    audit = validate_pcsi(pcsi_panels['County'])
    audit.to_csv(OUTPUT_DIR / '01b_CPSI_range_audit.csv', index=False, encoding='utf-8-sig')
    panel_file_names = {'County': '02a_county_CPSI_panel.csv', 'City': '02b_city_CPSI_panel.csv', 'Province': '02c_province_CPSI_panel.csv', 'National': '02d_national_CPSI_panel.csv'}
    for level, panel in pcsi_panels.items():
        panel.to_csv(OUTPUT_DIR / panel_file_names[level], index=False, encoding='utf-8-sig')
    all_pcsi = pd.concat(pcsi_panels.values(), ignore_index=True)
    pass
    annual_summary = summarize_pcsi_by_year(all_pcsi)
    annual_summary.to_csv(OUTPUT_DIR / '03_CPSI_annual_summary_all_levels.csv', index=False, encoding='utf-8-sig')
    trends = calculate_pcsi_trends(all_pcsi)
    trends.to_csv(OUTPUT_DIR / '04a_CPSI_Theil_Sen_trends_all_units.csv', index=False, encoding='utf-8-sig')
    summarize_pcsi_trends(trends).to_csv(OUTPUT_DIR / '04b_CPSI_trend_summary_by_level.csv', index=False, encoding='utf-8-sig')
    annual_transitions, pooled_transitions, endpoint_transitions = calculate_state_transitions(all_pcsi)
    annual_transitions.to_csv(OUTPUT_DIR / '05a_CPSI_state_transitions_annual.csv', index=False, encoding='utf-8-sig')
    pooled_transitions.to_csv(OUTPUT_DIR / '05b_CPSI_state_transitions_pooled.csv', index=False, encoding='utf-8-sig')
    endpoint_transitions.to_csv(OUTPUT_DIR / '05c_CPSI_state_transitions_initial_final.csv', index=False, encoding='utf-8-sig')
    state_persistence = calculate_county_state_persistence(pcsi_panels['County'])
    state_persistence.to_csv(OUTPUT_DIR / '06_county_CPSI_state_persistence.csv', index=False, encoding='utf-8-sig')
    calculate_rank_persistence(pcsi_panels['County']).to_csv(OUTPUT_DIR / '07_county_CPSI_rank_persistence.csv', index=False, encoding='utf-8-sig')
    pass
    spatial_modules = load_spatial_dependencies()
    geometry = load_geometry(spatial_modules)
    county_spatial, geometry, weights = build_weights_from_module2(pcsi_panels['County'], geometry, spatial_modules)
    pass
    global_table, join_table, bivariate_table, local_table = calculate_pcsi_spatial_statistics(county_spatial, weights, spatial_modules)
    global_table.to_csv(OUTPUT_DIR / '09_global_moran_CPSI_components.csv', index=False, encoding='utf-8-sig')
    join_table.to_csv(OUTPUT_DIR / '10_join_count_both_improved.csv', index=False, encoding='utf-8-sig')
    bivariate_table.to_csv(OUTPUT_DIR / '10b_bivariate_moran_QC_QP.csv', index=False, encoding='utf-8-sig')
    local_table.to_csv(OUTPUT_DIR / '11_local_moran_CPSI.csv', index=False, encoding='utf-8-sig')
    pass
    lisa_locking, lisa_lock_summary, lisa_transitions, lisa_annual = calculate_lisa_locking(local_table)
    lisa_locking.to_csv(OUTPUT_DIR / '12a_LISA_locking_county.csv', index=False, encoding='utf-8-sig')
    lisa_lock_summary.to_csv(OUTPUT_DIR / '12b_LISA_locking_summary.csv', index=False, encoding='utf-8-sig')
    lisa_transitions.to_csv(OUTPUT_DIR / '12c_LISA_transition_matrices.csv', index=False, encoding='utf-8-sig')
    lisa_annual.to_csv(OUTPUT_DIR / '12d_LISA_annual_persistence.csv', index=False, encoding='utf-8-sig')
    spatial_markov, spatial_markov_pooled = calculate_spatial_markov(county_spatial, weights, spatial_modules)
    spatial_markov.to_csv(OUTPUT_DIR / '13a_spatial_markov_annual.csv', index=False, encoding='utf-8-sig')
    spatial_markov_pooled.to_csv(OUTPUT_DIR / '13b_spatial_markov_pooled.csv', index=False, encoding='utf-8-sig')
    pass
    sensitivity_parameters, scaling_sensitivity = calculate_scaling_sensitivity(panels['County'], pcsi_panels['County'])
    sensitivity_parameters.to_csv(OUTPUT_DIR / '14a_scaling_sensitivity_parameters.csv', index=False, encoding='utf-8-sig')
    scaling_sensitivity.to_csv(OUTPUT_DIR / '14b_scaling_sensitivity_comparison.csv', index=False, encoding='utf-8-sig')
    leave_one_summary, leave_one_detail = calculate_leave_one_pollutant_out(panels['County'], primary_parameters, pcsi_panels['County'])
    leave_one_summary.to_csv(OUTPUT_DIR / '15a_leave_one_pollutant_out_summary.csv', index=False, encoding='utf-8-sig')
    leave_one_detail.to_csv(OUTPUT_DIR / '15b_leave_one_pollutant_out_county_year.csv', index=False, encoding='utf-8-sig')
    pass
    create_main_temporal_figure(annual_summary)
    create_diagnostic_pcsi_heatmap(annual_summary)
    create_main_transition_heatmap(pooled_transitions)
    create_global_moran_figure(global_table)
    create_main_pcsi_maps(geometry, county_spatial)
    create_main_state_maps(geometry, county_spatial)
    create_main_lisa_maps(geometry, local_table)
    create_diagnostic_final_year_maps(geometry, county_spatial)
    create_state_lock_map(geometry, state_persistence, lisa_locking)
    export_pcsi_geopackage(geometry, county_spatial, local_table, state_persistence, lisa_locking)
    write_data_manifest()
    write_readme()
    pass
    pass
    pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
