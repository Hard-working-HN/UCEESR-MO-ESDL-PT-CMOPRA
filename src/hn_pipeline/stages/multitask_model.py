from __future__ import annotations
import argparse
import copy
import json
import random
import re
import shutil
import warnings
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.nn.utils.rnn import pack_padded_sequence
from torch_geometric.nn import GATConv
warnings.filterwarnings('ignore', category=RuntimeWarning)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

@dataclass
class Config:
    data_path: str = str(PROJECT_ROOT / 'data' / 'processed' / 'model_data_total.csv')
    edge_path: str = str(PROJECT_ROOT / 'data' / 'processed' / 'module2_time_space_results' / '10a_spatial_weights_edge_list.csv')
    output_dir: str = str(PROJECT_ROOT / 'models' / 'formal_multitask')
    code_col: str = 'Code'
    year_col: str = 'Year'
    eci_col: str = 'ECI'
    cpi_col: str = 'CPI'
    city_features: List[str] = field(default_factory=lambda: ['PGDP', 'POPD', 'VAF_GDP', 'GIA_GDP', 'TSS_GDP', 'Mean_Light', 'Builtup', 'Road_Density', 'PIU_PIR_Ratio', 'Fiscal_Revenue_GDP'])
    energy_features: List[str] = field(default_factory=lambda: ['Energy_Intensity', 'Fossil_Energy_Share', 'Energy_Diversity_Index_Norm', 'Electricity_Share', 'Industrial_Coal_Share', 'TPSP_Petroleum_Share'])
    use_coal_share: bool = False
    coal_share_name: str = 'Coal_Share'
    start_year: int = 2013
    end_year: int = 2022
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    split_random_seed: int = 2026
    lookback: int = 3
    gat_hidden: int = 96
    gat_heads1: int = 4
    gat_heads2: int = 4
    gru_hidden: int = 96
    encoder_hidden: int = 128
    task_hidden: int = 64
    use_county_embedding: bool = False
    county_embedding_dim: int = 16
    n_counties: int = 0
    softshare_init_self: float = 0.9
    softshare_init_cross: float = 0.1
    dropout: float = 0.08
    seeds: List[int] = field(default_factory=lambda: [42, 2026, 3407])
    epochs: int = 333
    min_epochs: int = 123
    early_stopping_patience: int = 66
    learning_rate: float = 0.001
    weight_decay: float = 1e-05
    grad_clip: float = 5.0
    scheduler_patience: int = 25
    scheduler_factor: float = 0.5
    min_lr: float = 1e-06
    min_delta: float = 1e-07
    mc_samples: int = 1000
    uncertainty_quantile_low: float = 0.025
    uncertainty_quantile_high: float = 0.975
    scenario_percent_changes: List[int] = field(default_factory=lambda: [-20, -15, -10, -5, 5, 10, 15, 20])
    scenario_years: List[int] = field(default_factory=lambda: [2022])
    scenario_use_all_exogenous_features: bool = True
    scenario_clip_unit_interval_features: bool = True
    scenario_save_sample_level: bool = True
    models_to_run: List[str] = field(default_factory=lambda: ['Model3', 'Model5'])
    shap_background_multiplier: int = 2
    shap_permutations: int = 12
    shap_2d_permutations: int = 12
    shap_random_seed: int = 20260901
    shap_beeswarm_top_n: int = 20
    ale_1d_bins: int = 10
    ale_2d_bins: int = 5
    ale_2d_top_features: int = 6
    ale_sync_current_step_in_gru: bool = True
    level_epsilon: float = 1e-08
    strict_panel: bool = True
    add_self_loops_in_gat: bool = True
    graph_unknown_code_tolerance: float = 0.01
CITY_FEATURE_BLOCK: List[str] = ['PGDP', 'POPD', 'VAF_GDP', 'GIA_GDP', 'TSS_GDP', 'Mean_Light', 'Builtup', 'Road_Density', 'PIU_PIR_Ratio', 'Fiscal_Revenue_GDP']
ENERGY_FEATURE_BLOCK: List[str] = ['Energy_Intensity', 'Coal_Share', 'Electricity_Share', 'Fossil_Energy_Share', 'Energy_Diversity_Index_Norm', 'Industrial_Coal_Share', 'TPSP_Petroleum_Share']
BLUE_GREEN_FEATURE_BLOCK: List[str] = ['Forest', 'Grassland', 'Shrub', 'Water', 'Wetland', 'Annual NDVI', 'Summer NDVI', 'AMEVT', 'LAI_HV', 'LAI_LV', 'BG_AI', 'BG_Divsion']
MITIGATION_FEATURE_BLOCK: List[str] = ['EEI_GDP', 'EES_GEF', 'GPAI', 'GIPAI', 'GIPGS']
MODEL_FEATURE_SETS: Dict[str, List[str]] = {'Model1': CITY_FEATURE_BLOCK, 'Model2': CITY_FEATURE_BLOCK + ENERGY_FEATURE_BLOCK, 'Model3': CITY_FEATURE_BLOCK + ENERGY_FEATURE_BLOCK + BLUE_GREEN_FEATURE_BLOCK, 'Model4': CITY_FEATURE_BLOCK + ENERGY_FEATURE_BLOCK + MITIGATION_FEATURE_BLOCK, 'Model5': CITY_FEATURE_BLOCK + ENERGY_FEATURE_BLOCK + BLUE_GREEN_FEATURE_BLOCK + MITIGATION_FEATURE_BLOCK}
COLUMN_ALIASES: Dict[str, List[str]] = {'Code': ['code', 'county_code', 'countycode', 'administrative_code', 'county_code'], 'Year': ['year', 'year_alias', 'yr'], 'ECI': ['eci', 'energy_carbon_index', 'carbon_index'], 'CPI': ['cpi', 'comprehensive_pollution_index', 'pollution_index'], 'PGDP': ['pgdp', 'per_capita_gdp', 'gdp_per_capita'], 'POPD': ['popd', 'population_density', 'pop_density'], 'VAF_GDP': ['vaf_gdp', 'vaf/gdp'], 'GIA_GDP': ['gia_gdp', 'gia/gdp'], 'TSS_GDP': ['tss_gdp', 'tss/gdp'], 'Mean_Light': ['mean_light', 'meanlight', 'night_light', 'nighttime_light'], 'Builtup': ['builtup', 'built_up', 'builtup_area'], 'Road_Density': ['road_density', 'roaddensity'], 'PIU_PIR_Ratio': ['piu_pir_ratio', 'piu/pir', 'piu_pir'], 'Fiscal_Revenue_GDP': ['fiscal_revenue_gdp', 'fiscal_revenue/gdp'], 'Energy_Intensity': ['energy_intensity', 'energyintensity'], 'Fossil_Energy_Share': ['fossil_energy_share', 'fossil_share'], 'Coal_Share': ['coal_share', 'coalshare'], 'Energy_Diversity_Index_Norm': ['energy_diversity_index_norm', 'energy_diversity_norm'], 'Electricity_Share': ['electricity_share', 'electric_share'], 'Industrial_Coal_Share': ['industrial_coal_share', 'industry_coal_share'], 'TPSP_Petroleum_Share': ['tpsp_petroleum_share', 'petroleum_share', 'oil_share'], 'Origin_Code': ['origin_code', 'origincode', 'origin', 'from_code'], 'Neighbor_Code': ['neighbor_code', 'neighbour_code', 'neighborcode', 'neighbor', 'to_code'], 'Row_Standardized_Weight': ['row_standardized_weight', 'row_standardised_weight', 'weight', 'spatial_weight']}

def normalize_name(name: str) -> str:
    return re.sub('[^a-z0-9]+', '', str(name).strip().lower())

def resolve_column(columns: Sequence[str], requested: str, required: bool=True) -> Optional[str]:
    columns = list(columns)
    if requested in columns:
        return requested
    lookup = {normalize_name(c): c for c in columns}
    for candidate in [requested] + COLUMN_ALIASES.get(requested, []):
        key = normalize_name(candidate)
        if key in lookup:
            return lookup[key]
    if required:
        raise KeyError(f"Unable to find column '{requested}'. Available columns: {columns}")
    return None

def resolve_feature_list(columns: Sequence[str], requested: Sequence[str]) -> Tuple[List[str], Dict[str, str]]:
    real, mapping = ([], {})
    for name in requested:
        col = resolve_column(columns, name, required=True)
        real.append(col)
        mapping[name] = col
    return (real, mapping)

def canonicalize_code(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.replace('\\.0$', '', regex=True)
    numeric_like = s.str.fullmatch('\\d+')
    s.loc[numeric_like] = s.loc[numeric_like].str.zfill(6)
    return s

def read_csv_flexible(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'File not found: {p}')
    last_error = None
    for enc in ['utf-8-sig', 'utf-8', 'gb18030', 'gbk']:
        try:
            return pd.read_csv(p, encoding=enc, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error

def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_panel(raw: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict, Dict[str, object]]:
    columns = raw.columns.tolist()
    code_col = resolve_column(columns, cfg.code_col)
    year_col = resolve_column(columns, cfg.year_col)
    eci_col = resolve_column(columns, cfg.eci_col)
    cpi_col = resolve_column(columns, cfg.cpi_col)
    requested_energy = list(cfg.energy_features)
    if cfg.use_coal_share:
        requested_energy = [cfg.coal_share_name if x == 'Fossil_Energy_Share' else x for x in requested_energy]
    city_real, city_map = resolve_feature_list(columns, cfg.city_features)
    energy_real, energy_map = resolve_feature_list(columns, requested_energy)
    df = raw.copy()
    df[code_col] = canonicalize_code(df[code_col])
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    if df[year_col].isna().any():
        raise ValueError('The Year column contains values that cannot be converted to integers.')
    df[year_col] = df[year_col].astype(int)
    duplicate_count = int(df.duplicated([code_col, year_col]).sum())
    if duplicate_count > 0:
        raise ValueError(f'Found {duplicate_count} duplicate Code-Year rows.')
    df = df[df[year_col].between(cfg.start_year, cfg.end_year)].copy()
    numeric_cols = list(dict.fromkeys(city_real + energy_real + [eci_col, cpi_col]))
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if (df[eci_col].notna() & (df[eci_col] <= 0)).any():
        raise ValueError('ECI contains nonpositive values and cannot be log transformed.')
    if (df[cpi_col].notna() & (df[cpi_col] <= 0)).any():
        raise ValueError('CPI contains nonpositive values and cannot be log transformed.')
    all_codes = sorted(df[code_col].dropna().unique().tolist())
    all_years = list(range(cfg.start_year, cfg.end_year + 1))
    expected = len(all_codes) * len(all_years)
    counts = df.groupby(code_col)[year_col].nunique()
    panel_complete = len(df) == expected and len(counts) == len(all_codes) and bool((counts == len(all_years)).all())
    if cfg.strict_panel and (not panel_complete):
        raise ValueError(f'The data are not a strictly balanced 2013-2022 panel. County count={len(all_codes)}, rows={len(df)}, expected={expected}。')
    full_index = pd.MultiIndex.from_product([all_codes, all_years], names=[code_col, year_col])
    df = df.set_index([code_col, year_col]).reindex(full_index).reset_index().sort_values([code_col, year_col]).reset_index(drop=True)
    df['Log_ECI'] = np.log(df[eci_col])
    df['Log_CPI'] = np.log(df[cpi_col])
    df['True_ECI_Previous'] = df.groupby(code_col)[eci_col].shift(1)
    df['True_CPI_Previous'] = df.groupby(code_col)[cpi_col].shift(1)
    df['True_Carbon_Improvement'] = -np.log(df[eci_col] / df['True_ECI_Previous'])
    df['True_Pollution_Improvement'] = -np.log(df[cpi_col] / df['True_CPI_Previous'])
    ci = df['True_Carbon_Improvement']
    pi = df['True_Pollution_Improvement']
    valid_state = ci.notna() & pi.notna()
    state = np.full(len(df), np.nan)
    state[valid_state & (ci <= 0) & (pi <= 0)] = 0
    state[valid_state & (ci <= 0) & (pi > 0)] = 1
    state[valid_state & (ci > 0) & (pi <= 0)] = 2
    state[valid_state & (ci > 0) & (pi > 0)] = 3
    df['True_State'] = state
    df['Lag_Log_ECI'] = df.groupby(code_col)['Log_ECI'].shift(1)
    df['Lag_Log_CPI'] = df.groupby(code_col)['Log_CPI'].shift(1)
    df['Lag_Carbon_Improvement'] = df.groupby(code_col)['True_Carbon_Improvement'].shift(1)
    df['Lag_Pollution_Improvement'] = df.groupby(code_col)['True_Pollution_Improvement'].shift(1)
    lag_state = df.groupby(code_col)['True_State'].shift(1)
    for s in range(4):
        df[f'Lag_State_{s}'] = ((lag_state == s) & lag_state.notna()).astype(float)
    df['Lag_State_Available_Mask'] = lag_state.notna().astype(float)
    df['Lag_Level_Available_Mask'] = (df['Lag_Log_ECI'].notna() & df['Lag_Log_CPI'].notna()).astype(float)
    current_complete = df[city_real + energy_real + [eci_col, cpi_col]].notna().all(axis=1).astype(float)
    df['Lag_History_Complete_Mask'] = current_complete.groupby(df[code_col]).shift(1).fillna(0).astype(float)
    mapping = {'code_col': code_col, 'year_col': year_col, 'eci_col': eci_col, 'cpi_col': cpi_col, 'city_features': city_real, 'energy_features': energy_real}
    resolved = {cfg.code_col: code_col, cfg.year_col: year_col, cfg.eci_col: eci_col, cfg.cpi_col: cpi_col, **city_map, **energy_map}
    audit = {'shape_raw': [int(raw.shape[0]), int(raw.shape[1])], 'n_counties': int(len(all_codes)), 'n_years': int(len(all_years)), 'panel_complete': bool(panel_complete), 'duplicate_code_year': duplicate_count, 'resolved_columns': resolved, 'direct_targets': [eci_col, cpi_col], 'derived_improvement_formula': {'carbon': '-log(Pred_ECI_t / True_ECI_t-1)', 'pollution': '-log(Pred_CPI_t / True_CPI_t-1)'}, 'state_definition': {'0': 'Carbon<=0 and Pollution<=0', '1': 'Carbon<=0 and Pollution>0', '2': 'Carbon>0 and Pollution<=0', '3': 'Carbon>0 and Pollution>0'}, 'true_state_counts': {str(k): int(v) for k, v in df['True_State'].dropna().astype(int).value_counts().sort_index().items()}}
    return (df, audit, mapping)

def assign_splits(df: pd.DataFrame, mapping: Dict[str, object], cfg: Config) -> Tuple[pd.DataFrame, Dict]:
    out = df.copy()
    code_col = str(mapping['code_col'])
    year_col = str(mapping['year_col'])
    eci_col = str(mapping['eci_col'])
    cpi_col = str(mapping['cpi_col'])
    level_valid = out[eci_col].notna() & out[cpi_col].notna()
    ratios = np.asarray([cfg.train_ratio, cfg.validation_ratio, cfg.test_ratio], dtype=float)
    if (ratios <= 0).any() or not np.isclose(ratios.sum(), 1.0):
        raise ValueError('train_ratio, validation_ratio, and test_ratio must be positive and sum to one.')
    county_codes = np.asarray(sorted(out.loc[level_valid, code_col].dropna().unique().tolist()), dtype=object)
    n_counties = len(county_codes)
    if n_counties < 3:
        raise ValueError('Fewer than three valid counties are available, so three splits cannot be created.')
    rng = np.random.default_rng(int(cfg.split_random_seed))
    rng.shuffle(county_codes)
    n_train = max(1, int(np.floor(n_counties * cfg.train_ratio)))
    n_validation = max(1, int(np.floor(n_counties * cfg.validation_ratio)))
    if n_train + n_validation >= n_counties:
        n_validation = max(1, n_counties - n_train - 1)
    n_test = n_counties - n_train - n_validation
    if n_test < 1:
        n_train -= 1
        n_test = 1
    train_counties = set(county_codes[:n_train].tolist())
    validation_counties = set(county_codes[n_train:n_train + n_validation].tolist())
    test_counties = set(county_codes[n_train + n_validation:].tolist())
    county_split_mapping = {**{code: 'train' for code in train_counties}, **{code: 'validation' for code in validation_counties}, **{code: 'test' for code in test_counties}}
    out['__county_split__'] = out[code_col].map(county_split_mapping).fillna('')
    out['__level_split__'] = ''
    out.loc[level_valid, '__level_split__'] = out.loc[level_valid, '__county_split__']
    train_mask = level_valid & out['__county_split__'].eq('train')
    val_mask = level_valid & out['__county_split__'].eq('validation')
    test_mask = level_valid & out['__county_split__'].eq('test')
    state_valid = out['True_Carbon_Improvement'].notna() & out['True_Pollution_Improvement'].notna() & out['True_State'].notna()
    out['__state_split__'] = ''
    out.loc[state_valid, '__state_split__'] = out.loc[state_valid, '__county_split__']
    state_train = state_valid & out['__county_split__'].eq('train')
    state_val = state_valid & out['__county_split__'].eq('validation')
    state_test = state_valid & out['__county_split__'].eq('test')
    counts_by_year = {}
    for year, group in out.loc[level_valid].groupby(year_col, sort=True):
        counts = group['__level_split__'].value_counts()
        counts_by_year[str(int(year))] = {'train': int(counts.get('train', 0)), 'validation': int(counts.get('validation', 0)), 'test': int(counts.get('test', 0))}
    audit = {'method': 'random group split by county', 'rule': 'all years of the same county belong to the same split', 'random_seed': int(cfg.split_random_seed), 'requested_ratios': {'train': float(cfg.train_ratio), 'validation': float(cfg.validation_ratio), 'test': float(cfg.test_ratio)}, 'county_counts': {'total': int(n_counties), 'train': int(len(train_counties)), 'validation': int(len(validation_counties)), 'test': int(len(test_counties))}, 'level_regression': {'n_train': int(train_mask.sum()), 'n_validation': int(val_mask.sum()), 'n_test': int(test_mask.sum()), 'counts_by_year': counts_by_year}, 'derived_state': {'n_train': int(state_train.sum()), 'n_validation': int(state_val.sum()), 'n_test': int(state_test.sum())}, 'county_overlap': {'train_validation': int(len(train_counties & validation_counties)), 'train_test': int(len(train_counties & test_counties)), 'validation_test': int(len(validation_counties & test_counties))}}
    return (out, audit)

class FeatureStandardizer:

    def __init__(self, columns: Sequence[str]):
        self.columns = list(columns)
        self.median_: Dict[str, float] = {}
        self.mean_: Dict[str, float] = {}
        self.std_: Dict[str, float] = {}

    def fit(self, frame: pd.DataFrame) -> 'FeatureStandardizer':
        for col in self.columns:
            s = pd.to_numeric(frame[col], errors='coerce').astype(float)
            median = float(s.median()) if s.notna().any() else 0.0
            filled = s.fillna(median)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            if not np.isfinite(std) or std < 1e-12:
                std = 1.0
            self.median_[col] = median
            self.mean_[col] = mean
            self.std_[col] = std
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        arrays = []
        for col in self.columns:
            s = pd.to_numeric(frame[col], errors='coerce').astype(float)
            s = s.fillna(self.median_[col])
            arrays.append(((s - self.mean_[col]) / self.std_[col]).to_numpy(dtype=np.float32))
        return np.column_stack(arrays).astype(np.float32)

    def to_dict(self) -> Dict:
        return {'columns': self.columns, 'median': self.median_, 'mean': self.mean_, 'std': self.std_, 'fit_scope': 'county-group training rows only'}

class LevelTargetStandardizer:

    def fit(self, frame: pd.DataFrame, eci_col: str, cpi_col: str) -> 'LevelTargetStandardizer':
        eci = pd.to_numeric(frame[eci_col], errors='coerce').dropna().to_numpy(float)
        cpi = pd.to_numeric(frame[cpi_col], errors='coerce').dropna().to_numpy(float)
        if len(eci) == 0 or len(cpi) == 0:
            raise ValueError('The training set has no ECI/CPI values available for target scaling.')
        if np.any(eci <= 0):
            raise ValueError('Training ECI contains nonpositive values and cannot be log transformed.')
        if np.any(cpi <= 0):
            raise ValueError('Training CPI contains nonpositive values and cannot be log transformed.')
        log_eci = np.log(eci)
        log_cpi = np.log(cpi)
        self.eci_mean = float(np.mean(log_eci))
        self.eci_std = float(np.std(log_eci, ddof=0))
        self.cpi_mean = float(np.mean(log_cpi))
        self.cpi_std = float(np.std(log_cpi, ddof=0))
        if not np.isfinite(self.eci_std) or self.eci_std < 1e-12:
            self.eci_std = 1.0
        if not np.isfinite(self.cpi_std) or self.cpi_std < 1e-12:
            self.cpi_std = 1.0
        return self

    def eci_to_z(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if np.any(y <= 0):
            raise ValueError('ECI targets contain nonpositive values and cannot be log transformed.')
        log_y = np.log(y)
        return ((log_y - self.eci_mean) / self.eci_std).astype(np.float32)

    def cpi_to_z(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if np.any(y <= 0):
            raise ValueError('CPI targets contain nonpositive values and cannot be log transformed.')
        log_y = np.log(y)
        return ((log_y - self.cpi_mean) / self.cpi_std).astype(np.float32)

    def eci_log_inverse_np(self, z: np.ndarray) -> np.ndarray:
        return (z * self.eci_std + self.eci_mean).astype(np.float32)

    def cpi_log_inverse_np(self, z: np.ndarray) -> np.ndarray:
        return (z * self.cpi_std + self.cpi_mean).astype(np.float32)

    def eci_inverse_np(self, z: np.ndarray) -> np.ndarray:
        log_level = self.eci_log_inverse_np(z)
        return np.exp(log_level).astype(np.float32)

    def cpi_inverse_np(self, z: np.ndarray) -> np.ndarray:
        log_level = self.cpi_log_inverse_np(z)
        return np.exp(log_level).astype(np.float32)

    def eci_log_inverse_torch(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.eci_std + self.eci_mean

    def cpi_log_inverse_torch(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.cpi_std + self.cpi_mean

    def eci_inverse_torch(self, z: torch.Tensor) -> torch.Tensor:
        log_level = self.eci_log_inverse_torch(z)
        return torch.exp(log_level)

    def cpi_inverse_torch(self, z: torch.Tensor) -> torch.Tensor:
        log_level = self.cpi_log_inverse_torch(z)
        return torch.exp(log_level)

    @property
    def eci_geometric_center(self) -> float:
        return float(np.exp(self.eci_mean))

    @property
    def cpi_geometric_center(self) -> float:
        return float(np.exp(self.cpi_mean))

    def to_dict(self) -> Dict:
        return {'log_ECI': {'mean': self.eci_mean, 'std': self.eci_std}, 'log_CPI': {'mean': self.cpi_mean, 'std': self.cpi_std}, 'model_targets': ['ln(ECI)', 'ln(CPI)'], 'fit_scope': 'county-group training rows only', 'loss_space': 'z-score standardized ln(ECI) and ln(CPI)', 'inverse_transform': {'ECI': 'exp(predicted ln(ECI))', 'CPI': 'exp(predicted ln(CPI))'}}

def load_graph(edge_path: str, node_codes: Sequence[str], cfg: Config) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    raw = read_csv_flexible(edge_path)
    columns = raw.columns.tolist()
    origin_col = resolve_column(columns, 'Origin_Code')
    neighbor_col = resolve_column(columns, 'Neighbor_Code')
    weight_col = resolve_column(columns, 'Row_Standardized_Weight')
    edge = raw[[origin_col, neighbor_col, weight_col]].copy()
    edge[origin_col] = canonicalize_code(edge[origin_col])
    edge[neighbor_col] = canonicalize_code(edge[neighbor_col])
    edge[weight_col] = pd.to_numeric(edge[weight_col], errors='coerce')
    if edge[weight_col].isna().any():
        raise ValueError('The spatial weight column contains missing or nonnumeric values.')
    code_to_idx = {str(code): i for i, code in enumerate(node_codes)}
    known = edge[origin_col].isin(code_to_idx) & edge[neighbor_col].isin(code_to_idx)
    unknown_count = int((~known).sum())
    unknown_ratio = unknown_count / max(len(edge), 1)
    if unknown_ratio > cfg.graph_unknown_code_tolerance:
        raise ValueError(f'Unknown-code share in spatial edges={unknown_ratio:.2%}, exceeding threshold={cfg.graph_unknown_code_tolerance:.2%}。')
    edge = edge.loc[known].copy()
    if edge.duplicated([origin_col, neighbor_col]).any():
        edge = edge.groupby([origin_col, neighbor_col], as_index=False)[weight_col].sum()
    src = edge[origin_col].map(code_to_idx).to_numpy(np.int64)
    dst = edge[neighbor_col].map(code_to_idx).to_numpy(np.int64)
    weight = edge[weight_col].to_numpy(np.float32)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_attr = torch.tensor(weight.reshape(-1, 1), dtype=torch.float32)
    degree = np.bincount(src, minlength=len(node_codes))
    audit = {'n_nodes': int(len(node_codes)), 'n_edges_raw': int(len(raw)), 'n_edges_used': int(edge_index.shape[1]), 'unknown_edges_dropped': unknown_count, 'unknown_edge_ratio': float(unknown_ratio), 'zero_out_degree_nodes': int((degree == 0).sum()), 'weight_min': float(weight.min()) if len(weight) else None, 'weight_max': float(weight.max()) if len(weight) else None, 'weight_mean': float(weight.mean()) if len(weight) else None}
    return (edge_index, edge_attr, audit)

@dataclass
class YearBatch:
    year: int
    codes: List[str]
    x_current: torch.Tensor
    x_sequence: torch.Tensor
    lengths: torch.Tensor
    y_eci: torch.Tensor
    y_cpi: torch.Tensor
    y_eci_z: torch.Tensor
    y_cpi_z: torch.Tensor
    true_eci_previous: torch.Tensor
    true_cpi_previous: torch.Tensor
    true_carbon_improvement: torch.Tensor
    true_pollution_improvement: torch.Tensor
    true_state: torch.Tensor
    level_train_mask: torch.Tensor
    level_validation_mask: torch.Tensor
    level_test_mask: torch.Tensor
    state_train_mask: torch.Tensor
    state_validation_mask: torch.Tensor
    state_test_mask: torch.Tensor
    level_target_mask: torch.Tensor
    state_target_mask: torch.Tensor

    def to(self, device: torch.device) -> 'YearBatch':
        kwargs = {}
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                kwargs[key] = value.to(device)
            else:
                kwargs[key] = value
        return YearBatch(**kwargs)

def build_input_matrix(frame: pd.DataFrame, scaler: FeatureStandardizer, continuous_columns: Sequence[str]) -> np.ndarray:
    continuous = scaler.transform(frame[list(continuous_columns)])
    return continuous.astype(np.float32)

def build_year_batches(df: pd.DataFrame, mapping: Dict[str, object], feature_scaler: FeatureStandardizer, target_scaler: LevelTargetStandardizer, cfg: Config) -> Tuple[Dict[int, YearBatch], List[str]]:
    code_col = str(mapping['code_col'])
    year_col = str(mapping['year_col'])
    eci_col = str(mapping['eci_col'])
    cpi_col = str(mapping['cpi_col'])
    exogenous = list(mapping['city_features']) + list(mapping['energy_features'])
    continuous_columns = list(exogenous)
    feature_names = list(exogenous)
    codes = sorted(df[code_col].unique().tolist())
    by_year: Dict[int, pd.DataFrame] = {}
    for year in range(cfg.start_year, cfg.end_year + 1):
        by_year[year] = df[df[year_col] == year].set_index(code_col).reindex(codes).copy()
    batches = {}
    for year in range(cfg.start_year, cfg.end_year + 1):
        current = by_year[year]
        x_current = build_input_matrix(current, feature_scaler, continuous_columns)
        sequence_years = list(range(max(cfg.start_year, year - cfg.lookback + 1), year + 1))
        sequence_arrays = [build_input_matrix(by_year[y], feature_scaler, continuous_columns) for y in sequence_years]
        x_sequence = np.stack(sequence_arrays, axis=1).astype(np.float32)
        lengths = np.full(len(codes), len(sequence_years), dtype=np.int64)
        y_eci = pd.to_numeric(current[eci_col], errors='coerce').to_numpy(dtype=np.float32)
        y_cpi = pd.to_numeric(current[cpi_col], errors='coerce').to_numpy(dtype=np.float32)
        level_valid = np.isfinite(y_eci) & np.isfinite(y_cpi)
        y_eci_fill = np.where(np.isfinite(y_eci), y_eci, target_scaler.eci_geometric_center).astype(np.float32)
        y_cpi_fill = np.where(np.isfinite(y_cpi), y_cpi, target_scaler.cpi_geometric_center).astype(np.float32)
        true_eci_previous = pd.to_numeric(current['True_ECI_Previous'], errors='coerce').to_numpy(dtype=np.float32)
        true_cpi_previous = pd.to_numeric(current['True_CPI_Previous'], errors='coerce').to_numpy(dtype=np.float32)
        true_carbon_improvement = pd.to_numeric(current['True_Carbon_Improvement'], errors='coerce').to_numpy(dtype=np.float32)
        true_pollution_improvement = pd.to_numeric(current['True_Pollution_Improvement'], errors='coerce').to_numpy(dtype=np.float32)
        true_state = pd.to_numeric(current['True_State'], errors='coerce').to_numpy(float)
        state_valid = np.isfinite(true_eci_previous) & np.isfinite(true_cpi_previous) & np.isfinite(true_carbon_improvement) & np.isfinite(true_pollution_improvement) & np.isfinite(true_state)
        true_eci_previous_fill = np.where(np.isfinite(true_eci_previous), true_eci_previous, target_scaler.eci_geometric_center).astype(np.float32)
        true_cpi_previous_fill = np.where(np.isfinite(true_cpi_previous), true_cpi_previous, target_scaler.cpi_geometric_center).astype(np.float32)
        true_carbon_improvement_fill = np.where(np.isfinite(true_carbon_improvement), true_carbon_improvement, 0.0).astype(np.float32)
        true_pollution_improvement_fill = np.where(np.isfinite(true_pollution_improvement), true_pollution_improvement, 0.0).astype(np.float32)
        true_state_fill = np.where(np.isfinite(true_state), true_state, 0).astype(np.int64)
        level_split = current['__level_split__'].fillna('').astype(str).to_numpy()
        state_split = current['__state_split__'].fillna('').astype(str).to_numpy()
        batches[year] = YearBatch(year=year, codes=codes, x_current=torch.tensor(x_current, dtype=torch.float32), x_sequence=torch.tensor(x_sequence, dtype=torch.float32), lengths=torch.tensor(lengths, dtype=torch.long), y_eci=torch.tensor(y_eci_fill, dtype=torch.float32), y_cpi=torch.tensor(y_cpi_fill, dtype=torch.float32), y_eci_z=torch.tensor(target_scaler.eci_to_z(y_eci_fill), dtype=torch.float32), y_cpi_z=torch.tensor(target_scaler.cpi_to_z(y_cpi_fill), dtype=torch.float32), true_eci_previous=torch.tensor(true_eci_previous_fill, dtype=torch.float32), true_cpi_previous=torch.tensor(true_cpi_previous_fill, dtype=torch.float32), true_carbon_improvement=torch.tensor(true_carbon_improvement_fill, dtype=torch.float32), true_pollution_improvement=torch.tensor(true_pollution_improvement_fill, dtype=torch.float32), true_state=torch.tensor(true_state_fill, dtype=torch.long), level_train_mask=torch.tensor(level_valid & (level_split == 'train'), dtype=torch.bool), level_validation_mask=torch.tensor(level_valid & (level_split == 'validation'), dtype=torch.bool), level_test_mask=torch.tensor(level_valid & (level_split == 'test'), dtype=torch.bool), state_train_mask=torch.tensor(state_valid & (state_split == 'train'), dtype=torch.bool), state_validation_mask=torch.tensor(state_valid & (state_split == 'validation'), dtype=torch.bool), state_test_mask=torch.tensor(state_valid & (state_split == 'test'), dtype=torch.bool), level_target_mask=torch.tensor(level_valid, dtype=torch.bool), state_target_mask=torch.tensor(state_valid, dtype=torch.bool))
    return (batches, feature_names)

class SpatialGAT(nn.Module):

    def __init__(self, input_dim: int, cfg: Config):
        super().__init__()
        if cfg.gat_hidden % cfg.gat_heads2 != 0:
            raise ValueError('gat_hidden must be divisible by gat_heads2.')
        second_out = cfg.gat_hidden // cfg.gat_heads2
        self.gat1 = GATConv(in_channels=input_dim, out_channels=cfg.gat_hidden, heads=cfg.gat_heads1, concat=True, dropout=cfg.dropout, edge_dim=1, add_self_loops=cfg.add_self_loops_in_gat, fill_value='mean')
        self.norm1 = nn.LayerNorm(cfg.gat_hidden * cfg.gat_heads1)
        self.gat2 = GATConv(in_channels=cfg.gat_hidden * cfg.gat_heads1, out_channels=second_out, heads=cfg.gat_heads2, concat=True, dropout=cfg.dropout, edge_dim=1, add_self_loops=cfg.add_self_loops_in_gat, fill_value='mean')
        self.norm2 = nn.LayerNorm(cfg.gat_hidden)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.gat1(x, edge_index, edge_attr=edge_attr)
        h = self.dropout(F.elu(self.norm1(h)))
        h = self.gat2(h, edge_index, edge_attr=edge_attr)
        return F.elu(self.norm2(h))

class TemporalGRU(nn.Module):

    def __init__(self, input_dim: int, cfg: Config):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=cfg.gru_hidden, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(cfg.gru_hidden)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x_sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x_sequence, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.dropout(self.norm(h_n[-1]))

class TaskEncoder(nn.Module):

    def __init__(self, input_dim: int, cfg: Config):
        super().__init__()
        self.spatial = SpatialGAT(input_dim, cfg)
        self.temporal = TemporalGRU(input_dim, cfg)
        self.use_county_embedding = bool(cfg.use_county_embedding)
        fusion_input_dim = cfg.gat_hidden + cfg.gru_hidden
        if self.use_county_embedding:
            if int(cfg.n_counties) <= 0:
                raise ValueError('n_counties must be positive when county embeddings are enabled.')
            self.county_embedding = nn.Embedding(int(cfg.n_counties), int(cfg.county_embedding_dim))
            fusion_input_dim += int(cfg.county_embedding_dim)
        else:
            self.county_embedding = None
        self.fusion = nn.Sequential(nn.Linear(fusion_input_dim, cfg.encoder_hidden), nn.LayerNorm(cfg.encoder_hidden), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.encoder_hidden, cfg.encoder_hidden), nn.LayerNorm(cfg.encoder_hidden), nn.GELU())

    def forward(self, x_current: torch.Tensor, x_sequence: torch.Tensor, lengths: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h_spatial = self.spatial(x_current, edge_index, edge_attr)
        h_temporal = self.temporal(x_sequence, lengths)
        fusion_parts = [h_spatial, h_temporal]
        if self.use_county_embedding:
            county_index = torch.arange(x_current.shape[0], dtype=torch.long, device=x_current.device)
            fusion_parts.append(self.county_embedding(county_index))
        return self.fusion(torch.cat(fusion_parts, dim=-1))

class CrossStitchUnit(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        initial_alpha = torch.tensor([[cfg.softshare_init_self, cfg.softshare_init_cross], [cfg.softshare_init_cross, cfg.softshare_init_self]], dtype=torch.float32)
        self.alpha = nn.Parameter(initial_alpha)

    def forward(self, h_eci: torch.Tensor, h_cpi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_eci_soft = self.alpha[0, 0] * h_eci + self.alpha[0, 1] * h_cpi
        h_cpi_soft = self.alpha[1, 0] * h_eci + self.alpha[1, 1] * h_cpi
        return (h_eci_soft, h_cpi_soft)

class LevelRegressionHead(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(cfg.encoder_hidden, cfg.task_hidden), nn.LayerNorm(cfg.task_hidden), nn.GELU(), nn.Dropout(cfg.dropout))
        self.output = nn.Linear(cfg.task_hidden, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.hidden(h)
        return self.output(z).squeeze(-1)

class SoftShareLevelModel(nn.Module):

    def __init__(self, input_dim: int, cfg: Config):
        super().__init__()
        self.eci_encoder = TaskEncoder(input_dim, cfg)
        self.cpi_encoder = TaskEncoder(input_dim, cfg)
        self.cross_stitch = CrossStitchUnit(cfg)
        self.eci_head = LevelRegressionHead(cfg)
        self.cpi_head = LevelRegressionHead(cfg)

    def forward(self, x_current: torch.Tensor, x_sequence: torch.Tensor, lengths: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Dict[str, torch.Tensor]:
        h_eci = self.eci_encoder(x_current, x_sequence, lengths, edge_index, edge_attr)
        h_cpi = self.cpi_encoder(x_current, x_sequence, lengths, edge_index, edge_attr)
        h_eci_soft, h_cpi_soft = self.cross_stitch(h_eci, h_cpi)
        pred_eci_z = self.eci_head(h_eci_soft)
        pred_cpi_z = self.cpi_head(h_cpi_soft)
        return {'pred_eci_z': pred_eci_z, 'pred_cpi_z': pred_cpi_z}

def safe_corr(func, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) < 1e-15 or np.std(y_pred) < 1e-15:
        return float('nan')
    try:
        return float(func(y_true, y_pred)[0])
    except Exception:
        return float('nan')

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {'MAE': float(mean_absolute_error(y_true, y_pred)), 'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))), 'MSE': float(mean_squared_error(y_true, y_pred)), 'R2': float(r2_score(y_true, y_pred)), 'Pearson': safe_corr(pearsonr, y_true, y_pred), 'Spearman': safe_corr(spearmanr, y_true, y_pred)}

def derive_improvements_np(pred_eci_current: np.ndarray, true_eci_previous: np.ndarray, pred_cpi_current: np.ndarray, true_cpi_previous: np.ndarray, epsilon: float=1e-08) -> Tuple[np.ndarray, np.ndarray]:
    pred_eci_current = np.clip(np.asarray(pred_eci_current, dtype=float), epsilon, None)
    true_eci_previous = np.clip(np.asarray(true_eci_previous, dtype=float), epsilon, None)
    pred_cpi_current = np.clip(np.asarray(pred_cpi_current, dtype=float), epsilon, None)
    true_cpi_previous = np.clip(np.asarray(true_cpi_previous, dtype=float), epsilon, None)
    carbon = np.log(true_eci_previous) - np.log(pred_eci_current)
    pollution = np.log(true_cpi_previous) - np.log(pred_cpi_current)
    return (carbon.astype(np.float32), pollution.astype(np.float32))

def get_level_mask(batch: YearBatch, split: str) -> torch.Tensor:
    mapping = {'train': batch.level_train_mask, 'validation': batch.level_validation_mask, 'test': batch.level_test_mask, 'all': batch.level_target_mask}
    if split not in mapping:
        raise ValueError(f'Unknown split: {split}')
    return mapping[split]

@torch.no_grad()
def build_deterministic_level_cache(model: SoftShareLevelModel, batches: Dict[int, YearBatch], edge_index: torch.Tensor, edge_attr: torch.Tensor, target_scaler: LevelTargetStandardizer, device: torch.device) -> Dict[int, Dict[str, torch.Tensor]]:
    model.eval()
    edge_index_d = edge_index.to(device)
    edge_attr_d = edge_attr.to(device)
    cache = {}
    for year, batch_cpu in sorted(batches.items()):
        batch = batch_cpu.to(device)
        out = model(batch.x_current, batch.x_sequence, batch.lengths, edge_index_d, edge_attr_d)
        cache[year] = {'pred_eci': target_scaler.eci_inverse_torch(out['pred_eci_z']), 'pred_cpi': target_scaler.cpi_inverse_torch(out['pred_cpi_z'])}
    return cache

def evaluate_level_cache(cache: Dict[int, Dict[str, torch.Tensor]], batches: Dict[int, YearBatch], split: str, target_scaler: LevelTargetStandardizer, epsilon: float=1e-08) -> Dict:
    true_eci = []
    pred_eci = []
    true_cpi = []
    pred_cpi = []
    for year, batch in sorted(batches.items()):
        mask = get_level_mask(batch, split)
        if int(mask.sum()) == 0:
            continue
        device = cache[year]['pred_eci'].device
        mask_d = mask.to(device)
        true_eci.append(batch.y_eci[mask].cpu().numpy())
        true_cpi.append(batch.y_cpi[mask].cpu().numpy())
        pred_eci.append(cache[year]['pred_eci'][mask_d].detach().cpu().numpy())
        pred_cpi.append(cache[year]['pred_cpi'][mask_d].detach().cpu().numpy())
    y_eci = np.concatenate(true_eci)
    p_eci = np.concatenate(pred_eci)
    y_cpi = np.concatenate(true_cpi)
    p_cpi = np.concatenate(pred_cpi)
    eci_metrics = regression_metrics(y_eci, p_eci)
    cpi_metrics = regression_metrics(y_cpi, p_cpi)
    y_eci_safe = np.clip(y_eci, epsilon, None)
    p_eci_safe = np.clip(p_eci, epsilon, None)
    y_cpi_safe = np.clip(y_cpi, epsilon, None)
    p_cpi_safe = np.clip(p_cpi, epsilon, None)
    true_eci_log_z = (np.log(y_eci_safe) - target_scaler.eci_mean) / target_scaler.eci_std
    pred_eci_log_z = (np.log(p_eci_safe) - target_scaler.eci_mean) / target_scaler.eci_std
    true_cpi_log_z = (np.log(y_cpi_safe) - target_scaler.cpi_mean) / target_scaler.cpi_std
    pred_cpi_log_z = (np.log(p_cpi_safe) - target_scaler.cpi_mean) / target_scaler.cpi_std
    eci_log_z_mse = float(np.mean((true_eci_log_z - pred_eci_log_z) ** 2))
    cpi_log_z_mse = float(np.mean((true_cpi_log_z - pred_cpi_log_z) ** 2))
    return {'ECI': eci_metrics, 'CPI': cpi_metrics, 'ECI_Log_Z_MSE': eci_log_z_mse, 'CPI_Log_Z_MSE': cpi_log_z_mse, 'Validation_Objective': eci_log_z_mse + cpi_log_z_mse}

def train_single_seed(seed: int, batches: Dict[int, YearBatch], edge_index: torch.Tensor, edge_attr: torch.Tensor, target_scaler: LevelTargetStandardizer, input_dim: int, cfg: Config, device: torch.device, seed_dir: Path) -> Dict:
    set_random_seed(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    model = SoftShareLevelModel(input_dim, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.scheduler_factor, patience=cfg.scheduler_patience, min_lr=cfg.min_lr)
    edge_index_d = edge_index.to(device)
    edge_attr_d = edge_attr.to(device)
    best_validation_objective = float('inf')
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        years = sorted(batches)
        random.Random(seed + epoch).shuffle(years)
        epoch_losses = []
        for year in years:
            batch = batches[year].to(device)
            mask = batch.level_train_mask
            if int(mask.sum()) == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            out = model(batch.x_current, batch.x_sequence, batch.lengths, edge_index_d, edge_attr_d)
            loss_eci = F.mse_loss(out['pred_eci_z'][mask], batch.y_eci_z[mask])
            loss_cpi = F.mse_loss(out['pred_cpi_z'][mask], batch.y_cpi_z[mask])
            loss = loss_eci + loss_cpi
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            epoch_losses.append([float(loss.item()), float(loss_eci.item()), float(loss_cpi.item())])
        validation_cache = build_deterministic_level_cache(model, batches, edge_index, edge_attr, target_scaler, device)
        validation_metrics = evaluate_level_cache(validation_cache, batches, 'validation', target_scaler, cfg.level_epsilon)
        validation_objective = validation_metrics['Validation_Objective']
        scheduler.step(validation_objective)
        improved = validation_objective < best_validation_objective - cfg.min_delta
        if improved:
            best_validation_objective = float(validation_objective)
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            patience_counter = 0
        else:
            patience_counter += 1
        losses_array = np.asarray(epoch_losses)
        history.append({'Seed': seed, 'Epoch': epoch, 'Train_Total_MSE_Z': float(losses_array[:, 0].mean()) if len(losses_array) else np.nan, 'Train_Log_ECI_MSE_Z': float(losses_array[:, 1].mean()) if len(losses_array) else np.nan, 'Train_Log_CPI_MSE_Z': float(losses_array[:, 2].mean()) if len(losses_array) else np.nan, 'Validation_Objective': float(validation_objective), 'Validation_ECI_R2': validation_metrics['ECI']['R2'], 'Validation_CPI_R2': validation_metrics['CPI']['R2'], 'Validation_ECI_Pearson': validation_metrics['ECI']['Pearson'], 'Validation_CPI_Pearson': validation_metrics['CPI']['Pearson'], 'Learning_Rate': float(optimizer.param_groups[0]['lr'])})
        if epoch == 1 or epoch % 10 == 0:
            alpha = model.cross_stitch.alpha.detach().cpu().numpy()
            pass
        if epoch >= cfg.min_epochs and patience_counter >= cfg.early_stopping_patience:
            pass
            break
    if best_state is None:
        raise RuntimeError('Training did not produce a best model.')
    model.load_state_dict(best_state)
    model.to(device)
    deterministic_cache = build_deterministic_level_cache(model, batches, edge_index, edge_attr, target_scaler, device)
    level_metrics = {split: evaluate_level_cache(deterministic_cache, batches, split, target_scaler, cfg.level_epsilon) for split in ['train', 'validation', 'test']}
    alpha = model.cross_stitch.alpha.detach().cpu().numpy().tolist()
    checkpoint = {'model_state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()}, 'seed': int(seed), 'best_epoch': int(best_epoch), 'best_validation_objective': float(best_validation_objective), 'input_dim': int(input_dim), 'target_scaler': target_scaler.to_dict(), 'cross_stitch_alpha': alpha, 'config': asdict(cfg)}
    checkpoint_path = seed_dir / 'best_model.pt'
    torch.save(checkpoint, checkpoint_path)
    history_df = pd.DataFrame(history)
    history_df.to_csv(seed_dir / 'training_history.csv', index=False, encoding='utf-8-sig')
    save_json(seed_dir / 'level_metrics.json', level_metrics)
    save_json(seed_dir / 'cross_stitch_weights.json', {'alpha': alpha, 'definition': [['alpha_ECI_ECI', 'alpha_ECI_CPI'], ['alpha_CPI_ECI', 'alpha_CPI_CPI']], 'warning': 'Cross-Stitch represents task-level representation sharing and is not equivalent to input feature importance.'})
    return {'seed': seed, 'checkpoint_path': str(checkpoint_path), 'best_epoch': best_epoch, 'best_validation_objective': best_validation_objective, 'history': history_df, 'level_metrics': level_metrics, 'cross_stitch_alpha': alpha}

def load_trained_model(checkpoint_path: str | Path, input_dim: int, cfg: Config, device: torch.device) -> SoftShareLevelModel:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = SoftShareLevelModel(input_dim, cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def prediction_frame_for_split(model: SoftShareLevelModel, batches: Dict[int, YearBatch], edge_index: torch.Tensor, edge_attr: torch.Tensor, target_scaler: LevelTargetStandardizer, split: str, device: torch.device, seed: Optional[int]=None, epsilon: float=1e-08) -> pd.DataFrame:
    model.eval()
    edge_index_d = edge_index.to(device)
    edge_attr_d = edge_attr.to(device)
    rows = []
    for year, batch_cpu in sorted(batches.items()):
        mask_cpu = get_level_mask(batch_cpu, split)
        if int(mask_cpu.sum()) == 0:
            continue
        batch = batch_cpu.to(device)
        mask = mask_cpu.to(device)
        out = model(batch.x_current, batch.x_sequence, batch.lengths, edge_index_d, edge_attr_d)
        pred_eci = target_scaler.eci_inverse_torch(out['pred_eci_z'])[mask].detach().cpu().numpy()
        pred_cpi = target_scaler.cpi_inverse_torch(out['pred_cpi_z'])[mask].detach().cpu().numpy()
        true_eci = batch.y_eci[mask].detach().cpu().numpy()
        true_cpi = batch.y_cpi[mask].detach().cpu().numpy()
        prev_eci = batch.true_eci_previous[mask].detach().cpu().numpy()
        prev_cpi = batch.true_cpi_previous[mask].detach().cpu().numpy()
        carbon = np.full(len(pred_eci), np.nan, dtype=np.float32)
        pollution = np.full(len(pred_cpi), np.nan, dtype=np.float32)
        valid_prev = np.isfinite(prev_eci) & np.isfinite(prev_cpi) & (prev_eci > 0) & (prev_cpi > 0)
        if valid_prev.any():
            c, p = derive_improvements_np(pred_eci[valid_prev], prev_eci[valid_prev], pred_cpi[valid_prev], prev_cpi[valid_prev], epsilon)
            carbon[valid_prev] = c
            pollution[valid_prev] = p
        indices = np.where(mask_cpu.numpy())[0]
        frame = pd.DataFrame({'Code': [batch_cpu.codes[i] for i in indices], 'Year': int(year), 'True_ECI': true_eci, 'Pred_ECI': pred_eci, 'True_CPI': true_cpi, 'Pred_CPI': pred_cpi, 'True_ECI_Previous_Year': prev_eci, 'True_CPI_Previous_Year': prev_cpi, 'Pred_Carbon_Improvement': carbon, 'Pred_Pollution_Improvement': pollution})
        if seed is not None:
            frame['Seed'] = int(seed)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def evaluate_prediction_frame(frame: pd.DataFrame) -> Dict:
    return {'ECI': regression_metrics(frame['True_ECI'].to_numpy(float), frame['Pred_ECI'].to_numpy(float)), 'CPI': regression_metrics(frame['True_CPI'].to_numpy(float), frame['Pred_CPI'].to_numpy(float)), 'n_samples': int(len(frame))}

def enable_mc_dropout(model: SoftShareLevelModel) -> None:
    model.train()

def _summary_from_mc_array(arr: np.ndarray, prefix: str, cfg: Config) -> Dict[str, np.ndarray]:
    return {f'{prefix}_Mean': np.mean(arr, axis=0).astype(np.float32), f'{prefix}_Std': np.std(arr, axis=0, ddof=0).astype(np.float32), f'{prefix}_P025': np.quantile(arr, cfg.uncertainty_quantile_low, axis=0).astype(np.float32), f'{prefix}_P975': np.quantile(arr, cfg.uncertainty_quantile_high, axis=0).astype(np.float32)}

@torch.no_grad()
def mc_dropout_uncertainty_for_split(model: SoftShareLevelModel, batches: Dict[int, YearBatch], edge_index: torch.Tensor, edge_attr: torch.Tensor, target_scaler: LevelTargetStandardizer, split: str, cfg: Config, device: torch.device) -> pd.DataFrame:
    if cfg.mc_samples < 2:
        raise ValueError('mc_samples must be at least two; 1000 is recommended.')
    edge_index_d = edge_index.to(device)
    edge_attr_d = edge_attr.to(device)
    rows = []
    for year, batch_cpu in sorted(batches.items()):
        mask_cpu = get_level_mask(batch_cpu, split)
        n = int(mask_cpu.sum())
        if n == 0:
            continue
        pass
        batch = batch_cpu.to(device)
        mask = mask_cpu.to(device)
        true_eci = batch.y_eci[mask].detach().cpu().numpy().astype(np.float32)
        true_cpi = batch.y_cpi[mask].detach().cpu().numpy().astype(np.float32)
        prev_eci = batch.true_eci_previous[mask].detach().cpu().numpy().astype(np.float32)
        prev_cpi = batch.true_cpi_previous[mask].detach().cpu().numpy().astype(np.float32)
        eci_draws = np.empty((cfg.mc_samples, n), dtype=np.float32)
        cpi_draws = np.empty((cfg.mc_samples, n), dtype=np.float32)
        valid_prev = np.isfinite(prev_eci) & np.isfinite(prev_cpi) & (prev_eci > 0) & (prev_cpi > 0)
        if valid_prev.any():
            carbon_draws = np.full((cfg.mc_samples, n), np.nan, dtype=np.float32)
            pollution_draws = np.full((cfg.mc_samples, n), np.nan, dtype=np.float32)
        else:
            carbon_draws = None
            pollution_draws = None
        enable_mc_dropout(model)
        for draw in range(cfg.mc_samples):
            out = model(batch.x_current, batch.x_sequence, batch.lengths, edge_index_d, edge_attr_d)
            pred_eci = target_scaler.eci_inverse_torch(out['pred_eci_z'])[mask].detach().cpu().numpy().astype(np.float32)
            pred_cpi = target_scaler.cpi_inverse_torch(out['pred_cpi_z'])[mask].detach().cpu().numpy().astype(np.float32)
            eci_draws[draw] = pred_eci
            cpi_draws[draw] = pred_cpi
            if valid_prev.any():
                carbon, pollution = derive_improvements_np(pred_eci[valid_prev], prev_eci[valid_prev], pred_cpi[valid_prev], prev_cpi[valid_prev], cfg.level_epsilon)
                carbon_draws[draw, valid_prev] = carbon
                pollution_draws[draw, valid_prev] = pollution
        model.eval()
        data = {'Code': [batch_cpu.codes[i] for i in np.where(mask_cpu.numpy())[0]], 'Year': int(year), 'True_ECI': true_eci, 'True_CPI': true_cpi, 'True_ECI_Previous_Year': prev_eci, 'True_CPI_Previous_Year': prev_cpi}
        data.update(_summary_from_mc_array(eci_draws, 'MC_ECI', cfg))
        data.update(_summary_from_mc_array(cpi_draws, 'MC_CPI', cfg))
        if valid_prev.any():
            carbon_stats = {'MC_Carbon_Improvement_Mean': np.full(n, np.nan, dtype=np.float32), 'MC_Carbon_Improvement_Std': np.full(n, np.nan, dtype=np.float32), 'MC_Carbon_Improvement_P025': np.full(n, np.nan, dtype=np.float32), 'MC_Carbon_Improvement_P975': np.full(n, np.nan, dtype=np.float32), 'MC_Pollution_Improvement_Mean': np.full(n, np.nan, dtype=np.float32), 'MC_Pollution_Improvement_Std': np.full(n, np.nan, dtype=np.float32), 'MC_Pollution_Improvement_P025': np.full(n, np.nan, dtype=np.float32), 'MC_Pollution_Improvement_P975': np.full(n, np.nan, dtype=np.float32)}
            c_valid = carbon_draws[:, valid_prev]
            p_valid = pollution_draws[:, valid_prev]
            carbon_stats['MC_Carbon_Improvement_Mean'][valid_prev] = np.mean(c_valid, axis=0)
            carbon_stats['MC_Carbon_Improvement_Std'][valid_prev] = np.std(c_valid, axis=0, ddof=0)
            carbon_stats['MC_Carbon_Improvement_P025'][valid_prev] = np.quantile(c_valid, cfg.uncertainty_quantile_low, axis=0)
            carbon_stats['MC_Carbon_Improvement_P975'][valid_prev] = np.quantile(c_valid, cfg.uncertainty_quantile_high, axis=0)
            carbon_stats['MC_Pollution_Improvement_Mean'][valid_prev] = np.mean(p_valid, axis=0)
            carbon_stats['MC_Pollution_Improvement_Std'][valid_prev] = np.std(p_valid, axis=0, ddof=0)
            carbon_stats['MC_Pollution_Improvement_P025'][valid_prev] = np.quantile(p_valid, cfg.uncertainty_quantile_low, axis=0)
            carbon_stats['MC_Pollution_Improvement_P975'][valid_prev] = np.quantile(p_valid, cfg.uncertainty_quantile_high, axis=0)
            data.update(carbon_stats)
        else:
            for col in ['MC_Carbon_Improvement_Mean', 'MC_Carbon_Improvement_Std', 'MC_Carbon_Improvement_P025', 'MC_Carbon_Improvement_P975', 'MC_Pollution_Improvement_Mean', 'MC_Pollution_Improvement_Std', 'MC_Pollution_Improvement_P025', 'MC_Pollution_Improvement_P975']:
                data[col] = np.full(n, np.nan, dtype=np.float32)
        frame = pd.DataFrame(data)
        frame['ECI_Interval_Width'] = frame['MC_ECI_P975'] - frame['MC_ECI_P025']
        frame['CPI_Interval_Width'] = frame['MC_CPI_P975'] - frame['MC_CPI_P025']
        frame['ECI_Relative_Interval_Width'] = frame['ECI_Interval_Width'] / np.clip(np.abs(frame['MC_ECI_Mean']), cfg.level_epsilon, None)
        frame['CPI_Relative_Interval_Width'] = frame['CPI_Interval_Width'] / np.clip(np.abs(frame['MC_CPI_Mean']), cfg.level_epsilon, None)
        frame['ECI_Covered_By_MC_Interval'] = ((frame['True_ECI'] >= frame['MC_ECI_P025']) & (frame['True_ECI'] <= frame['MC_ECI_P975'])).astype(int)
        frame['CPI_Covered_By_MC_Interval'] = ((frame['True_CPI'] >= frame['MC_CPI_P025']) & (frame['True_CPI'] <= frame['MC_CPI_P975'])).astype(int)
        rows.append(frame)
        del eci_draws
        del cpi_draws
        if carbon_draws is not None:
            del carbon_draws
        if pollution_draws is not None:
            del pollution_draws
    model.eval()
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def evaluate_mc_dropout_uncertainty(frame: pd.DataFrame) -> Dict:
    if frame.empty:
        return {}
    metrics = {'ECI': regression_metrics(frame['True_ECI'].to_numpy(float), frame['MC_ECI_Mean'].to_numpy(float)), 'CPI': regression_metrics(frame['True_CPI'].to_numpy(float), frame['MC_CPI_Mean'].to_numpy(float)), 'uncertainty': {'Mean_ECI_Std': float(frame['MC_ECI_Std'].mean()), 'Median_ECI_Std': float(frame['MC_ECI_Std'].median()), 'Mean_CPI_Std': float(frame['MC_CPI_Std'].mean()), 'Median_CPI_Std': float(frame['MC_CPI_Std'].median()), 'Mean_ECI_Interval_Width': float(frame['ECI_Interval_Width'].mean()), 'Mean_CPI_Interval_Width': float(frame['CPI_Interval_Width'].mean()), 'Mean_ECI_Relative_Interval_Width': float(frame['ECI_Relative_Interval_Width'].mean()), 'Mean_CPI_Relative_Interval_Width': float(frame['CPI_Relative_Interval_Width'].mean()), 'ECI_Empirical_Interval_Coverage': float(frame['ECI_Covered_By_MC_Interval'].mean()), 'CPI_Empirical_Interval_Coverage': float(frame['CPI_Covered_By_MC_Interval'].mean())}, 'n_samples': int(len(frame))}
    valid_improvement = np.isfinite(frame['True_ECI_Previous_Year'].to_numpy(float)) & np.isfinite(frame['True_CPI_Previous_Year'].to_numpy(float)) & (frame['True_ECI_Previous_Year'].to_numpy(float) > 0) & (frame['True_CPI_Previous_Year'].to_numpy(float) > 0)
    if valid_improvement.any():
        true_carbon, true_pollution = derive_improvements_np(frame.loc[valid_improvement, 'True_ECI'].to_numpy(float), frame.loc[valid_improvement, 'True_ECI_Previous_Year'].to_numpy(float), frame.loc[valid_improvement, 'True_CPI'].to_numpy(float), frame.loc[valid_improvement, 'True_CPI_Previous_Year'].to_numpy(float), 1e-08)
        metrics['Carbon_Improvement'] = regression_metrics(true_carbon, frame.loc[valid_improvement, 'MC_Carbon_Improvement_Mean'].to_numpy(float))
        metrics['Pollution_Improvement'] = regression_metrics(true_pollution, frame.loc[valid_improvement, 'MC_Pollution_Improvement_Mean'].to_numpy(float))
        metrics['uncertainty']['Mean_Carbon_Improvement_Std'] = float(frame.loc[valid_improvement, 'MC_Carbon_Improvement_Std'].mean())
        metrics['uncertainty']['Mean_Pollution_Improvement_Std'] = float(frame.loc[valid_improvement, 'MC_Pollution_Improvement_Std'].mean())
    return metrics

def infer_unit_interval_feature(train_series: pd.Series) -> bool:
    values = pd.to_numeric(train_series, errors='coerce').dropna()
    if values.empty:
        return False
    return bool(values.min() >= -1e-08 and values.max() <= 1.0 + 1e-08)

def apply_relative_scenario(panel: pd.DataFrame, feature: str, percent_change: float, train_mask: pd.Series, clip_unit_interval: bool) -> Tuple[pd.DataFrame, Dict]:
    scenario = panel.copy()
    original = pd.to_numeric(scenario[feature], errors='coerce').astype(float)
    factor = 1.0 + percent_change / 100.0
    changed = original * factor
    train_series = pd.to_numeric(panel.loc[train_mask, feature], errors='coerce')
    is_unit_interval = infer_unit_interval_feature(train_series)
    clipped_mask = np.zeros(len(scenario), dtype=bool)
    if clip_unit_interval and is_unit_interval:
        before_clip = changed.copy()
        changed = changed.clip(lower=0.0, upper=1.0)
        clipped_mask = np.abs(changed.to_numpy(float) - before_clip.to_numpy(float)) > 1e-12
    scenario[feature] = changed
    original_np = original.to_numpy(float)
    changed_np = changed.to_numpy(float)
    actual_pct = np.full(len(original_np), np.nan, dtype=float)
    nonzero = np.isfinite(original_np) & np.isfinite(changed_np) & (np.abs(original_np) > 1e-12)
    actual_pct[nonzero] = (changed_np[nonzero] - original_np[nonzero]) / original_np[nonzero] * 100.0
    finite = np.isfinite(actual_pct)
    audit = {'Feature': feature, 'Requested_Percent_Change': float(percent_change), 'Applied_Factor': float(factor), 'Inferred_Unit_Interval': bool(is_unit_interval), 'Clipped_N': int(clipped_mask.sum()), 'Clipped_Rate': float(clipped_mask.mean()), 'Mean_Actual_Percent_Change': float(np.nanmean(actual_pct[finite])) if finite.any() else np.nan}
    return (scenario, audit)

@torch.no_grad()
def predict_specific_year(model: SoftShareLevelModel, batch_cpu: YearBatch, edge_index: torch.Tensor, edge_attr: torch.Tensor, target_scaler: LevelTargetStandardizer, device: torch.device, epsilon: float) -> pd.DataFrame:
    model.eval()
    batch = batch_cpu.to(device)
    out = model(batch.x_current, batch.x_sequence, batch.lengths, edge_index.to(device), edge_attr.to(device))
    pred_eci = target_scaler.eci_inverse_torch(out['pred_eci_z']).detach().cpu().numpy()
    pred_cpi = target_scaler.cpi_inverse_torch(out['pred_cpi_z']).detach().cpu().numpy()
    prev_eci = batch.true_eci_previous.detach().cpu().numpy()
    prev_cpi = batch.true_cpi_previous.detach().cpu().numpy()
    carbon = np.full(len(pred_eci), np.nan, dtype=np.float32)
    pollution = np.full(len(pred_cpi), np.nan, dtype=np.float32)
    valid_prev = np.isfinite(prev_eci) & np.isfinite(prev_cpi) & (prev_eci > 0) & (prev_cpi > 0)
    if valid_prev.any():
        c, p = derive_improvements_np(pred_eci[valid_prev], prev_eci[valid_prev], pred_cpi[valid_prev], prev_cpi[valid_prev], epsilon)
        carbon[valid_prev] = c
        pollution[valid_prev] = p
    return pd.DataFrame({'Code': batch_cpu.codes, 'Year': int(batch_cpu.year), 'Pred_ECI': pred_eci, 'Pred_CPI': pred_cpi, 'Pred_Carbon_Improvement': carbon, 'Pred_Pollution_Improvement': pollution})

def run_scenario_analysis(panel: pd.DataFrame, mapping: Dict[str, object], feature_scaler: FeatureStandardizer, target_scaler: LevelTargetStandardizer, cfg: Config, edge_index: torch.Tensor, edge_attr: torch.Tensor, model: SoftShareLevelModel, baseline_batches: Dict[int, YearBatch], output_dir: Path, device: torch.device) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scenario_dir = output_dir / 'scenario_analysis'
    scenario_dir.mkdir(parents=True, exist_ok=True)
    exogenous = list(mapping['city_features']) + list(mapping['energy_features'])
    scenario_features = list(exogenous)
    code_col = str(mapping['code_col'])
    year_col = str(mapping['year_col'])
    train_mask = panel['__level_split__'] == 'train'
    baseline_by_year = {}
    for year in cfg.scenario_years:
        if year not in baseline_batches:
            raise ValueError(f'Scenario year {year} is outside the model year range.')
        baseline_by_year[year] = predict_specific_year(model, baseline_batches[year], edge_index, edge_attr, target_scaler, device, cfg.level_epsilon)
    summary_rows = []
    audit_rows = []
    sample_outputs = []
    total_scenarios = len(scenario_features) * len(cfg.scenario_percent_changes)
    scenario_counter = 0
    for feature in scenario_features:
        for pct in cfg.scenario_percent_changes:
            scenario_counter += 1
            pass
            scenario_panel, audit = apply_relative_scenario(panel, feature, pct, train_mask, cfg.scenario_clip_unit_interval_features)
            audit_rows.append(audit)
            scenario_batches, _ = build_year_batches(scenario_panel, mapping, feature_scaler, target_scaler, cfg)
            for year in cfg.scenario_years:
                baseline = baseline_by_year[year].copy()
                scenario_pred = predict_specific_year(model, scenario_batches[year], edge_index, edge_attr, target_scaler, device, cfg.level_epsilon)
                merged = baseline.merge(scenario_pred, on=['Code', 'Year'], suffixes=('_Baseline', '_Scenario'), how='inner', validate='one_to_one')
                merged['Feature'] = feature
                merged['Requested_Percent_Change'] = float(pct)
                merged['Direction'] = 'Increase' if pct > 0 else 'Decrease'
                merged['Intensity'] = abs(float(pct))
                merged['Delta_ECI'] = merged['Pred_ECI_Scenario'] - merged['Pred_ECI_Baseline']
                merged['Pct_Delta_ECI'] = merged['Delta_ECI'] / np.clip(np.abs(merged['Pred_ECI_Baseline']), cfg.level_epsilon, None) * 100.0
                merged['Delta_CPI'] = merged['Pred_CPI_Scenario'] - merged['Pred_CPI_Baseline']
                merged['Pct_Delta_CPI'] = merged['Delta_CPI'] / np.clip(np.abs(merged['Pred_CPI_Baseline']), cfg.level_epsilon, None) * 100.0
                merged['Delta_Carbon_Improvement'] = merged['Pred_Carbon_Improvement_Scenario'] - merged['Pred_Carbon_Improvement_Baseline']
                merged['Delta_Pollution_Improvement'] = merged['Pred_Pollution_Improvement_Scenario'] - merged['Pred_Pollution_Improvement_Baseline']
                summary_rows.append({'Feature': feature, 'Year': int(year), 'Requested_Percent_Change': float(pct), 'Direction': 'Increase' if pct > 0 else 'Decrease', 'Intensity': abs(float(pct)), 'N': int(len(merged)), 'Mean_Delta_ECI': float(merged['Delta_ECI'].mean()), 'Median_Delta_ECI': float(merged['Delta_ECI'].median()), 'Mean_Abs_Delta_ECI': float(merged['Delta_ECI'].abs().mean()), 'Mean_Pct_Delta_ECI': float(merged['Pct_Delta_ECI'].mean()), 'Mean_Delta_CPI': float(merged['Delta_CPI'].mean()), 'Median_Delta_CPI': float(merged['Delta_CPI'].median()), 'Mean_Abs_Delta_CPI': float(merged['Delta_CPI'].abs().mean()), 'Mean_Pct_Delta_CPI': float(merged['Pct_Delta_CPI'].mean()), 'Mean_Delta_Carbon_Improvement': float(merged['Delta_Carbon_Improvement'].mean()), 'Mean_Delta_Pollution_Improvement': float(merged['Delta_Pollution_Improvement'].mean()), 'Share_ECI_Increase': float((merged['Delta_ECI'] > 0).mean()), 'Share_ECI_Decrease': float((merged['Delta_ECI'] < 0).mean()), 'Share_CPI_Increase': float((merged['Delta_CPI'] > 0).mean()), 'Share_CPI_Decrease': float((merged['Delta_CPI'] < 0).mean()), 'Clipped_Rate': float(audit['Clipped_Rate']), 'Mean_Actual_Percent_Change': float(audit['Mean_Actual_Percent_Change'])})
                if cfg.scenario_save_sample_level:
                    sample_outputs.append(merged)
    summary = pd.DataFrame(summary_rows)
    audit_df = pd.DataFrame(audit_rows)
    summary.to_csv(scenario_dir / 'scenario_summary.csv', index=False, encoding='utf-8-sig')
    audit_df.to_csv(scenario_dir / 'scenario_input_audit.csv', index=False, encoding='utf-8-sig')
    if cfg.scenario_save_sample_level and sample_outputs:
        sample_df = pd.concat(sample_outputs, ignore_index=True)
        sample_df.to_csv(scenario_dir / 'scenario_sample_level_predictions.csv', index=False, encoding='utf-8-sig')
    sensitivity = build_sensitivity_summary(summary, cfg)
    sensitivity.to_csv(scenario_dir / 'sensitivity_summary.csv', index=False, encoding='utf-8-sig')
    return (summary, sensitivity)

def build_sensitivity_summary(scenario_summary: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for (year, feature), g in scenario_summary.groupby(['Year', 'Feature'], sort=True):
        g = g.sort_values('Requested_Percent_Change')
        x = g['Mean_Actual_Percent_Change'].to_numpy(float)
        y_eci = g['Mean_Pct_Delta_ECI'].to_numpy(float)
        y_cpi = g['Mean_Pct_Delta_CPI'].to_numpy(float)
        y_ci = g['Mean_Delta_Carbon_Improvement'].to_numpy(float)
        y_pi = g['Mean_Delta_Pollution_Improvement'].to_numpy(float)

        def slope(values: np.ndarray) -> float:
            valid = np.isfinite(x) & np.isfinite(values)
            if valid.sum() < 2:
                return np.nan
            return float(np.polyfit(x[valid], values[valid], 1)[0])

        def corr(values: np.ndarray) -> float:
            valid = np.isfinite(x) & np.isfinite(values)
            if valid.sum() < 3:
                return np.nan
            if np.std(x[valid]) < 1e-12 or np.std(values[valid]) < 1e-12:
                return np.nan
            return float(np.corrcoef(x[valid], values[valid])[0, 1])
        plus5 = g[g['Requested_Percent_Change'] == 5]
        minus5 = g[g['Requested_Percent_Change'] == -5]

        def local_symmetric(column: str) -> float:
            if len(plus5) != 1 or len(minus5) != 1:
                return np.nan
            return float((plus5[column].iloc[0] - minus5[column].iloc[0]) / 10.0)
        eci_global_slope = slope(y_eci)
        cpi_global_slope = slope(y_cpi)
        eci_local = local_symmetric('Mean_Pct_Delta_ECI')
        cpi_local = local_symmetric('Mean_Pct_Delta_CPI')
        rows.append({'Year': int(year), 'Feature': feature, 'ECI_Global_Elasticity_Slope': eci_global_slope, 'CPI_Global_Elasticity_Slope': cpi_global_slope, 'ECI_Local_Elasticity_at_5pct': eci_local, 'CPI_Local_Elasticity_at_5pct': cpi_local, 'Carbon_Improvement_Global_Slope': slope(y_ci), 'Pollution_Improvement_Global_Slope': slope(y_pi), 'Carbon_Improvement_Local_Sensitivity_at_5pct': local_symmetric('Mean_Delta_Carbon_Improvement'), 'Pollution_Improvement_Local_Sensitivity_at_5pct': local_symmetric('Mean_Delta_Pollution_Improvement'), 'ECI_Response_Correlation': corr(y_eci), 'CPI_Response_Correlation': corr(y_cpi), 'Max_Abs_Mean_Pct_Delta_ECI': float(np.nanmax(np.abs(y_eci))), 'Max_Abs_Mean_Pct_Delta_CPI': float(np.nanmax(np.abs(y_cpi)))})
    sensitivity = pd.DataFrame(rows)
    sensitivity['Combined_Absolute_Elasticity'] = sensitivity['ECI_Global_Elasticity_Slope'].abs() + sensitivity['CPI_Global_Elasticity_Slope'].abs()
    sensitivity['Combined_Max_Response'] = sensitivity['Max_Abs_Mean_Pct_Delta_ECI'] + sensitivity['Max_Abs_Mean_Pct_Delta_CPI']
    sensitivity = sensitivity.sort_values(['Year', 'Combined_Absolute_Elasticity'], ascending=[True, False]).reset_index(drop=True)
    sensitivity['Sensitivity_Rank'] = sensitivity.groupby('Year')['Combined_Absolute_Elasticity'].rank(method='min', ascending=False).astype(int)
    return sensitivity

def plot_training_history(history_df: pd.DataFrame, path: Path) -> None:
    if history_df.empty:
        return
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for seed, g in history_df.groupby('Seed'):
        ax.plot(g['Epoch'], g['Validation_Objective'], label=f'Seed {seed}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation objective')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)

def plot_sensitivity_ranking(sensitivity: pd.DataFrame, scenario_dir: Path) -> None:
    if sensitivity.empty:
        return
    plt.rcParams['font.family'] = 'Times New Roman'
    for year, g in sensitivity.groupby('Year'):
        g = g.sort_values('Combined_Absolute_Elasticity', ascending=True)
        fig_height = max(5.5, 0.34 * len(g) + 1.2)
        fig, ax = plt.subplots(figsize=(9, fig_height))
        ax.barh(g['Feature'], g['Combined_Absolute_Elasticity'])
        ax.set_xlabel('Combined absolute elasticity')
        ax.set_ylabel('')
        fig.tight_layout()
        fig.savefig(scenario_dir / f'sensitivity_ranking_{year}.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

def _pad_sequence_tensor(sequence: torch.Tensor, lookback: int) -> torch.Tensor:
    if sequence.shape[1] == lookback:
        return sequence.clone()
    if sequence.shape[1] > lookback:
        return sequence[:, -lookback:, :].clone()
    padded = sequence.new_zeros((sequence.shape[0], lookback, sequence.shape[2]))
    padded[:, :sequence.shape[1], :] = sequence
    return padded

def collect_shap_background_pool(batches: Dict[int, YearBatch], background_size: int, cfg: Config) -> Dict[str, object]:
    candidates = []
    for year, batch in sorted(batches.items()):
        mask = batch.level_train_mask.numpy()
        lengths = batch.lengths.numpy()
        idx = np.where(mask & (lengths == cfg.lookback))[0]
        for i in idx:
            candidates.append((int(year), int(i)))
    if not candidates:
        for year, batch in sorted(batches.items()):
            idx = np.where(batch.level_train_mask.numpy())[0]
            for i in idx:
                candidates.append((int(year), int(i)))
    if not candidates:
        raise RuntimeError('No training samples are available for the SHAP background set.')
    rng = np.random.default_rng(cfg.shap_random_seed)
    replace = len(candidates) < background_size
    selected_positions = rng.choice(len(candidates), size=background_size, replace=replace)
    selected = [candidates[int(i)] for i in selected_positions]
    x_current = []
    x_sequence = []
    codes = []
    years = []
    lengths = []
    for year, i in selected:
        batch = batches[year]
        x_current.append(batch.x_current[i].numpy())
        padded = _pad_sequence_tensor(batch.x_sequence[i:i + 1], cfg.lookback)[0]
        x_sequence.append(padded.numpy())
        codes.append(batch.codes[i])
        years.append(year)
        lengths.append(int(batch.lengths[i]))
    return {'x_current': np.stack(x_current).astype(np.float32), 'x_sequence': np.stack(x_sequence).astype(np.float32), 'lengths': np.asarray(lengths, dtype=np.int64), 'codes': codes, 'years': years, 'sampling_with_replacement': bool(replace), 'n_candidates': int(len(candidates))}

@torch.no_grad()
def _predict_log_targets_from_tensors(model: SoftShareLevelModel, x_current: torch.Tensor, x_sequence: torch.Tensor, lengths: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, target_scaler: LevelTargetStandardizer) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    out = model(x_current, x_sequence, lengths, edge_index, edge_attr)
    return (target_scaler.eci_log_inverse_torch(out['pred_eci_z']), target_scaler.cpi_log_inverse_torch(out['pred_cpi_z']))

@torch.no_grad()
def graph_aware_shap_for_batch(model: SoftShareLevelModel, batch_cpu: YearBatch, background_pool: Dict[str, object], explain_feature_indices: Sequence[int], edge_index: torch.Tensor, edge_attr: torch.Tensor, target_scaler: LevelTargetStandardizer, cfg: Config, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    rng = np.random.default_rng(cfg.shap_random_seed + int(batch_cpu.year))
    test_current = batch_cpu.x_current.to(device)
    test_sequence = _pad_sequence_tensor(batch_cpu.x_sequence, cfg.lookback).to(device)
    test_lengths = batch_cpu.lengths.to(device)
    edge_index_d = edge_index.to(device)
    edge_attr_d = edge_attr.to(device)
    n_nodes, _ = test_current.shape
    explain_feature_indices = [int(j) for j in explain_feature_indices]
    n_explained = len(explain_feature_indices)
    test_mask = batch_cpu.level_test_mask.to(device)
    selected_idx = torch.nonzero(test_mask, as_tuple=False).squeeze(-1)
    if selected_idx.numel() == 0:
        raise RuntimeError(f'{batch_cpu.year} has no test samples available for SHAP interpretation.')
    pool_current = torch.tensor(background_pool['x_current'], dtype=torch.float32, device=device)
    pool_sequence = torch.tensor(background_pool['x_sequence'], dtype=torch.float32, device=device)
    pool_size = int(pool_current.shape[0])
    shap_eci = torch.zeros((selected_idx.numel(), n_explained), dtype=torch.float32, device=device)
    shap_cpi = torch.zeros_like(shap_eci)
    base_eci = torch.zeros(selected_idx.numel(), device=device)
    base_cpi = torch.zeros(selected_idx.numel(), device=device)
    for _ in range(cfg.shap_permutations):
        draw = rng.choice(pool_size, size=n_nodes, replace=pool_size < n_nodes)
        draw_t = torch.tensor(draw, dtype=torch.long, device=device)
        background_current = pool_current.index_select(0, draw_t)
        background_sequence = pool_sequence.index_select(0, draw_t)
        work_current = test_current.clone()
        work_sequence = test_sequence.clone()
        for j in explain_feature_indices:
            work_current[:, j] = background_current[:, j]
            work_sequence[:, :, j] = background_sequence[:, :, j]
        previous_eci, previous_cpi = _predict_log_targets_from_tensors(model, work_current, work_sequence, test_lengths, edge_index_d, edge_attr_d, target_scaler)
        base_eci += previous_eci.index_select(0, selected_idx)
        base_cpi += previous_cpi.index_select(0, selected_idx)
        order = rng.permutation(n_explained)
        for explained_position in order:
            position = int(explained_position)
            j = explain_feature_indices[position]
            work_current[:, j] = test_current[:, j]
            work_sequence[:, :, j] = test_sequence[:, :, j]
            current_eci, current_cpi = _predict_log_targets_from_tensors(model, work_current, work_sequence, test_lengths, edge_index_d, edge_attr_d, target_scaler)
            shap_eci[:, position] += (current_eci - previous_eci).index_select(0, selected_idx)
            shap_cpi[:, position] += (current_cpi - previous_cpi).index_select(0, selected_idx)
            previous_eci = current_eci
            previous_cpi = current_cpi
    divisor = float(cfg.shap_permutations)
    shap_eci /= divisor
    shap_cpi /= divisor
    base_eci /= divisor
    base_cpi /= divisor
    pred_eci, pred_cpi = _predict_log_targets_from_tensors(model, test_current, test_sequence, test_lengths, edge_index_d, edge_attr_d, target_scaler)
    pred_eci = pred_eci.index_select(0, selected_idx)
    pred_cpi = pred_cpi.index_select(0, selected_idx)
    return {'selected_indices': selected_idx.detach().cpu().numpy(), 'shap_eci': shap_eci.detach().cpu().numpy(), 'shap_cpi': shap_cpi.detach().cpu().numpy(), 'base_eci': base_eci.detach().cpu().numpy(), 'base_cpi': base_cpi.detach().cpu().numpy(), 'pred_log_eci': pred_eci.detach().cpu().numpy(), 'pred_log_cpi': pred_cpi.detach().cpu().numpy()}

@torch.no_grad()
def graph_aware_shap_interaction_for_batch(model: SoftShareLevelModel, batch_cpu: YearBatch, background_pool: Dict[str, object], explain_feature_indices: Sequence[int], feature_pairs: Sequence[Tuple[str, str]], explained_feature_names: Sequence[str], edge_index: torch.Tensor, edge_attr: torch.Tensor, target_scaler: LevelTargetStandardizer, cfg: Config, device: torch.device) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
    model.eval()
    test_current = batch_cpu.x_current.to(device)
    test_sequence = _pad_sequence_tensor(batch_cpu.x_sequence, cfg.lookback).to(device)
    test_lengths = batch_cpu.lengths.to(device)
    selected_idx = torch.nonzero(batch_cpu.level_test_mask.to(device), as_tuple=False).squeeze(-1)
    if selected_idx.numel() == 0:
        raise RuntimeError(f'{batch_cpu.year} has no test samples available for 2D SHAP.')
    explain_feature_indices = [int(j) for j in explain_feature_indices]
    name_to_full_index = {str(name): int(index) for name, index in zip(explained_feature_names, explain_feature_indices)}
    pool_current = torch.as_tensor(background_pool['x_current'], dtype=torch.float32, device=device)
    pool_sequence = torch.as_tensor(background_pool['x_sequence'], dtype=torch.float32, device=device)
    pool_size = int(pool_current.shape[0])
    n_nodes = int(test_current.shape[0])
    edge_index_d = edge_index.to(device)
    edge_attr_d = edge_attr.to(device)
    outputs: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for pair_number, (feature_a, feature_b) in enumerate(feature_pairs):
        if feature_a not in name_to_full_index or feature_b not in name_to_full_index:
            raise ValueError(f'A 2D SHAP variable is not in the interpretable exogenous feature set: {feature_a}, {feature_b}')
        ia = name_to_full_index[feature_a]
        ib = name_to_full_index[feature_b]
        remaining = [j for j in explain_feature_indices if j not in {ia, ib}]
        rng = np.random.default_rng(cfg.shap_random_seed + int(batch_cpu.year) * 1009 + (pair_number + 1) * 7919)
        interaction_eci = torch.zeros(selected_idx.numel(), dtype=torch.float32, device=device)
        interaction_cpi = torch.zeros_like(interaction_eci)
        for _ in range(cfg.shap_2d_permutations):
            draw = rng.choice(pool_size, size=n_nodes, replace=pool_size < n_nodes)
            draw_t = torch.as_tensor(draw, dtype=torch.long, device=device)
            background_current = pool_current.index_select(0, draw_t)
            background_sequence = pool_sequence.index_select(0, draw_t)
            base_current = test_current.clone()
            base_sequence = test_sequence.clone()
            for j in explain_feature_indices:
                base_current[:, j] = background_current[:, j]
                base_sequence[:, :, j] = background_sequence[:, :, j]
            coalition_size = int(rng.integers(0, len(remaining) + 1))
            if coalition_size > 0:
                coalition = rng.choice(remaining, size=coalition_size, replace=False).tolist()
                for j in coalition:
                    base_current[:, j] = test_current[:, j]
                    base_sequence[:, :, j] = test_sequence[:, :, j]

            def predict_with_pair(use_a: bool, use_b: bool):
                current = base_current.clone()
                sequence = base_sequence.clone()
                if use_a:
                    current[:, ia] = test_current[:, ia]
                    sequence[:, :, ia] = test_sequence[:, :, ia]
                if use_b:
                    current[:, ib] = test_current[:, ib]
                    sequence[:, :, ib] = test_sequence[:, :, ib]
                return _predict_log_targets_from_tensors(model, current, sequence, test_lengths, edge_index_d, edge_attr_d, target_scaler)
            p00 = predict_with_pair(False, False)
            p10 = predict_with_pair(True, False)
            p01 = predict_with_pair(False, True)
            p11 = predict_with_pair(True, True)
            delta_eci = 0.5 * (p11[0] - p10[0] - p01[0] + p00[0])
            delta_cpi = 0.5 * (p11[1] - p10[1] - p01[1] + p00[1])
            interaction_eci += delta_eci.index_select(0, selected_idx)
            interaction_cpi += delta_cpi.index_select(0, selected_idx)
        divisor = float(cfg.shap_2d_permutations)
        outputs[feature_a, feature_b] = {'selected_indices': selected_idx.detach().cpu().numpy(), 'shap_interaction_eci': (interaction_eci / divisor).detach().cpu().numpy(), 'shap_interaction_cpi': (interaction_cpi / divisor).detach().cpu().numpy()}
    return outputs

def input_matrix_on_display_scale(x_current: np.ndarray, feature_names: Sequence[str], feature_scaler: FeatureStandardizer) -> np.ndarray:
    display = np.asarray(x_current, dtype=float).copy()
    for j, feature in enumerate(feature_names):
        if feature in feature_scaler.mean_:
            display[:, j] = display[:, j] * feature_scaler.std_[feature] + feature_scaler.mean_[feature]
    return display

def plot_overall_shap_beeswarm(shap_eci: np.ndarray, shap_cpi: np.ndarray, display_values: np.ndarray, feature_names: Sequence[str], output_path: Path, top_n: int, random_seed: int) -> None:
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'pdf.fonttype': 42, 'font.size': 8, 'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': 0.8})
    combined_importance = np.mean(np.abs(shap_eci), axis=0) + np.mean(np.abs(shap_cpi), axis=0)
    show_n = min(int(top_n), len(feature_names))
    selected = np.argsort(combined_importance)[-show_n:]
    selected = selected[np.argsort(combined_importance[selected])]
    fig_height = max(5.5, 0.32 * show_n + 1.4)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, fig_height), sharey=True)
    rng = np.random.default_rng(random_seed)
    cmap = mpl.colormaps['coolwarm']
    for ax, values, title in zip(axes, [shap_eci, shap_cpi], ['ECI', 'CPI']):
        for row, j in enumerate(selected):
            color_value = display_values[:, j]
            finite = np.isfinite(color_value)
            if finite.any():
                q05, q95 = np.nanquantile(color_value[finite], [0.05, 0.95])
                scale = max(float(q95 - q05), 1e-12)
                normalized = np.clip((color_value - q05) / scale, 0.0, 1.0)
            else:
                normalized = np.full(len(color_value), 0.5)
            jitter = rng.normal(0.0, 0.1, size=values.shape[0])
            ax.scatter(values[:, j], row + jitter, c=normalized, cmap=cmap, vmin=0.0, vmax=1.0, s=8, alpha=0.58, linewidths=0, rasterized=True)
        ax.axvline(0.0, color='#666666', linewidth=0.8, linestyle='--')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(f'SHAP value: change in ln({title})')
        ax.grid(axis='x', color='#E5E5E5', linewidth=0.5)
    axes[0].set_yticks(np.arange(show_n))
    axes[0].set_yticklabels([feature_names[j] for j in selected])
    axes[0].set_ylabel('')
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(['Low', 'High'])
    cbar.set_label('Feature value')
    fig.subplots_adjust(left=0.23, right=0.92, bottom=0.1, top=0.93, wspace=0.12)
    fig.savefig(output_path.with_suffix('.png'), dpi=600, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return np.asarray([], dtype=float)
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, int(n_bins) + 1)))
    return edges if len(edges) >= 2 else np.asarray([], dtype=float)

def _standardized_value(raw_value: float, feature: str, feature_scaler: FeatureStandardizer) -> float:
    return float((raw_value - feature_scaler.mean_[feature]) / feature_scaler.std_[feature])

@torch.no_grad()
def _ale_prediction_with_updates(model: SoftShareLevelModel, batch_cpu: YearBatch, updates: Sequence[Tuple[int, np.ndarray, float]], edge_index: torch.Tensor, edge_attr: torch.Tensor, target_scaler: LevelTargetStandardizer, cfg: Config, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    x_current = batch_cpu.x_current.to(device).clone()
    x_sequence = _pad_sequence_tensor(batch_cpu.x_sequence, cfg.lookback).to(device)
    lengths = batch_cpu.lengths.to(device)
    for feature_index, node_indices, standardized_value in updates:
        nodes = torch.tensor(node_indices, dtype=torch.long, device=device)
        x_current[nodes, int(feature_index)] = float(standardized_value)
        if cfg.ale_sync_current_step_in_gru:
            steps = lengths.index_select(0, nodes) - 1
            valid = steps >= 0
            if bool(valid.any()):
                valid_nodes = nodes[valid]
                valid_steps = steps[valid]
                x_sequence[valid_nodes, valid_steps, int(feature_index)] = float(standardized_value)
    log_eci, log_cpi = _predict_log_targets_from_tensors(model, x_current, x_sequence, lengths, edge_index.to(device), edge_attr.to(device), target_scaler)
    return (log_eci.detach().cpu().numpy(), log_cpi.detach().cpu().numpy())

def compute_ale_1d_for_batch(model: SoftShareLevelModel, batch_cpu: YearBatch, exogenous_features: Sequence[str], feature_names: Sequence[str], display_values: np.ndarray, feature_scaler: FeatureStandardizer, edge_index: torch.Tensor, edge_attr: torch.Tensor, target_scaler: LevelTargetStandardizer, cfg: Config, device: torch.device) -> pd.DataFrame:
    rows = []
    test_nodes = np.where(batch_cpu.level_test_mask.numpy())[0]
    for feature in exogenous_features:
        j = feature_names.index(feature)
        raw = display_values[:, j]
        edges = _quantile_edges(raw[test_nodes], cfg.ale_1d_bins)
        if len(edges) < 2:
            continue
        local_eci = []
        local_cpi = []
        counts = []
        centers = []
        for k in range(len(edges) - 1):
            if k == len(edges) - 2:
                in_bin = (raw >= edges[k]) & (raw <= edges[k + 1])
            else:
                in_bin = (raw >= edges[k]) & (raw < edges[k + 1])
            nodes = np.intersect1d(np.where(in_bin)[0], test_nodes)
            if len(nodes) == 0:
                local_eci.append(0.0)
                local_cpi.append(0.0)
                counts.append(0)
                centers.append((edges[k] + edges[k + 1]) / 2.0)
                continue
            low = _standardized_value(edges[k], feature, feature_scaler)
            high = _standardized_value(edges[k + 1], feature, feature_scaler)
            pred_low = _ale_prediction_with_updates(model, batch_cpu, [(j, nodes, low)], edge_index, edge_attr, target_scaler, cfg, device)
            pred_high = _ale_prediction_with_updates(model, batch_cpu, [(j, nodes, high)], edge_index, edge_attr, target_scaler, cfg, device)
            local_eci.append(float(np.mean(pred_high[0][nodes] - pred_low[0][nodes])))
            local_cpi.append(float(np.mean(pred_high[1][nodes] - pred_low[1][nodes])))
            counts.append(int(len(nodes)))
            centers.append((edges[k] + edges[k + 1]) / 2.0)
        counts_np = np.asarray(counts, dtype=float)
        ale_eci = np.cumsum(local_eci) - 0.5 * np.asarray(local_eci)
        ale_cpi = np.cumsum(local_cpi) - 0.5 * np.asarray(local_cpi)
        if counts_np.sum() > 0:
            ale_eci -= np.average(ale_eci, weights=counts_np)
            ale_cpi -= np.average(ale_cpi, weights=counts_np)
        for k in range(len(edges) - 1):
            rows.append({'Year': int(batch_cpu.year), 'Feature': feature, 'Bin': int(k + 1), 'Lower': float(edges[k]), 'Upper': float(edges[k + 1]), 'Center': float(centers[k]), 'Count': int(counts[k]), 'ALE_Log_ECI': float(ale_eci[k]), 'ALE_Log_CPI': float(ale_cpi[k])})
    return pd.DataFrame(rows)

def _double_center_surface(surface: np.ndarray, counts: np.ndarray) -> np.ndarray:
    centered = np.asarray(surface, dtype=float).copy()
    weights = np.asarray(counts, dtype=float)
    for _ in range(4):
        row_weights = weights.sum(axis=1)
        for i in range(centered.shape[0]):
            if row_weights[i] > 0:
                row_mean = np.sum(centered[i] * weights[i]) / row_weights[i]
                centered[i] -= row_mean
        col_weights = weights.sum(axis=0)
        for j in range(centered.shape[1]):
            if col_weights[j] > 0:
                col_mean = np.sum(centered[:, j] * weights[:, j]) / col_weights[j]
                centered[:, j] -= col_mean
    if weights.sum() > 0:
        centered -= np.sum(centered * weights) / weights.sum()
    return centered

def compute_ale_2d_for_batch(model: SoftShareLevelModel, batch_cpu: YearBatch, selected_features: Sequence[str], feature_names: Sequence[str], display_values: np.ndarray, feature_scaler: FeatureStandardizer, edge_index: torch.Tensor, edge_attr: torch.Tensor, target_scaler: LevelTargetStandardizer, cfg: Config, device: torch.device) -> pd.DataFrame:
    rows = []
    test_nodes = np.where(batch_cpu.level_test_mask.numpy())[0]
    for feature_a, feature_b in combinations(selected_features, 2):
        ia = feature_names.index(feature_a)
        ib = feature_names.index(feature_b)
        raw_a = display_values[:, ia]
        raw_b = display_values[:, ib]
        edges_a = _quantile_edges(raw_a[test_nodes], cfg.ale_2d_bins)
        edges_b = _quantile_edges(raw_b[test_nodes], cfg.ale_2d_bins)
        if len(edges_a) < 2 or len(edges_b) < 2:
            continue
        shape = (len(edges_a) - 1, len(edges_b) - 1)
        local_eci = np.zeros(shape, dtype=float)
        local_cpi = np.zeros(shape, dtype=float)
        counts = np.zeros(shape, dtype=int)
        for ka in range(shape[0]):
            cond_a = (raw_a >= edges_a[ka]) & (raw_a <= edges_a[ka + 1] if ka == shape[0] - 1 else raw_a < edges_a[ka + 1])
            for kb in range(shape[1]):
                cond_b = (raw_b >= edges_b[kb]) & (raw_b <= edges_b[kb + 1] if kb == shape[1] - 1 else raw_b < edges_b[kb + 1])
                nodes = np.intersect1d(np.where(cond_a & cond_b)[0], test_nodes)
                counts[ka, kb] = len(nodes)
                if len(nodes) == 0:
                    continue
                a0 = _standardized_value(edges_a[ka], feature_a, feature_scaler)
                a1 = _standardized_value(edges_a[ka + 1], feature_a, feature_scaler)
                b0 = _standardized_value(edges_b[kb], feature_b, feature_scaler)
                b1 = _standardized_value(edges_b[kb + 1], feature_b, feature_scaler)
                p00 = _ale_prediction_with_updates(model, batch_cpu, [(ia, nodes, a0), (ib, nodes, b0)], edge_index, edge_attr, target_scaler, cfg, device)
                p10 = _ale_prediction_with_updates(model, batch_cpu, [(ia, nodes, a1), (ib, nodes, b0)], edge_index, edge_attr, target_scaler, cfg, device)
                p01 = _ale_prediction_with_updates(model, batch_cpu, [(ia, nodes, a0), (ib, nodes, b1)], edge_index, edge_attr, target_scaler, cfg, device)
                p11 = _ale_prediction_with_updates(model, batch_cpu, [(ia, nodes, a1), (ib, nodes, b1)], edge_index, edge_attr, target_scaler, cfg, device)
                local_eci[ka, kb] = float(np.mean(p11[0][nodes] - p10[0][nodes] - p01[0][nodes] + p00[0][nodes]))
                local_cpi[ka, kb] = float(np.mean(p11[1][nodes] - p10[1][nodes] - p01[1][nodes] + p00[1][nodes]))
        surface_eci = np.cumsum(np.cumsum(local_eci, axis=0), axis=1)
        surface_cpi = np.cumsum(np.cumsum(local_cpi, axis=0), axis=1)
        surface_eci = _double_center_surface(surface_eci, counts)
        surface_cpi = _double_center_surface(surface_cpi, counts)
        for ka in range(shape[0]):
            for kb in range(shape[1]):
                if counts[ka, kb] == 0:
                    continue
                rows.append({'Year': int(batch_cpu.year), 'Feature_1': feature_a, 'Feature_2': feature_b, 'Bin_1': int(ka + 1), 'Bin_2': int(kb + 1), 'Lower_1': float(edges_a[ka]), 'Upper_1': float(edges_a[ka + 1]), 'Center_1': float((edges_a[ka] + edges_a[ka + 1]) / 2.0), 'Lower_2': float(edges_b[kb]), 'Upper_2': float(edges_b[kb + 1]), 'Center_2': float((edges_b[kb] + edges_b[kb + 1]) / 2.0), 'Count': int(counts[ka, kb]), 'ALE2D_Log_ECI': float(surface_eci[ka, kb]), 'ALE2D_Log_CPI': float(surface_cpi[ka, kb])})
    return pd.DataFrame(rows)

def run_shap_and_ale(model: SoftShareLevelModel, batches: Dict[int, YearBatch], feature_names: Sequence[str], exogenous_features: Sequence[str], feature_scaler: FeatureStandardizer, target_scaler: LevelTargetStandardizer, edge_index: torch.Tensor, edge_attr: torch.Tensor, cfg: Config, output_dir: Path, device: torch.device) -> None:
    explanation_dir = output_dir / 'shap_ale'
    explanation_dir.mkdir(parents=True, exist_ok=True)
    explained_feature_names = list(exogenous_features)
    missing_explainers = [feature for feature in explained_feature_names if feature not in feature_names]
    if missing_explainers:
        raise ValueError(f'Exogenous interpretation feature is not in the model input: {missing_explainers}')
    explained_feature_indices = [list(feature_names).index(feature) for feature in explained_feature_names]
    test_count = int(sum((int(batch.level_test_mask.sum()) for batch in batches.values())))
    if test_count <= 0:
        raise RuntimeError('The test set is empty, so SHAP and ALE cannot be computed.')
    background_size = int(cfg.shap_background_multiplier * test_count)
    background_pool = collect_shap_background_pool(batches, background_size, cfg)
    pd.DataFrame({'Code': background_pool['codes'], 'Year': background_pool['years'], 'Length': background_pool['lengths']}).to_csv(explanation_dir / 'shap_background_index.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(explanation_dir / 'shap_background_graph_pool.npz', x_current=np.asarray(background_pool['x_current'], dtype=np.float32), x_sequence=np.asarray(background_pool['x_sequence'], dtype=np.float32), lengths=np.asarray(background_pool['lengths'], dtype=np.int64), codes=np.asarray(background_pool['codes'], dtype=str), years=np.asarray(background_pool['years'], dtype=np.int64), feature_names=np.asarray(feature_names, dtype=str), exogenous_features=np.asarray(explained_feature_names, dtype=str), exogenous_feature_indices=np.asarray(explained_feature_indices, dtype=np.int64))
    background_display_full = input_matrix_on_display_scale(np.asarray(background_pool['x_current'], dtype=np.float32), feature_names, feature_scaler)
    background_original = pd.DataFrame(background_display_full[:, explained_feature_indices], columns=explained_feature_names)
    background_original.insert(0, 'Year', np.asarray(background_pool['years'], dtype=np.int64))
    background_original.insert(0, 'Code', np.asarray(background_pool['codes'], dtype=str))
    background_original.to_csv(explanation_dir / 'shap_background_original_features.csv', index=False, encoding='utf-8-sig')
    long_frames = []
    additivity_rows = []
    shap_eci_all = []
    shap_cpi_all = []
    display_all = []
    code_all = []
    year_all = []
    base_eci_all = []
    base_cpi_all = []
    pred_eci_all = []
    pred_cpi_all = []
    test_original_frames = []
    for year, batch in sorted(batches.items()):
        if int(batch.level_test_mask.sum()) == 0:
            continue
        result = graph_aware_shap_for_batch(model, batch, background_pool, explained_feature_indices, edge_index, edge_attr, target_scaler, cfg, device)
        idx = result['selected_indices']
        display_full = input_matrix_on_display_scale(batch.x_current.numpy(), feature_names, feature_scaler)
        display = display_full[idx][:, explained_feature_indices]
        shap_eci = result['shap_eci']
        shap_cpi = result['shap_cpi']
        shap_eci_all.append(shap_eci)
        shap_cpi_all.append(shap_cpi)
        display_all.append(display)
        codes = [batch.codes[int(i)] for i in idx]
        code_all.extend(codes)
        year_all.extend([int(year)] * len(codes))
        base_eci_all.append(result['base_eci'])
        base_cpi_all.append(result['base_cpi'])
        pred_eci_all.append(result['pred_log_eci'])
        pred_cpi_all.append(result['pred_log_cpi'])
        n = len(idx)
        long_frames.append(pd.DataFrame({'Code': np.repeat(codes, len(explained_feature_names)), 'Year': np.repeat(int(year), n * len(explained_feature_names)), 'Feature': np.tile(np.asarray(explained_feature_names, dtype=object), n), 'Feature_Value': display.reshape(-1), 'SHAP_Log_ECI': shap_eci.reshape(-1), 'SHAP_Log_CPI': shap_cpi.reshape(-1)}))
        original_frame = pd.DataFrame(display, columns=explained_feature_names)
        original_frame.insert(0, 'Year', int(year))
        original_frame.insert(0, 'Code', codes)
        test_original_frames.append(original_frame)
        np.savez_compressed(explanation_dir / f'test_graph_{int(year)}.npz', x_current=batch.x_current.numpy().astype(np.float32), x_sequence=_pad_sequence_tensor(batch.x_sequence, cfg.lookback).numpy().astype(np.float32), lengths=batch.lengths.numpy().astype(np.int64), test_mask=batch.level_test_mask.numpy().astype(bool), codes=np.asarray(batch.codes, dtype=str), year=np.asarray([int(year)], dtype=np.int64), y_eci=batch.y_eci.numpy().astype(np.float32), y_cpi=batch.y_cpi.numpy().astype(np.float32), feature_names=np.asarray(feature_names, dtype=str), exogenous_features=np.asarray(explained_feature_names, dtype=str), exogenous_feature_indices=np.asarray(explained_feature_indices, dtype=np.int64), edge_index=edge_index.cpu().numpy().astype(np.int64), edge_attr=edge_attr.cpu().numpy().astype(np.float32))
        residual_eci = result['pred_log_eci'] - result['base_eci'] - shap_eci.sum(axis=1)
        residual_cpi = result['pred_log_cpi'] - result['base_cpi'] - shap_cpi.sum(axis=1)
        additivity_rows.extend([{'Code': code, 'Year': int(year), 'Base_Log_ECI': float(be), 'Pred_Log_ECI': float(pe), 'Additivity_Residual_ECI': float(re), 'Base_Log_CPI': float(bp), 'Pred_Log_CPI': float(pp), 'Additivity_Residual_CPI': float(rp)} for code, be, pe, re, bp, pp, rp in zip(codes, result['base_eci'], result['pred_log_eci'], residual_eci, result['base_cpi'], result['pred_log_cpi'], residual_cpi)])
    shap_long = pd.concat(long_frames, ignore_index=True)
    shap_long.to_csv(explanation_dir / 'shap_values_test_long.csv', index=False, encoding='utf-8-sig')
    pd.concat(test_original_frames, ignore_index=True).to_csv(explanation_dir / 'test_original_features.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(additivity_rows).to_csv(explanation_dir / 'shap_additivity_check.csv', index=False, encoding='utf-8-sig')
    shap_eci_matrix = np.vstack(shap_eci_all)
    shap_cpi_matrix = np.vstack(shap_cpi_all)
    display_matrix = np.vstack(display_all)
    importance = pd.concat([pd.DataFrame({'Target': 'ECI', 'Feature': explained_feature_names, 'Mean_Abs_SHAP': np.mean(np.abs(shap_eci_matrix), axis=0), 'Mean_SHAP': np.mean(shap_eci_matrix, axis=0)}), pd.DataFrame({'Target': 'CPI', 'Feature': explained_feature_names, 'Mean_Abs_SHAP': np.mean(np.abs(shap_cpi_matrix), axis=0), 'Mean_SHAP': np.mean(shap_cpi_matrix, axis=0)})], ignore_index=True)
    importance['Rank'] = importance.groupby('Target')['Mean_Abs_SHAP'].rank(method='min', ascending=False).astype(int)
    importance.to_csv(explanation_dir / 'shap_importance.csv', index=False, encoding='utf-8-sig')
    plot_overall_shap_beeswarm(shap_eci_matrix, shap_cpi_matrix, display_matrix, explained_feature_names, explanation_dir / 'overall_shap_beeswarm', cfg.shap_beeswarm_top_n, cfg.shap_random_seed)
    np.savez_compressed(explanation_dir / 'shap_values_test_matrices.npz', codes=np.asarray(code_all, dtype=str), years=np.asarray(year_all, dtype=np.int64), feature_names=np.asarray(explained_feature_names, dtype=str), feature_values=display_matrix.astype(np.float32), shap_log_eci=shap_eci_matrix.astype(np.float32), shap_log_cpi=shap_cpi_matrix.astype(np.float32), base_log_eci=np.concatenate(base_eci_all).astype(np.float32), base_log_cpi=np.concatenate(base_cpi_all).astype(np.float32), pred_log_eci=np.concatenate(pred_eci_all).astype(np.float32), pred_log_cpi=np.concatenate(pred_cpi_all).astype(np.float32))
    combined = importance.groupby('Feature', as_index=False)['Mean_Abs_SHAP'].sum().sort_values('Mean_Abs_SHAP', ascending=False)
    if cfg.ale_2d_top_features <= 0:
        selected_2d = list(exogenous_features)
    else:
        selected_2d = combined.head(cfg.ale_2d_top_features)['Feature'].tolist()
    pd.DataFrame({'Feature': selected_2d, 'Selection_Order': np.arange(1, len(selected_2d) + 1)}).to_csv(explanation_dir / 'ale_2d_selected_features.csv', index=False, encoding='utf-8-sig')
    selected_pairs = list(combinations(selected_2d, 2))
    pd.DataFrame([{'Feature_1': feature_a, 'Feature_2': feature_b, 'Pair_Order': pair_order} for pair_order, (feature_a, feature_b) in enumerate(selected_pairs, start=1)]).to_csv(explanation_dir / 'shap_2d_selected_pairs.csv', index=False, encoding='utf-8-sig')
    shap_2d_frames = []
    for year, batch in sorted(batches.items()):
        if int(batch.level_test_mask.sum()) == 0 or not selected_pairs:
            continue
        display_full = input_matrix_on_display_scale(batch.x_current.numpy(), feature_names, feature_scaler)
        pair_outputs = graph_aware_shap_interaction_for_batch(model=model, batch_cpu=batch, background_pool=background_pool, explain_feature_indices=explained_feature_indices, feature_pairs=selected_pairs, explained_feature_names=explained_feature_names, edge_index=edge_index, edge_attr=edge_attr, target_scaler=target_scaler, cfg=cfg, device=device)
        for (feature_a, feature_b), pair_result in pair_outputs.items():
            idx = pair_result['selected_indices']
            ia = list(feature_names).index(feature_a)
            ib = list(feature_names).index(feature_b)
            shap_2d_frames.append(pd.DataFrame({'Code': [batch.codes[int(i)] for i in idx], 'Year': int(year), 'Feature_1': feature_a, 'Feature_2': feature_b, 'Feature_Value_1': display_full[idx, ia], 'Feature_Value_2': display_full[idx, ib], 'SHAP_Interaction_Log_ECI': pair_result['shap_interaction_eci'], 'SHAP_Interaction_Log_CPI': pair_result['shap_interaction_cpi']}))
    if shap_2d_frames:
        shap_2d_long = pd.concat(shap_2d_frames, ignore_index=True)
    else:
        shap_2d_long = pd.DataFrame(columns=['Code', 'Year', 'Feature_1', 'Feature_2', 'Feature_Value_1', 'Feature_Value_2', 'SHAP_Interaction_Log_ECI', 'SHAP_Interaction_Log_CPI'])
    shap_2d_long.to_csv(explanation_dir / 'shap_2d_interaction_values_long.csv', index=False, encoding='utf-8-sig')
    if not shap_2d_long.empty:
        shap_2d_importance = shap_2d_long.assign(Abs_ECI=lambda d: d['SHAP_Interaction_Log_ECI'].abs(), Abs_CPI=lambda d: d['SHAP_Interaction_Log_CPI'].abs()).groupby(['Feature_1', 'Feature_2'], as_index=False).agg(Mean_Abs_SHAP_Interaction_ECI=('Abs_ECI', 'mean'), Mean_Abs_SHAP_Interaction_CPI=('Abs_CPI', 'mean'), Mean_SHAP_Interaction_ECI=('SHAP_Interaction_Log_ECI', 'mean'), Mean_SHAP_Interaction_CPI=('SHAP_Interaction_Log_CPI', 'mean'))
        shap_2d_importance['Combined_Mean_Abs_Interaction'] = shap_2d_importance['Mean_Abs_SHAP_Interaction_ECI'] + shap_2d_importance['Mean_Abs_SHAP_Interaction_CPI']
        shap_2d_importance = shap_2d_importance.sort_values('Combined_Mean_Abs_Interaction', ascending=False)
    else:
        shap_2d_importance = pd.DataFrame()
    shap_2d_importance.to_csv(explanation_dir / 'shap_2d_interaction_importance.csv', index=False, encoding='utf-8-sig')
    ale_1d_frames = []
    ale_2d_frames = []
    for year, batch in sorted(batches.items()):
        if int(batch.level_test_mask.sum()) == 0:
            continue
        display_full = input_matrix_on_display_scale(batch.x_current.numpy(), feature_names, feature_scaler)
        ale_1d_frames.append(compute_ale_1d_for_batch(model, batch, exogenous_features, feature_names, display_full, feature_scaler, edge_index, edge_attr, target_scaler, cfg, device))
        ale_2d_frames.append(compute_ale_2d_for_batch(model, batch, selected_2d, feature_names, display_full, feature_scaler, edge_index, edge_attr, target_scaler, cfg, device))
    pd.concat(ale_1d_frames, ignore_index=True).to_csv(explanation_dir / 'ale_1d.csv', index=False, encoding='utf-8-sig')
    if ale_2d_frames and any((not frame.empty for frame in ale_2d_frames)):
        pd.concat(ale_2d_frames, ignore_index=True).to_csv(explanation_dir / 'ale_2d.csv', index=False, encoding='utf-8-sig')
    else:
        pd.DataFrame().to_csv(explanation_dir / 'ale_2d.csv', index=False, encoding='utf-8-sig')
    combined_rank = {feature: rank for rank, feature in enumerate(combined['Feature'].tolist(), start=1)}
    selected_pair_set = {tuple(sorted((feature_a, feature_b))) for feature_a, feature_b in selected_pairs}
    pair_catalog = pd.DataFrame([{'Feature_1': feature_a, 'Feature_2': feature_b, 'Combined_SHAP_Rank_1': combined_rank[feature_a], 'Combined_SHAP_Rank_2': combined_rank[feature_b], 'Initial_2D_SHAP_Calculated': tuple(sorted((feature_a, feature_b))) in selected_pair_set, 'Initial_2D_ALE_Calculated': tuple(sorted((feature_a, feature_b))) in selected_pair_set} for feature_a, feature_b in combinations(explained_feature_names, 2)])
    pair_catalog.to_csv(explanation_dir / 'all_exogenous_pair_catalog.csv', index=False, encoding='utf-8-sig')
    save_json(explanation_dir / 'explanation_metadata.json', {'test_samples': test_count, 'background_samples': background_size, 'background_multiplier': cfg.shap_background_multiplier, 'background_candidates': background_pool['n_candidates'], 'background_sampling_with_replacement': background_pool['sampling_with_replacement'], 'shap_method': 'graph-aware grouped Monte-Carlo permutation Shapley; each feature group contains the full-graph current column and all GRU time steps', 'explained_inputs_only': 'original exogenous variables', 'county_embedding_explained': False, 'county_embedding_enabled': bool(cfg.use_county_embedding), 'county_embedding_dim': int(cfg.county_embedding_dim) if cfg.use_county_embedding else 0, 'county_embedding_interpretation': 'County identity embedding is disabled; SHAP/ALE explain original exogenous variables only.', 'explained_features': explained_feature_names, 'non_explained_model_inputs': [feature for feature in feature_names if feature not in explained_feature_names], 'shap_permutations': cfg.shap_permutations, 'shap_2d_method': 'graph-aware Monte-Carlo Shapley interaction; off-diagonal interaction value equals 0.5 times the second-order difference', 'shap_2d_permutations': cfg.shap_2d_permutations, 'shap_2d_pairs': [list(pair) for pair in selected_pairs], 'shap_output_space': 'ln(ECI) and ln(CPI)', 'ale_1d_features': list(exogenous_features), 'ale_1d_bins': cfg.ale_1d_bins, 'ale_2d_features': selected_2d, 'ale_2d_bins': cfg.ale_2d_bins, 'ale_interpretation': 'test-set graph-aware local perturbation; values are model responses, not causal effects'})

def run_single_model(cfg: Config, model_name: str) -> Dict[str, object]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    pass
    raw = read_csv_flexible(cfg.data_path)
    panel, data_audit, mapping = prepare_panel(raw, cfg)
    panel, split_audit = assign_splits(panel, mapping, cfg)
    save_json(output_dir / 'data_audit.json', data_audit)
    save_json(output_dir / 'split_audit.json', split_audit)
    code_col = str(mapping['code_col'])
    panel[[code_col, '__county_split__']].drop_duplicates().sort_values(code_col).rename(columns={'__county_split__': 'Split'}).to_csv(output_dir / 'county_split_assignments.csv', index=False, encoding='utf-8-sig')
    eci_col = str(mapping['eci_col'])
    cpi_col = str(mapping['cpi_col'])
    node_codes = sorted(panel[code_col].unique().tolist())
    cfg.n_counties = int(len(node_codes))
    edge_index, edge_attr, graph_audit = load_graph(cfg.edge_path, node_codes, cfg)
    save_json(output_dir / 'graph_audit.json', graph_audit)
    exogenous = list(mapping['city_features']) + list(mapping['energy_features'])
    continuous_columns = list(exogenous)
    train_rows = panel[panel['__level_split__'] == 'train'].copy()
    feature_scaler = FeatureStandardizer(continuous_columns).fit(train_rows[continuous_columns])
    target_scaler = LevelTargetStandardizer().fit(train_rows, eci_col, cpi_col)
    save_json(output_dir / 'feature_scaler.json', feature_scaler.to_dict())
    save_json(output_dir / 'target_scaler.json', target_scaler.to_dict())
    batches, feature_names = build_year_batches(panel, mapping, feature_scaler, target_scaler, cfg)
    save_json(output_dir / 'feature_names.json', {'feature_names': feature_names, 'scenario_features': exogenous, 'county_embedding_enabled': bool(cfg.use_county_embedding), 'county_embedding_dim': int(cfg.county_embedding_dim) if cfg.use_county_embedding else 0, 'n_counties': int(cfg.n_counties), 'county_embedding_in_feature_names': False, 'county_embedding_note': 'County identity embedding is disabled.'})
    input_dim = len(feature_names)
    pass
    pass
    pass
    results = []
    seeds_root = output_dir / 'seed_models'
    for seed in cfg.seeds:
        pass
        pass
        result = train_single_seed(seed=seed, batches=batches, edge_index=edge_index, edge_attr=edge_attr, target_scaler=target_scaler, input_dim=input_dim, cfg=cfg, device=device, seed_dir=seeds_root / f'seed_{seed}')
        results.append(result)
    results = sorted(results, key=lambda x: x['best_validation_objective'])
    best = results[0]
    pass
    best_model = load_trained_model(best['checkpoint_path'], input_dim, cfg, device)
    best_checkpoint = torch.load(best['checkpoint_path'], map_location='cpu')
    torch.save(best_checkpoint, output_dir / 'best_model.pt')
    best_metrics = {}
    for split in ['train', 'validation', 'test']:
        frame = prediction_frame_for_split(best_model, batches, edge_index, edge_attr, target_scaler, split, device, seed=int(best['seed']), epsilon=cfg.level_epsilon)
        frame.to_csv(output_dir / f'predictions_best_seed_{split}.csv', index=False, encoding='utf-8-sig')
        best_metrics[split] = evaluate_prediction_frame(frame)
    mc_metrics = {}
    pass
    for split in ['train', 'validation', 'test']:
        mc_frame = mc_dropout_uncertainty_for_split(best_model, batches, edge_index, edge_attr, target_scaler, split, cfg, device)
        mc_frame.to_csv(output_dir / f'mc_dropout_uncertainty_{split}.csv', index=False, encoding='utf-8-sig')
        mc_metrics[split] = evaluate_mc_dropout_uncertainty(mc_frame)
    save_json(output_dir / 'uncertainty_analysis.json', {'method': 'MC Dropout on the validation-selected best regression model', 'best_seed': int(best['seed']), 'mc_samples': int(cfg.mc_samples), 'dropout_rate': float(cfg.dropout), 'interval_quantiles': [cfg.uncertainty_quantile_low, cfg.uncertainty_quantile_high], 'interpretation': 'Approximate model/prediction uncertainty obtained from stochastic forward passes with Dropout enabled at inference.', 'metrics': mc_metrics})
    alpha = best_model.cross_stitch.alpha.detach().cpu().numpy().tolist()
    save_json(output_dir / 'cross_stitch_weights.json', {'best_seed': int(best['seed']), 'alpha': alpha, 'definition': [['alpha_ECI_ECI', 'alpha_ECI_CPI'], ['alpha_CPI_ECI', 'alpha_CPI_CPI']], 'warning': 'Cross-Stitch reflects representation sharing between tasks and is not equivalent to variable importance.'})
    seed_rows = []
    histories = []
    for result in results:
        histories.append(result['history'])
        lm = result['level_metrics']
        seed_rows.append({'Seed': int(result['seed']), 'Best_Epoch': int(result['best_epoch']), 'Validation_Objective': float(result['best_validation_objective']), 'Validation_ECI_R2': lm['validation']['ECI']['R2'], 'Validation_CPI_R2': lm['validation']['CPI']['R2'], 'Test_ECI_R2': lm['test']['ECI']['R2'], 'Test_CPI_R2': lm['test']['CPI']['R2']})
    pd.DataFrame(seed_rows).to_csv(output_dir / 'metrics_by_seed.csv', index=False, encoding='utf-8-sig')
    history_df = pd.concat(histories, ignore_index=True)
    history_df.to_csv(output_dir / 'training_history.csv', index=False, encoding='utf-8-sig')
    save_json(output_dir / 'regression_metrics.json', {'best_seed': int(best['seed']), 'best_validation_objective': float(best['best_validation_objective']), 'best_seed_metrics': best_metrics, 'mc_dropout_metrics': mc_metrics})
    pass
    run_shap_and_ale(model=best_model, batches=batches, feature_names=feature_names, exogenous_features=exogenous, feature_scaler=feature_scaler, target_scaler=target_scaler, edge_index=edge_index, edge_attr=edge_attr, cfg=cfg, output_dir=output_dir, device=device)
    pass
    scenario_summary, sensitivity = run_scenario_analysis(panel=panel, mapping=mapping, feature_scaler=feature_scaler, target_scaler=target_scaler, cfg=cfg, edge_index=edge_index, edge_attr=edge_attr, model=best_model, baseline_batches=batches, output_dir=output_dir, device=device)
    model_config = asdict(cfg)
    model_config.update({'model_name': model_name, 'selected_exogenous_features': exogenous, 'input_dim': int(input_dim), 'node_order': node_codes, 'direct_model_targets': ['ln(ECI)', 'ln(CPI)'], 'architecture': 'dual independent ST-GAT-GRU encoders without county identity embeddings -> Cross-Stitch soft sharing -> ln(ECI)/ln(CPI) dual regression', 'county_embedding_enabled': bool(cfg.use_county_embedding), 'county_embedding_dim': int(cfg.county_embedding_dim) if cfg.use_county_embedding else 0, 'n_counties': int(cfg.n_counties), 'county_embedding_role': 'disabled', 'split_method': 'random group split by county', 'split_ratios': {'train': float(cfg.train_ratio), 'validation': float(cfg.validation_ratio), 'test': float(cfg.test_ratio)}, 'split_random_seed': int(cfg.split_random_seed), 'loss': 'MSE(ln(ECI)_z) + MSE(ln(CPI)_z)', 'classification_head': None, 'historical_auxiliary_inputs_included': False, 'historical_auxiliary_inputs_removed': ['Lag_Log_ECI', 'Lag_Log_CPI', 'Lag_Carbon_Improvement', 'Lag_Pollution_Improvement', 'Lag_State_0', 'Lag_State_1', 'Lag_State_2', 'Lag_State_3', 'Lag_State_Available_Mask', 'Lag_Level_Available_Mask', 'Lag_History_Complete_Mask'], 'mc_dropout_inference': True, 'uncertainty_method': 'MC Dropout with 1000 stochastic forward passes on the best model', 'sensitivity_method': 'One-at-a-time relative perturbation scenarios', 'scenario_features': exogenous, 'shap_method': 'graph-aware grouped Monte-Carlo permutation Shapley', 'shap_explained_inputs': 'all model inputs; the model contains original exogenous variables only', 'shap_2d': 'pairwise graph-aware Monte-Carlo SHAP interaction among the highest combined-SHAP exogenous features', 'ale_1d': 'all selected exogenous features on the full test graph', 'ale_2d': 'pairwise ALE among the highest combined-SHAP exogenous features'})
    save_json(output_dir / 'model_config.json', model_config)
    pass
    outputs = ['best_model.pt', 'regression_metrics.json', 'uncertainty_analysis.json', 'mc_dropout_uncertainty_train.csv', 'mc_dropout_uncertainty_validation.csv', 'mc_dropout_uncertainty_test.csv', 'predictions_best_seed_train.csv', 'predictions_best_seed_validation.csv', 'predictions_best_seed_test.csv', 'metrics_by_seed.csv', 'training_history.csv', 'county_split_assignments.csv', 'split_audit.json', 'cross_stitch_weights.json', 'model_config.json', 'scenario_analysis/scenario_summary.csv', 'scenario_analysis/scenario_sample_level_predictions.csv', 'scenario_analysis/sensitivity_summary.csv', 'scenario_analysis/scenario_input_audit.csv', 'shap_ale/shap_values_test_long.csv', 'shap_ale/shap_importance.csv', 'shap_ale/shap_additivity_check.csv', 'shap_ale/shap_background_index.csv', 'shap_ale/shap_background_graph_pool.npz', 'shap_ale/shap_background_original_features.csv', 'shap_ale/test_original_features.csv', 'shap_ale/test_graph_2022.npz', 'shap_ale/shap_values_test_matrices.npz', 'shap_ale/shap_2d_interaction_values_long.csv', 'shap_ale/shap_2d_interaction_importance.csv', 'shap_ale/shap_2d_selected_pairs.csv', 'shap_ale/all_exogenous_pair_catalog.csv', 'shap_ale/ale_1d.csv', 'shap_ale/ale_2d.csv', 'shap_ale/ale_2d_selected_features.csv', 'shap_ale/overall_shap_beeswarm.png', 'shap_ale/overall_shap_beeswarm.pdf']
    for name in outputs:
        pass
    return {'Model': model_name, 'N_Exogenous_Features': int(len(exogenous)), 'N_Total_Model_Inputs': int(input_dim), 'Best_Seed': int(best['seed']), 'Best_Epoch': int(best['best_epoch']), 'Validation_Objective': float(best['best_validation_objective']), 'Validation_ECI_R2': float(best_metrics['validation']['ECI']['R2']), 'Validation_CPI_R2': float(best_metrics['validation']['CPI']['R2']), 'Test_ECI_R2': float(best_metrics['test']['ECI']['R2']), 'Test_CPI_R2': float(best_metrics['test']['CPI']['R2']), 'Output_Directory': str(output_dir)}

def run(cfg: Config) -> None:
    root_output = Path(cfg.output_dir)
    root_output.mkdir(parents=True, exist_ok=True)
    unknown = [name for name in cfg.models_to_run if name not in MODEL_FEATURE_SETS]
    if unknown:
        raise ValueError(f'Unknown model name: {unknown}. Allowed values={list(MODEL_FEATURE_SETS)}')
    save_json(root_output / 'model_feature_definitions.json', {name: features for name, features in MODEL_FEATURE_SETS.items()})
    comparison_rows = []
    for model_name in cfg.models_to_run:
        model_cfg = copy.deepcopy(cfg)
        model_cfg.city_features = list(MODEL_FEATURE_SETS[model_name])
        model_cfg.energy_features = []
        model_cfg.use_coal_share = False
        model_cfg.output_dir = str(root_output / model_name)
        pass
        pass
        pass
        comparison_rows.append(run_single_model(model_cfg, model_name))
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(root_output / 'model_comparison.csv', index=False, encoding='utf-8-sig')
    pass

def _load_saved_feature_scaler(path: Path) -> FeatureStandardizer:
    with path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    scaler = FeatureStandardizer(payload['columns'])
    scaler.median_ = {str(k): float(v) for k, v in payload['median'].items()}
    scaler.mean_ = {str(k): float(v) for k, v in payload['mean'].items()}
    scaler.std_ = {str(k): float(v) for k, v in payload['std'].items()}
    return scaler

def _load_saved_target_scaler(path: Path) -> LevelTargetStandardizer:
    with path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    scaler = LevelTargetStandardizer()
    scaler.eci_mean = float(payload['log_ECI']['mean'])
    scaler.eci_std = float(payload['log_ECI']['std'])
    scaler.cpi_mean = float(payload['log_CPI']['mean'])
    scaler.cpi_std = float(payload['log_CPI']['std'])
    return scaler

def _saved_npz_to_year_batch(npz_path: Path) -> YearBatch:
    with np.load(npz_path, allow_pickle=False) as saved:
        x_current = torch.tensor(saved['x_current'], dtype=torch.float32)
        x_sequence = torch.tensor(saved['x_sequence'], dtype=torch.float32)
        lengths = torch.tensor(saved['lengths'], dtype=torch.long)
        test_mask = torch.tensor(saved['test_mask'], dtype=torch.bool)
        codes = saved['codes'].astype(str).tolist()
        year = int(saved['year'][0])
        y_eci = torch.tensor(saved['y_eci'], dtype=torch.float32)
        y_cpi = torch.tensor(saved['y_cpi'], dtype=torch.float32)
    n = int(x_current.shape[0])
    zeros = torch.zeros(n, dtype=torch.float32)
    false_mask = torch.zeros(n, dtype=torch.bool)
    return YearBatch(year=year, codes=codes, x_current=x_current, x_sequence=x_sequence, lengths=lengths, y_eci=y_eci, y_cpi=y_cpi, y_eci_z=zeros.clone(), y_cpi_z=zeros.clone(), true_eci_previous=zeros.clone(), true_cpi_previous=zeros.clone(), true_carbon_improvement=zeros.clone(), true_pollution_improvement=zeros.clone(), true_state=torch.zeros(n, dtype=torch.long), level_train_mask=false_mask.clone(), level_validation_mask=false_mask.clone(), level_test_mask=test_mask, state_train_mask=false_mask.clone(), state_validation_mask=false_mask.clone(), state_test_mask=false_mask.clone(), level_target_mask=test_mask.clone(), state_target_mask=false_mask.clone())

def run_arbitrary_pair_from_saved_model(cfg: Config, model_name: str, feature_a: str, feature_b: str) -> None:
    if model_name not in MODEL_FEATURE_SETS:
        raise ValueError(f'Unknown model: {model_name}; allowed values={list(MODEL_FEATURE_SETS)}')
    if feature_a == feature_b:
        raise ValueError('Two different variables are required for 2D interpretation.')
    model_dir = Path(cfg.output_dir) / model_name
    explanation_dir = model_dir / 'shap_ale'
    required = [model_dir / 'best_model.pt', model_dir / 'model_config.json', model_dir / 'feature_scaler.json', model_dir / 'target_scaler.json', explanation_dir / 'shap_background_graph_pool.npz']
    missing = [str(path) for path in required if not path.exists()]
    graph_paths = sorted(explanation_dir.glob('test_graph_*.npz'))
    if not graph_paths:
        missing.append(str(explanation_dir / 'test_graph_*.npz'))
    if missing:
        raise FileNotFoundError('Files required for saved-model pair analysis are incomplete:\n' + '\n'.join(missing))
    with (model_dir / 'model_config.json').open('r', encoding='utf-8') as handle:
        stored_config = json.load(handle)
    analysis_cfg = copy.deepcopy(cfg)
    for key in Config.__dataclass_fields__:
        if key in stored_config:
            setattr(analysis_cfg, key, stored_config[key])
    analysis_cfg.shap_2d_permutations = cfg.shap_2d_permutations
    analysis_cfg.ale_2d_bins = cfg.ale_2d_bins
    with np.load(explanation_dir / 'shap_background_graph_pool.npz', allow_pickle=False) as saved_background:
        background_pool = {'x_current': saved_background['x_current'].copy(), 'x_sequence': saved_background['x_sequence'].copy(), 'lengths': saved_background['lengths'].copy(), 'codes': saved_background['codes'].astype(str).tolist(), 'years': saved_background['years'].astype(int).tolist()}
        feature_names = saved_background['feature_names'].astype(str).tolist()
        exogenous_features = saved_background['exogenous_features'].astype(str).tolist()
        exogenous_indices = saved_background['exogenous_feature_indices'].astype(int).tolist()
    invalid = [feature for feature in (feature_a, feature_b) if feature not in exogenous_features]
    if invalid:
        raise ValueError(f'Variable does not belong to {model_name} original exogenous inputs: {invalid}; allowed values={exogenous_features}')
    feature_scaler = _load_saved_feature_scaler(model_dir / 'feature_scaler.json')
    target_scaler = _load_saved_target_scaler(model_dir / 'target_scaler.json')
    device = get_device()
    model = load_trained_model(model_dir / 'best_model.pt', len(feature_names), analysis_cfg, device)
    custom_dir = explanation_dir / 'custom_pairs'
    custom_dir.mkdir(parents=True, exist_ok=True)
    safe_a = re.sub('[^A-Za-z0-9._-]+', '_', feature_a).strip('_')
    safe_b = re.sub('[^A-Za-z0-9._-]+', '_', feature_b).strip('_')
    pair_stem = f'{safe_a}__{safe_b}'
    shap_frames = []
    ale_frames = []
    for graph_path in graph_paths:
        batch = _saved_npz_to_year_batch(graph_path)
        with np.load(graph_path, allow_pickle=False) as saved_graph:
            edge_index = torch.tensor(saved_graph['edge_index'], dtype=torch.long)
            edge_attr = torch.tensor(saved_graph['edge_attr'], dtype=torch.float32)
        display_full = input_matrix_on_display_scale(batch.x_current.numpy(), feature_names, feature_scaler)
        interaction = graph_aware_shap_interaction_for_batch(model=model, batch_cpu=batch, background_pool=background_pool, explain_feature_indices=exogenous_indices, feature_pairs=[(feature_a, feature_b)], explained_feature_names=exogenous_features, edge_index=edge_index, edge_attr=edge_attr, target_scaler=target_scaler, cfg=analysis_cfg, device=device)[feature_a, feature_b]
        idx = interaction['selected_indices']
        ia = feature_names.index(feature_a)
        ib = feature_names.index(feature_b)
        shap_frames.append(pd.DataFrame({'Code': [batch.codes[int(i)] for i in idx], 'Year': int(batch.year), 'Feature_1': feature_a, 'Feature_2': feature_b, 'Feature_Value_1': display_full[idx, ia], 'Feature_Value_2': display_full[idx, ib], 'SHAP_Interaction_Log_ECI': interaction['shap_interaction_eci'], 'SHAP_Interaction_Log_CPI': interaction['shap_interaction_cpi']}))
        ale_frames.append(compute_ale_2d_for_batch(model=model, batch_cpu=batch, selected_features=[feature_a, feature_b], feature_names=feature_names, display_values=display_full, feature_scaler=feature_scaler, edge_index=edge_index, edge_attr=edge_attr, target_scaler=target_scaler, cfg=analysis_cfg, device=device))
    shap_output = custom_dir / f'{pair_stem}_shap2d.csv'
    ale_output = custom_dir / f'{pair_stem}_ale2d.csv'
    pd.concat(shap_frames, ignore_index=True).to_csv(shap_output, index=False, encoding='utf-8-sig')
    valid_ale = [frame for frame in ale_frames if not frame.empty]
    (pd.concat(valid_ale, ignore_index=True) if valid_ale else pd.DataFrame()).to_csv(ale_output, index=False, encoding='utf-8-sig')
    save_json(custom_dir / f'{pair_stem}_metadata.json', {'model': model_name, 'feature_1': feature_a, 'feature_2': feature_b, 'checkpoint': str(model_dir / 'best_model.pt'), 'retraining_performed': False, 'explained_inputs_only': 'original exogenous variables', 'shap_2d_permutations': analysis_cfg.shap_2d_permutations, 'ale_2d_bins': analysis_cfg.ale_2d_bins, 'shap_output': str(shap_output), 'ale_output': str(ale_output)})
    catalog_path = explanation_dir / 'all_exogenous_pair_catalog.csv'
    if catalog_path.exists():
        catalog = pd.read_csv(catalog_path)
        same_pair = (catalog['Feature_1'] == feature_a) & (catalog['Feature_2'] == feature_b) | (catalog['Feature_1'] == feature_b) & (catalog['Feature_2'] == feature_a)
        for column in ['Custom_2D_SHAP_Calculated', 'Custom_2D_ALE_Calculated']:
            if column not in catalog:
                catalog[column] = False
            catalog.loc[same_pair, column] = True
        catalog.to_csv(catalog_path, index=False, encoding='utf-8-sig')
    pass

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Progressive Model1-Model5 ln(ECI)/ln(CPI) regression with MC Dropout, OAT scenarios, and graph-aware SHAP/ALE')
    parser.add_argument('--data', default=None, help='Path to model_data_total.csv')
    parser.add_argument('--edges', default=None, help='Path to the spatial weight edge-list CSV')
    parser.add_argument('--output', default=None, help='Output directory')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--models', nargs='+', choices=list(MODEL_FEATURE_SETS.keys()), default=None, help='Models to run.')
    parser.add_argument('--lookback', type=int, default=None)
    parser.add_argument('--train-ratio', type=float, default=None, help='Training county fraction.')
    parser.add_argument('--validation-ratio', type=float, default=None, help='Validation county fraction.')
    parser.add_argument('--test-ratio', type=float, default=None, help='Test county fraction.')
    parser.add_argument('--split-seed', type=int, default=None, help='Random seed for county-level splitting.')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate used for training regularization and MC Dropout uncertainty analysis.')
    parser.add_argument('--seeds', nargs='+', type=int, default=None, help='Training random seeds; the best model is selected by validation performance and uncertainty uses MC Dropout from that model.')
    parser.add_argument('--mc-samples', type=int, default=None, help='Number of MC Dropout stochastic forward passes.')
    parser.add_argument('--shap-permutations', type=int, default=None, help='Number of graph-aware Monte Carlo Shapley feature permutations.')
    parser.add_argument('--shap-2d-permutations', type=int, default=None, help='Number of graph-aware 2D SHAP coalition samples per feature pair.')
    parser.add_argument('--ale-1d-bins', type=int, default=None, help='Number of quantile bins for 1D ALE.')
    parser.add_argument('--ale-2d-bins', type=int, default=None, help='Number of quantile bins per dimension for 2D ALE.')
    parser.add_argument('--ale-2d-top-features', type=int, default=None, help='Number of top combined-SHAP features used for 2D SHAP and 2D ALE; zero uses all exogenous feature pairs.')
    parser.add_argument('--scenario-years', nargs='+', type=int, default=None, help='Years used for scenario analysis.')
    parser.add_argument('--no-scenario-sample-output', action='store_true', help='Do not save city-level scenario results; save summaries only.')
    parser.add_argument('--pair-only', nargs=3, metavar=('MODEL', 'FEATURE_1', 'FEATURE_2'), default=None, help='Without retraining, use a saved model to compute 2D SHAP and 2D ALE for any two exogenous variables.')
    return parser.parse_args()

def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    if args.data is not None:
        cfg.data_path = args.data
    if args.edges is not None:
        cfg.edge_path = args.edges
    if args.output is not None:
        cfg.output_dir = args.output
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.models is not None:
        cfg.models_to_run = list(args.models)
    if args.lookback is not None:
        cfg.lookback = args.lookback
    if args.train_ratio is not None:
        cfg.train_ratio = float(args.train_ratio)
    if args.validation_ratio is not None:
        cfg.validation_ratio = float(args.validation_ratio)
    if args.test_ratio is not None:
        cfg.test_ratio = float(args.test_ratio)
    if args.split_seed is not None:
        cfg.split_random_seed = int(args.split_seed)
    ratios = [cfg.train_ratio, cfg.validation_ratio, cfg.test_ratio]
    if any((ratio <= 0 for ratio in ratios)) or not np.isclose(sum(ratios), 1.0):
        raise ValueError('Training, validation, and test county fractions must be positive and sum to one.')
    if args.dropout is not None:
        cfg.dropout = args.dropout
    if args.seeds is not None:
        if len(args.seeds) < 1:
            raise ValueError('At least one training seed is required.')
        cfg.seeds = list(args.seeds)
    if args.mc_samples is not None:
        if args.mc_samples < 2:
            raise ValueError('mc_samples must be at least two.')
        cfg.mc_samples = int(args.mc_samples)
    if args.shap_permutations is not None:
        if args.shap_permutations < 1:
            raise ValueError('shap_permutations must be at least one.')
        cfg.shap_permutations = int(args.shap_permutations)
    if args.shap_2d_permutations is not None:
        if args.shap_2d_permutations < 1:
            raise ValueError('shap_2d_permutations must be at least one.')
        cfg.shap_2d_permutations = int(args.shap_2d_permutations)
    if args.ale_1d_bins is not None:
        if args.ale_1d_bins < 2:
            raise ValueError('ale_1d_bins must be at least two.')
        cfg.ale_1d_bins = int(args.ale_1d_bins)
    if args.ale_2d_bins is not None:
        if args.ale_2d_bins < 2:
            raise ValueError('ale_2d_bins must be at least two.')
        cfg.ale_2d_bins = int(args.ale_2d_bins)
    if args.ale_2d_top_features is not None:
        if args.ale_2d_top_features < 0:
            raise ValueError('ale_2d_top_features cannot be negative.')
        cfg.ale_2d_top_features = int(args.ale_2d_top_features)
    if args.scenario_years is not None:
        cfg.scenario_years = list(args.scenario_years)
    if args.no_scenario_sample_output:
        cfg.scenario_save_sample_level = False
    return cfg
if __name__ == '__main__':
    args = parse_args()
    cfg = config_from_args(args)
    if args.pair_only is not None:
        run_arbitrary_pair_from_saved_model(cfg, model_name=args.pair_only[0], feature_a=args.pair_only[1], feature_b=args.pair_only[2])
    else:
        run(cfg)
