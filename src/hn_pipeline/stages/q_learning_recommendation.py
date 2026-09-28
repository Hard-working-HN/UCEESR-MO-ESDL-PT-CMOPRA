from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import sys
import time
import traceback
from dataclasses import dataclass, asdict, field, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROOT32 = PROJECT_ROOT / 'data' / 'processed'
ROOT34 = PROJECT_ROOT / 'data' / 'processed'
ROOT35 = PROJECT_ROOT / 'outputs'
FEATURES = ['PGDP', 'POPD', 'VAF_GDP', 'GIA_GDP', 'TSS_GDP', 'Mean_Light', 'Builtup', 'Road_Density', 'PIU_PIR_Ratio', 'Fiscal_Revenue_GDP', 'Energy_Intensity', 'Coal_Share', 'Electricity_Share', 'Fossil_Energy_Share', 'Energy_Diversity_Index_Norm', 'Industrial_Coal_Share', 'TPSP_Petroleum_Share', 'Forest', 'Grassland', 'Shrub', 'Water', 'Wetland', 'Annual NDVI', 'Summer NDVI', 'AMEVT', 'LAI_HV', 'LAI_LV', 'BG_AI', 'BG_Divsion', 'EEI_GDP', 'EES_GEF', 'GPAI', 'GIPAI', 'GIPGS']
OUTCOMES = ['ECI', 'CPI', 'O3', 'PM10']
POLICIES = ['CETP', 'LCCP', 'ECERFP', 'ZWC', 'CEP', 'CEPDU', 'CACP', 'SCP', 'LECCDC', 'NGFRIPZ']
STRUCTURAL = ['PGDP', 'POPD', 'VAF_GDP', 'GIA_GDP', 'Builtup', 'Forest']
BLOCKS = {'energy': ['ECI', 'Energy_Intensity', 'Coal_Share', 'Electricity_Share', 'Fossil_Energy_Share', 'Energy_Diversity_Index_Norm', 'Industrial_Coal_Share', 'TPSP_Petroleum_Share', 'GIA_GDP'], 'pollution': ['CPI', 'O3', 'PM10'], 'capacity': ['PGDP', 'Builtup', 'GIPAI', 'EEI_GDP', 'EES_GEF', 'Forest', 'Grassland', 'Water', 'Wetland', 'Annual NDVI'], 'potential': ['Preliminary_ECI_pct', 'Preliminary_CPI_pct']}
FAMILIES = {'efficiency': ['Energy_Intensity'], 'energy_structure': ['Coal_Share', 'Electricity_Share', 'Fossil_Energy_Share', 'Energy_Diversity_Index_Norm', 'Industrial_Coal_Share', 'TPSP_Petroleum_Share'], 'technology': ['GPAI', 'GIPAI', 'GIPGS'], 'governance': ['EEI_GDP', 'EES_GEF'], 'ecological_conditions': ['Annual NDVI', 'Summer NDVI', 'AMEVT', 'LAI_HV', 'LAI_LV']}
FAMILIES['joint_energy'] = FAMILIES['efficiency'] + FAMILIES['energy_structure']
FAMILIES['joint_energy_technology_governance'] = FAMILIES['joint_energy'] + FAMILIES['technology'] + FAMILIES['governance']
FAMILIES['joint_all'] = FAMILIES['joint_energy_technology_governance'] + FAMILIES['ecological_conditions']
ADJUSTABLE = list(dict.fromkeys((x for v in FAMILIES.values() for x in v)))

@dataclass
class Settings:
    panel: str = str(PROJECT_ROOT / 'data' / 'processed' / 'model_data_total.csv')
    policy: str = str(PROJECT_ROOT / 'data' / 'processed' / 'policy_conditions.csv')
    edge: str = str(PROJECT_ROOT / 'data' / 'processed' / 'module2_time_space_results' / '10a_spatial_weights_edge_list.csv')
    dual_model: str = str(PROJECT_ROOT / 'models' / 'formal_multitask' / 'Model5')
    o3_model: str = str(PROJECT_ROOT / 'models' / 'o3' / 'Model5')
    pm10_model: str = str(PROJECT_ROOT / 'models' / 'pm10' / 'Model5')
    shape: str = str(PROJECT_ROOT / 'data' / 'spatial' / 'county_2023.shp')
    shape_code: str = 'code'
    mechanism_dir: str = str(ROOT35 / 'Policy_Mechanism_34_Results/run_20260918_171105_422124')
    boundary_dir: str = str(ROOT35 / 'Policy_Boundary_Results/run_20260921_095238_103577')
    output: str = str(PROJECT_ROOT / 'outputs' / 'community_assessment')
    year: int = 2022
    history_start: int = 2013
    seed: int = 20260921
    algorithm: str = 'auto'
    resolution: float = 1.0
    min_community_size: int = 20
    peer_count: int = 50
    benchmark_quantile: float = 0.25
    block_weights: dict = field(default_factory=lambda: dict(energy=0.4, pollution=0.25, capacity=0.15, potential=0.2))
    support_low: float = 0.05
    support_high: float = 0.95
    annual_change_quantile: float = 0.9
    adjustment_years: int = 1
    fractions: tuple = (0.25, 0.5)
    mc_fraction: float = 0.5
    mc_all_doses: bool = True
    reference_joint_quantile: float = 0.5
    reference_warning_n: int = 5
    support_distance_quantile: float = 0.95
    min_reference_counties: int = 5
    freeze_unsupported_updates: bool = True
    observed_inequalities: tuple = (('Coal_Share', 'Fossil_Energy_Share'), ('GIPAI', 'GPAI'))
    o3_budget_mode: str = 'historical_trend'
    o3_trend_window_years: int = 0
    o3_trend_min_years: int = 3
    o3_trend_multiplier: float = 1.0
    o3_history_sensitivity_multipliers: tuple = (0.5, 1.0, 1.5)
    o3_tolerance_pct: float = 0.0
    o3_sensitivity_thresholds: tuple = (0.0, 0.5, 1.0, 2.0)
    improvement_tolerance_pct: float = 0.01
    baseline_error_flag_pct: float = 30.0
    mc_samples: int = 100
    device: str = 'auto'
    torch_threads: int = 4
    sensitivity_repeats: int = 3
    max_communities: int = 0
    export_geo: bool = False
    smoke: bool = False
    stage: str = 'full'

def log(s):
    pass

def read_csv(path, **kwargs):
    for enc in ('utf-8-sig', 'utf-8', 'gb18030'):
        try:
            kwargs.setdefault('low_memory', False)
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Unable to decode CSV: {path}')

def codes(s):
    v = s.astype('string').str.strip().str.replace('\\.0$', '', regex=True)
    if v.isna().any() or not v.str.fullmatch('\\d{1,6}').all():
        raise ValueError('County codes are missing or are not one to six digits.')
    return v.str.zfill(6).astype(str)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def save_json(path, obj):

    def clean(v):
        if isinstance(v, np.generic):
            return clean(v.item())
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, np.ndarray):
            return clean(v.tolist())
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, float) and (not math.isfinite(v)):
            return None
        return v
    Path(path).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def robust_z(frame):
    a = np.asarray(frame, float)
    med = np.nanmedian(a, axis=0)
    if not np.isfinite(med).all():
        raise ValueError('Community or matching features contain an entirely missing column.')
    a = np.where(np.isfinite(a), a, med)
    lo, hi = np.quantile(a, [0.01, 0.99], axis=0)
    a = np.clip(a, lo, hi)
    sd = np.std(a, axis=0)
    return (a - np.mean(a, axis=0)) / np.where(sd > 1e-10, sd, 1.0)

def split_sources(directory):
    p = Path(directory)
    saved = p / 'county_split_assignments.csv'
    if saved.exists():
        return [saved]
    files = [p / f'predictions_{part}.csv' for part in ['train', 'validation', 'test']]
    if not all((f.is_file() for f in files)):
        raise FileNotFoundError(f'{p}: missing county split assignments or split prediction files.')
    return files

def read_splits(directory):
    files = split_sources(directory)
    if len(files) == 1:
        result = read_csv(files[0], usecols=['Code', 'Split'])
    else:
        result = pd.concat([read_csv(f, usecols=['Code', 'Split']) for f in files]).drop_duplicates()
    result['Code'] = codes(result.Code)
    if result.Code.duplicated().any():
        raise ValueError('A county appears in more than one train, validation, or test split.')
    if not result.Split.isin(['train', 'validation', 'test']).all():
        raise ValueError('Unknown sample split label.')
    return result

def validate_settings(c):
    if c.o3_budget_mode not in ('historical_trend', 'fixed'):
        raise ValueError('o3_budget_mode must be historical_trend or fixed.')
    if c.o3_trend_min_years < 3 or c.o3_trend_window_years < 0:
        raise ValueError('Historical trend estimation requires at least three valid years and a nonnegative window.')
    if not np.isfinite(c.o3_trend_multiplier) or c.o3_trend_multiplier < 0:
        raise ValueError('The historical threshold multiplier must be finite and nonnegative.')
    if any((not np.isfinite(x) or x < 0 for x in [c.o3_tolerance_pct, *c.o3_sensitivity_thresholds, *c.o3_history_sensitivity_multipliers])):
        raise ValueError('O3 thresholds and sensitivity multipliers must be finite and nonnegative.')
    if c.adjustment_years < 1 or c.mc_samples < 2:
        raise ValueError('adjustment_years>=1、mc_samples>=2。')
    if c.mc_fraction not in c.fractions or not 0 < c.reference_joint_quantile < 1:
        raise ValueError('The MC fraction must be included in the preset fractions and the reference quantile must be in (0, 1).')
    if not 0 <= c.support_low < c.support_high <= 1:
        raise ValueError('Invalid historical support quantiles.')
    if c.min_community_size < 2 or c.peer_count < 2 or c.min_reference_counties < 2:
        raise ValueError('Community size, peer count, and support count must each be at least two.')
    if any((f <= 0 or f > 1 for f in c.fractions)):
        raise ValueError('Donor interpolation fractions must be in (0, 1].')
    if set(c.block_weights) != set(BLOCKS) or any((v < 0 for v in c.block_weights.values())) or sum(c.block_weights.values()) <= 0:
        raise ValueError('Invalid community feature block weights.')

def load_data(c, out):
    panel = read_csv(c.panel)
    if not set(['Code', 'Year'] + FEATURES + OUTCOMES).issubset(panel):
        raise ValueError('The indicator panel is missing required fields: ' + str(set(['Code', 'Year'] + FEATURES + OUTCOMES) - set(panel)))
    panel['Code'] = codes(panel.Code)
    panel['Year'] = pd.to_numeric(panel.Year, errors='raise').astype(int)
    if panel.duplicated(['Code', 'Year']).any():
        raise ValueError('The indicator panel contains duplicate Code-Year rows.')
    for f in FEATURES + OUTCOMES:
        panel[f] = pd.to_numeric(panel[f], errors='coerce').replace([np.inf, -np.inf], np.nan)
    panel = panel[panel.Year.between(c.history_start, c.year)].sort_values(['Code', 'Year']).copy()
    ids = sorted(panel.Code.unique())
    expected = pd.MultiIndex.from_product([ids, range(c.history_start, c.year + 1)], names=['Code', 'Year'])
    if not expected.equals(pd.MultiIndex.from_frame(panel[['Code', 'Year']])):
        raise ValueError('The historical panel is incomplete; missing years are not fabricated.')
    base = panel[panel.Year.eq(c.year)].set_index('Code').reindex(ids).copy()
    if base[OUTCOMES].isna().any().any() or (base[OUTCOMES] <= 0).any().any():
        raise ValueError('All four baseline outcomes must be positive and complete.')
    base['Feature_Missing_Count'] = base[FEATURES].isna().sum(axis=1)
    base.to_csv(out / '01_county_baseline.csv', encoding='utf-8-sig')
    audit = panel[FEATURES + OUTCOMES].agg(['count', 'min', 'max']).T
    audit.index.name = 'Feature'
    audit['missing_rows'] = panel[FEATURES + OUTCOMES].isna().sum()
    audit.to_csv(out / 'data_columns_audit.csv', encoding='utf-8-sig')
    edge = read_csv(c.edge)
    for col in ['Origin_Code', 'Neighbor_Code']:
        edge[col] = codes(edge[col])
    w = pd.to_numeric(edge.Row_Standardized_Weight, errors='raise')
    if (~np.isfinite(w) | (w <= 0)).any() or edge.duplicated(['Origin_Code', 'Neighbor_Code']).any():
        raise ValueError('Spatial edge weights are invalid or duplicated.')
    if not set(edge.Origin_Code).union(edge.Neighbor_Code) <= set(ids):
        raise ValueError('The spatial edge list contains codes absent from the indicator panel.')
    if edge.Origin_Code.eq(edge.Neighbor_Code).any():
        raise ValueError('The spatial edge list contains self-loops.')
    index = {s: i for i, s in enumerate(ids)}
    src = edge.Origin_Code.map(index).to_numpy(int)
    dst = edge.Neighbor_Code.map(index).to_numpy(int)
    e = pd.DataFrame({'i': np.minimum(src, dst), 'j': np.maximum(src, dst), 'spatial': w.to_numpy() / 2})
    undirected = e.groupby(['i', 'j'], as_index=False).spatial.sum()
    audit_graph = {'n_nodes': len(ids), 'directed_edges': len(edge), 'undirected_edges': len(undirected), 'row_sum_max_error': float(abs(edge.groupby('Origin_Code').Row_Standardized_Weight.sum() - 1).max()), 'spatial_relation': 'provided edge-list; not assumed polygon Queen adjacency', 'community_symmetrization': '(w_ij+w_ji)/2; absent reverse direction=0', 'prediction_graph': 'unchanged original directed graph, including original weights'}
    save_json(out / 'spatial_graph_audit.json', audit_graph)
    return (panel, base, ids, edge, undirected)

def historical_o3_budgets(panel, ids, c, out):
    start = max(c.history_start, c.year - c.o3_trend_window_years + 1) if c.o3_trend_window_years else c.history_start
    rows = []
    for code, group in panel[panel.Year.between(start, c.year)].groupby('Code'):
        group = group.sort_values('Year')
        valid = group[np.isfinite(group.O3) & (group.O3 > 0)]
        n = len(valid)
        rate = np.nan
        slope = np.nan
        r2 = np.nan
        if n >= c.o3_trend_min_years:
            t = valid.Year.to_numpy(float)
            y = np.log(valid.O3.to_numpy(float))
            t = t - t.mean()
            yc = y - y.mean()
            if np.dot(t, t) > 0:
                slope = float(np.dot(t, yc) / np.dot(t, t))
                rate = float(100 * np.expm1(slope))
                sst = float(np.dot(yc, yc))
                r2 = 1 - float(np.sum((yc - slope * t) ** 2)) / sst if sst > 1e-20 else 1.0
        historical = max(0.0, rate) if np.isfinite(rate) else np.nan
        budget = historical * c.o3_trend_multiplier if c.o3_budget_mode == 'historical_trend' else c.o3_tolerance_pct
        rows.append({'Code': code, 'Historical_Start_Year': start, 'Historical_End_Year': c.year, 'Valid_Historical_Years': n, 'Invalid_Historical_Years': len(group) - n, 'O3_Log_Trend_Per_Year': slope, 'O3_Historical_Annual_Change_pct': rate, 'O3_Trend_R2': r2, 'O3_Historical_Positive_Trend_pct': historical, 'O3_Budget_pct': budget, 'O3_Budget_Mode': c.o3_budget_mode, 'O3_Trend_Multiplier': c.o3_trend_multiplier, 'Trend_Supported': np.isfinite(rate)})
    table = pd.DataFrame(rows).set_index('Code').reindex(ids)
    table.to_csv(out / 'county_O3_historical_thresholds.csv', encoding='utf-8-sig')
    if table.O3_Budget_pct.isna().any() or not table.Trend_Supported.fillna(False).all():
        raise ValueError('Some counties lack sufficient historical O3 observations; thresholds are not fabricated. See county_O3_historical_thresholds.csv.')
    return table

def multiplier_excess_column(value):
    return f'O3_History_x{float(value):g}_Max_Excess_pct_points'

def preliminary(base, c, out):
    z = robust_z(base[STRUCTURAL])
    n = len(base)
    if n < 3:
        raise ValueError('Insufficient county count.')
    k = min(c.peer_count, n - 1)
    nbr = NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(z, return_distance=False)
    peers = np.stack([row[row != i][:k] for i, row in enumerate(nbr)])
    result = pd.DataFrame(index=base.index)
    for y in ['ECI', 'CPI']:
        bench = np.quantile(base[y].to_numpy()[peers], c.benchmark_quantile, axis=1)
        result[f'Peer_Benchmark_{y}'] = bench
        result[f'Preliminary_{y}_pct'] = 100 * (1 - bench / base[y])
    result['N_Peers'] = k
    result['Peer_Codes'] = [';'.join(base.index[row]) for row in peers]
    result.to_csv(out / '02_county_preliminary_potential.csv', encoding='utf-8-sig')
    return base.join(result)

def make_graph(base, e, weights, seed=None):
    ii, jj = (e.i.to_numpy(int), e.j.to_numpy(int))
    graph = nx.Graph()
    graph.add_nodes_from(range(len(base)))
    total = np.zeros(len(e))
    distances = {}
    rng = np.random.default_rng(seed)
    for block, cols in BLOCKS.items():
        z = robust_z(base[cols])
        if seed is not None:
            z = z[:, rng.integers(0, len(cols), len(cols))]
        d = np.sqrt(np.mean((z[ii] - z[jj]) ** 2, axis=1))
        scale = np.median(d[d > 1e-12]) if (d > 1e-12).any() else 1.0
        d = d / scale
        distances[block] = d
        total += weights.get(block, 0.0) / sum(weights.values()) * d
    w = e.spatial.to_numpy() * np.exp(-np.minimum(total, 700))
    graph.add_weighted_edges_from(zip(ii, jj, w))
    return (graph, w, distances)

def partition(graph, c, resolution, seed, algorithm=None):
    algo = algorithm or c.algorithm
    if algo in ('auto', 'leiden'):
        try:
            import igraph as ig
            import leidenalg as la
        except ImportError:
            if algo == 'leiden':
                raise ImportError('Install igraph and leidenalg, or set --algorithm louvain.')
            algo = 'louvain'
        else:
            edges = list(graph.edges())
            g = ig.Graph(n=len(graph), edges=edges, directed=False)
            p = la.find_partition(g, la.RBConfigurationVertexPartition, weights=[graph[a][b]['weight'] for a, b in edges], resolution_parameter=resolution, seed=seed)
            communities = [set(x) for x in p]
            algo = 'leiden'
    if algo == 'louvain':
        communities = nx.community.louvain_communities(graph, weight='weight', resolution=resolution, seed=seed)
    if algo not in ('leiden', 'louvain'):
        raise ValueError(f'Unknown algorithm: {algo}')
    communities = [set(comp) for group in communities for comp in nx.connected_components(graph.subgraph(group))]
    while True:
        mapping = {i: k for k, group in enumerate(communities) for i in group}
        changed = False
        for k in sorted(range(len(communities)), key=lambda q: len(communities[q])):
            if not communities[k] or len(communities[k]) >= c.min_community_size:
                continue
            totals = {}
            for i in communities[k]:
                for j, data in graph[i].items():
                    other = mapping[j]
                    if other != k:
                        totals[other] = totals.get(other, 0.0) + data['weight']
            if totals:
                target = max(totals, key=lambda q: (totals[q], -q))
                communities[target] |= communities[k]
                communities[k] = set()
                changed = True
                break
        if not changed:
            break
        communities = [x for x in communities if x]
    communities = sorted(communities, key=min)
    labels = np.zeros(len(graph), int)
    for k, group in enumerate(communities, 1):
        labels[list(group)] = k
    return (labels, algo)

def community_analysis(base, e, c, out):
    g, w, ds = make_graph(base, e, c.block_weights)
    labels, algo = partition(g, c, c.resolution, c.seed)
    log(f'Community algorithm={algo}; communities={len(set(labels))}; minimum-size merge={c.min_community_size}')
    edge_out = e.copy()
    edge_out['Origin_Code'] = base.index[e.i.to_numpy()]
    edge_out['Neighbor_Code'] = base.index[e.j.to_numpy()]
    edge_out['Weight'] = w
    for k, d in ds.items():
        edge_out[f'Distance_{k}'] = d
    edge_out.to_csv(out / '03_potential_aware_edge_list.csv', index=False, encoding='utf-8-sig')
    base['Community_ID'] = labels
    assignments = base[['Community_ID']].copy()
    sizes = pd.Series(labels).value_counts()
    assignments['Community_N'] = [sizes[v] for v in labels]
    assignments['Small_Community'] = assignments.Community_N < c.min_community_size
    assignments.to_csv(out / '04_community_assignments.csv', encoding='utf-8-sig')
    profiles, types = ([], {})
    for group, rows in base.groupby('Community_ID'):
        m = rows[FEATURES + OUTCOMES].mean()
        carbon = 'high_carbon' if rows.ECI.median() > base.ECI.median() else 'low_carbon'
        pollution = 'high_pollution' if rows.CPI.median() > base.CPI.median() else 'low_pollution'
        pressure = 'coal_dependence' if rows.Coal_Share.mean() > base.Coal_Share.median() else 'lower_coal_dependence'
        types[group] = f'{carbon}_{pollution}_{pressure}'
        members = np.flatnonzero(labels == group)
        row = {'Community_ID': group, 'Community_Type': types[group], 'N_Counties': len(rows), 'Graph_Connected_Components': nx.number_connected_components(g.subgraph(members)), 'Small_Community': len(rows) < c.min_community_size, 'Province_Count_Descriptive_Only': rows.Province_ID.nunique() if 'Province_ID' in rows else np.nan}
        for f, value in m.items():
            row[f'{f}_mean'] = value
            row[f'{f}_median'] = rows[f].median()
        profiles.append(row)
    base['Community_Type'] = base.Community_ID.map(types)
    pd.DataFrame(profiles).to_csv(out / '05_community_profiles.csv', index=False, encoding='utf-8-sig')
    z = np.column_stack([robust_z(base[v]) / math.sqrt(len(v)) * math.sqrt(c.block_weights[k]) for k, v in BLOCKS.items()])
    tests = []
    variants = [('main', c.resolution, c.block_weights, None, None)]
    variants += [(f'resolution_{r}', r, c.block_weights, None, None) for r in (0.5, 1.5)]
    variants += [('without_potential', c.resolution, {**c.block_weights, 'potential': 0.0}, None, None), ('equal_weights', c.resolution, {k: 1.0 for k in BLOCKS}, None, None), ('alternative_louvain', c.resolution, c.block_weights, None, 'louvain'), ('spatial_binary', c.resolution, c.block_weights, None, None)]
    variants += [(f'feature_bootstrap_{i}', c.resolution, c.block_weights, c.seed + i + 1, None) for i in range(c.sensitivity_repeats)]
    variants += [(f'seed_{i}', c.resolution, c.block_weights, None, None) for i in range(c.sensitivity_repeats)]
    sensitivity_assignments = assignments[['Community_ID']].rename(columns={'Community_ID': 'main'})
    for no, (name, resolution, weights, jitter, method) in enumerate(variants):
        ee = e.copy()
        if name == 'spatial_binary':
            ee['spatial'] = 1.0
        gg, _, _ = make_graph(base, ee, weights, jitter)
        ll, used = (labels, algo) if name == 'main' else partition(gg, c, resolution, c.seed + no, method)
        groups = [set(np.flatnonzero(ll == k)) for k in np.unique(ll)]
        tests.append({'Variant': name, 'Algorithm': used, 'Resolution': resolution, 'N_Communities': len(groups), 'ARI_vs_Main': adjusted_rand_score(labels, ll), 'NMI_vs_Main': normalized_mutual_info_score(labels, ll), 'Modularity': nx.community.modularity(gg, groups, weight='weight', resolution=resolution), 'Silhouette': silhouette_score(z, ll, sample_size=min(1500, len(z)), random_state=c.seed) if 1 < len(groups) < len(z) else np.nan, 'Small_Communities': sum((len(q) < c.min_community_size for q in groups))})
        sensitivity_assignments[name] = ll
    pd.DataFrame(tests).to_csv(out / 'community_stability.csv', index=False, encoding='utf-8-sig')
    sensitivity_assignments.to_csv(out / 'community_sensitivity_assignments.csv', encoding='utf-8-sig')
    return (base, algo)

def build_model_classes():
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch_geometric.nn import GATConv

    class SpatialGAT(nn.Module):

        def __init__(self, d, cfg):
            super().__init__()
            self.gat1 = GATConv(d, cfg.gat_hidden, heads=cfg.gat_heads1, concat=True, dropout=cfg.dropout, edge_dim=1, add_self_loops=cfg.add_self_loops_in_gat, fill_value='mean')
            self.norm1 = nn.LayerNorm(cfg.gat_hidden * cfg.gat_heads1)
            self.gat2 = GATConv(cfg.gat_hidden * cfg.gat_heads1, cfg.gat_hidden // cfg.gat_heads2, heads=cfg.gat_heads2, concat=True, dropout=cfg.dropout, edge_dim=1, add_self_loops=cfg.add_self_loops_in_gat, fill_value='mean')
            self.norm2 = nn.LayerNorm(cfg.gat_hidden)
            self.dropout = nn.Dropout(cfg.dropout)

        def forward(self, x, ei, ea):
            h = self.dropout(F.elu(self.norm1(self.gat1(x, ei, edge_attr=ea))))
            return F.elu(self.norm2(self.gat2(h, ei, edge_attr=ea)))

    class TemporalGRU(nn.Module):

        def __init__(self, d, cfg):
            super().__init__()
            self.gru = nn.GRU(d, cfg.gru_hidden, num_layers=1, batch_first=True)
            self.norm = nn.LayerNorm(cfg.gru_hidden)
            self.dropout = nn.Dropout(cfg.dropout)

        def forward(self, x, lens):
            packed = pack_padded_sequence(x, lens.detach().cpu(), batch_first=True, enforce_sorted=False)
            _, h = self.gru(packed)
            return self.dropout(self.norm(h[-1]))

    class TaskEncoder(nn.Module):

        def __init__(self, d, cfg):
            super().__init__()
            if cfg.use_county_embedding:
                raise ValueError('This version supports only Model5 without county identity embeddings.')
            self.spatial = SpatialGAT(d, cfg)
            self.temporal = TemporalGRU(d, cfg)
            self.fusion = nn.Sequential(nn.Linear(cfg.gat_hidden + cfg.gru_hidden, cfg.encoder_hidden), nn.LayerNorm(cfg.encoder_hidden), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.encoder_hidden, cfg.encoder_hidden), nn.LayerNorm(cfg.encoder_hidden), nn.GELU())

        def forward(self, x, seq, lens, ei, ea):
            return self.fusion(torch.cat([self.spatial(x, ei, ea), self.temporal(seq, lens)], dim=-1))

    class LevelRegressionHead(nn.Module):

        def __init__(self, cfg):
            super().__init__()
            self.hidden = nn.Sequential(nn.Linear(cfg.encoder_hidden, cfg.task_hidden), nn.LayerNorm(cfg.task_hidden), nn.GELU(), nn.Dropout(cfg.dropout))
            self.output = nn.Linear(cfg.task_hidden, 1)

        def forward(self, x):
            return self.output(self.hidden(x)).squeeze(-1)

    class CrossStitchUnit(nn.Module):

        def __init__(self):
            super().__init__()
            self.alpha = nn.Parameter(torch.eye(2))

        def forward(self, a, b):
            return (self.alpha[0, 0] * a + self.alpha[0, 1] * b, self.alpha[1, 0] * a + self.alpha[1, 1] * b)

    class Dual(nn.Module):

        def __init__(self, d, cfg):
            super().__init__()
            self.eci_encoder, self.cpi_encoder = (TaskEncoder(d, cfg), TaskEncoder(d, cfg))
            self.cross_stitch = CrossStitchUnit()
            self.eci_head, self.cpi_head = (LevelRegressionHead(cfg), LevelRegressionHead(cfg))

        def forward(self, *args):
            a, b = self.cross_stitch(self.eci_encoder(*args), self.cpi_encoder(*args))
            return torch.stack([self.eci_head(a), self.cpi_head(b)], dim=1)

    class Single(nn.Module):

        def __init__(self, d, cfg):
            super().__init__()
            self.encoder = TaskEncoder(d, cfg)
            self.head = LevelRegressionHead(cfg)

        def forward(self, *args):
            return self.head(self.encoder(*args))[:, None]
    return (Dual, Single)

class Predictor:

    def __init__(self, panel, base, edge, c, out):
        import torch
        self.torch = torch
        torch.set_num_threads(c.torch_threads)
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if c.device == 'auto' else c.device)
        self.c = c
        self.ids = base.index.tolist()
        self.base_raw = base[FEATURES].to_numpy(float)
        self.o3_budget_table = historical_o3_budgets(panel, self.ids, c, out)
        self.o3_thresholds = self.o3_budget_table.O3_Budget_pct.to_numpy(float)
        self.o3_historical_rates = self.o3_budget_table.O3_Historical_Positive_Trend_pct.to_numpy(float)
        self.members = []
        self.split_map = {}
        dual, single = build_model_classes()
        index = {code: i for i, code in enumerate(self.ids)}
        self.ei = torch.tensor(np.stack([edge.Origin_Code.map(index), edge.Neighbor_Code.map(index)]), dtype=torch.long, device=self.device)
        self.ea = torch.tensor(edge.Row_Standardized_Weight.to_numpy(np.float32)[:, None], device=self.device)
        for directory, targets, cls in [(c.dual_model, ['ECI', 'CPI'], dual), (c.o3_model, ['O3'], single), (c.pm10_model, ['PM10'], single)]:
            p = Path(directory)
            cfg = json.loads((p / 'model_config.json').read_text(encoding='utf-8-sig'))
            fs = json.loads((p / 'feature_scaler.json').read_text(encoding='utf-8-sig'))
            ts = json.loads((p / 'target_scaler.json').read_text(encoding='utf-8-sig'))
            names = json.loads((p / 'feature_names.json').read_text(encoding='utf-8-sig'))
            if isinstance(names, dict):
                names = names['feature_names']
            if names != fs['columns'] or set(names) != set(FEATURES):
                raise ValueError(f'{p}: saved feature order or set is inconsistent.')
            if c.year > cfg['end_year'] or c.year < cfg['start_year']:
                raise ValueError('Saved models are used only for conditional scenarios in observed years and do not automatically extrapolate into the future.')
            saved_codes = cfg.get('county_index_codes') or cfg.get('county_codes')
            if saved_codes is not None and list(map(str, saved_codes)) != self.ids:
                raise ValueError('The saved model node order does not match the current data.')
            model = cls(len(names), SimpleNamespace(**cfg))
            ck = torch.load(p / 'best_model.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(ck['model_state_dict'], strict=True)
            model.to(self.device).eval()
            pos = [FEATURES.index(f) for f in names]
            med, mean, std = [np.array([fs[k][f] for f in names]) for k in ['median', 'mean', 'std']]
            if not np.isfinite(np.r_[med, mean, std]).all() or (std <= 0).any():
                raise ValueError('Saved scaler metadata are invalid.')
            years = list(range(max(cfg['start_year'], c.year - cfg['lookback'] + 1), c.year + 1))
            arr = []
            for year in years:
                raw = panel[panel.Year.eq(year)].set_index('Code').reindex(self.ids)[names].to_numpy(float)
                arr.append((np.where(np.isfinite(raw), raw, med) - mean) / std)
            seq = torch.tensor(np.stack(arr, axis=1), dtype=torch.float32, device=self.device)
            lens = torch.full((len(base),), len(years), dtype=torch.long, device=self.device)
            if len(targets) == 2:
                tm = np.array([ts['log_' + t]['mean'] for t in targets])
                sd = np.array([ts['log_' + t]['std'] for t in targets])
            else:
                tm, sd = (np.array([ts['mean_log']]), np.array([ts['std_log']]))
            self.members.append(dict(model=model, pos=pos, med=med, mean=mean, std=std, seq=seq, lens=lens, targets=targets, tm=tm, sd=sd, path=p))
            splits = read_splits(p).set_index('Code').Split.reindex(self.ids)
            if splits.isna().any():
                raise ValueError(f'{p}: some model graph counties have no train, validation, or test assignment.')
            self.split_map['_'.join(targets) + '_Split'] = splits.to_numpy()
        self.baseline = self.predict(self.base_raw)
        pred = pd.DataFrame({'Code': self.ids, 'Year': c.year})
        for j, y in enumerate(OUTCOMES):
            pred['Observed_' + y] = base[y].to_numpy()
            pred['Predicted_' + y] = self.baseline[:, j]
            pred['Baseline_Error_' + y + '_pct'] = 100 * (self.baseline[:, j] / base[y].to_numpy() - 1)
        for name, values in self.split_map.items():
            pred[name] = values
        pred.to_csv(out / 'baseline_predictions_and_errors.csv', index=False, encoding='utf-8-sig')
        self.reconcile(out)
        log(f'Restored all models; device={self.device}; full graph nodes={len(base)}')

    def predict(self, raw, seed=None):
        torch = self.torch
        result = []
        with torch.no_grad():
            for j, m in enumerate(self.members):
                m['model'].train(seed is not None)
                if seed is not None:
                    torch.manual_seed(seed + j * 100003)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed + j * 100003)
                x = raw[:, m['pos']]
                x = (np.where(np.isfinite(x), x, m['med']) - m['mean']) / m['std']
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
                seq = m['seq'].clone()
                seq[:, -1, :] = x
                z = m['model'](x, seq, m['lens'], self.ei, self.ea).cpu().numpy()
                result.append(np.exp(z * m['sd'] + m['tm']))
                m['model'].eval()
        result = np.column_stack(result)
        if not np.isfinite(result).all() or (result <= 0).any():
            raise FloatingPointError('Model output overflowed or became nonpositive; outputs were not arbitrarily clipped.')
        return result

    def reconcile(self, out):
        rows = []
        for m in self.members:
            files = list(m['path'].glob('predictions*test*.csv')) + list(m['path'].glob('predictions*validation*.csv')) + list(m['path'].glob('predictions*train*.csv'))
            for p in files:
                df = read_csv(p)
                if not {'Code', 'Year'} <= set(df):
                    continue
                df['Code'] = codes(df.Code)
                df = df[df.Year.eq(self.c.year)]
                if df.Code.duplicated().any():
                    raise ValueError(f'{p}: saved predictions contain duplicate codes within a year.')
                df = df.set_index('Code')
                common = [s for s in self.ids if s in df.index]
                ix = [self.ids.index(s) for s in common]
                for y in m['targets']:
                    col = 'Pred_' + y if 'Pred_' + y in df else 'Pred_Index'
                    if col not in df or not common:
                        continue
                    reference = df.loc[common, col].to_numpy(float)
                    error = abs(self.baseline[ix, OUTCOMES.index(y)] / reference - 1)
                    rows.append({'File': str(p), 'Outcome': y, 'N': len(common), 'Max_Relative_Error': error.max()})
        pd.DataFrame(rows, columns=['File', 'Outcome', 'N', 'Max_Relative_Error']).to_csv(out / 'saved_prediction_reconciliation.csv', index=False, encoding='utf-8-sig')
        if set((r['Outcome'] for r in rows)) != set(OUTCOMES):
            raise ValueError('Original prediction files required to validate all four outcomes were not found.')
        for y in OUTCOMES:
            if sum((r['N'] for r in rows if r['Outcome'] == y)) != len(self.ids):
                raise ValueError(f'{y}: saved predictions do not uniquely cover all baseline counties.')
        if max((r['Max_Relative_Error'] for r in rows)) > 0.001:
            raise ValueError('Restored predictions differ from saved predictions by more than 0.1%; scenario evaluation is stopped.')

class FeasibleSpace:

    def __init__(self, panel, base, predictor, c, out):
        self.c, self.base = (c, base)
        self.raw = base[FEATURES].to_numpy(float)
        self.bounds = {}
        self.support = {}
        self.logs = []
        self.inequalities = []
        relation_audit = []
        for first, second in c.observed_inequalities:
            if first not in FEATURES or second not in FEATURES:
                raise ValueError('Unknown joint constraint variable.')
            valid = panel[[first, second]].dropna()
            violations = int((valid[first] > valid[second] + 1e-10).sum())
            active = bool(len(valid) > 0 and violations == 0)
            if active:
                self.inequalities.append((FEATURES.index(first), FEATURES.index(second)))
            relation_audit.append({'Relation': f'{first} <= {second}', 'N_Observed': len(valid), 'N_Violations': violations, 'Preserved_In_Scenarios': active, 'Interpretation': 'preserved empirical relation, not an assumed accounting identity'})
        pd.DataFrame(relation_audit).to_csv(out / 'joint_feature_relations_audit.csv', index=False, encoding='utf-8-sig')
        for member in predictor.members:
            sf = read_splits(member['path'])
            train = set(sf.loc[sf.Split.eq('train'), 'Code'])
            ss = panel[panel.Code.isin(train)][FEATURES].to_numpy(float)
            med = np.nanmedian(ss, axis=0)
            ss = np.where(np.isfinite(ss), ss, med)
            mu, sd = (ss.mean(axis=0), ss.std(axis=0))
            sd = np.where(sd > 1e-10, sd, 1.0)
            z = (ss - mu) / sd
            codearr = panel.loc[panel.Code.isin(train), 'Code'].to_numpy()
            nn = NearestNeighbors(n_neighbors=min(30, len(z)), n_jobs=1).fit(z)
            d, ind = nn.kneighbors(z)
            different = codearr[ind] != codearr[:, None]
            loo = np.min(np.where(different, d, np.inf), axis=1) / math.sqrt(len(FEATURES))
            if not np.isfinite(loo).all():
                raise ValueError('Leave-one-county support distance references are insufficient; increase the neighbor count.')
            threshold = float(np.quantile(loo, c.support_distance_quantile))
            self.support['+'.join(member['targets'])] = (nn, med, mu, sd, threshold)
        boundrows = []
        changes = panel.groupby('Code')[ADJUSTABLE].diff().abs()
        historical = panel.assign(Community_ID=panel.Code.map(base.Community_ID))
        for group, rows in base.groupby('Community_ID'):
            hist = historical[historical.Community_ID.eq(group)]
            idx = hist.index
            lo = hist[FEATURES].quantile(c.support_low).to_numpy()
            hi = hist[FEATURES].quantile(c.support_high).to_numpy()
            cap = np.zeros(len(FEATURES))
            for f in ADJUSTABLE:
                v = changes.loc[idx, f].dropna()
                cap[FEATURES.index(f)] = v.quantile(c.annual_change_quantile) * c.adjustment_years if len(v) else 0.0
            cap = np.nan_to_num(cap, nan=0.0)
            ref = hist[FEATURES].to_numpy(float)
            med = np.nanmedian(ref, axis=0)
            if not np.isfinite(med).all():
                raise ValueError(f'Community{group}contains an entirely missing feature.')
            ref = np.where(np.isfinite(ref), ref, med)
            mu, sd = (ref.mean(axis=0), ref.std(axis=0))
            sd = np.where(sd > 1e-10, sd, 1.0)
            nn = NearestNeighbors(n_neighbors=1, n_jobs=1).fit((ref - mu) / sd)
            refcodes = hist.Code.to_numpy()
            nbor = NearestNeighbors(n_neighbors=min(30, len(ref)), n_jobs=1).fit((ref - mu) / sd)
            dd, ii = nbor.kneighbors((ref - mu) / sd)
            dd = np.min(np.where(refcodes[ii] != refcodes[:, None], dd, np.inf), axis=1) / math.sqrt(len(FEATURES))
            threshold = float(np.quantile(dd[np.isfinite(dd)], c.support_distance_quantile)) if np.isfinite(dd).any() else np.nan
            self.bounds[group] = (lo, hi, cap, nn, med, mu, sd, threshold, hist.Code.nunique())
            for k, f in enumerate(FEATURES):
                boundrows.append({'Community_ID': group, 'Feature': f, 'Historical_Q_Low': lo[k], 'Historical_Q_High': hi[k], 'Max_Change_Budget': cap[k], 'Adjustable': f in ADJUSTABLE, 'N_Reference_Counties': hist.Code.nunique(), 'N_Reference_Rows': len(hist)})
        pd.DataFrame(boundrows).to_csv(out / '06_community_feasible_bounds.csv', index=False, encoding='utf-8-sig')
        save_json(out / 'support_distance_thresholds.json', {'training': {k: v[-1] for k, v in self.support.items()}, 'community': {str(k): v[-2] for k, v in self.bounds.items()}, 'method': 'standardized full-feature nearest-neighbor distance; cutoff from leave-county reference distances', 'caveat': 'empirical support is not proof of all physical/accounting feasibility'})

    def constrain(self, group, idx, proposal, features):
        old = self.raw[idx].copy()
        pos = [FEATURES.index(f) for f in features]
        lo, hi, cap, *_ = self.bounds[group]
        delta = proposal - old
        missing = ~np.isfinite(old[:, pos]).all(axis=1) | ~np.isfinite(proposal[:, pos]).all(axis=1)
        delta[~np.isfinite(delta)] = 0
        keep = np.ones(len(FEATURES), bool)
        keep[pos] = False
        delta[:, keep] = 0
        scale = np.ones(len(idx))
        for j in pos:
            d = delta[:, j]
            changed = abs(d) > 1e-12
            allowance = np.where(d >= 0, hi[j] - old[:, j], old[:, j] - lo[j])
            ratio = np.minimum(cap[j], np.maximum(allowance, 0)) / np.maximum(abs(d), 1e-12)
            scale = np.minimum(scale, np.where(changed, ratio, 1))
        scale[missing] = 0
        new = old + delta * np.clip(scale, 0, 1)[:, None]
        difficulty = np.nanmean(abs(new[:, [FEATURES.index(f) for f in ADJUSTABLE]] - old[:, [FEATURES.index(f) for f in ADJUSTABLE]]) / np.maximum(cap[[FEATURES.index(f) for f in ADJUSTABLE]], 1e-12), axis=1)
        return (new, scale, difficulty)

    def check(self, group, new):
        lo, hi, cap, nn, med, mu, sd, cut, n = self.bounds[group]
        fill = np.where(np.isfinite(new), new, med)
        dist = nn.kneighbors((fill - mu) / sd, return_distance=True)[0][:, 0] / math.sqrt(len(FEATURES))
        ok = (dist <= cut + 1e-10) & (n >= self.c.min_reference_counties) & np.isfinite(new).all(axis=1)
        for first, second in self.inequalities:
            ok &= new[:, first] <= new[:, second] + 1e-10
        worst = dist / max(cut, 1e-12) if np.isfinite(cut) else np.full(len(new), np.inf)
        for nn, med, mu, sd, threshold in self.support.values():
            z = (np.where(np.isfinite(new), new, med) - mu) / sd
            d = nn.kneighbors(z, n_neighbors=1)[0][:, 0] / math.sqrt(len(FEATURES))
            ok &= d <= threshold + 1e-10
            worst = np.maximum(worst, d / max(threshold, 1e-12))
        return (ok, worst)

def evaluate_scenario(group, sid, family, raw, idx, scale, difficulty, predictor, space, base, c):
    changed = np.any(~np.isclose(raw[idx], space.raw[idx], rtol=0, atol=1e-12, equal_nan=True), axis=1)
    pred = predictor.predict(raw) if changed.any() else predictor.baseline.copy()
    imp = 100 * (1 - pred / predictor.baseline)
    supported, ratio = space.check(group, raw[idx])
    o3_change = -imp[:, 2]
    excess = o3_change - predictor.o3_thresholds
    feasible = bool(supported[changed].all() and np.max(excess) <= 1e-06)
    rec = {'Community_ID': group, 'Scenario_ID': sid, 'Family': family, 'N_Counties': len(idx), 'Support_Rate': supported.mean(), 'Max_Support_Distance_Ratio': ratio.max(), 'Min_Applied_Fraction': np.min(scale), 'Mean_Applied_Fraction': np.mean(scale), 'Mean_Adjustment_Difficulty': np.mean(difficulty), 'Within_Checked_Limits': feasible, 'Max_O3_Change_All_Counties_pct': o3_change.max(), 'O3_Budget_Mode': c.o3_budget_mode, 'Max_O3_Threshold_Excess_All_Counties_pct_points': excess.max(), 'O3_Budget_Min_Community_pct': predictor.o3_thresholds[idx].min(), 'O3_Budget_Mean_Community_pct': predictor.o3_thresholds[idx].mean(), 'O3_Budget_Max_Community_pct': predictor.o3_thresholds[idx].max(), 'O3_Violating_Counties_All': int((excess > 1e-06).sum()), 'Support_Failed_Counties': int((~supported).sum()), 'Changed_Unsupported_Counties': int((changed & ~supported).sum()), 'Changed_Counties': int(changed.sum())}
    for factor in set(list(c.o3_history_sensitivity_multipliers) + [c.o3_trend_multiplier]):
        rec[multiplier_excess_column(factor)] = float(np.max(o3_change - factor * predictor.o3_historical_rates))
    for j, y in enumerate(OUTCOMES):
        rec[f'{y}_improvement_pct'] = imp[idx, j].mean()
        rec[f'{y}_improvement_median_pct'] = np.median(imp[idx, j])
    rec['Dual_Improvement_Share'] = np.mean((imp[idx, 0] > c.improvement_tolerance_pct) & (imp[idx, 1] > c.improvement_tolerance_pct))
    detail = pd.DataFrame({'Code': base.index[idx], 'Community_ID': group, 'Scenario_ID': sid, 'Family': family, 'Supported': supported, 'Support_Distance_Ratio': ratio, 'Applied_Fraction': scale, 'Adjustment_Difficulty': difficulty, 'Community_Scenario_Within_Checked_Limits': feasible})
    detail['Own_Inputs_Changed'] = changed
    for j, y in enumerate(OUTCOMES):
        detail[f'Observed_{y}'] = base.iloc[idx][y].to_numpy()
        detail[f'Predicted_Baseline_{y}'] = predictor.baseline[idx, j]
        detail[f'Predicted_CF_{y}'] = pred[idx, j]
        detail[f'{y}_improvement_pct'] = imp[idx, j]
        detail[f'Residual_Anchored_CF_{y}'] = base.iloc[idx][y].to_numpy() * pred[idx, j] / predictor.baseline[idx, j]
    detail['O3_change_pct'] = -imp[idx, 2]
    detail['O3_Budget_pct'] = predictor.o3_thresholds[idx]
    detail['O3_Threshold_Excess_pct_points'] = excess[idx]
    return (rec, detail, pred)

def posthoc_context(panel, base, c, out):
    sources = []
    if Path(c.policy).exists():
        p = read_csv(c.policy)
        p['Code'] = codes(p.Code)
        if p.duplicated(['Code', 'Year']).any():
            raise ValueError('The policy panel contains duplicate Code-Year rows.')
        if not set(POLICIES) <= set(p):
            raise ValueError('Required policy columns are missing.')
        p = p[p.Year.between(c.history_start, c.year) & p.Code.isin(base.index)].copy()
        expected = pd.MultiIndex.from_frame(panel[['Code', 'Year']])
        present = pd.MultiIndex.from_frame(p[['Code', 'Year']])
        if not expected.isin(present).all():
            raise ValueError('The policy panel does not fully cover the indicator panel; missing values cannot be treated as no policy coverage.')
        p['Community_ID'] = p.Code.map(base.Community_ID)
        if p[POLICIES].isna().any().any() or not p[POLICIES].isin([0, 1]).all().all():
            raise ValueError('Policy coverage must be complete and binary.')
        p['Policy_Count'] = p[POLICIES].sum(axis=1)
        p.groupby(['Community_ID', 'Year'])[POLICIES + ['Policy_Count']].mean().to_csv(out / 'posthoc_policy_coverage.csv', encoding='utf-8-sig')
        p = p.sort_values(['Code', 'Year'])
        support = []
        for policy in POLICIES:
            previous = p.groupby('Code')[policy].shift()
            entries = p[p[policy].eq(1) & previous.eq(0)].groupby('Community_ID').Code.nunique()
            for group in sorted(base.Community_ID.unique()):
                current = p[p.Community_ID.eq(group) & p.Year.eq(c.year)]
                support.append({'Community_ID': group, 'Policy': policy, 'Covered_N': int(current[policy].sum()), 'Uncovered_N': int(current[policy].eq(0).sum()), 'Observed_Adopting_Counties': int(entries.get(group, 0)), 'Interpretation': 'historical support only; not a county-specific causal reward'})
        pd.DataFrame(support).to_csv(out / 'posthoc_policy_action_support.csv', index=False, encoding='utf-8-sig')
    membership = Path(c.boundary_dir) / 'diagnostics/county_group_membership.csv'
    if membership.exists():
        m = read_csv(membership)
        m['Code'] = codes(m.Code)
        m = m.merge(base[['Community_ID']], left_on='Code', right_index=True, how='inner', validate='many_to_one')
        g = m.groupby(['Community_ID', 'stratifier', 'group']).Code.nunique().rename('N_Counties').reset_index()
        g['Share'] = g.N_Counties / g.Community_ID.map(base.Community_ID.value_counts())
        g.to_csv(out / 'posthoc_boundary_alignment.csv', index=False, encoding='utf-8-sig')
    paths = [Path(c.mechanism_dir) / 'pathway_evidence_summary.csv', Path(c.boundary_dir) / 'boundary_summary.csv', membership]
    for p in paths:
        sources.append({'Path': str(p), 'Exists': p.exists(), 'SHA256': digest(p) if p.exists() else '', 'Use': 'posthoc interpretation only; no ATT enters clustering or scenario objective'})
    pd.DataFrame(sources).to_csv(out / 'posthoc_evidence_sources.csv', index=False, encoding='utf-8-sig')

def export_geo(master, c, out):
    import geopandas as gpd
    shape = gpd.read_file(c.shape)
    shape['Code'] = codes(shape[c.shape_code])
    shape = shape[['Code', 'geometry']].dissolve(by='Code').reset_index()
    shape = shape.merge(master, on='Code', how='right', validate='one_to_one')
    if shape.geometry.isna().any():
        raise ValueError('The shapefile and analysis counties contain unmatched codes; a misleading complete map will not be exported.')
    shape.to_file(out / 'county_community_potential.gpkg', layer='potential', driver='GPKG')

def construct_references(base, c, out):
    z = robust_z(base[STRUCTURAL])
    zfeatures = robust_z(base[FEATURES])
    donor_index = np.full(len(base), -1, int)
    rows = []
    for group in sorted(base.Community_ID.unique()):
        idx = np.flatnonzero(base.Community_ID.to_numpy() == group)
        distances = cdist(z[idx], z[idx])
        np.fill_diagonal(distances, np.inf)
        for local, i in enumerate(idx):
            peers = idx[np.argsort(distances[local], kind='stable')[:min(c.peer_count, len(idx) - 1)]]
            row = {'Code': base.index[i], 'Community_ID': group, 'N_Comparable_Counties': len(peers), 'Observed_ECI': base.ECI.iloc[i], 'Observed_CPI': base.CPI.iloc[i], 'N_Joint_Reference_Counties': 0, 'Reference_Code': '', 'Reference_Status': 'no_peer_county', 'Independent_Marginal_References_Not_A_Joint_Target': True}
            for y in ['ECI', 'CPI']:
                q = np.quantile(base.iloc[peers][y], c.benchmark_quantile) if len(peers) else np.nan
                row[f'{y}_Marginal_Benchmark'] = q
                row[f'{y}_Reference_Gap_pct'] = 100 * (1 - q / base.iloc[i][y])
                row[f'{y}_Joint_Reference_Gap_pct'] = np.nan
            if len(peers):
                peer_data = base.iloc[peers]
                qeci = peer_data.ECI.quantile(c.reference_joint_quantile)
                qcpi = peer_data.CPI.quantile(c.reference_joint_quantile)
                reference_pool = peers[(peer_data.ECI.to_numpy() <= qeci) & (peer_data.CPI.to_numpy() <= qcpi)]
                row['Peer_ECI_Joint_Cutoff'] = qeci
                row['Peer_CPI_Joint_Cutoff'] = qcpi
                row['N_Joint_Reference_Counties'] = len(reference_pool)
                row['Comparable_County_Codes'] = ';'.join(base.index[peers])
                row['Reference_Pool_Codes'] = ';'.join(base.index[reference_pool])
                row['Reference_Status'] = 'no_joint_reference' if not len(reference_pool) else 'available'
                if len(reference_pool):
                    center = np.median(zfeatures[reference_pool], axis=0)
                    deviation = np.mean((zfeatures[reference_pool] - center) ** 2, axis=1)
                    j = int(reference_pool[int(np.argmin(deviation))])
                    donor_index[i] = j
                    row['Reference_Code'] = base.index[j]
                    row['Reference_Structural_Distance'] = float(np.sqrt(np.mean((z[i] - z[j]) ** 2)))
                    for y in ['ECI', 'CPI']:
                        row[f'{y}_Joint_Reference_Level'] = base.iloc[j][y]
                        row[f'{y}_Joint_Reference_Gap_pct'] = 100 * (1 - base.iloc[j][y] / base.iloc[i][y])
            row['Small_Reference_Pool'] = row['N_Joint_Reference_Counties'] < c.reference_warning_n
            rows.append(row)
    references = pd.DataFrame(rows).set_index('Code').reindex(base.index)
    references.to_csv(out / '07_county_reference_gaps.csv', encoding='utf-8-sig')
    summary = references.groupby('Community_ID')[['ECI_Reference_Gap_pct', 'CPI_Reference_Gap_pct', 'ECI_Joint_Reference_Gap_pct', 'CPI_Joint_Reference_Gap_pct']].agg(['count', 'mean', 'median', 'min', 'max'])
    summary.columns = ['_'.join(x) for x in summary.columns]
    summary['N_Counties'] = references.groupby('Community_ID').size()
    summary['N_With_Joint_Reference'] = references.Reference_Status.eq('available').groupby(references.Community_ID).sum()
    summary['Interpretation'] = 'reference opportunity gaps, not identified attainable or causal effects'
    summary.to_csv(out / '08_community_reference_gaps.csv', encoding='utf-8-sig')
    return (references, donor_index)

class PairedMC:

    def __init__(self, predictor, c):
        self.predictor, self.c = (predictor, c)
        self.seeds = [c.seed + 400000 + i for i in range(c.mc_samples)]
        self.cache = None

    def evaluate(self, raw, idx):
        p, c = (self.predictor, self.c)
        if np.array_equal(raw, p.base_raw, equal_nan=True):
            return (np.zeros((c.mc_samples, len(idx), len(OUTCOMES))), np.full(c.mc_samples, -p.o3_thresholds.min()))
        if self.cache is None:
            log(f'Caching paired MC baseline: {c.mc_samples} draws')
            self.cache = []
            for r, seed in enumerate(self.seeds):
                self.cache.append(p.predict(p.base_raw, seed))
                if (r + 1) % 25 == 0:
                    log(f'  MC baseline {r + 1}/{c.mc_samples}')
        draws = []
        excess = []
        for r, seed in enumerate(self.seeds):
            cf = p.predict(raw, seed)
            imp = 100 * (1 - cf / self.cache[r])
            draws.append(imp[idx])
            excess.append(float(np.max(-imp[:, 2] - p.o3_thresholds)))
            if (r + 1) % 25 == 0:
                log(f'  MC scenario {r + 1}/{c.mc_samples}')
        return (np.stack(draws), np.array(excess))

def fixed_scenario_inputs(baseline, idx, donors, features, fraction):
    proposed = baseline[idx].copy()
    available = donors[idx] >= 0
    positions = [FEATURES.index(f) for f in features]
    if available.any():
        target = baseline[donors[idx][available]][:, positions]
        source = baseline[idx[available]][:, positions]
        proposed[np.ix_(available, positions)] = (1 - fraction) * source + fraction * target
    return (proposed, available)

def response_types(eci, cpi, eps):
    a, b = (np.asarray(eci), np.asarray(cpi))
    return np.select([(a > eps) & (b > eps), (a > eps) & (b >= -eps), (b > eps) & (a >= -eps), (a > eps) & (b < -eps) | (b > eps) & (a < -eps), (abs(a) <= eps) & (abs(b) <= eps)], ['dual_improvement', 'carbon_only', 'pollution_only', 'tradeoff', 'no_change'], default='no_dual_benefit')

def assess_scenarios(panel, base, references, donors, predictor, space, c, out):
    group_ids = sorted(base.Community_ID.unique())
    if c.max_communities:
        group_ids = group_ids[:c.max_communities]
    baseline = predictor.base_raw.copy()
    mc = PairedMC(predictor, c)
    summary_rows = []
    county_frames = []
    mc_rows = []
    constraints = []
    sensitivity = []
    external_rows = []
    archives = out / 'community_scenario_inputs'
    archives.mkdir()
    for gi, group in enumerate(group_ids, 1):
        idx = np.flatnonzero(base.Community_ID.to_numpy() == group)
        log(f'Community {gi}/{len(group_ids)}: ID={group}, counties={len(idx)}')
        local_refs = references.iloc[idx]
        families = {k: FAMILIES[k] for k in ['efficiency', 'joint_energy']} if c.smoke else FAMILIES
        fractions = [c.mc_fraction] if c.smoke else list(c.fractions)
        plans = [('baseline', 'baseline', 0.0, [])] + [(f'{family}_align_{fraction:g}', family, fraction, features) for family, features in families.items() for fraction in fractions]
        local_summaries = []
        local_details = []
        input_arrays = []
        input_ids = []
        for pi, (sid, family, fraction, features) in enumerate(plans, 1):
            if family == 'baseline':
                new = baseline[idx].copy()
                scale = np.ones(len(idx))
                difficulty = np.zeros(len(idx))
                frozen = np.zeros(len(idx), bool)
                available = donors[idx] >= 0
            else:
                proposed, available = fixed_scenario_inputs(baseline, idx, donors, features, fraction)
                new, scale, difficulty = space.constrain(group, idx, proposed, features)
                supported, _ = space.check(group, new)
                changed = np.any(~np.isclose(new, baseline[idx], rtol=0, atol=1e-12, equal_nan=True), axis=1)
                frozen = ~supported & changed if c.freeze_unsupported_updates else np.zeros(len(idx), bool)
                new[frozen] = baseline[idx][frozen]
                scale[frozen] = 0.0
                difficulty[frozen] = 0.0
            raw = baseline.copy()
            raw[idx] = new
            rec, detail, pred = evaluate_scenario(group, sid, family, raw, idx, scale, difficulty, predictor, space, base, c)
            rec['Alignment_Fraction'] = fraction
            rec['Predeclared_Primary_Dose'] = bool(fraction == c.mc_fraction)
            rec['N_No_Joint_Reference'] = int((~available).sum())
            rec['N_Unsupported_Updates_Frozen'] = int(frozen.sum())
            rec['Interpretation'] = 'response to a fixed reference-alignment scenario, not an optimal policy effect'
            detail['Alignment_Fraction'] = fraction
            detail['Reference_Code'] = local_refs.Reference_Code.to_numpy()
            detail['Reference_Status'] = local_refs.Reference_Status.to_numpy()
            detail['N_Joint_Reference_Counties'] = local_refs.N_Joint_Reference_Counties.to_numpy()
            detail['Unsupported_Update_Frozen'] = frozen
            detail['Predeclared_Primary_Dose'] = bool(fraction == c.mc_fraction)
            detail['Response_Type'] = response_types(detail.ECI_improvement_pct, detail.CPI_improvement_pct, c.improvement_tolerance_pct)
            detail['O3_Local_Threshold_Passed'] = detail.O3_Threshold_Excess_pct_points <= 1e-06
            rec['Share_Local_O3_Threshold_Passed'] = detail.O3_Local_Threshold_Passed.mean()
            for column, values in predictor.split_map.items():
                detail[column] = values[idx]
            for y in OUTCOMES:
                detail[y + '_MC_low'] = np.nan
                detail[y + '_MC_high'] = np.nan
            detail['MC_O3_Exceedance_Frequency'] = np.nan
            detail['MC_All_County_O3_Constraint_Frequency'] = np.nan
            detail['MC_Dual_Improvement_Frequency'] = np.nan
            detail['MC_Status'] = 'not_scheduled_for_this_prespecified_dose'
            if c.mc_all_doses or fraction == c.mc_fraction or family == 'baseline':
                log(f'  Scenario {pi}/{len(plans)}: {sid}; paired MC')
                draws, excess = mc.evaluate(raw, idx)
                qlow, qhigh = np.quantile(draws, [0.025, 0.975], axis=0)
                graph_frequency = float((excess <= 1e-06).mean())
                for j, y in enumerate(OUTCOMES):
                    detail[y + '_MC_low'] = qlow[:, j]
                    detail[y + '_MC_high'] = qhigh[:, j]
                    lo, hi = np.quantile(draws[:, :, j].mean(axis=1), [0.025, 0.975])
                    mc_rows.append({'Community_ID': group, 'Scenario_ID': sid, 'Family': family, 'Alignment_Fraction': fraction, 'Outcome': y, 'Mean_improvement_pct': rec[y + '_improvement_pct'], 'MC_low': lo, 'MC_high': hi, 'MC_Samples': c.mc_samples if family != 'baseline' else 0, 'MC_All_County_O3_Constraint_Frequency': graph_frequency, 'Interval_Type': 'fixed-scenario paired dropout interval; not causal or sampling CI'})
                detail['MC_O3_Exceedance_Frequency'] = (-draws[:, :, 2] - predictor.o3_thresholds[idx] > 1e-06).mean(axis=0)
                detail['MC_All_County_O3_Constraint_Frequency'] = graph_frequency
                detail['MC_Dual_Improvement_Frequency'] = ((draws[:, :, 0] > c.improvement_tolerance_pct) & (draws[:, :, 1] > c.improvement_tolerance_pct)).mean(axis=0)
                detail['MC_Status'] = 'evaluated_prespecified_scenario' if family != 'baseline' else 'exact_no_change'
            else:
                log(f'  Scenario {pi}/{len(plans)}: {sid}; point predictions')
            flags = []
            error = np.max(abs(predictor.baseline[idx] / base.iloc[idx][OUTCOMES].to_numpy(float) - 1), axis=1) * 100
            for k, r in enumerate(detail.itertuples()):
                f = []
                if not r.Supported:
                    f.append('insufficient_covariate_support')
                if r.Reference_Status != 'available':
                    f.append('no_joint_reference')
                if r.N_Joint_Reference_Counties < c.reference_warning_n:
                    f.append('small_reference_pool')
                if len(idx) < c.min_community_size:
                    f.append('small_community')
                if r.Unsupported_Update_Frozen:
                    f.append('unsupported_input_update_frozen')
                if not r.O3_Local_Threshold_Passed:
                    f.append('local_O3_budget_exceeded')
                if rec['O3_Violating_Counties_All'] > 0:
                    f.append('graph_wide_O3_budget_exceeded')
                if np.isfinite(r.O3_MC_low) and -r.O3_MC_low > r.O3_Budget_pct + 1e-06:
                    f.append('local_O3_MC_risk')
                if np.isfinite(r.MC_All_County_O3_Constraint_Frequency) and r.MC_All_County_O3_Constraint_Frequency < 0.95:
                    f.append('graph_wide_O3_MC_risk')
                if error[k] > c.baseline_error_flag_pct:
                    f.append('large_baseline_prediction_error')
                if c.mc_samples < 100 and r.MC_Status == 'evaluated_prespecified_scenario':
                    f.append('few_MC_draws')
                flags.append(';'.join(f) or 'within_checked_support')
            detail['Evidence_Support_Flag'] = flags
            local_summaries.append(rec)
            local_details.append(detail)
            input_ids.append(sid)
            input_arrays.append(new)
            outside = np.ones(len(base), bool)
            outside[idx] = False
            imp = 100 * (1 - pred / predictor.baseline)
            ext = {'Community_ID': group, 'Scenario_ID': sid, 'Family': family, 'N_Outside_Counties': int(outside.sum()), 'N_Outside_O3_Threshold_Exceeded': int((-imp[outside, 2] - predictor.o3_thresholds[outside] > 1e-06).sum())}
            for j, y in enumerate(OUTCOMES):
                ext[y + '_Mean_Outside_improvement_pct'] = float(imp[outside, j].mean()) if outside.any() else np.nan
                ext[y + '_MaxAbs_Outside_Change_pct'] = float(abs(imp[outside, j]).max()) if outside.any() else np.nan
            external_rows.append(ext)
            rules = [('fixed', t, np.nan, rec['Max_O3_Change_All_Counties_pct'] - t) for t in c.o3_sensitivity_thresholds]
            rules += [('historical_trend', np.nan, factor, rec[multiplier_excess_column(factor)]) for factor in sorted(set(list(c.o3_history_sensitivity_multipliers) + [c.o3_trend_multiplier]))]
            for mode, threshold, factor, excess in rules:
                sensitivity.append({'Community_ID': group, 'Scenario_ID': sid, 'Family': family, 'Alignment_Fraction': fraction, 'O3_Mode': mode, 'Fixed_Threshold_pct': threshold, 'History_Multiplier': factor, 'Max_Excess_pct_points': excess, 'O3_Threshold_Passed': excess <= 1e-06, 'Support_Update_Check_Passed': rec['Changed_Unsupported_Counties'] == 0})
        summary = pd.DataFrame(local_summaries)
        details = pd.concat(local_details, ignore_index=True)
        summary.to_csv(archives / f'community_{group:04d}_responses.csv', index=False, encoding='utf-8-sig')
        details.to_csv(archives / f'community_{group:04d}_county_responses.csv', index=False, encoding='utf-8-sig')
        np.savez_compressed(archives / f'community_{group:04d}_inputs.npz', scenario_ids=np.array(input_ids), codes=np.array(base.index[idx], dtype=str), feature_names=np.array(FEATURES), inputs=np.stack(input_arrays))
        summary_rows.extend(local_summaries)
        county_frames.append(details)
        assessed = summary[summary.Family.ne('baseline')]
        constraints.append({'Community_ID': group, 'N_Counties': len(idx), 'N_Preset_Scenarios': len(assessed), 'N_Scenarios_Within_Checked_Limits': int(assessed.Within_Checked_Limits.sum()), 'N_Scenarios_Exceeding_O3': int(assessed.O3_Violating_Counties_All.gt(0).sum()), 'N_Scenarios_Limited_By_Support': int(assessed.N_Unsupported_Updates_Frozen.gt(0).sum()), 'N_Scenarios_Limited_By_Adjustment_Budget': int(assessed.Mean_Applied_Fraction.lt(0.999).sum()), 'N_Counties_No_Joint_Reference': int((donors[idx] < 0).sum()), 'Interpretation': 'diagnostics of prespecified scenarios; not causal contribution or policy ranking'})
    summary = pd.DataFrame(summary_rows)
    details = pd.concat(county_frames, ignore_index=True)
    summary.to_csv(out / '09_community_scenario_responses.csv', index=False, encoding='utf-8-sig')
    details.to_csv(out / '10_county_scenario_responses.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(mc_rows).to_csv(out / '11_community_scenario_MC_intervals.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(constraints).to_csv(out / '12_community_constraints.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(sensitivity).to_csv(out / '13_O3_threshold_sensitivity.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(external_rows).to_csv(out / '16_outside_community_responses.csv', index=False, encoding='utf-8-sig')
    primary = summary[summary.Predeclared_Primary_Dose]
    primary.to_csv(out / '15_prespecified_primary_scenario_comparison.csv', index=False, encoding='utf-8-sig')
    distribution = details.groupby(['Community_ID', 'Family', 'Alignment_Fraction'])[[y + '_improvement_pct' for y in OUTCOMES]].agg(['count', 'mean', 'median', 'min', 'max'])
    distribution.columns = ['_'.join(x) for x in distribution.columns]
    distribution.to_csv(out / '17_county_response_distributions.csv', encoding='utf-8-sig')
    split_frames = []
    for column in predictor.split_map:
        frame = details[details.Predeclared_Primary_Dose].groupby(['Family', 'Alignment_Fraction', column])[[y + '_improvement_pct' for y in OUTCOMES]].agg(['count', 'mean', 'median'])
        frame.columns = ['_'.join(x) for x in frame.columns]
        frame = frame.reset_index().rename(columns={column: 'Split'})
        frame['Model_Split'] = column
        split_frames.append(frame)
    pd.concat(split_frames).to_csv(out / '18_responses_by_model_split.csv', index=False, encoding='utf-8-sig')
    master = base[['Community_ID', 'Community_Type'] + FEATURES + OUTCOMES].copy()
    master = master.join(references.drop(columns=['Community_ID', 'Observed_ECI', 'Observed_CPI']))
    master = master.join(predictor.o3_budget_table[['O3_Budget_pct', 'O3_Historical_Annual_Change_pct']])
    master['Scenario_Assessed'] = master.Community_ID.isin(group_ids)
    master['Scenario_Response_File'] = '10_county_scenario_responses.csv'
    master['Policy_Action_Assigned'] = False
    master['Interpretation'] = 'reference gaps and fixed-scenario responses; no optimal policy or county-specific causal reward'
    master = master.reset_index()
    master.to_csv(out / '14_section46_to47_state_handoff.csv', index=False, encoding='utf-8-sig')
    return (master, summary)

def runtime_preflight():
    pass

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config')
    p.add_argument('--output')
    p.add_argument('--year', type=int)
    p.add_argument('--stage', choices=['audit', 'prepare', 'full'])
    p.add_argument('--algorithm', choices=['auto', 'leiden', 'louvain'])
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--mc-samples', type=int)
    p.add_argument('--max-communities', type=int)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--export-geo', action='store_true')
    a = p.parse_args()
    d = asdict(Settings())
    if a.config:
        overrides = json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
        unknown = set(overrides) - set(d)
        if unknown:
            raise ValueError('The configuration contains unknown or obsolete parameters: ' + str(unknown))
        d.update(overrides)
    for k, v in vars(a).items():
        if k not in ('config', 'smoke', 'export_geo') and v is not None:
            d[k] = v
    if a.smoke:
        d['smoke'] = True
    if a.export_geo:
        d['export_geo'] = True
    if d['smoke']:
        d['max_communities'] = 1
        d['mc_samples'] = 4
        d['sensitivity_repeats'] = 1
    return Settings(**d)

def main():
    runtime_preflight()
    c = parse_args()
    validate_settings(c)
    random.seed(c.seed)
    np.random.seed(c.seed)
    out = Path(c.output) / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'settings': asdict(c), 'started': datetime.now().isoformat(), 'scope': '4.6 community identification and condition-based assessment only', 'no_FCII': True, 'no_policy_assignment': True, 'no_optimization': True, 'no_scenario_selection_by_benefit': True, 'unit': 'county Code; administrative IDs descriptive only', 'reference_gap': '100*(1-observed_peer_benchmark/observed_baseline); separate marginal gaps are not a jointly attainable state', 'reference_protocol': 'leave-self-out structural peers; predeclared joint low-quantile set; actual sample nearest its feature median', 'scenario_protocol': 'all prespecified families and alignment fractions retained regardless of sign or limit violations', 'time_basis': f'{c.year} snapshot; only current features and GRU final step changed', 'prediction_change': '100*(1-predicted_scenario/predicted_baseline); positive=improvement', 'anchoring': 'Observed_CF=Observed_Baseline*Predicted_CF/Predicted_Baseline; assumes unchanged multiplicative residual', 'O3_budget': 'positive part of each county historical log-linear annual O3 trend; all model-graph counties checked', 'uncertainty': 'paired MC dropout for predeclared doses (all doses by default); not causal or sampling CIs; budget treated as fixed', 'community_scope': 'one community aligned at a time; outside/historical inputs fixed; cannot sum into a joint national rollout', 'community_validation': 'contemporary descriptive communities; reference-gap differences are not independent proof of community validity', 'input_files': {}, 'python': sys.version, 'platform': platform.platform()}
    try:
        needed = [Path(c.panel), Path(c.edge), Path(__file__)]
        if Path(c.policy).exists():
            needed.append(Path(c.policy))
        for directory in [c.dual_model, c.o3_model, c.pm10_model]:
            needed += [Path(directory) / f for f in ['model_config.json', 'feature_names.json', 'feature_scaler.json', 'target_scaler.json', 'best_model.pt']] + split_sources(directory)
        for path in needed:
            if not path.is_file():
                raise FileNotFoundError(path)
            manifest['input_files'][str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
        manifest['versions'] = {}
        for pkg in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'networkx', 'torch', 'torch-geometric', 'igraph', 'leidenalg']:
            try:
                manifest['versions'][pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                manifest['versions'][pkg] = 'not installed'
        save_json(out / 'analysis_manifest.json', manifest)
        log(f'Output: {out}')
        panel, base, ids, edge, e = load_data(c, out)
        if c.stage == 'audit':
            Predictor(panel, base, edge, c, out)
        else:
            base = preliminary(base, c, out)
            base, algo = community_analysis(base, e, c, out)
            manifest['community_algorithm'] = algo
            references, donors = construct_references(base, c, out)
            posthoc_context(panel, base, c, out)
            if c.stage == 'full':
                predictor = Predictor(panel, base, edge, c, out)
                space = FeasibleSpace(panel, base, predictor, c, out)
                master, summary = assess_scenarios(panel, base, references, donors, predictor, space, c, out)
                if c.export_geo:
                    export_geo(master, c, out)
                manifest['assessed_counties'] = int(master.Scenario_Assessed.sum())
                manifest['scenario_rows'] = len(summary)
        if c.smoke:
            manifest['status'] = 'smoke_test_complete_not_publication_results'
        elif c.stage == 'full' and manifest.get('assessed_counties', 0) < len(base):
            manifest['status'] = 'complete_selected_communities_only'
        else:
            manifest['status'] = 'complete_' + c.stage
    except Exception:
        manifest['status'] = 'failed'
        manifest['error'] = traceback.format_exc()
        raise
    finally:
        manifest['finished'] = datetime.now().isoformat()
        manifest['outputs'] = [str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()]
        save_json(out / 'analysis_manifest.json', manifest)
    log(f'Finished {c.stage}: {out}')
from itertools import combinations
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

@dataclass
class RLSettings(Settings):
    freeze_unsupported_updates: bool = False
    reward_mode: str = 'mean'
    trend_budget_years: float = 1.0
    reverse_trend_fraction: float = 0.25
    trend_min_years: int = 3
    trend_zero_tolerance: float = 1e-12
    desired_recommended_units: int = 8
    physical_bounds: dict = field(default_factory=lambda: {'PGDP': [0, None], 'POPD': [0, None], 'Road_Density': [0, None], 'Energy_Intensity': [0, None], 'GPAI': [0, None], 'GIPAI': [0, None]})
    output: str = str(PROJECT_ROOT / 'outputs' / 'q_learning')
    section46_run: str = ''
    decision_level: str = 'community'
    max_new_policies: int = 2
    max_units: int = 0
    unit_ids: tuple = ()
    ridge_alpha: float = 100.0
    transition_test_fraction: float = 0.2
    temporal_train_end: int = 2020
    min_policy_adoptions: int = 5
    allow_pair_extrapolation: bool = True
    min_pair_adoptions: int = 1
    n_uncertainty_draws: int = 30
    reward_quantile: float = 0.1
    reward_eci_weight: float = 0.5
    reward_cpi_weight: float = 0.5
    policy_count_penalty: float = 0.0
    pair_extrapolation_penalty: float = 0.0
    support_freeze_penalty: float = 0.0
    o3_excess_penalty: float = 1.0
    hard_o3_constraint: bool = True
    q_learning_rate: float = 0.25
    q_replay_epochs: int = 500
    q_tolerance: float = 1e-07
    evidence_min_pre: int = 1
    att_min_pre_values: tuple = (1, 2, 3)
    active_att_min_pre: int = 1
    att_features: tuple = tuple(FEATURES)
    att_recipe_file: str = ''
    att_uncertainty: str = 'fixed_att_paired_dropout'
    att_min_complete_fraction: float = 1e-09
    att_min_treated: int = 1
    att_export_count_benchmarks: bool = False
    att_export_boundary_context: bool = True
    history_fallback: bool = True
    history_min_adoptions: int = 1
    history_max_post: int = 5
    history_min_rows: int = 20
    history_min_noaddition_rows: int = 5
    history_fallback_penalty: float = 0.0
    pair_additive_fallback: bool = True
    scenario_targets: tuple = (('ECI',), ('CPI',), ('O3',), ('PM10',), ('ECI', 'CPI'), ('ECI', 'O3'), ('ECI', 'PM10'), ('CPI', 'O3'), ('CPI', 'PM10'), ('O3', 'PM10'), ('ECI', 'CPI', 'O3'), ('ECI', 'CPI', 'PM10'), ('ECI', 'O3', 'PM10'), ('CPI', 'O3', 'PM10'), ('ECI', 'CPI', 'O3', 'PM10'))
    o3_exceedance_limits: tuple = (0.05, 0.1)
    risk_scope: str = 'decision_unit'
    risk_numeric_tolerance_pct: float = 1e-06
    reference_run: str = ''
    new_communities: str = ''
    similarity_top_k: int = 5
    similarity_min_feature_fraction: float = 0.8
    similarity_include_policy: bool = False
    smoke: bool = False

def read47_args():
    p = argparse.ArgumentParser(description='Two-policy model-based RL; separate community/city decisions.')
    p.add_argument('--config', type=str)
    p.add_argument('--stage', choices=['transition', 'full', 'similarity', 'att_audit'], default=None)
    p.add_argument('--reference-run')
    p.add_argument('--new-communities')
    p.add_argument('--decision-level', choices=['community', 'city'])
    p.add_argument('--max-units', type=int)
    p.add_argument('--draws', type=int)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--output')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    d = asdict(RLSettings())
    if args.config:
        extra = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        if set(extra) - set(d):
            raise ValueError('Unknown settings: ' + str(set(extra) - set(d)))
        d.update(extra)
    for k, v in vars(args).items():
        if k in ('config', 'smoke') or v is None:
            continue
        d[{'draws': 'n_uncertainty_draws'}.get(k, k)] = v
    if args.smoke:
        d.update(smoke=True, max_units=1, n_uncertainty_draws=2)
    c = RLSettings(**d)
    if c.max_new_policies != 2:
        raise ValueError('This program implements at most two new policies.')
    if not (c.ridge_alpha > 0 and 0 < c.transition_test_fraction < 0.5 and (c.min_policy_adoptions >= 1)):
        raise ValueError('Invalid transition settings.')
    if c.n_uncertainty_draws < 2 or not 0 < c.reward_quantile <= 0.5:
        raise ValueError('Use at least two uncertainty draws and a reward quantile in (0,.5].')
    if not 0 < c.q_learning_rate <= 1 or c.q_replay_epochs < 1 or c.q_tolerance <= 0:
        raise ValueError('Invalid Q-learning settings.')
    if min(c.reward_eci_weight, c.reward_cpi_weight) < 0 or c.reward_eci_weight + c.reward_cpi_weight <= 0:
        raise ValueError('Invalid environment reward weights.')
    if min(c.policy_count_penalty, c.pair_extrapolation_penalty, c.support_freeze_penalty, c.o3_excess_penalty) < 0:
        raise ValueError('Penalty coefficients must be nonnegative.')
    if c.risk_scope not in ('all_graph', 'decision_unit'):
        raise ValueError('risk_scope must be all_graph or decision_unit.')
    if not c.o3_exceedance_limits or any((not 0 <= v <= 1 for v in c.o3_exceedance_limits)):
        raise ValueError('Invalid exceedance proportions.')
    if not c.scenario_targets or any((not 1 <= len(t) <= 4 or len(set(t)) != len(t) or (not set(t) <= set(OUTCOMES)) for t in c.scenario_targets)):
        raise ValueError('Targets must contain one to four different outcomes.')
    if c.similarity_top_k < 1 or not 0 < c.similarity_min_feature_fraction <= 1:
        raise ValueError('Invalid similarity settings.')
    if not c.att_min_pre_values or not set(c.att_min_pre_values) <= {1, 2, 3}:
        raise ValueError('att_min_pre_values must be drawn from 1,2,3.')
    if not c.att_features or len(set(c.att_features)) != len(c.att_features) or (not set(c.att_features) <= set(FEATURES)):
        raise ValueError('att_features must be unique saved-model feature names.')
    if c.att_uncertainty not in ('fixed_att_paired_dropout', 'independent_normal_sensitivity'):
        raise ValueError('Unsupported ATT uncertainty option.')
    if not 0 < c.att_min_complete_fraction <= 1 or c.att_min_treated < 1:
        raise ValueError('Invalid ATT completeness/sample settings.')
    if c.history_min_adoptions < 1 or c.history_max_post < 0 or c.history_min_rows < 2 or (c.history_min_noaddition_rows < 1) or (c.history_fallback_penalty < 0):
        raise ValueError('Invalid historical fallback settings.')
    if c.reward_mode not in ('mean', 'q10'):
        raise ValueError('reward_mode must be mean or q10.')
    if c.freeze_unsupported_updates:
        raise ValueError('This version clips each feature independently, never freezes whole vectors.')
    if c.decision_level != 'community' or c.risk_scope != 'decision_unit':
        raise ValueError('This version requires community decisions and community-only O3 risk denominator.')
    if not c.hard_o3_constraint:
        raise ValueError('O3 community-risk constraint must remain enabled.')
    if not np.isfinite(c.trend_budget_years) or c.trend_budget_years <= 0:
        raise ValueError('trend_budget_years must be finite and positive.')
    if not np.isfinite(c.reverse_trend_fraction) or not 0 <= c.reverse_trend_fraction <= 1:
        raise ValueError('reverse_trend_fraction must be between 0 and 1.')
    if c.trend_min_years < 2 or c.trend_zero_tolerance < 0 or (not np.isfinite(c.trend_zero_tolerance)):
        raise ValueError('Invalid trend estimation settings.')
    if c.desired_recommended_units < 0:
        raise ValueError('desired_recommended_units is an audit count, must be nonnegative.')
    return c

def policy_panel47(panel, base, ids, c, out):
    pol = read_csv(c.policy)
    if not {'Code', 'Year', *POLICIES} <= set(pol):
        raise ValueError('Policy panel columns missing.')
    pol['Code'] = codes(pol.Code)
    pol['Year'] = pd.to_numeric(pol.Year, errors='raise').astype(int)
    if pol.duplicated(['Code', 'Year']).any():
        raise ValueError('Duplicate policy Code-Year.')
    for name in POLICIES:
        pol[name] = pd.to_numeric(pol[name], errors='raise')
        if not pol[name].isin([0, 1]).all():
            raise ValueError('Policies must be complete binary indicators.')
    target = panel[['Code', 'Year'] + FEATURES].merge(pol[['Code', 'Year'] + POLICIES], on=['Code', 'Year'], how='left', validate='one_to_one')
    if target[POLICIES].isna().any().any():
        raise ValueError('Incomplete policy matches.')
    target = target.sort_values(['Year', 'Code']).copy()
    groups = target.groupby('Code', sort=False)
    for p in POLICIES:
        if (groups[p].diff().dropna() < 0).any():
            raise ValueError('Policy reversal detected; this persistent-policy environment does not model withdrawal.')
        target['age_' + p] = groups[p].cumsum().to_numpy()
        first = target.sort_values('Year').groupby('Code')[p].first()
        target['initial_' + p] = target.Code.map(first)
    current = target[target.Year.eq(c.year)].set_index('Code').reindex(ids)
    contexts = FEATURES + POLICIES + ['age_' + p for p in POLICIES] + ['initial_' + p for p in POLICIES]
    next_rows = target[['Code', 'Year'] + FEATURES + POLICIES].copy()
    next_rows['Year'] -= 1
    next_rows = next_rows.rename(columns={k: 'next_' + k for k in FEATURES + POLICIES})
    trans = target.merge(next_rows, on=['Code', 'Year'], how='inner', validate='one_to_one')
    before = trans[FEATURES].to_numpy(float)
    after = trans[['next_' + f for f in FEATURES]].to_numpy(float)
    actions = trans[['next_' + p for p in POLICIES]].to_numpy(int) - trans[POLICIES].to_numpy(int)
    if not np.isin(actions, [0, 1]).all():
        raise ValueError('Non-monotone action transitions.')
    apos = [FEATURES.index(f) for f in ADJUSTABLE]
    delta = after[:, apos] - before[:, apos]
    valid = np.isfinite(delta).all(axis=1)
    audit = trans[['Code', 'Year']].copy()
    audit['Target_Year'] = audit.Year + 1
    audit['New_Policy_Count'] = actions.sum(axis=1)
    audit['Valid_Target_Changes'] = valid
    audit.to_csv(out / '02_transition_row_audit.csv', index=False, encoding='utf-8-sig')
    trans = trans.loc[valid].reset_index(drop=True)
    actions, delta = (actions[valid], delta[valid])
    if len(trans) < 20:
        raise ValueError('Insufficient valid adjacent-year transition rows.')
    return (trans, actions, delta, current, contexts)

def evidence_context47(c, out):
    file = Path(c.mechanism_dir) / 'pathway_evidence_summary.csv'
    if file.exists():
        evidence = read_csv(file)
        evidence = evidence[evidence.min_pre.eq(c.evidence_min_pre)].copy()
        evidence['RL_Use'] = 'context_only;not_added_to_model_response;estimands_and_scales_must_not_be_mixed'
        evidence.to_csv(out / '06_prior_pathway_evidence_context.csv', index=False, encoding='utf-8-sig')
    file = Path(c.boundary_dir) / 'boundary_summary.csv'
    if file.exists():
        evidence = read_csv(file)
        evidence = evidence[evidence.min_pre.eq(c.evidence_min_pre)].copy()
        evidence['RL_Use'] = 'advisory_subgroup_evidence;not_a_deterministic_action_ban'
        evidence.to_csv(out / '07_prior_boundary_evidence_context.csv', index=False, encoding='utf-8-sig')

def verify_ATT_feature_scale(panel, c, out):
    manifest_path = Path(c.mechanism_dir) / 'analysis_manifest.json'
    if not manifest_path.exists():
        raise ValueError('Mechanism analysis manifest is required for scale verification.')
    metadata = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    source = metadata.get('configuration', {}).get('model_file')
    if not source or not Path(source).exists():
        raise ValueError('Original DID feature panel cannot be located from manifest.')
    original = read_csv(source, usecols=['Code', 'Year'] + list(c.att_features))
    original['Code'] = codes(original.Code)
    original['Year'] = pd.to_numeric(original.Year, errors='raise').astype(int)
    if original.duplicated(['Code', 'Year']).any():
        raise ValueError('Original DID panel has duplicate Code-Year.')
    merged = panel[['Code', 'Year'] + list(c.att_features)].merge(original, on=['Code', 'Year'], suffixes=('_model', '_DID'), how='inner', validate='one_to_one')
    if len(merged) != len(panel):
        raise ValueError('DID/model feature panels do not cover the same requested history.')
    report = []
    for f in c.att_features:
        a = pd.to_numeric(merged[f + '_model'], errors='coerce').to_numpy(float)
        b = pd.to_numeric(merged[f + '_DID'], errors='coerce').to_numpy(float)
        equal = np.isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=True)
        valid = np.isfinite(a) & np.isfinite(b)
        report.append({'Feature': f, 'Matched_Rows': len(a), 'Mismatches': int((~equal).sum()), 'Max_Absolute_Difference': float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else np.nan, 'DID_Panel': str(source), 'Prediction_Panel': c.panel})
    pd.DataFrame(report).to_csv(out / '29_ATT_input_scale_reconciliation.csv', index=False, encoding='utf-8-sig')
    if any((r['Mismatches'] for r in report)):
        raise ValueError('DID and model raw feature values differ. Check scaling/versions; see 29_ATT_input_scale_reconciliation.csv.')

class DIDMechanismATT:

    def __init__(self, c, out):
        self.c, self.out = (c, out)
        path = Path(c.mechanism_dir) / 'overall.csv'
        df = read_csv(path)
        required = {'design', 'min_pre', 'role', 'variable', 'att', 'status', 'unit', 'n_treated'}
        if not required <= set(df):
            raise ValueError('Mechanism ATT file lacks ' + str(required - set(df)))
        env_context = df[df.role.eq('paired_environment') & pd.to_numeric(df.min_pre, errors='raise').eq(c.active_att_min_pre)].copy()
        env_context['Use'] = 'paired_sample_total_environment_ATT_context_only_not_added_or_averaged_across_mechanisms'
        env_context.to_csv(out / '36_environment_ATT_context.csv', index=False, encoding='utf-8-sig')
        df = df[df.role.eq('mechanism') & pd.to_numeric(df.min_pre, errors='raise').eq(c.active_att_min_pre)].copy()
        df = df[df.variable.isin(c.att_features)].copy()
        if not df.unit.eq('original_indicator_units').all():
            raise ValueError('Mechanism ATT must be ORIGINAL feature units; do not use log or standardized ATT.')
        df['ATT_Source_File'] = str(path)
        if c.att_recipe_file:
            rp = read_csv(c.att_recipe_file)
            need = {'from_mask', 'to_mask', 'min_pre', 'variable', 'att', 'status', 'unit', 'n_treated'}
            if not need <= set(rp):
                raise ValueError('Recipe mechanism ATT requires ' + str(need))
            if 'role' in rp:
                rp = rp[rp.role.eq('mechanism')].copy()
            rp = rp[pd.to_numeric(rp.min_pre, errors='raise').eq(c.active_att_min_pre) & rp.variable.isin(c.att_features)].copy()
            for col in ('from_mask', 'to_mask'):
                values = pd.to_numeric(rp[col], errors='raise')
                if not ((values >= 0) & (values < 2 ** len(POLICIES)) & (values == np.floor(values))).all():
                    raise ValueError('Invalid portfolio bitmask.')
                rp[col] = values.astype(int)
            for a, b in rp[['from_mask', 'to_mask']].drop_duplicates().itertuples(index=False, name=None):
                if a & b != a or not 1 <= bin(b ^ a).count('1') <= 2:
                    raise ValueError('Recipe ATT must represent addition of one or two policies.')
            if not rp.unit.eq('original_indicator_units').all():
                raise ValueError('Recipe ATT is not in raw feature units.')
            rp['design'] = ['recipe_' + str(a) + '_to_' + str(b) for a, b in zip(rp.from_mask, rp.to_mask)]
            rp['design_kind'] = 'recipe'
            rp['role'] = 'mechanism'
            rp['ATT_Source_File'] = str(c.att_recipe_file)
            df = pd.concat([df, rp], ignore_index=True)
        if df.duplicated(['design', 'variable']).any():
            raise ValueError('Duplicate ATT rows: one overall estimate per design, min_pre and mechanism is required.')
        for col in ('att', 'se', 'n_treated', 'p_value', 'q_BH', 'pretrend_p'):
            if col not in df:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors='coerce')
        self.table = df
        self.vectors = {}
        self.rows = {}
        for design, g in df.groupby('design', sort=True):
            gg = g.set_index('variable').reindex(c.att_features)
            known = (gg.status.eq('estimated') & np.isfinite(gg.att) & gg.n_treated.ge(c.att_min_treated)).to_numpy()
            fraction = float(known.mean())
            valid = bool(fraction >= c.att_min_complete_fraction and known.any())
            values = np.where(known, gg.att.to_numpy(float), 0.0)
            valid_se = np.isfinite(gg.se.to_numpy(float)) & (gg.se.to_numpy(float) >= 0)
            errors = np.where(known & valid_se, gg.se.to_numpy(float), 0.0)
            self.vectors[str(design)] = {'point': values, 'se': errors, 'known': known, 'valid': valid, 'fraction': fraction, 'n_treated': float(gg.loc[known, 'n_treated'].min()) if known.any() else 0.0, 'quality_flags': ';'.join(sorted(set(g.get('quality_flags', pd.Series(dtype=str)).dropna().astype(str)))), 'pretrend_rejected': bool((g.pretrend_p < 0.05).any()), 'se_missing': bool((known & ~valid_se).any())}
            self.rows[str(design)] = gg
        used = df.copy()
        used['ATT_Min_Pre'] = c.active_att_min_pre
        used['Adjustment_Interpretation'] = 'raw_post_period_average_displacement_not_annual_rate'
        used.to_csv(out / '30_mechanism_ATT_source_rows.csv', index=False, encoding='utf-8-sig')
        audits = []
        for d in sorted(set(POLICIES) | set(self.vectors)):
            v = self.vectors.get(d)
            audits.append({'Design': d, 'ATT_Min_Pre': c.active_att_min_pre, 'Enabled': bool(v and v['valid']), 'Complete_Feature_Fraction': v['fraction'] if v else 0.0, 'Minimum_Source_Treated': v['n_treated'] if v else 0.0, 'Missing_Mechanisms': ';'.join((f for i, f in enumerate(c.att_features) if not v or not v['known'][i])), 'Source_Pretrend_Rejection': v['pretrend_rejected'] if v else False, 'Source_Quality_Flags': v['quality_flags'] if v else 'no_ATT_rows'})
        pd.DataFrame(audits).to_csv(out / '31_ATT_design_availability.csv', index=False, encoding='utf-8-sig')
        self.enabled = [i for i, p in enumerate(POLICIES) if p in self.vectors and self.vectors[p]['valid']]
        for d, v in self.vectors.items():
            if d.startswith('recipe_') and v['valid']:
                a, b = map(int, d.removeprefix('recipe_').split('_to_'))
                self.enabled.extend((i for i in range(len(POLICIES)) if (b ^ a) & 1 << i))
        self.enabled = sorted(set(self.enabled))
        self.single_counts = np.array([self.vectors.get(p, {}).get('n_treated', 0) for p in POLICIES])
        self.pair_counts = {p: 0 for p in combinations(range(len(POLICIES)), 2)}
        self.draw_cache = {}
        save_json(out / '32_ATT_method.json', {'active_min_pre': c.active_att_min_pre, 'features': list(c.att_features), 'DID_source': str(path), 'uncertainty': c.att_uncertainty, 'unknown_joint_covariance': 'not_recovered_from_marginal_SE_or_CI', 'default_interval': 'paired_dropout_conditional_on_fixed_ATT_not_total_uncertainty', 'pair_rule': 'exact_background_recipe_mechanism_ATT_required_for_two_actual_new_policies', 'no_single_ATT_addition': True, 'no_environment_ATT_double_counting': True, 'no_pooling_across_min_pre': True, 'missing_mechanism_rule': 'package_unavailable_by_default;partial_mode_freezes_missing_features_and_is_labeled', 'no_significance_filter': True, 'transportability': 'pooled_ATT_applied_to_new_context_is_an_assumption'})

    def plan(self, current, actions, count_additions=0):
        plans = []
        old = current[POLICIES].to_numpy(int)
        for i, code in enumerate(current.index.astype(str)):
            before = sum((int(v) << j for j, v in enumerate(old[i])))
            added = np.flatnonzero(actions[i] > 0).tolist()
            if count_additions:
                n = int(old[i].sum())
                d = f'count_{n}_to_{n + count_additions}'
                label = 'unspecified_policy_identities'
                applicable = True
            elif not added:
                d = 'STOP'
                label = 'none'
                applicable = False
            else:
                after = before | sum((1 << j for j in added))
                exact = f'recipe_{before}_to_{after}'
                d = exact if exact in self.vectors or len(added) == 2 else POLICIES[added[0]]
                label = '+'.join((POLICIES[j] for j in added))
                applicable = True
            v = self.vectors.get(d)
            okay = not applicable or bool(v and v['valid'])
            plans.append({'Code': code, 'ATT_Design': d, 'Actual_New_Policies': label, 'ATT_Min_Pre': self.c.active_att_min_pre, 'ATT_Available': okay, 'Applicable': applicable, 'ATT_Feature_Fraction': v['fraction'] if v else 1.0 if not applicable else 0.0, 'ATT_Source_Min_Treated': v['n_treated'] if v else 0.0, 'ATT_Reason': '' if okay else 'missing_exact_joint_mechanism_ATT' if d.startswith('recipe_') and d not in self.vectors else 'missing_or_incomplete_mechanism_ATT', 'ATT_Transport_Type': 'none' if not applicable else 'count_average_policy_identity_unspecified' if count_additions else 'exact_recipe_average_transport' if d.startswith('recipe_') else 'pooled_single_policy_average_transport', 'ATT_Quality_Flags': v['quality_flags'] if v else '', 'ATT_Pretrend_Rejected': v['pretrend_rejected'] if v else False, 'ATT_SE_Missing': v['se_missing'] if v else False})
        return pd.DataFrame(plans)

    def effect(self, current, actions, model_number=0, count_additions=0):
        plan = self.plan(current, actions, count_additions)
        result = np.zeros((len(current), len(self.c.att_features)))
        for i, r in plan.iterrows():
            if not r.Applicable or not r.ATT_Available:
                continue
            v = self.vectors[r.ATT_Design]
            key = (r.ATT_Design, model_number)
            if key not in self.draw_cache:
                value = v['point'].copy()
                if model_number and self.c.att_uncertainty == 'independent_normal_sensitivity':
                    token = f'{self.c.seed}:{self.c.active_att_min_pre}:{r.ATT_Design}:{model_number}'
                    seed = int(hashlib.sha256(token.encode()).hexdigest()[:16], 16)
                    value += np.random.default_rng(seed).normal(size=len(value)) * v['se']
                self.draw_cache[key] = value
            result[i] = self.draw_cache[key]
        return result

class HistoricalFallback:

    def __init__(self, panel, base, current, c, out, needed):
        self.c = c
        self.enabled = set()
        self.estimator = None
        self.ids = base.index.astype(str).tolist()
        self.position = {v: i for i, v in enumerate(self.ids)}
        self.features = list(c.att_features)
        self.effects = {}
        self.counts = np.zeros(len(POLICIES), int)
        years = sorted(panel.Year.unique())
        nt = len(years)
        nc = len(self.ids)
        if panel.duplicated(['Code', 'Year']).any():
            raise ValueError('Duplicate feature Code-Year.')
        keys = pd.MultiIndex.from_product([self.ids, years], names=['Code', 'Year'])
        if len(panel) != nc * nt or not keys.isin(pd.MultiIndex.from_frame(panel[['Code', 'Year']])).all():
            raise ValueError('Historical fallback requires a balanced panel.')
        if np.any(np.diff(np.asarray(years, dtype=int)) != 1):
            raise ValueError('Historical years must be consecutive.')
        ordered = panel.set_index(['Code', 'Year']).reindex(keys)
        values = ordered[self.features].to_numpy(float).reshape(nc, nt, -1)
        pol = read_csv(c.policy)
        pol['Code'] = codes(pol.Code)
        pol['Year'] = pd.to_numeric(pol.Year, errors='raise').astype(int)
        if pol.duplicated(['Code', 'Year']).any():
            raise ValueError('Duplicate policy Code-Year.')
        keys = pd.MultiIndex.from_product([self.ids, years], names=['Code', 'Year'])
        pol = pol.set_index(['Code', 'Year']).reindex(keys)
        if not pol[POLICIES].isin([0, 1]).all().all():
            raise ValueError('Incomplete historical policies.')
        p = pol[POLICIES].to_numpy(int).reshape(nc, nt, -1)
        if (np.diff(p, axis=1) < 0).any():
            raise ValueError('Policy withdrawals are outside the persistent-entry design.')
        contexts = []
        actions = []
        targets = []
        code_rows = []
        year_rows = []
        horizons = []
        for t in range(int(c.active_att_min_pre), nt):
            a = p[:, t] - p[:, t - 1]
            alive = np.ones(nc, bool)
            sums = np.zeros((nc, len(self.features)))
            counts = np.zeros_like(sums)
            h = np.zeros(nc, int)
            for s in range(t, min(nt, t + c.history_max_post + 1)):
                alive &= np.all(p[:, s] == p[:, t], axis=1)
                good = alive[:, None] & np.isfinite(values[:, s])
                sums += np.where(good, values[:, s], 0.0)
                counts += good
                h += alive
            target = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0) - values[:, t - 1]
            context = np.column_stack([values[:, t - 1], p[:, t - 1], p[:, :t].sum(axis=1), p[:, 0], np.full(nc, years[t - 1])])
            valid = np.isfinite(target).all(axis=1)
            contexts.append(context[valid])
            actions.append(a[valid])
            targets.append(target[valid])
            code_rows.extend(np.array(self.ids)[valid])
            year_rows.extend([years[t]] * int(valid.sum()))
            horizons.extend(h[valid])
        raw_counts = [int((np.diff(p[:, :, j], axis=1) > 0).any(axis=1).sum()) for j in range(len(POLICIES))]
        self.n_control = 0
        if contexts:
            x = np.concatenate(contexts)
            a = np.concatenate(actions)
            y = np.concatenate(targets)
            code_rows = np.array(code_rows)
            year_rows = np.array(year_rows)
            self.counts = np.array([len(np.unique(code_rows[a[:, j] > 0])) for j in range(len(POLICIES))])
            self.n_control = int((a.sum(axis=1) == 0).sum())
        else:
            x = np.empty((0, 1))
            a = np.empty((0, len(POLICIES)))
            y = np.empty((0, len(self.features)))
        enough = len(x) >= c.history_min_rows and self.n_control >= c.history_min_noaddition_rows
        eligible = {j for j in needed if self.counts[j] >= c.history_min_adoptions}
        support = []
        for j, name in enumerate(POLICIES):
            reason = 'DID_complete_no_backup_needed' if j not in needed else 'no_observed_new_entry' if raw_counts[j] == 0 else 'no_sufficient_min_pre_and_valid_target_entries' if j not in eligible else 'insufficient_model_training_or_noaddition_rows' if not enough else 'historical_association_backup_available'
            support.append({'Policy': name, 'ATT_Min_Pre': c.active_att_min_pre, 'Raw_New_Entry_Cities': raw_counts[j], 'Qualifying_New_Entry_Cities': int(self.counts[j]), 'Needs_Backup': j in needed, 'History_Enabled': bool(c.history_fallback and enough and (j in eligible)), 'Reason': reason if c.history_fallback else 'historical_backup_disabled'})
        pd.DataFrame(support).to_csv(out / '41_historical_fallback_support.csv', index=False, encoding='utf-8-sig')
        diagnostics = []
        if c.history_fallback and enough and eligible:
            if len(np.unique(code_rows)) >= 5:
                train, test = next(GroupShuffleSplit(n_splits=1, test_size=c.transition_test_fraction, random_state=c.seed).split(x, groups=code_rows))
                diagnostic = self.fit(x[train], a[train], y[train], c)
                predictions = self.predict(diagnostic, x[test], a[test])
                for j, f in enumerate(self.features):
                    diagnostics.append({'Validation': 'heldout_Code', 'Feature': f, 'N': len(test), 'MAE': mean_absolute_error(y[test, j], predictions[:, j]), 'Constant_Mean_MAE': mean_absolute_error(y[test, j], np.full(len(test), y[train, j].mean()))})
            early = np.flatnonzero(year_rows + np.asarray(horizons) - 1 <= c.temporal_train_end)
            late = np.flatnonzero(year_rows > c.temporal_train_end)
            if len(early) >= c.history_min_rows and len(late):
                diagnostic = self.fit(x[early], a[early], y[early], c)
                predictions = self.predict(diagnostic, x[late], a[late])
                for j, f in enumerate(self.features):
                    diagnostics.append({'Validation': 'later_entry_years_association_only', 'Feature': f, 'N': len(late), 'MAE': mean_absolute_error(y[late, j], predictions[:, j]), 'Constant_Mean_MAE': mean_absolute_error(y[late, j], np.full(len(late), y[early, j].mean()))})
            self.estimator = self.fit(x, a, y, c)
            self.enabled = eligible
            cx = np.column_stack([base[self.features].to_numpy(float), current[POLICIES].to_numpy(int), p.sum(axis=1), p[:, 0], np.full(nc, c.year)])
            none = np.zeros((nc, len(POLICIES)))
            baseline = self.predict(self.estimator, cx, none)
            for j in self.enabled:
                action = none.copy()
                action[:, j] = 1 - current[POLICIES[j]].to_numpy(int)
                self.effects[j] = self.predict(self.estimator, cx, action) - baseline
            import joblib
            joblib.dump(self.estimator, out / 'historical_fallback_model.joblib')
            pd.DataFrame({'Code': code_rows, 'Entry_Year': year_rows, 'Observed_Followup_Years': horizons, 'New_Policy_Count': a.sum(axis=1)}).to_csv(out / '43_historical_training_rows.csv', index=False, encoding='utf-8-sig')
        if not diagnostics:
            diagnostics = [{'Validation': 'not_performed', 'Reason': 'backup_not_needed_or_insufficient_support'}]
        pd.DataFrame(diagnostics).to_csv(out / '42_historical_prediction_checks.csv', index=False, encoding='utf-8-sig')
        save_json(out / '44_historical_method.json', {'min_pre': c.active_att_min_pre, 'features': self.features, 'enabled_policies': [POLICIES[j] for j in sorted(self.enabled)], 'target': 'available_post_entry_mean_level_minus_pre_entry_level_until_next_policy_change', 'max_post': c.history_max_post, 'noaddition_training_rows': self.n_control, 'prediction': 'fitted_delta(single_new_policy)-fitted_delta(no_new_policy)', 'causal_ATT': False, 'uncertainty': 'backup_parameters_fixed;not_propagated_by_dropout', 'horizon_caveat': 'available_followup_differs;not_identical_to_DID_cohort_weighted_ATT', 'validation_caveat': 'outcome_prediction_checks_not_counterfactual_policy_validation;early_followup_ends_before_temporal_cutoff'})

    @staticmethod
    def design(z, a):
        interaction = (a[:, :, None] * z[:, None, :]).reshape(len(z), -1)
        pair = np.column_stack([a[:, i] * a[:, j] for i, j in combinations(range(len(POLICIES)), 2)])
        return np.column_stack([z, a, interaction, pair])

    @classmethod
    def fit(cls, x, a, y, c):
        med = np.nanmedian(x, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        filled = np.where(np.isfinite(x), x, med)
        mu = filled.mean(axis=0)
        sd = filled.std(axis=0)
        sd = np.where(sd > 1e-10, sd, 1.0)
        ys = y.std(axis=0)
        ys = np.where(ys > 1e-10, ys, 1.0)
        model = Ridge(alpha=c.ridge_alpha).fit(cls.design((filled - mu) / sd, a), y / ys)
        return {'median': med, 'mean': mu, 'std': sd, 'target_std': ys, 'model': model}

    @classmethod
    def predict(cls, fit, x, a):
        z = (np.where(np.isfinite(x), x, fit['median']) - fit['mean']) / fit['std']
        return fit['model'].predict(cls.design(z, a)) * fit['target_std']

class HybridMechanism(DIDMechanismATT):

    def __init__(self, c, out, panel, base, current):
        super().__init__(c, out)
        needed = {j for j, p in enumerate(POLICIES) if p not in self.vectors or not self.vectors[p]['known'].all()}
        self.history = HistoricalFallback(panel, base, current, c, out, needed)
        self.enabled = list(range(len(POLICIES)))
        self.cache = {}
        availability = []
        for j, p in enumerate(POLICIES):
            v = self.vectors.get(p)
            availability.append({'Policy': p, 'ATT_Min_Pre': c.active_att_min_pre, 'Pooled_Single_ATT_Feature_Fraction': v['fraction'] if v else 0.0, 'History_Backup_Available': j in self.history.enabled, 'Qualifying_Historical_Adopters': int(self.history.counts[j]), 'Interpretation': 'candidate_menu_only;exact_background_and_all_mechanisms_checked_per_city'})
        pd.DataFrame(availability).to_csv(out / '45_final_action_sources.csv', index=False, encoding='utf-8-sig')
        save_json(out / '32_ATT_method.json', {'active_min_pre': c.active_att_min_pre, 'priority': ['matching_joint_mechanism_ATT', 'same_min_pre_single_components_under_additivity', 'historical_association_only_for_missing_single_ATT_with_new_entries', 'unavailable'], 'fallback_per_indicator': True, 'pair_additive_fallback': c.pair_additive_fallback, 'no_environment_ATT_addition': True, 'uncertainty': c.att_uncertainty, 'interpretation': 'hybrid_conditional_simulation_not_all_outputs_are_ATT', 'missing_SE': 'point_effect_retained;unpropagated_uncertainty_flagged', 'no_joint_covariance_reconstruction': True})

    def begin_candidate(self):
        self.cache.clear()

    def single(self, code, before, j, k):
        for design in (f'recipe_{before}_to_{before | 1 << j}', POLICIES[j]):
            v = self.vectors.get(design)
            if v is not None and v['known'][k]:
                return (v['point'][k], [(design, k)], 'single_ATT', v['n_treated'])
        if j in self.history.enabled and code in self.history.position:
            value = self.history.effects[j][self.history.position[code], k]
            if np.isfinite(value):
                return (value, [], 'history_model', float(self.history.counts[j]))
        return (np.nan, [], 'unavailable', 0.0)

    def resolve(self, code, old, added, count_additions):
        before = sum((int(v) << j for j, v in enumerate(old)))
        key = (code, before, tuple(added), count_additions)
        if key in self.cache:
            return self.cache[key]
        applicable = bool(added or count_additions)
        nf = len(self.c.att_features)
        values = np.zeros(nf)
        sources = []
        parts = []
        sample = []
        designs = set()
        hpol = set()
        if count_additions:
            design = f'count_{sum(old)}_to_{sum(old) + count_additions}'
        elif not added:
            design = 'STOP'
        else:
            design = f'recipe_{before}_to_{before | sum((1 << j for j in added))}'
        joint = self.vectors.get(design)
        for k, f in enumerate(self.c.att_features):
            component = []
            n = 0.0
            if not applicable:
                source = 'none'
                value = 0.0
            elif joint is not None and joint['known'][k]:
                source = 'count_ATT' if count_additions else 'joint_ATT' if len(added) == 2 else 'single_ATT'
                value = joint['point'][k]
                component = [(design, k)]
                n = joint['n_treated']
            elif count_additions:
                value = np.nan
                source = 'unavailable'
            elif len(added) == 1:
                value, component, source, n = self.single(code, before, added[0], k)
                if source == 'history_model':
                    hpol.add(POLICIES[added[0]])
            elif self.c.pair_additive_fallback:
                singles = [self.single(code, before, j, k) for j in added]
                if all((np.isfinite(s[0]) for s in singles)):
                    value = sum((s[0] for s in singles))
                    component = [part for s in singles for part in s[1]]
                    nh = sum((s[2] == 'history_model' for s in singles))
                    source = 'additive_ATT' if nh == 0 else 'additive_history' if nh == len(singles) else 'additive_mixed'
                    n = min((s[3] for s in singles))
                    hpol.update((POLICIES[j] for j, s in zip(added, singles) if s[2] == 'history_model'))
                else:
                    value = np.nan
                    source = 'unavailable'
            else:
                value = np.nan
                source = 'unavailable'
            values[k] = value
            sources.append(source)
            parts.append(component)
            sample.append(n)
            designs.update((d for d, _ in component))
        known = np.isfinite(values)
        fraction = float(known.mean())
        valid = bool(known.any() and fraction >= self.c.att_min_complete_fraction)
        quality = []
        rejected = False
        missing_se = False
        for d in designs:
            v = self.vectors[d]
            quality.append(v['quality_flags'])
            rejected |= v['pretrend_rejected']
            missing_se |= v['se_missing']
        counts = {s: sources.count(s) for s in sorted(set(sources))}
        nh = sum((s in ('history_model', 'additive_history', 'additive_mixed') for s in sources))
        na = sum((s.startswith('additive_') for s in sources))
        did_sources = ';'.join(sorted(designs))
        if not count_additions and len(added) == 1 and (joint is None or not joint['known'].any()):
            design = POLICIES[added[0]]
        meta = {'Code': code, 'ATT_Design': design, 'ATT_Component_Designs': did_sources, 'Actual_New_Policies': 'unspecified_policy_identities' if count_additions else '+'.join((POLICIES[j] for j in added)) or 'none', 'ATT_Min_Pre': self.c.active_att_min_pre, 'ATT_Available': valid, 'Mechanism_Effect_Available': valid, 'Applicable': applicable, 'ATT_Feature_Fraction': fraction, 'ATT_Source_Min_Treated': min([sample[k] for k in range(nf) if known[k] and sample[k] > 0], default=0.0), 'ATT_Reason': '' if valid else 'insufficient_mechanism_evidence_after_ATT_and_history_fallback', 'Missing_Mechanisms': ';'.join((f for k, f in enumerate(self.c.att_features) if not known[k])), 'ATT_Transport_Type': ';'.join(sorted(set(sources))), 'Effect_Source_Mode': ';'.join(sorted(set(sources))), 'Effect_Source_Counts': json.dumps(counts, ensure_ascii=False), 'History_Policies': ';'.join(sorted(hpol)), 'Mechanism_Source_JSON': json.dumps(dict(zip(self.c.att_features, sources)), ensure_ascii=False), 'History_Mechanisms': ';'.join((f for f, s in zip(self.c.att_features, sources) if s in ('history_model', 'additive_history', 'additive_mixed'))), 'Additive_Mechanisms': ';'.join((f for f, s in zip(self.c.att_features, sources) if s.startswith('additive_'))), 'N_History_Mechanisms': nh, 'N_Additive_Mechanisms': na, 'ATT_Quality_Flags': ';'.join(sorted(set(quality) - {''})), 'ATT_Pretrend_Rejected': rejected, 'ATT_SE_Missing': missing_se}
        result = {'point': np.where(known, values, 0.0), 'parts': parts, 'meta': meta}
        self.cache[key] = result
        return result

    def plan(self, current, actions, count_additions=0):
        return pd.DataFrame([self.resolve(str(code), current.iloc[i][POLICIES].to_numpy(int), np.flatnonzero(actions[i] > 0).tolist(), count_additions)['meta'] for i, code in enumerate(current.index)])

    def effect(self, current, actions, model_number=0, count_additions=0):
        output = np.zeros((len(current), len(self.c.att_features)))
        for i, code in enumerate(current.index):
            r = self.resolve(str(code), current.iloc[i][POLICIES].to_numpy(int), np.flatnonzero(actions[i] > 0).tolist(), count_additions)
            if not r['meta']['Applicable'] or not r['meta']['ATT_Available']:
                continue
            value = r['point'].copy()
            if model_number and self.c.att_uncertainty == 'independent_normal_sensitivity':
                for k, components in enumerate(r['parts']):
                    for d, j in components:
                        token = (d, model_number)
                        if token not in self.draw_cache:
                            seed = int(hashlib.sha256(f'{self.c.seed}:{self.c.active_att_min_pre}:{d}:{model_number}'.encode()).hexdigest()[:16], 16)
                            self.draw_cache[token] = np.random.default_rng(seed).normal(size=len(value)) * self.vectors[d]['se']
                        value[k] += self.draw_cache[token][j]
            output[i] = value
        return output

class TwoPolicyEnvironment47:

    def __init__(self, predictor, space, transition, current, base, c, out):
        self.p, self.space, self.t, self.current, self.base, self.c, self.out = (predictor, space, transition, current, base, c, out)
        self.apos = [FEATURES.index(f) for f in ADJUSTABLE]
        self.seeds = [c.seed + 900000 + i for i in range(c.n_uncertainty_draws)]
        self.mc_baseline = [predictor.predict(predictor.base_raw, seed) for seed in self.seeds]
        self.enabled = transition.enabled

    def available(self, idx):
        covered = self.current.iloc[idx][POLICIES].to_numpy(int)
        return [j for j in self.enabled if (covered[:, j] == 0).any()]

    def evaluate(self, unit, group, idx, selected, count_additions=0):
        self.t.begin_candidate()
        selected = tuple(sorted(selected))
        names = [POLICIES[j] for j in selected]
        portfolio = '+'.join(names) if names else 'STOP'
        old = self.p.base_raw
        covered = self.current.iloc[idx][POLICIES].to_numpy(int)
        a = np.zeros_like(covered)
        if selected:
            a[:, selected] = 1 - covered[:, selected]
        if count_additions:
            a[:] = 0
            a[:, :count_additions] = 1
            portfolio = f'COUNT_ADD_{count_additions}'
        att_plan = self.t.plan(self.current.iloc[idx], a, count_additions)
        att_complete = bool(not selected or (att_plan.Applicable & att_plan.ATT_Available).any())
        pair_unsupported = False
        pair_count = np.nan
        if len(selected) == 2 and (not count_additions) and (a[:, selected].sum(axis=1) == 2).any():
            pair_plan = att_plan[a.sum(axis=1) == 2]
            direct = pair_plan.Effect_Source_Mode.eq('joint_ATT')
            pair_count = float(pair_plan.loc[direct, 'ATT_Source_Min_Treated'].min()) if direct.any() else np.nan
            pair_unsupported = bool(pair_plan.N_Additive_Mechanisms.gt(0).any())
        applicable = a.sum(axis=1) > 0
        cost = self.c.policy_count_penalty * float(a.sum(axis=1).mean())

        def transform(model_num):
            raw = old.copy()
            proposed = old[idx].copy()
            effect = self.t.effect(self.current.iloc[idx], a, model_num, count_additions)
            proposed[:, self.apos] = proposed[:, self.apos] + effect
            proposed[~applicable] = old[idx][~applicable]
            new, scale, difficulty = self.space.constrain(group, idx, proposed, ADJUSTABLE)
            changed = np.any(~np.isclose(new, old[idx], rtol=0, atol=1e-12, equal_nan=True), axis=1)
            supported, ratio = self.space.check(group, new)
            frozen = changed & ~supported
            if self.c.freeze_unsupported_updates:
                new[frozen] = old[idx][frozen]
                scale[frozen] = 0
            raw[idx] = new
            actual = np.any(~np.isclose(new, old[idx], rtol=0, atol=1e-12, equal_nan=True), axis=1)
            return (raw, actual, frozen, scale, difficulty, ratio)
        if selected and att_complete:
            raw, changed, frozen, scale, difficulty, ratio = transform(0)
            point = self.p.predict(raw) if changed.any() else self.p.baseline.copy()
        else:
            raw = old.copy()
            changed = frozen = np.zeros(len(idx), bool)
            scale = np.ones(len(idx))
            difficulty = np.zeros(len(idx))
            ratio = np.zeros(len(idx))
            point = self.p.baseline.copy()
        imp = 100 * (1 - point / self.p.baseline)
        excess = -imp[:, 2] - self.p.o3_thresholds
        means = imp[idx].mean(axis=0)
        freeze_rate = float(frozen.mean())
        risk = max(0.0, float(excess.max()))
        penalty = cost + self.c.pair_extrapolation_penalty * pair_unsupported + self.c.support_freeze_penalty * freeze_rate
        utility_mean = self.c.reward_eci_weight * means[0] + self.c.reward_cpi_weight * means[1] - penalty - self.c.o3_excess_penalty * risk
        draws = []
        utilities = []
        violations = []
        fraction_draws = []
        risk_idx = np.arange(len(old)) if self.c.risk_scope == 'all_graph' else idx
        for k, seed in enumerate(self.seeds):
            if selected and att_complete:
                mcraw, mcchanged, *_ = transform(k + 1)
                predicted = self.p.predict(mcraw, seed) if mcchanged.any() else self.mc_baseline[k]
                di = 100 * (1 - predicted / self.mc_baseline[k])
            else:
                di = np.zeros_like(self.p.baseline)
            dm = di[idx].mean(axis=0)
            de = -di[:, 2] - self.p.o3_thresholds
            draws.append(di[idx])
            violations.append(float(de.max()))
            fraction_draws.append(float(np.mean(de[risk_idx] > self.c.risk_numeric_tolerance_pct)))
            utilities.append(self.c.reward_eci_weight * dm[0] + self.c.reward_cpi_weight * dm[1] - penalty - self.c.o3_excess_penalty * max(0.0, float(de.max())))
        draws = np.stack(draws)
        robust = float(np.quantile(utilities, self.c.reward_quantile))
        permitted = not selected or (changed.any() and (not pair_unsupported or self.c.allow_pair_extrapolation) and (not self.c.hard_o3_constraint or risk <= 1e-06))
        reasons = []
        if selected and (not changed.any()):
            reasons.append('no_actual_supported_input_change')
        if pair_unsupported:
            reasons.append('unobserved_or_sparse_simultaneous_pair')
        if risk > 1e-06:
            reasons.append('O3_exceeds_historical_limit_somewhere_in_graph')
        if frozen.any():
            reasons.append('some_input_updates_frozen')
        if not selected:
            robust = utility_mean = 0.0
        rec = {'Decision_Unit': str(unit), 'Decision_Level': self.c.decision_level, 'Community_ID': int(group), 'Portfolio': portfolio, 'N_Selected_Policies': len(selected), 'N_Cities': len(idx), 'N_Eligible_Cities': int(applicable.sum()), 'N_Changed_Cities': int(changed.sum()), 'N_Frozen_Updates': int(frozen.sum()), 'Mean_Applied_Scale': float(scale.mean()), 'Historical_Pair_Adoptions': pair_count, 'Pair_Extrapolated': bool(pair_unsupported), 'Mean_Reward': float(utility_mean), 'Robust_Reward': robust, 'Feasible': bool(permitted), 'Policy_Burden_Proxy': cost, 'O3_Exceeding_Cities_All_Graph': int((excess > 1e-06).sum()), 'Max_O3_Excess_All_Graph_pct_points': float(excess.max()), 'MC_Graph_O3_Within_Limit_Frequency': float(np.mean(np.array(violations) <= 1e-06)), 'Reasons': ';'.join(reasons), 'Evidence_Type': 'model_based_conditional_policy_simulation_not_causal_ATT'}
        rec.update(ATT_Min_Pre=self.c.active_att_min_pre, ATT_Complete=att_complete, Joint_ATT_Source_Min_Treated=pair_count, Historical_Pair_Adoptions=np.nan, N_ATT_Missing_Cities=int((~att_plan.ATT_Available).sum()), ATT_Designs=';'.join(sorted(set(att_plan.ATT_Design) - {'STOP'})), ATT_Missing_Reasons=';'.join(sorted(set(att_plan.ATT_Reason) - {''})), ATT_Estimated_Feature_Min_Fraction=float(att_plan.ATT_Feature_Fraction.min()), ATT_Transport_Interpretation='count_average_not_named_package' if count_additions else 'hybrid_mechanism_guided_simulation', Effect_Source_Modes=';'.join(sorted(set(att_plan.Effect_Source_Mode))), ATT_Component_Designs=';'.join(sorted(set(';'.join(att_plan.ATT_Component_Designs).split(';')) - {''})), N_History_Backed_Cities=int(att_plan.N_History_Mechanisms.gt(0).sum()), History_Backed_City_Fraction=float(att_plan.N_History_Mechanisms.gt(0).mean()), N_Additive_Cities=int(att_plan.N_Additive_Mechanisms.gt(0).sum()), ATT_Uncertainty=self.c.att_uncertainty, Any_ATT_Pretrend_Rejected=bool(att_plan.ATT_Pretrend_Rejected.any()), Evidence_Type='hybrid_ATT_or_history_guided_model_response_not_total_causal_effect')
        detail = pd.DataFrame({'Code': np.array(self.p.ids)[idx], 'Decision_Unit': str(unit), 'Community_ID': group, 'Portfolio': portfolio, 'New_Policies_For_City': ['+'.join((POLICIES[j] for j in selected if a[i, j])) or 'none' for i in range(len(idx))], 'Own_Inputs_Changed': changed, 'Input_Update_Frozen': frozen, 'Policy_Count_For_City': a.sum(axis=1), 'Applied_Scale': scale, 'Support_Distance_Ratio': ratio, 'O3_Budget_pct': self.p.o3_thresholds[idx]})
        if count_additions:
            detail['New_Policies_For_City'] = 'unspecified_policy_identities'
        detail = detail.merge(att_plan.drop(columns='Actual_New_Policies'), on='Code', how='left', validate='one_to_one')
        requested = self.t.effect(self.current.iloc[idx], a, 0, count_additions)
        mechanism_columns = {}
        for j, f in enumerate(ADJUSTABLE):
            known = ~att_plan.Missing_Mechanisms.str.split(';').apply(lambda xs: f in xs)
            mechanism_columns[f + '_Mechanism_requested_delta'] = np.where(known, requested[:, j], np.nan)
            mechanism_columns[f + '_Applied_delta'] = raw[idx, FEATURES.index(f)] - old[idx, FEATURES.index(f)]
        detail = detail.copy()
        means_draw = draws.mean(axis=1)
        detail = pd.concat([detail, pd.DataFrame(mechanism_columns, index=detail.index)], axis=1).copy()
        diagnostics = self.space.audit(group, idx, raw[idx], requested)
        detail = pd.concat([detail, diagnostics], axis=1).copy()
        detail['Unmodelled_Paths_Held_At_Baseline'] = detail.Missing_Mechanisms.ne('')
        rec['N_Partial_Mechanism_Cities'] = int((att_plan.Applicable & att_plan.Missing_Mechanisms.ne('')).sum())
        rec['N_Outside_Training_Range_Cities'] = int(diagnostics.Outside_Training_Range.sum())
        rec['N_Legal_Bound_Clipped_Cities'] = int(diagnostics.Legal_Bound_Clipped.sum())
        rec['Input_Constraint_Mode'] = 'county_feature_trend_budget_then_community_history_minmax'
        for label in ('Trend_Budget_Capped', 'Reverse_Trend', 'Trend_Unavailable', 'Flat_Trend', 'Baseline_Missing', 'Community_Range_Clipped', 'Community_Range_Unavailable'):
            rec['N_' + label + '_Cities'] = int(diagnostics[label].sum())
        rec['Trend_Budget_Years'] = self.c.trend_budget_years
        rec['Reverse_Trend_Fraction'] = self.c.reverse_trend_fraction
        rec['Improvement_Mean_Draws_JSON'] = json.dumps(means_draw.tolist())
        rec['O3_Exceedance_Fraction_Draws_JSON'] = json.dumps(fraction_draws)
        rec['O3_Exceedance_Fraction'] = float(np.mean(excess[risk_idx] > self.c.risk_numeric_tolerance_pct))
        rec['O3_Risk_Denominator'] = len(risk_idx)
        rec['O3_Risk_Scope'] = self.c.risk_scope
        rec['O3_Exceeding_Cities_Risk_Scope'] = int((excess[risk_idx] > self.c.risk_numeric_tolerance_pct).sum())
        for j, y in enumerate(OUTCOMES):
            rec[y + '_change_pct'] = -float(means[j])
            rec[y + '_change_p05'] = float(np.quantile(-means_draw[:, j], 0.05))
            rec[y + '_change_p95'] = float(np.quantile(-means_draw[:, j], 0.95))
            detail[y + '_change_pct'] = -imp[idx, j]
            detail[y + '_change_p05'] = np.quantile(-draws[:, :, j], 0.05, axis=0)
            detail[y + '_change_p95'] = np.quantile(-draws[:, :, j], 0.95, axis=0)
            detail[y + '_predicted_baseline'] = self.p.baseline[idx, j]
            detail[y + '_predicted_scenario'] = point[idx, j]
        detail = detail.copy()
        detail['O3_Exceeds_Local_Limit'] = excess[idx] > 1e-06
        detail['Interval_Type'] = self.c.att_uncertainty + '_not_causal_CI'
        if not att_complete:
            for y in OUTCOMES:
                for suffix in ('_change_pct', '_change_p05', '_change_p95'):
                    rec[y + suffix] = np.nan
                for suffix in ('_change_pct', '_change_p05', '_change_p95', '_predicted_scenario'):
                    detail[y + suffix] = np.nan
            detail['Own_Inputs_Changed'] = False
            detail['Response_Status'] = 'package_response_unavailable_due_to_missing_ATT'
            rec['O3_Exceedance_Fraction'] = np.nan
            rec['MC_Graph_O3_Within_Limit_Frequency'] = np.nan
        else:
            detail['Response_Status'] = 'partial_path_conditional_simulation_unmodelled_paths_fixed'
        return (rec, detail)

def score_scenario(raw, targets, limit, c):
    r = dict(raw)
    p = [OUTCOMES.index(y) for y in targets]
    draws = np.asarray(json.loads(r.pop('Improvement_Mean_Draws_JSON')), float)
    riskdraw = np.asarray(json.loads(r.pop('O3_Exceedance_Fraction_Draws_JSON')), float)
    target_draw = draws[:, p].mean(axis=1)
    benefit = -np.mean([r[y + '_change_pct'] for y in targets])
    count_penalty = float(r['Policy_Burden_Proxy'])
    pair_penalty = c.pair_extrapolation_penalty * bool(r['Pair_Extrapolated'])
    freeze_penalty = c.support_freeze_penalty * r['N_Frozen_Updates'] / r['N_Cities']
    history_penalty = c.history_fallback_penalty * r.get('History_Backed_City_Fraction', 0.0)
    fixed = count_penalty + pair_penalty + freeze_penalty + history_penalty
    risk_penalty = c.o3_excess_penalty * 100 * max(0.0, r['O3_Exceedance_Fraction'] - limit)
    rewarddraw = target_draw - fixed - c.o3_excess_penalty * 100 * np.maximum(0.0, riskdraw - limit)
    stop = r['N_Selected_Policies'] == 0
    base_ok = stop or (r.get('ATT_Complete', True) and r['N_Changed_Cities'] > 0 and (not r['Pair_Extrapolated'] or c.allow_pair_extrapolation))
    risk_available = bool(np.isfinite(r['O3_Exceedance_Fraction']))
    risk_ok = risk_available and r['O3_Exceedance_Fraction'] <= limit + 1e-12
    reasons = []
    if not r.get('ATT_Complete', True):
        reasons.append(r.get('ATT_Missing_Reasons', 'missing_ATT'))
    if not stop and r['N_Changed_Cities'] == 0 and r.get('ATT_Complete', True):
        reasons.append('no_actual_change_after_independent_constraints')
        if r.get('N_Flat_Trend_Cities', 0):
            reasons.append('flat_trend_blocks_some_requested_features')
        if r.get('N_Trend_Unavailable_Cities', 0):
            reasons.append('missing_trend_blocks_some_requested_features')
        if r.get('N_Community_Range_Clipped_Cities', 0):
            reasons.append('community_history_bounds_clip_requested_features')
    if r['Pair_Extrapolated']:
        reasons.append('pair_extrapolation')
    if not risk_available:
        reasons.append('O3_risk_not_estimable')
    elif not risk_ok:
        reasons.append('O3_exceedance_proportion_above_budget')
    if r['N_Frozen_Updates']:
        reasons.append('some_updates_frozen')
    r.update(Targets='+'.join(targets), O3_Exceedance_Limit=limit, Target_Improvement_Mean=float(benefit), Target_Improvement_Q10=float(np.quantile(target_draw, 0.1)), Policy_Count_Penalty=count_penalty, Pair_Extrapolation_Penalty=pair_penalty, Frozen_Update_Penalty=freeze_penalty, History_Fallback_Penalty=history_penalty, O3_Proportion_Penalty=risk_penalty, Mean_Reward=float(benefit - fixed - risk_penalty), Robust_Reward=float(np.quantile(rewarddraw, c.reward_quantile)), Target_Improvement_Conservative=float(np.quantile(target_draw, c.reward_quantile)), MC_Risk_Budget_Pass_Frequency=float(np.mean(riskdraw <= limit + 1e-12)), O3_Allowed_Exceeding_Cities=int(np.floor(limit * r['O3_Risk_Denominator'] + 1e-12)), Feasible=bool(base_ok and (risk_ok or not c.hard_o3_constraint)), Reasons=';'.join(reasons))
    if c.reward_mode == 'mean':
        r['Robust_Reward'] = r['Mean_Reward']
    r['Reward_Mode'] = c.reward_mode
    r['Selection_Reward'] = r['Robust_Reward']
    if stop:
        r.update(Mean_Reward=0.0, Robust_Reward=0.0, Selection_Reward=0.0, Feasible=True)
    if not r.get('ATT_Complete', True):
        r.update(Mean_Reward=np.nan, Robust_Reward=np.nan, Target_Improvement_Mean=np.nan, Target_Improvement_Q10=np.nan, Target_Improvement_Conservative=np.nan, Feasible=False)
    if not all((np.isfinite(r[y + '_change_pct']) for y in OUTCOMES)):
        r.update(Feasible=False, Mean_Reward=np.nan, Robust_Reward=np.nan, Selection_Reward=np.nan)
        r['Reasons'] += ';nonfinite_or_unavailable_model_response'
    r['Selection_Reward'] = r['Robust_Reward']
    return r
SIM_BLOCKS = {'development': FEATURES[:10], 'energy': FEATURES[10:17], 'BG': FEATURES[17:29], 'IGT': FEATURES[29:], 'environment': OUTCOMES, 'policy_coverage': POLICIES}

def export_reference(base, current, units, out):
    rows = []
    for unit, group, idx in units:
        row = {'Unit_ID': str(unit), 'Community_ID': group, 'N_Cities': len(idx)}
        for f in FEATURES + OUTCOMES:
            row[f] = float(base.iloc[idx][f].mean())
        for f in POLICIES:
            row[f] = float(current.iloc[idx][f].mean())
        rows.append(row)
    ref = pd.DataFrame(rows)
    ref.to_csv(out / '20_similarity_reference_profiles.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(columns=['Unit_ID'] + FEATURES + OUTCOMES + POLICIES).to_csv(out / '21_new_community_input_template.csv', index=False, encoding='utf-8-sig')
    return ref

def similarity_match(ref, query, recommendations, c, out):
    ref = ref.copy()
    query = query.copy()
    for df in (ref, query):
        if 'Unit_ID' not in df:
            raise ValueError('Similarity input requires Unit_ID.')
        df['Unit_ID'] = df.Unit_ID.astype(str)
        if df.Unit_ID.duplicated().any():
            raise ValueError('Duplicate similarity Unit_ID.')
    cols = FEATURES + OUTCOMES + (POLICIES if c.similarity_include_policy else [])
    x = ref.reindex(columns=cols).apply(pd.to_numeric, errors='coerce').to_numpy(float)
    z = query.reindex(columns=cols).apply(pd.to_numeric, errors='coerce').to_numpy(float)
    for frame in (ref, query):
        coverage = frame.reindex(columns=POLICIES).apply(pd.to_numeric, errors='coerce').to_numpy(float)
        if np.any(np.isfinite(coverage) & ((coverage < 0) | (coverage > 1))):
            raise ValueError('Policy coverage must be fractions in [0,1], not percentages.')
    med = np.nanmedian(x, axis=0)
    sd = np.nanstd(x, axis=0)
    active = np.isfinite(med) & (sd > 1e-10)
    if not active.any():
        raise ValueError('No varying reference features for similarity.')
    scale = np.where(active, sd, 1.0)
    xs = (x - med) / scale
    zs = (z - med) / scale
    blocks = {b: [cols.index(f) for f in fs if f in cols and active[cols.index(f)]] for b, fs in SIM_BLOCKS.items()}
    save_json(out / '22_similarity_scaler.json', {'columns': cols, 'median': med.tolist(), 'reference_std': scale.tolist(), 'active': active.tolist(), 'blocks': blocks, 'metric': 'equal_block_weight_standardized_RMS', 'similarity': 'exp(-distance)', 'caveat': 'score_not_probability_or_policy_effect;matching_requires_harmonized_units'})

    def distances(row):
        result = []
        for xr in xs:
            valid = np.isfinite(row) & np.isfinite(xr) & active
            fraction = valid.sum() / active.sum()
            bd = []
            for indices in blocks.values():
                if not indices:
                    continue
                keep = [j for j in indices if valid[j]]
                if len(keep) < max(1, int(np.ceil(len(indices) * 0.5))):
                    bd = []
                    break
                bd.append(float(np.mean((row[keep] - xr[keep]) ** 2)))
            distance = np.sqrt(np.mean(bd)) if bd and fraction >= c.similarity_min_feature_fraction else np.inf
            result.append((distance, fraction))
        return result
    loo = []
    for i, row in enumerate(xs):
        ds = [d for j, (d, _) in enumerate(distances(row)) if i != j and np.isfinite(d)]
        if ds:
            loo.append(min(ds))
    cutoff = float(np.quantile(loo, 0.95)) if len(loo) >= 5 else np.nan
    matches = []
    suggestions = []
    for i, row in enumerate(zs):
        ranked = sorted(enumerate(distances(row)), key=lambda a: a[1][0])
        valid = [(j, d, f) for j, (d, f) in ranked if np.isfinite(d)][:c.similarity_top_k]
        if not valid:
            matches.append({'Query_Unit_ID': query.iloc[i].Unit_ID, 'Match_Status': 'insufficient_comparable_features'})
            continue
        for rank, (j, d, fraction) in enumerate(valid, 1):
            reference = ref.iloc[j].Unit_ID
            flag = 'reference_range_warning' if np.isfinite(cutoff) and d > cutoff else 'within_empirical_distance_range' if np.isfinite(cutoff) else 'distance_cutoff_unavailable'
            match = {'Query_Unit_ID': query.iloc[i].Unit_ID, 'Reference_Unit_ID': reference, 'Rank': rank, 'Distance': d, 'Similarity_Score': float(np.exp(-d)), 'Comparable_Feature_Fraction': fraction, 'Reference_Nearest_Distance_Q95': cutoff, 'Match_Status': flag}
            constant = np.isfinite(med) & ~active & np.isfinite(z[i])
            match['Reference_Constant_Feature_Mismatches'] = ';'.join((cols[k] for k in range(len(cols)) if constant[k] and (not np.isclose(z[i, k], med[k], rtol=1e-05, atol=1e-08))))
            if match['Reference_Constant_Feature_Mismatches']:
                match['Match_Status'] += ';constant_reference_feature_mismatch'
            matches.append(match)
            if recommendations is None:
                continue
            for _, r in recommendations[recommendations.Decision_Unit.astype(str).eq(reference)].iterrows():
                policies = [] if r.Portfolio == 'STOP' else r.Portfolio.split('+')
                fully_covered = [p for p in policies if p in query and pd.notna(query.iloc[i][p]) and (float(query.iloc[i][p]) >= 1 - 1e-10)]
                suggestions.append({**match, 'Scenario_ID': r.Scenario_ID, 'Reference_Portfolio': r.Portfolio, 'Reference_Targets': r.Targets, 'Reference_Robust_Reward': r.Robust_Reward, 'Reference_Recommendation_Status': r.Recommendation_Status, 'Already_Fully_Covered_In_Query': ';'.join(fully_covered), 'Candidate_Policies_For_Local_Review': '+'.join((p for p in policies if p not in fully_covered)) or 'none', 'Transfer_Status': 'analogy_only_local_policy_equivalence_and_effect_validation_required'})
    pd.DataFrame(matches).to_csv(out / '23_new_community_similarity_matches.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(suggestions).to_csv(out / '24_analog_policy_references.csv', index=False, encoding='utf-8-sig')

def q_learning_package(records, available, c):
    u = {m: float(r['Robust_Reward']) if np.isfinite(r['Robust_Reward']) else 0.0 for m, r in records.items()}
    feasible = {m: bool(r['Feasible']) for m, r in records.items()}
    if any((feasible[m] and (not np.isfinite(records[m]['Robust_Reward'])) for m in records)):
        raise ValueError('Feasible final portfolios require finite rewards.')
    live = [(a,) for a in available if feasible[a,] or any((len(m) == 2 and a in m and feasible[m] for m in records))]
    states = [()] + live

    def legal(s):
        choices = [-1] if feasible[s] else []
        for a in available:
            if a in s:
                continue
            nxt = tuple(sorted(s + (a,)))
            if not s and nxt in live or (s and nxt in records and feasible[nxt]):
                choices.append(a)
        return choices
    q = {(s, a): 0.0 for s in states for a in legal(s)}
    replay = list(q)
    rng = np.random.default_rng(c.seed)
    trace = []

    def target(s, a):
        if a == -1:
            return 0.0
        nxt = tuple(sorted(s + (a,)))
        return u[nxt] - u[s] + (0.0 if len(nxt) == 2 else max((q[nxt, b] for b in legal(nxt))))
    for epoch in range(c.q_replay_epochs):
        for k in rng.permutation(len(replay)):
            s, a = replay[k]
            q[s, a] += c.q_learning_rate * (target(s, a) - q[s, a])
        residual = max((abs(q[s, a] - target(s, a)) for s, a in replay))
        if epoch % 10 == 0 or residual < c.q_tolerance:
            trace.append({'Replay_Epoch': epoch + 1, 'Bellman_Residual': residual})
        if residual < c.q_tolerance:
            break
    selected = ()
    sequence = []
    for _ in range(2):
        choices = legal(selected)
        best = max((q[selected, a] for a in choices))
        a = next((a for a in choices if q[selected, a] >= best - c.q_tolerance))
        if a == -1:
            break
        sequence.append(POLICIES[a])
        selected = tuple(sorted(selected + (a,)))
    if not feasible[selected]:
        raise RuntimeError('Attempted to submit infeasible portfolio.')
    reachable = [m for m in records if feasible[m]]
    oracle = min(reachable, key=lambda m: (-u[m], len(m), m))
    regret = max(0.0, u[oracle] - u[selected])
    check = {'RL_Portfolio': records[selected]['Portfolio'], 'RL_Sequence': '>'.join(sequence) or 'STOP', 'Oracle_Portfolio': records[oracle]['Portfolio'], 'RL_Robust_Reward': u[selected], 'Oracle_Robust_Reward': u[oracle], 'Optimization_Regret': regret, 'Bellman_Residual': residual, 'Converged': residual < c.q_tolerance, 'Agreement_Within_Tolerance': regret <= 10 * c.q_tolerance, 'N_Feasible_Final_Portfolios': len(reachable), 'N_Pairs_With_No_Feasible_Singleton': sum((len(m) == 2 and (not any((feasible[a,] for a in m))) for m in reachable)), 'Algorithm': 'tabular_Q_learning_final_package_constraints', 'Sequence_Interpretation': 'tentative_selection_not_real_world_implementation_order'}
    qt = pd.DataFrame([{'State': records[s]['Portfolio'], 'State_Submittable': feasible[s], 'Action': 'SUBMIT' if a == -1 else POLICIES[a], 'Q_Value': v} for (s, a), v in q.items()])
    return (selected, check, qt, pd.DataFrame(trace), reachable)

class TrendBudgetFeasibleSpace:

    def __init__(self, panel, base, predictor, c, out):
        self.c, self.base, self.raw = (c, base, base[FEATURES].to_numpy(float))
        self.apos = [FEATURES.index(f) for f in c.att_features]
        self.legal = {}
        for feature, bounds in c.physical_bounds.items():
            if feature not in FEATURES or len(bounds) != 2:
                raise ValueError('physical_bounds requires feature: [lower or null, upper or null].')
            low = -np.inf if bounds[0] is None else float(bounds[0])
            high = np.inf if bounds[1] is None else float(bounds[1])
            if low > high or np.isnan(low) or np.isnan(high):
                raise ValueError('Invalid physical bounds.')
            self.legal[FEATURES.index(feature)] = (low, high)
            values = panel[feature].dropna().to_numpy(float)
            if ((values < low - 1e-10) | (values > high + 1e-10)).any():
                raise ValueError(f'{feature}: history violates physical bounds; check raw units/sign.')
        self.training = []
        for member in predictor.members:
            splits = read_splits(member['path'])
            train = set(splits.loc[splits.Split.eq('train'), 'Code'])
            sample = panel[panel.Code.isin(train)][FEATURES]
            self.training.append((sample.min().to_numpy(float), sample.max().to_numpy(float)))
        self.community = {}
        groups = panel.Code.map(base.Community_ID)
        for group in base.Community_ID.unique():
            ref = panel.loc[groups.eq(group), FEATURES]
            self.community[group] = (ref.min().to_numpy(float), ref.max().to_numpy(float))
        ids = base.index.astype(str).tolist()
        years = sorted(panel.Year.unique())
        keys = pd.MultiIndex.from_product([ids, years], names=['Code', 'Year'])
        if panel.duplicated(['Code', 'Year']).any():
            raise ValueError('Duplicate history Code-Year.')
        history = panel.set_index(['Code', 'Year']).reindex(keys)
        values = history[FEATURES].to_numpy(float).reshape(len(ids), len(years), len(FEATURES))
        x = np.asarray(years, dtype=float)[None, :, None]
        good = np.isfinite(values)
        n = good.sum(axis=1)
        xm = np.divide(np.where(good, x, 0.0).sum(axis=1), n, out=np.zeros_like(n, dtype=float), where=n > 0)
        ym = np.divide(np.where(good, values, 0.0).sum(axis=1), n, out=np.zeros_like(n, dtype=float), where=n > 0)
        dx = np.where(good, x - xm[:, None, :], 0.0)
        dy = np.where(good, values - ym[:, None, :], 0.0)
        den = np.sum(dx * dx, axis=1)
        slope = np.divide(np.sum(dx * dy, axis=1), den, out=np.full_like(den, np.nan), where=(den > 0) & (n >= c.trend_min_years))
        mean_abs = np.divide(np.where(good, np.abs(values), 0.0).sum(axis=1), n, out=np.ones_like(n, dtype=float), where=n > 0)
        tolerance = c.trend_zero_tolerance * np.maximum(1.0, mean_abs)
        slope = np.where(np.isfinite(slope) & (np.abs(slope) <= tolerance), 0.0, slope)
        self.slope, self.n, self.tolerance = (slope, n, tolerance)
        trendrows = []
        for i, code in enumerate(ids):
            for j, f in enumerate(FEATURES):
                trendrows.append({'Code': code, 'Community_ID': int(base.iloc[i].Community_ID), 'Feature': f, 'Historical_Annual_Slope': slope[i, j], 'N_Trend_Years': int(n[i, j]), 'Trend_Zero_Tolerance': tolerance[i, j], 'Same_Direction_Budget': abs(slope[i, j]) * c.trend_budget_years, 'Opposite_Direction_Budget': abs(slope[i, j]) * c.trend_budget_years * c.reverse_trend_fraction, 'Trend_Status': 'missing' if not np.isfinite(slope[i, j]) else 'flat' if slope[i, j] == 0 else 'estimated'})
        pd.DataFrame(trendrows).to_csv(out / '07_county_feature_trend_budgets.csv', index=False, encoding='utf-8-sig')
        rows = []
        for group, (low, high) in self.community.items():
            subset = self.raw[base.Community_ID.to_numpy() == group]
            for j, f in enumerate(FEATURES):
                legal_lo, legal_hi = self.legal.get(j, (-np.inf, np.inf))
                lower = max(low[j], legal_lo)
                upper = min(high[j], legal_hi)
                finite = subset[:, j][np.isfinite(subset[:, j])]
                if np.isfinite(lower) and np.isfinite(upper):
                    if lower > upper or ((finite < lower - 1e-10) | (finite > upper + 1e-10)).any():
                        raise ValueError(f'Community {group}, {f}: baseline not inside historical/legal intersection.')
                rows.append({'Community_ID': group, 'Feature': f, 'Historical_Min': low[j], 'Historical_Max': high[j], 'Applied_Lower_Bound': lower, 'Applied_Upper_Bound': upper, 'Use': 'hard_per_indicator_range;not_joint_empirical_support'})
        pd.DataFrame(rows).to_csv(out / '06_community_historical_bounds.csv', index=False, encoding='utf-8-sig')
        save_json(out / 'support_distance_thresholds.json', {'mode': 'independent_trend_budget_then_legal_and_community_minmax', 'trend_period': [int(min(years)), int(max(years))], 'trend_estimator': 'within_Code_feature_OLS_on_year_raw_units', 'trend_budget_years': c.trend_budget_years, 'opposite_fraction': c.reverse_trend_fraction, 'trend_min_years': c.trend_min_years, 'zero_or_missing_trend': 'hold_only_that_feature_at_baseline', 'common_scale': False, 'historical_range_veto': True, 'joint_distance_check': False, 'physical_bounds': c.physical_bounds, 'interpretation': 'one_year_budget_is_a_scenario_constraint_not_annualized_DID_ATT', 'training_range_use': 'diagnostic_only;community_minmax_is_not_full_joint_support'})

    def project(self, group, idx, proposal, features):
        old = self.raw[idx].copy()
        new = old.copy()
        shape = old.shape
        flags = {name: np.zeros(shape, dtype=bool) for name in ['Trend_Budget_Capped', 'Reverse_Trend', 'Trend_Unavailable', 'Flat_Trend', 'Baseline_Missing', 'Community_Range_Clipped', 'Community_Range_Unavailable', 'Legal_Bound_Clipped']}
        budgets = np.full(shape, np.nan)
        trend_limited = old.copy()
        lo, hi = self.community[group]
        for f in features:
            j = FEATURES.index(f)
            d = proposal[:, j] - old[:, j]
            request = np.isfinite(d) & (np.abs(d) > 1e-12)
            beta = self.slope[idx, j]
            known = np.isfinite(beta)
            reverse = request & known & (d * beta < 0)
            cap = np.where(known, np.abs(beta) * self.c.trend_budget_years, 0.0)
            cap = cap * np.where(reverse, self.c.reverse_trend_fraction, 1.0)
            budgets[:, j] = np.where(known, cap, np.nan)
            finite = np.isfinite(old[:, j]) & np.isfinite(proposal[:, j])
            step = np.where(finite, np.sign(d) * np.minimum(np.abs(d), cap), 0.0)
            limited = old[:, j] + step
            flags['Baseline_Missing'][:, j] = ~np.isfinite(old[:, j])
            flags['Reverse_Trend'][:, j] = reverse
            flags['Trend_Unavailable'][:, j] = request & ~known
            flags['Flat_Trend'][:, j] = request & known & (beta == 0)
            flags['Trend_Budget_Capped'][:, j] = request & known & (np.abs(d) > cap + 1e-12)
            trend_limited[:, j] = limited
            clipped = limited.copy()
            if j in self.legal:
                clipped = np.clip(clipped, *self.legal[j])
                flags['Legal_Bound_Clipped'][:, j] = finite & ~np.isclose(clipped, limited, rtol=0, atol=1e-12)
            if np.isfinite(lo[j]) and np.isfinite(hi[j]):
                bounded = np.clip(clipped, lo[j], hi[j])
                flags['Community_Range_Clipped'][:, j] = finite & ~np.isclose(bounded, clipped, rtol=0, atol=1e-12)
            else:
                bounded = old[:, j].copy()
                flags['Community_Range_Unavailable'][:, j] = request
            new[finite, j] = bounded[finite]
            actual = np.abs(new[:, j] - old[:, j])
            if np.any(finite & (actual > cap + 1e-09 * np.maximum(1.0, np.abs(old[:, j])))):
                raise RuntimeError('Independent trend budget violated after range clipping.')
        return (new, flags, budgets, trend_limited)

    def constrain(self, group, idx, proposal, features):
        new, _, _, _ = self.project(group, idx, proposal, features)
        return (new, np.full(len(idx), np.nan), np.zeros(len(idx)))

    def check(self, group, new):
        return (np.ones(len(new), bool), np.full(len(new), np.nan))

    def audit(self, group, idx, new, requested):
        proposed = self.raw[idx].copy()
        proposed[:, self.apos] += requested
        expected, flags, budgets, limited = self.project(group, idx, proposed, self.c.att_features)
        if not np.allclose(expected, new, rtol=1e-10, atol=1e-10, equal_nan=True):
            raise RuntimeError('Point-prediction inputs do not match independent constraint audit.')
        low, high = self.community[group]
        community = np.isfinite(new) & ((new < low - 1e-10) | (new > high + 1e-10))
        train = np.zeros_like(new, dtype=bool)
        for low, high in self.training:
            train |= np.isfinite(new) & ((new < low) | (new > high))
        names = lambda mask: [';'.join((f for f, b in zip(FEATURES, row) if b)) for row in mask]
        result = {'Outside_Training_Range': train.any(axis=1), 'Outside_Training_Range_Features': names(train), 'Outside_Community_History_Range': community.any(axis=1), 'Outside_Community_History_Features': names(community), 'Input_Extrapolation_Interpretation': 'community_minmax_enforced;training_ranges_diagnostic;no_joint_support_guarantee'}
        for label, mask in flags.items():
            result[label] = mask.any(axis=1)
            result[label + '_Features'] = names(mask)
        for j, f in enumerate(FEATURES):
            if f not in self.c.att_features:
                continue
            result[f + '_Historical_Annual_Slope'] = self.slope[idx, j]
            result[f + '_Directional_Budget'] = budgets[:, j]
            result[f + '_After_Trend_Budget_delta'] = limited[:, j] - self.raw[idx, j]
        return pd.DataFrame(result)

def export_stop_audit(table, c, out):
    actual = table.copy()
    added = actual.N_Selected_Policies.gt(0) & actual.Feasible.astype(str).str.lower().isin(['true', '1'])
    rows = []
    for (mp, targets, limit), g in actual.groupby(['ATT_Min_Pre', 'Targets', 'O3_Exceedance_Limit']):
        chosen = g[g.N_Selected_Policies.gt(0)]
        rows.append({'ATT_Min_Pre': mp, 'Targets': targets, 'O3_Exceedance_Limit': limit, 'N_Decision_Units': g.Decision_Unit.nunique(), 'N_Recommended_Units': chosen.Decision_Unit.nunique(), 'N_STOP_Units': g.loc[g.N_Selected_Policies.eq(0), 'Decision_Unit'].nunique(), 'Desired_Count_Audit_Only': c.desired_recommended_units, 'Desired_Count_Met': chosen.Decision_Unit.nunique() >= c.desired_recommended_units})
    pd.DataFrame(rows).to_csv(out / '50_recommendation_count_by_scenario.csv', index=False, encoding='utf-8-sig')
    chosen = actual.loc[added]
    n = int(chosen.Decision_Unit.nunique())
    save_json(out / '51_recommendation_count_audit.json', {'N_Distinct_Units_With_Any_Recommendation': n, 'Desired_Count_Audit_Only': c.desired_recommended_units, 'Desired_Count_Met_Any_Scenario': n >= c.desired_recommended_units, 'count_definition': 'distinct decision units with any non-STOP in any requested scenario', 'no_automatic_relaxation_or_forced_recommendation': True})
    cols = [k for k in ['ATT_Min_Pre', 'Scenario_ID', 'Decision_Unit', 'Community_ID', 'Targets', 'O3_Exceedance_Limit', 'STOP_Reason', 'N_Candidates_Evidence_Missing', 'N_Candidates_No_Actual_Change', 'N_Candidates_Risk_Rejected', 'N_Candidates_With_Trend_Caps', 'N_Candidates_With_Flat_Trends', 'N_Candidates_With_Missing_Trends', 'N_Candidates_With_Community_Clipping', 'N_Feasible_New_Candidates', 'Best_New_Selection_Reward'] if k in actual]
    actual.loc[actual.N_Selected_Policies.eq(0), cols].to_csv(out / '52_STOP_reason_audit.csv', index=False, encoding='utf-8-sig')

def main_scenarios(c=None, out_override=None):
    global ADJUSTABLE
    c = c or read47_args()
    runtime_preflight()
    ADJUSTABLE = list(c.att_features)
    out = Path(out_override) if out_override else Path(c.output) / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'settings': asdict(c), 'algorithm': 'tabular_Q_learning_with_exhaustive_verification', 'input_mode': 'county_feature_trend_budget_and_community_history_bounds', 'reward_mode': c.reward_mode, 'desired_count_is_not_constraint': True, 'uncertainty': c.att_uncertainty + '_not_causal_CI', 'policy_driver': 'joint_ATT_then_additive_single_ATT_with_history_backup_for_missing_components', 'ATT_Min_Pre': c.active_att_min_pre, 'constraint': 'only_final_submitted_package_checked_not_tentative_singletons', 'negative_environment_change_is_improvement': True, 'source_files': {}}
    try:
        if c.stage == 'similarity':
            if not c.reference_run or not c.new_communities:
                raise ValueError('Similarity stage requires --reference-run and --new-communities.')
            source = Path(c.reference_run)
            ref = read_csv(source / '20_similarity_reference_profiles.csv', dtype={'Unit_ID': str})
            rec = read_csv(source / '10_all_scenarios_recommendations.csv', dtype={'Decision_Unit': str})
            similarity_match(ref, read_csv(c.new_communities, dtype={'Unit_ID': str}), rec, c, out)
            manifest['status'] = 'complete_similarity_only'
            return
        for path in (c.panel, c.policy, c.edge, __file__, str(Path(c.section46_run) / '04_community_assignments.csv')):
            manifest['source_files'][path] = {'sha256': digest(Path(path))}
        log('Scenario Q-learning output: ' + str(out))
        panel, base, ids, edge, _ = load_data(c, out)
        verify_ATT_feature_scale(panel, c, out)
        assignments = read_csv(Path(c.section46_run) / '04_community_assignments.csv')
        assignments['Code'] = codes(assignments.Code)
        if assignments.Code.duplicated().any() or set(assignments.Code) != set(ids):
            raise ValueError('Community assignment must exactly cover model nodes.')
        base['Community_ID'] = assignments.set_index('Code').reindex(ids).Community_ID.astype(int)
        pol = read_csv(c.policy)
        pol['Code'] = codes(pol.Code)
        pol['Year'] = pd.to_numeric(pol.Year, errors='raise').astype(int)
        if pol.duplicated(['Code', 'Year']).any():
            raise ValueError('Duplicate policy Code-Year.')
        current = pol[pol.Year.eq(c.year)].set_index('Code').reindex(ids)
        if not current[POLICIES].isin([0, 1]).all().all():
            raise ValueError('Incomplete binary baseline policies.')
        transition = HybridMechanism(c, out, panel, base, current)
        manifest['source_files'][str(Path(c.mechanism_dir) / 'overall.csv')] = {'sha256': digest(Path(c.mechanism_dir) / 'overall.csv')}
        if c.att_recipe_file:
            manifest['source_files'][c.att_recipe_file] = {'sha256': digest(Path(c.att_recipe_file))}
        evidence_context47(c, out)
        if c.stage in ('transition', 'att_audit'):
            manifest['status'] = 'complete_mechanism_source_audit_only'
            return
        predictor = Predictor(panel, base, edge, c, out)
        space = TrendBudgetFeasibleSpace(panel, base, predictor, c, out)
        env = TwoPolicyEnvironment47(predictor, space, transition, current, base, c, out)
        if c.decision_level == 'community':
            units = [(str(g), int(g), np.flatnonzero(base.Community_ID.to_numpy() == g)) for g in sorted(base.Community_ID.unique())]
        else:
            units = [(code, int(base.loc[code, 'Community_ID']), np.array([i])) for i, code in enumerate(ids)]
        total_units = len(units)
        if c.unit_ids:
            units = [u for u in units if u[0] in set(map(str, c.unit_ids))]
        if c.max_units:
            units = units[:c.max_units]
        if not units:
            raise ValueError('No selected decision units.')
        ref = export_reference(base, current, units, out)
        bank = out / 'candidate_bank'
        bank.mkdir()
        recs = []
        cityresults = []
        allchecks = []
        allcomparisons = []
        allcandidates = []
        tier_recommendations = []
        tier_city_results = []
        count_benchmarks = []
        count_city = []
        boundary_records = []
        membership_file = Path(c.boundary_dir) / 'diagnostics/county_group_membership.csv'
        boundary_file = Path(c.boundary_dir) / 'group_effects.csv'
        membership = read_csv(membership_file, dtype={'Code': str}) if c.att_export_boundary_context and membership_file.exists() else None
        bounds = read_csv(boundary_file) if membership is not None and boundary_file.exists() else None
        if bounds is not None:
            bounds = bounds[pd.to_numeric(bounds.min_pre, errors='coerce').eq(c.active_att_min_pre) & bounds.scope.eq('overall') & bounds.period.eq('post_all')].copy()
            if 'role' in bounds:
                bounds = bounds[bounds.role.eq('environment')]
            if bounds.duplicated(['design', 'stratifier', 'group', 'variable']).any():
                raise ValueError('Ambiguous boundary records.')
            bounds = bounds.rename(columns={'att': 'Boundary_environment_ATT_negative_log', 'p_value': 'Boundary_p', 'status': 'Boundary_status', 'quality_flags': 'Boundary_quality_flags'})
            membership['Code'] = codes(membership.Code)
            differences = Path(c.boundary_dir) / 'group_differences.csv'
            if differences.exists():
                dif = read_csv(differences)
                dif[pd.to_numeric(dif.min_pre, errors='coerce').eq(c.active_att_min_pre)].to_csv(out / '38_boundary_group_differences_context.csv', index=False, encoding='utf-8-sig')
        scenario_specs = [(f'pre{c.active_att_min_pre}__' + '+'.join(t) + f'__O3limit{int(round(l * 100))}', tuple(t), float(l)) for t in c.scenario_targets for l in c.o3_exceedance_limits]
        if len({s[0] for s in scenario_specs}) != len(scenario_specs):
            raise ValueError('Duplicate scenario IDs.')
        for n, (unit, group, idx) in enumerate(units, 1):
            log(f'Unit {n}/{len(units)}: {unit}; simulate responses once for all scenarios')
            available = env.available(idx)
            if c.smoke:
                available = available[:2]
            portfolios = [()] + [(a,) for a in available] + list(combinations(available, 2))
            raw = {}
            detail = {}
            for k, m in enumerate(portfolios, 1):
                raw[m], detail[m] = env.evaluate(unit, group, idx, m)
                if k == 1 or k % 10 == 0:
                    log(f'  candidate {k}/{len(portfolios)}')
            if bounds is not None:
                for m, d in detail.items():
                    if not m:
                        continue
                    link = d[['Code', 'ATT_Component_Designs']].copy()
                    link['ATT_Design'] = link.ATT_Component_Designs.str.split(';')
                    link = link.explode('ATT_Design')
                    link = link[link.ATT_Design.notna() & link.ATT_Design.ne('')].drop_duplicates(['Code', 'ATT_Design'])
                    link = link.merge(membership, on='Code', how='left')
                    sizes = link.groupby(['ATT_Design', 'stratifier', 'group'], dropna=False).size().reset_index(name='N_Applicable_Cities')
                    context = sizes.merge(bounds, left_on=['ATT_Design', 'stratifier', 'group'], right_on=['design', 'stratifier', 'group'], how='left')
                    context['Decision_Unit'] = unit
                    context['Candidate_Menu'] = raw[m]['Portfolio']
                    context['ATT_Min_Pre'] = c.active_att_min_pre
                    context['Use'] = '2013_baseline_subgroup_environment_evidence_context_only_not_added_to_reward'
                    boundary_records.append(context)
            if c.att_export_count_benchmarks:
                for addition in (1, 2):
                    cr, cd = env.evaluate(unit, group, idx, tuple(range(addition)), count_additions=addition)
                    cd = cd.copy()
                    cd['Policy_Identity_Interpretation'] = 'count_transition_average_not_specific_policy_package'
                    cd.to_csv(bank / f'{unit}_COUNT_ADD_{addition}_city_response.csv', index=False, encoding='utf-8-sig')
                    for sid, targets, limit in scenario_specs:
                        scored = score_scenario(cr, targets, limit, c)
                        scored['Scenario_ID'] = sid
                        scored['Comparison_Use'] = 'separate_count_ATT_benchmark_not_ranked_as_named_policy_recipe'
                        count_benchmarks.append(scored)
            rawsave = pd.DataFrame(raw.values()).drop(columns=['Feasible', 'Mean_Reward', 'Robust_Reward', 'Reasons'])
            rawsave.to_csv(bank / f'{unit}_raw_response_bank.csv', index=False, encoding='utf-8-sig')
            pd.concat(detail.values(), ignore_index=True).to_csv(bank / f'{unit}_city_candidates.csv', index=False, encoding='utf-8-sig')
            for sid, targets, limit in scenario_specs:
                dest = out / 'scenarios' / sid
                dest.mkdir(parents=True, exist_ok=True)
                records = {m: score_scenario(r, targets, limit, c) for m, r in raw.items()}
                selected, check, qt, trace, reachable = q_learning_package(records, available, c)
                quality = 'model_optimal_verified' if check['Converged'] and check['Agreement_Within_Tolerance'] else 'not_verified_do_not_claim_optimal'
                stop_reason = ('no_feasible_new_package' if len(reachable) == 1 else 'no_positive_net_robust_gain') if not selected else ''
                result = {**records[selected], **check, 'Scenario_ID': sid, 'Recommendation_Status': quality, 'STOP_Reason': stop_reason}
                new_records = [r for m, r in records.items() if m]
                feasible_new = [r for r in new_records if r['Feasible']]
                result.update(N_Candidates_Evidence_Missing=sum((not r.get('ATT_Complete', True) for r in new_records)), N_Candidates_No_Actual_Change=sum((r.get('ATT_Complete', True) and r['N_Changed_Cities'] == 0 for r in new_records)), N_Candidates_With_Trend_Caps=sum((r.get('N_Trend_Budget_Capped_Cities', 0) > 0 for r in new_records)), N_Candidates_With_Flat_Trends=sum((r.get('N_Flat_Trend_Cities', 0) > 0 for r in new_records)), N_Candidates_With_Missing_Trends=sum((r.get('N_Trend_Unavailable_Cities', 0) > 0 for r in new_records)), N_Candidates_With_Community_Clipping=sum((r.get('N_Community_Range_Clipped_Cities', 0) > 0 for r in new_records)), N_Candidates_Risk_Rejected=sum((np.isfinite(r['O3_Exceedance_Fraction']) and r['O3_Exceedance_Fraction'] > limit for r in new_records)), N_Feasible_New_Candidates=len(feasible_new), Best_New_Selection_Reward=max((r['Robust_Reward'] for r in feasible_new), default=np.nan))
                recs.append(result)
                allchecks.append({'Scenario_ID': sid, 'Decision_Unit': unit, **check})
                chosen = detail[selected].copy()
                chosen['Scenario_ID'] = sid
                chosen['Targets'] = '+'.join(targets)
                chosen['Recommendation_Status'] = quality
                chosen['O3_Exceedance_Limit'] = limit
                chosen['O3_Risk_Scope'] = c.risk_scope
                chosen['N_Improved_Outcomes'] = (chosen[[y + '_change_pct' for y in OUTCOMES]] < -c.improvement_tolerance_pct).sum(axis=1)
                chosen['N_Worsened_Outcomes'] = (chosen[[y + '_change_pct' for y in OUTCOMES]] > c.improvement_tolerance_pct).sum(axis=1)
                chosen['N_Improved_Targets'] = (chosen[[y + '_change_pct' for y in targets]] < -c.improvement_tolerance_pct).sum(axis=1)
                chosen['N_Worsened_Targets'] = (chosen[[y + '_change_pct' for y in targets]] > c.improvement_tolerance_pct).sum(axis=1)
                cityresults.append(chosen)
                tiers = []
                for count, label in ((1, 'best_single'), (2, 'best_double')):
                    pool = [m for m in reachable if len(m) == count]
                    winner = min(pool, key=lambda m: (-records[m]['Robust_Reward'], m)) if pool else None
                    tiers.append((label, winner, len(pool)))
                tiers.append(('actual_overall', selected, len(reachable)))
                for label, winner, nfeasible in tiers:
                    common = {'Scenario_ID': sid, 'Decision_Unit': unit, 'Community_ID': group, 'ATT_Min_Pre': c.active_att_min_pre, 'Decision_Level': c.decision_level, 'N_Cities': len(idx), 'Recommendation_Type': label, 'Targets': '+'.join(targets), 'O3_Exceedance_Limit': limit, 'N_Feasible_In_Category': nfeasible, 'Count_Definition': 'selected_policy_menu_count_not_each_city_actual_additions'}
                    if winner is None:
                        size = 1 if label == 'best_single' else 2
                        missing = [r for m, r in records.items() if len(m) == size and (not r.get('ATT_Complete', True))]
                        tier_recommendations.append({**common, 'Portfolio': '', 'Recommendation_Status': 'no_estimable_or_feasible_single' if label == 'best_single' else 'no_estimable_or_feasible_double', 'N_ATT_Unavailable_In_Category': len(missing), 'Missing_Evidence_Reasons': ';'.join(sorted(set((r.get('ATT_Missing_Reasons', '') for r in missing)) - {''})), 'Robust_Reward': np.nan, 'Beats_STOP': False, **{y + '_change_pct': np.nan for y in OUTCOMES}})
                        continue
                    source = 'Q_learning' if label == 'actual_overall' else 'exact_size_feasible_candidate_comparison'
                    tier_status = quality if label == 'actual_overall' else 'conditional_model_optimum'
                    tier_recommendations.append({**records[winner], **common, 'Recommendation_Status': tier_status, 'Selection_Source': source, 'Beats_STOP': bool(records[winner]['Robust_Reward'] > 0), 'Is_Actual_RL_Choice': winner == selected, 'Reward_Gap_To_Overall_Oracle': check['Oracle_Robust_Reward'] - records[winner]['Robust_Reward']})
                    td = detail[winner].copy()
                    td['Scenario_ID'] = sid
                    td['Recommendation_Type'] = label
                    td['Targets'] = '+'.join(targets)
                    td['Recommendation_Status'] = tier_status
                    td['O3_Exceedance_Limit'] = limit
                    td['O3_Risk_Scope'] = c.risk_scope
                    td['Category_Robust_Reward'] = records[winner]['Robust_Reward']
                    td['Beats_STOP'] = records[winner]['Robust_Reward'] > 0
                    td['N_Improved_Outcomes'] = (td[[y + '_change_pct' for y in OUTCOMES]] < -c.improvement_tolerance_pct).sum(axis=1)
                    td['N_Worsened_Outcomes'] = (td[[y + '_change_pct' for y in OUTCOMES]] > c.improvement_tolerance_pct).sum(axis=1)
                    tier_city_results.append(td)
                candidates = pd.DataFrame(records.values())
                candidates['Scenario_ID'] = sid
                allcandidates.append(candidates)
                qt.to_csv(dest / f'{unit}_Q_values.csv', index=False, encoding='utf-8-sig')
                trace.to_csv(dest / f'{unit}_learning_trace.csv', index=False, encoding='utf-8-sig')
                previous = 0.0
                for budget in range(3):
                    eligible = [m for m in reachable if len(m) <= budget]
                    exact = [m for m in reachable if len(m) == budget]
                    best = min(eligible, key=lambda m: (-records[m]['Robust_Reward'], len(m), m))
                    be = min(exact, key=lambda m: (-records[m]['Robust_Reward'], m)) if exact else None
                    allcomparisons.append({'Scenario_ID': sid, 'Decision_Unit': unit, 'Policy_Budget': budget, 'Best_AtMost_Portfolio': records[best]['Portfolio'], 'Best_AtMost_Robust_Reward': records[best]['Robust_Reward'], 'Increment_Over_Previous_Budget': records[best]['Robust_Reward'] - previous, 'Best_Exact_Portfolio': records[be]['Portfolio'] if be is not None else '', 'Best_Exact_Robust_Reward': records[be]['Robust_Reward'] if be is not None else np.nan, 'N_Feasible_Exact_Packages': len(exact)})
                    previous = records[best]['Robust_Reward']
            pd.DataFrame(recs).to_csv(out / '10_all_scenarios_recommendations.csv', index=False, encoding='utf-8-sig')
            pd.DataFrame(tier_recommendations).to_csv(out / '17_single_double_overall_recommendations.csv', index=False, encoding='utf-8-sig')
        tables = {'08_all_scenarios_candidates.csv': pd.concat(allcandidates, ignore_index=True), '10_all_scenarios_recommendations.csv': pd.DataFrame(recs), '11_all_scenarios_city_responses.csv': pd.concat(cityresults, ignore_index=True), '12_all_scenarios_budget_comparison.csv': pd.DataFrame(allcomparisons), '13_all_scenarios_Q_checks.csv': pd.DataFrame(allchecks), '17_single_double_overall_recommendations.csv': pd.DataFrame(tier_recommendations), '18_single_double_overall_city_responses.csv': pd.concat(tier_city_results, ignore_index=True)}
        if count_benchmarks:
            tables['35_count_ATT_scenario_benchmarks.csv'] = pd.DataFrame(count_benchmarks)
        if boundary_records:
            pd.concat(boundary_records, ignore_index=True).to_csv(out / '37_boundary_applicability_context.csv', index=False, encoding='utf-8-sig')
        for name, table in tables.items():
            table['ATT_Min_Pre'] = c.active_att_min_pre
            table.to_csv(out / name, index=False, encoding='utf-8-sig')
            for sid, _, _ in scenario_specs:
                table[table.Scenario_ID.eq(sid)].to_csv(out / 'scenarios' / sid / name, index=False, encoding='utf-8-sig')
        export_stop_audit(pd.DataFrame(recs), c, out)
        if c.new_communities:
            similarity_match(ref, read_csv(c.new_communities, dtype={'Unit_ID': str}), pd.DataFrame(recs), c, out)
        manifest.update(status='smoke_test_not_publication' if c.smoke else 'complete_full' if len(units) == total_units else 'complete_selected_units', processed_units=len(units), total_units=total_units, scenarios=[s[0] for s in scenario_specs], all_RL_checks_pass=all((r['Converged'] and r['Agreement_Within_Tolerance'] for r in allchecks)), risk_rule='fraction strictly greater than limit rejects; equality allowed', independent_unit_simulations_not_joint_national_deployment=True)
        if not manifest['all_RL_checks_pass']:
            manifest['status'] += ';optimization_check_failure'
        log('Completed. Main result: 10_all_scenarios_recommendations.csv')
    except Exception:
        manifest.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        manifest['finished'] = datetime.now().isoformat()
        save_json(out / 'analysis_manifest_47.json', manifest)

def run_pre123():
    c = read47_args()
    if c.stage == 'similarity':
        main_scenarios(c)
        return
    root = Path(c.output) / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'running', 'settings': asdict(c), 'min_pre_completed': [], 'expected_scenario_count': len(c.att_min_pre_values) * len(c.scenario_targets) * len(c.o3_exceedance_limits), 'meaning_of_min_pre': 'minimum_observed_pretreatment_years_not_lags_or_prediction_horizon', 'policy_driver': 'hybrid_mechanism_ATT', 'runtime': 'existing_ATT_with_labelled_historical_association_backup'}
    merged = {}
    try:
        for mp in c.att_min_pre_values:
            out = root / f'min_pre_{mp}'
            log(f'Loading existing DID ATT for minimum pre-treatment years={mp}')
            cc = replace(c, active_att_min_pre=int(mp), evidence_min_pre=int(mp), new_communities='')
            main_scenarios(cc, out)
            summary['min_pre_completed'].append(int(mp))
            for name in ('08_all_scenarios_candidates.csv', '10_all_scenarios_recommendations.csv', '11_all_scenarios_city_responses.csv', '12_all_scenarios_budget_comparison.csv', '13_all_scenarios_Q_checks.csv', '17_single_double_overall_recommendations.csv', '18_single_double_overall_city_responses.csv', '31_ATT_design_availability.csv', '35_count_ATT_scenario_benchmarks.csv', '37_boundary_applicability_context.csv', '41_historical_fallback_support.csv', '42_historical_prediction_checks.csv', '45_final_action_sources.csv'):
                path = out / name
                if path.exists():
                    frame = read_csv(path, dtype={'Code': str, 'Decision_Unit': str})
                    frame['ATT_Min_Pre'] = int(mp)
                    merged.setdefault(name, []).append(frame)
            if mp == c.att_min_pre_values[0] and (out / '20_similarity_reference_profiles.csv').exists():
                for name in ('20_similarity_reference_profiles.csv', '21_new_community_input_template.csv'):
                    read_csv(out / name, dtype={'Unit_ID': str}).to_csv(root / name, index=False, encoding='utf-8-sig')
        for name, frames in merged.items():
            pd.concat(frames, ignore_index=True).to_csv(root / name, index=False, encoding='utf-8-sig')
        if '10_all_scenarios_recommendations.csv' in merged:
            export_stop_audit(pd.concat(merged['10_all_scenarios_recommendations.csv'], ignore_index=True), c, root)
        if '13_all_scenarios_Q_checks.csv' in merged:
            checks = pd.concat(merged['13_all_scenarios_Q_checks.csv'], ignore_index=True)
            yes = lambda s: s.astype(str).str.lower().isin(['true', '1'])
            summary['all_RL_checks_pass'] = bool((yes(checks.Converged) & yes(checks.Agreement_Within_Tolerance)).all())
        if c.new_communities and '10_all_scenarios_recommendations.csv' in merged:
            ref = read_csv(root / '20_similarity_reference_profiles.csv', dtype={'Unit_ID': str})
            query = read_csv(c.new_communities, dtype={'Unit_ID': str})
            similarity_match(ref, query, pd.concat(merged['10_all_scenarios_recommendations.csv'], ignore_index=True), c, root)
        summary['status'] = 'complete_ATT_audit_only' if c.stage in ('transition', 'att_audit') else 'smoke_test_not_publication' if c.smoke else 'completed_requested_min_pre_runs'
        if summary.get('all_RL_checks_pass') is False:
            summary['status'] += ';optimization_check_failure'
        summary['child_manifests'] = [str(root / f'min_pre_{p}' / 'analysis_manifest_47.json') for p in summary['min_pre_completed']]
        summary['ATT_uncertainty'] = c.att_uncertainty
        pd.DataFrame(columns=['from_mask', 'to_mask', 'min_pre', 'role', 'variable', 'att', 'se', 'status', 'unit', 'n_treated', 'p_value', 'pretrend_p', 'quality_flags']).to_csv(root / '39_optional_joint_mechanism_ATT_template.csv', index=False, encoding='utf-8-sig')
        log('Hybrid ATT/history pre1/pre2/pre3 results: ' + str(root))
    except Exception:
        summary.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        summary['finished'] = datetime.now().isoformat()
        save_json(root / 'analysis_manifest_47.json', summary)
if __name__ == '__main__':
    run_pre123()
