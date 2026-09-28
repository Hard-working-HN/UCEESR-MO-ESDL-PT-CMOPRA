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
from dataclasses import dataclass, asdict, field
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
if __name__ == '__main__':
    main()
