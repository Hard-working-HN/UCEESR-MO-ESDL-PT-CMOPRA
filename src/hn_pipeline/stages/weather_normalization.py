import os
import gc
import time
import warnings
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pygam import LinearGAM, s
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
warnings.filterwarnings('ignore')
PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEATHER_INPUT_DIR = PROJECT_ROOT / 'data' / 'raw' / 'weather'
POLLUTANT_FILES = {'CO': str(WEATHER_INPUT_DIR / 'Day_CO.csv'), 'NO2': str(WEATHER_INPUT_DIR / 'Day_NO2.csv'), 'O3': str(WEATHER_INPUT_DIR / 'Day_O3.csv'), 'PM10': str(WEATHER_INPUT_DIR / 'Day_PM10.csv'), 'PM25': str(WEATHER_INPUT_DIR / 'Day_PM25.csv'), 'SO2': str(WEATHER_INPUT_DIR / 'Day_SO2.csv')}
CLIMATE_FILES = {'temp_c': str(WEATHER_INPUT_DIR / 'A_2m_temperature_C.csv'), 'rh': str(WEATHER_INPUT_DIR / 'relative_humidity.csv'), 'wind_speed': str(WEATHER_INPUT_DIR / 'wind_speed.csv'), 'wind_dir_sin': str(WEATHER_INPUT_DIR / 'wind_direction_sin.csv'), 'wind_dir_cos': str(WEATHER_INPUT_DIR / 'wind_direction_cos.csv'), 'pressure_hpa': str(WEATHER_INPUT_DIR / 'surface_pressure_hPa.csv'), 'radiation_wm2': str(WEATHER_INPUT_DIR / 'surface_solar_radiation_downwards_dailymean_Wm2.csv'), 'precip_mm': str(WEATHER_INPUT_DIR / 'total_precipitation_mm.csv')}
ROOT_RESULT_DIR = str(PROJECT_ROOT / 'outputs' / 'weather_normalization')
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLINES = 8
LAM = 0.75
N_SAMPLES = 300
MIN_VALID_ROWS = 260
LOW_R2_THRESHOLD = 0.6
USE_PARALLEL = True
N_WORKERS = min(max(1, cpu_count() - 2), 20)
BATCH_SIZE = 20
MAX_REGIONS = None
GLOBAL_POLLUTANT_WEEKLY = None
GLOBAL_CLIMATE_WEEKLY = None
GLOBAL_POLLUTANT_NAME = None

def read_wide_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}')
    df = pd.read_csv(path)
    if 'Date' not in df.columns:
        raise ValueError(f"{path} does not contain a 'Date' column")
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    value_cols = [c for c in df.columns if c != 'Date']
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors='coerce')
    return df

def aggregate_wide_to_weekly_mean(df):
    out = df.copy()
    out['Week'] = out['Date'].dt.to_period('W-SUN').dt.start_time
    value_cols = [c for c in out.columns if c not in ['Date', 'Week']]
    out = out.groupby('Week', as_index=False)[value_cols].mean()
    out = out.rename(columns={'Week': 'Date'})
    return out

def get_common_dates(dfs):
    common = set(dfs[0]['Date'])
    for df in dfs[1:]:
        common &= set(df['Date'])
    return sorted(common)

def get_common_regions(dfs):
    common = set([c for c in dfs[0].columns if c != 'Date'])
    for df in dfs[1:]:
        common &= set([c for c in df.columns if c != 'Date'])
    return sorted(common)

def subset_wide_df(df, common_dates, common_regions):
    out = df[df['Date'].isin(common_dates)].copy()
    out = out[['Date'] + common_regions].sort_values('Date').reset_index(drop=True)
    return out

def stable_region_seed(region, base_seed=42):
    h = hashlib.md5(region.encode('utf-8')).hexdigest()
    region_int = int(h[:8], 16)
    return (base_seed + region_int) % (2 ** 32 - 1)

def split_into_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def build_region_weekly_df(region, pollutant_df, climate_df_dict, pollutant_col_name):
    data = {'Date': pollutant_df['Date'].values, pollutant_col_name: pollutant_df[region].values}
    for var_name, df in climate_df_dict.items():
        data[var_name] = df[region].values
    df = pd.DataFrame(data)
    df = df.dropna().sort_values('Date').reset_index(drop=True)
    if len(df) == 0:
        return df
    df['trend_week'] = np.arange(len(df), dtype=float)
    iso = df['Date'].dt.isocalendar()
    df['week'] = iso.week.astype(int)
    df['year'] = iso.year.astype(int)
    weeks_per_year = 365.2425 / 7.0
    df['week_sin'] = np.sin(2 * np.pi * (df['week'] - 1) / weeks_per_year)
    df['week_cos'] = np.cos(2 * np.pi * (df['week'] - 1) / weeks_per_year)
    return df

def fit_gam_model_weekly(df_region, pollutant_col_name):
    feature_cols = ['trend_week', 'week_sin', 'week_cos', 'temp_c', 'rh', 'wind_speed', 'wind_dir_sin', 'wind_dir_cos', 'pressure_hpa', 'radiation_wm2', 'precip_mm']
    X = df_region[feature_cols].to_numpy(dtype=float)
    y_raw = df_region[pollutant_col_name].to_numpy(dtype=float)
    y = np.log1p(y_raw)
    idx = np.arange(len(df_region))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train, X_test = (X[train_idx], X[test_idx])
    y_train, y_test = (y[train_idx], y[test_idx])
    terms = s(0, n_splines=N_SPLINES) + s(1, n_splines=N_SPLINES) + s(2, n_splines=N_SPLINES) + s(3, n_splines=N_SPLINES) + s(4, n_splines=N_SPLINES) + s(5, n_splines=N_SPLINES) + s(6, n_splines=N_SPLINES) + s(7, n_splines=N_SPLINES) + s(8, n_splines=N_SPLINES) + s(9, n_splines=N_SPLINES) + s(10, n_splines=N_SPLINES)
    gam = LinearGAM(terms=terms, lam=LAM, fit_intercept=True)
    gam.fit(X_train, y_train)
    y_pred_test = np.expm1(gam.predict(X_test))
    y_pred_test = np.clip(y_pred_test, 0, None)
    y_test_raw = np.expm1(y_test)
    y_test_raw = np.clip(y_test_raw, 0, None)
    r2 = r2_score(y_test_raw, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_test))
    mean_val = float(df_region[pollutant_col_name].mean())
    std_val = float(df_region[pollutant_col_name].std(ddof=0))
    nrmse_mean = rmse / mean_val if mean_val > 0 else np.nan
    nrmse_sd = rmse / std_val if std_val > 0 else np.nan
    return (gam, feature_cols, r2, rmse, nrmse_mean, nrmse_sd)

def meteorological_normalisation_weekly_same_region_same_week(gam, df_region, feature_cols, n_samples=300, random_state=42):
    rng = np.random.default_rng(random_state)
    X_base = df_region[feature_cols].to_numpy(dtype=float)
    n = X_base.shape[0]
    met_col_idx = np.array([3, 4, 5, 6, 7, 8, 9, 10], dtype=int)
    week_arr = df_region['week'].to_numpy(dtype=int)
    week_to_indices = {w: np.where(week_arr == w)[0] for w in np.unique(week_arr)}
    pred_sum = np.zeros(n, dtype=float)
    for _ in range(n_samples):
        sampled_idx = np.empty(n, dtype=int)
        for i in range(n):
            w = week_arr[i]
            pool_idx = week_to_indices[w]
            if len(pool_idx) == 0:
                sampled_idx[i] = i
            else:
                sampled_idx[i] = rng.choice(pool_idx)
        X_tmp = X_base.copy()
        X_tmp[:, met_col_idx] = X_base[sampled_idx][:, met_col_idx]
        pred_log = gam.predict(X_tmp)
        pred = np.expm1(pred_log)
        pred = np.clip(pred, 0, None)
        pred_sum += pred
    deweathered = pred_sum / n_samples
    return deweathered

def init_worker(pollutant_weekly, climate_weekly, pollutant_name):
    global GLOBAL_POLLUTANT_WEEKLY, GLOBAL_CLIMATE_WEEKLY, GLOBAL_POLLUTANT_NAME
    GLOBAL_POLLUTANT_WEEKLY = pollutant_weekly
    GLOBAL_CLIMATE_WEEKLY = climate_weekly
    GLOBAL_POLLUTANT_NAME = pollutant_name

def process_one_region(region):
    start = time.time()
    pollutant_col_name = GLOBAL_POLLUTANT_NAME
    df_region = build_region_weekly_df(region, GLOBAL_POLLUTANT_WEEKLY, GLOBAL_CLIMATE_WEEKLY, pollutant_col_name)
    if len(df_region) == 0:
        elapsed = time.time() - start
        return {'status': 'empty', 'region': region, 'elapsed_seconds': elapsed}
    df_region[f'{pollutant_col_name}_deweathered'] = df_region[pollutant_col_name].values
    r2 = np.nan
    rmse = np.nan
    nrmse_mean = np.nan
    nrmse_sd = np.nan
    fallback_reason = 'none'
    error_msg = ''
    if len(df_region) < MIN_VALID_ROWS:
        fallback_reason = 'insufficient_data'
    else:
        try:
            gam, feature_cols, r2, rmse, nrmse_mean, nrmse_sd = fit_gam_model_weekly(df_region, pollutant_col_name)
            if pd.notna(r2) and r2 < LOW_R2_THRESHOLD:
                fallback_reason = 'low_r2'
            else:
                region_seed = stable_region_seed(region, RANDOM_STATE)
                df_region[f'{pollutant_col_name}_deweathered'] = meteorological_normalisation_weekly_same_region_same_week(gam=gam, df_region=df_region, feature_cols=feature_cols, n_samples=N_SAMPLES, random_state=region_seed)
                fallback_reason = 'none'
        except Exception as e:
            fallback_reason = 'model_error'
            error_msg = str(e)
    elapsed = time.time() - start
    weekly_series = pd.Series(data=df_region[f'{pollutant_col_name}_deweathered'].values, index=df_region['Date'].values)
    tmp_weekly = df_region[['Date', 'year', 'week', pollutant_col_name, f'{pollutant_col_name}_deweathered']].copy()
    tmp_weekly['Region'] = region
    tmp_weekly['R2_test'] = r2
    tmp_weekly['RMSE_test'] = rmse
    tmp_weekly['nRMSE_mean'] = nrmse_mean
    tmp_weekly['nRMSE_sd'] = nrmse_sd
    tmp_weekly['Fallback_Reason'] = fallback_reason
    tmp_weekly['Use_Observed_As_Result'] = 1 if fallback_reason != 'none' else 0
    tmp_weekly['Error_Message'] = error_msg
    tmp_annual = tmp_weekly.groupby('year', as_index=False).agg(Observed_Mean=(pollutant_col_name, 'mean'), Deweathered_Mean=(f'{pollutant_col_name}_deweathered', 'mean'), n_weeks=(pollutant_col_name, 'size'))
    tmp_annual['Region'] = region
    tmp_annual['R2_test'] = r2
    tmp_annual['RMSE_test'] = rmse
    tmp_annual['nRMSE_mean'] = nrmse_mean
    tmp_annual['nRMSE_sd'] = nrmse_sd
    tmp_annual['Fallback_Reason'] = fallback_reason
    tmp_annual['Use_Observed_As_Result'] = 1 if fallback_reason != 'none' else 0
    tmp_annual['Error_Message'] = error_msg
    perf = {'Region': region, 'n_valid_weeks': len(df_region), 'R2_test': r2, 'RMSE_test': rmse, 'nRMSE_mean': nrmse_mean, 'nRMSE_sd': nrmse_sd, 'elapsed_seconds': elapsed, 'Fallback_Reason': fallback_reason, 'Use_Observed_As_Result': 1 if fallback_reason != 'none' else 0, 'Low_R2_Fallback': 1 if fallback_reason == 'low_r2' else 0, 'Error_Message': error_msg}
    return {'status': 'ok', 'region': region, 'elapsed_seconds': elapsed, 'weekly_series': weekly_series, 'weekly_long': tmp_weekly, 'annual_long': tmp_annual, 'perf': perf}

def run_one_batch(batch_regions, pollutant_weekly, climate_weekly, pollutant_name, batch_id, total_batches):
    batch_results = {'weekly_dict': {}, 'weekly_long_list': [], 'annual_long_list': [], 'perf_list': []}
    total_in_batch = len(batch_regions)
    finished_in_batch = 0
    pass
    pass
    if USE_PARALLEL:
        pass
        with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=init_worker, initargs=(pollutant_weekly, climate_weekly, pollutant_name)) as executor:
            future_to_region = {executor.submit(process_one_region, region): region for region in batch_regions}
            for future in as_completed(future_to_region):
                finished_in_batch += 1
                result = future.result()
                region = result['region']
                if result['status'] == 'ok':
                    batch_results['weekly_dict'][region] = result['weekly_series']
                    batch_results['weekly_long_list'].append(result['weekly_long'])
                    batch_results['annual_long_list'].append(result['annual_long'])
                    batch_results['perf_list'].append(result['perf'])
                    pass
                else:
                    pass
    else:
        pass
        init_worker(pollutant_weekly, climate_weekly, pollutant_name)
        for region in batch_regions:
            finished_in_batch += 1
            result = process_one_region(region)
            if result['status'] == 'ok':
                batch_results['weekly_dict'][region] = result['weekly_series']
                batch_results['weekly_long_list'].append(result['weekly_long'])
                batch_results['annual_long_list'].append(result['annual_long'])
                batch_results['perf_list'].append(result['perf'])
                pass
    pass
    return batch_results

def process_one_pollutant(pollutant_name, pollutant_file, climate_weekly_base):
    start_time = time.time()
    pass
    pass
    pass
    result_dir = os.path.join(ROOT_RESULT_DIR, pollutant_name)
    ensure_dir(result_dir)
    out_weekly_obs = os.path.join(result_dir, f'{pollutant_name}_weekly_observed.csv')
    out_weekly_deweathered = os.path.join(result_dir, f'{pollutant_name}_weekly_deweathered_GAM.csv')
    out_weekly_long = os.path.join(result_dir, f'{pollutant_name}_observed_deweathered_weekly_long_GAM.csv')
    out_annual_long = os.path.join(result_dir, f'{pollutant_name}_observed_deweathered_annual_long_GAM.csv')
    out_model_perf = os.path.join(result_dir, f'{pollutant_name}_deweather_model_performance_weekly_GAM.csv')
    out_model_summary = os.path.join(result_dir, f'{pollutant_name}_deweather_model_summary_GAM.csv')
    out_low_r2_regions = os.path.join(result_dir, f'{pollutant_name}_low_r2_fallback_regions_GAM.csv')
    pollutant_daily = read_wide_csv(pollutant_file)
    pollutant_weekly = aggregate_wide_to_weekly_mean(pollutant_daily)
    all_dfs = [pollutant_weekly] + list(climate_weekly_base.values())
    common_dates = get_common_dates(all_dfs)
    common_regions = get_common_regions(all_dfs)
    if len(common_dates) == 0:
        raise ValueError(f'{pollutant_name} has no dates in common with the climate files.')
    if len(common_regions) == 0:
        raise ValueError(f'{pollutant_name} has no region columns in common with the climate files.')
    if MAX_REGIONS is not None:
        common_regions = common_regions[:MAX_REGIONS]
        pass
    pollutant_weekly = subset_wide_df(pollutant_weekly, common_dates, common_regions)
    climate_weekly = {k: subset_wide_df(df, common_dates, common_regions) for k, df in climate_weekly_base.items()}
    pollutant_weekly.to_csv(out_weekly_obs, index=False, encoding='utf-8-sig')
    weekly_deweathered_dict = {}
    weekly_long_records = []
    annual_long_records = []
    perf_records = []
    region_batches = list(split_into_batches(common_regions, BATCH_SIZE))
    total_batches = len(region_batches)
    pass
    pass
    pass
    pass
    for batch_id, batch_regions in enumerate(region_batches, start=1):
        batch_start = time.time()
        batch_results = run_one_batch(batch_regions=batch_regions, pollutant_weekly=pollutant_weekly, climate_weekly=climate_weekly, pollutant_name=pollutant_name, batch_id=batch_id, total_batches=total_batches)
        weekly_deweathered_dict.update(batch_results['weekly_dict'])
        weekly_long_records.extend(batch_results['weekly_long_list'])
        annual_long_records.extend(batch_results['annual_long_list'])
        perf_records.extend(batch_results['perf_list'])
        del batch_results
        gc.collect()
        batch_elapsed = time.time() - batch_start
        pass
        pass
    weekly_deweathered_out = pd.DataFrame({'Date': common_dates})
    for region, s_region in weekly_deweathered_dict.items():
        weekly_deweathered_out[region] = pd.Series(s_region).reindex(common_dates).values
    weekly_deweathered_out = weekly_deweathered_out[['Date'] + sorted([c for c in weekly_deweathered_out.columns if c != 'Date'])]
    weekly_deweathered_out.to_csv(out_weekly_deweathered, index=False, encoding='utf-8-sig')
    pollutant_col = pollutant_name
    pollutant_dew_col = f'{pollutant_name}_deweathered'
    if weekly_long_records:
        weekly_long_out = pd.concat(weekly_long_records, ignore_index=True)
        weekly_long_out = weekly_long_out[['Region', 'Date', 'year', 'week', pollutant_col, pollutant_dew_col, 'R2_test', 'RMSE_test', 'nRMSE_mean', 'nRMSE_sd', 'Fallback_Reason', 'Use_Observed_As_Result', 'Error_Message']].sort_values(['Region', 'Date'])
    else:
        weekly_long_out = pd.DataFrame()
    weekly_long_out.to_csv(out_weekly_long, index=False, encoding='utf-8-sig')
    if annual_long_records:
        annual_long_out = pd.concat(annual_long_records, ignore_index=True)
        annual_long_out = annual_long_out[['Region', 'year', 'Observed_Mean', 'Deweathered_Mean', 'n_weeks', 'R2_test', 'RMSE_test', 'nRMSE_mean', 'nRMSE_sd', 'Fallback_Reason', 'Use_Observed_As_Result', 'Error_Message']].sort_values(['Region', 'year'])
        annual_long_out = annual_long_out.rename(columns={'year': 'Year'})
    else:
        annual_long_out = pd.DataFrame()
    annual_long_out.to_csv(out_annual_long, index=False, encoding='utf-8-sig')
    perf_out = pd.DataFrame(perf_records)
    if not perf_out.empty:
        perf_out = perf_out.sort_values('Region')
    perf_out.to_csv(out_model_perf, index=False, encoding='utf-8-sig')
    if not perf_out.empty:
        low_r2_out = perf_out[perf_out['Fallback_Reason'] == 'low_r2'].copy()
    else:
        low_r2_out = pd.DataFrame()
    low_r2_out.to_csv(out_low_r2_regions, index=False, encoding='utf-8-sig')
    if not perf_out.empty:
        summary_df = pd.DataFrame({'Metric': ['pollutant', 'n_regions_total', 'n_regions_low_r2_fallback', 'n_regions_insufficient_data_fallback', 'n_regions_model_error_fallback', 'n_regions_all_fallback', 'mean_R2_test', 'median_R2_test', 'mean_RMSE_test', 'median_RMSE_test', 'mean_nRMSE_mean', 'median_nRMSE_mean', 'mean_nRMSE_sd', 'median_nRMSE_sd', 'mean_elapsed_seconds', 'median_elapsed_seconds'], 'Value': [pollutant_name, len(perf_out), (perf_out['Fallback_Reason'] == 'low_r2').sum(), (perf_out['Fallback_Reason'] == 'insufficient_data').sum(), (perf_out['Fallback_Reason'] == 'model_error').sum(), (perf_out['Use_Observed_As_Result'] == 1).sum(), perf_out['R2_test'].mean(), perf_out['R2_test'].median(), perf_out['RMSE_test'].mean(), perf_out['RMSE_test'].median(), perf_out['nRMSE_mean'].mean(), perf_out['nRMSE_mean'].median(), perf_out['nRMSE_sd'].mean(), perf_out['nRMSE_sd'].median(), perf_out['elapsed_seconds'].mean(), perf_out['elapsed_seconds'].median()]})
    else:
        summary_df = pd.DataFrame(columns=['Metric', 'Value'])
    summary_df.to_csv(out_model_summary, index=False, encoding='utf-8-sig')
    total_elapsed = time.time() - start_time
    pass
    pass
    pass
    if not perf_out.empty:
        pass
        pass
        pass
        pass
        pass
        pass
        pass
        pass
        pass
    pass
    pass
    del pollutant_daily
    del pollutant_weekly
    del climate_weekly
    del weekly_deweathered_dict
    del weekly_long_records
    del annual_long_records
    del perf_records
    del weekly_deweathered_out
    del weekly_long_out
    del annual_long_out
    del perf_out
    del summary_df
    gc.collect()

def main():
    total_start = time.time()
    ensure_dir(ROOT_RESULT_DIR)
    pass
    climate_daily = {k: read_wide_csv(v) for k, v in CLIMATE_FILES.items()}
    pass
    climate_weekly_base = {k: aggregate_wide_to_weekly_mean(df) for k, df in climate_daily.items()}
    del climate_daily
    gc.collect()
    for pollutant_name, pollutant_file in POLLUTANT_FILES.items():
        process_one_pollutant(pollutant_name=pollutant_name, pollutant_file=pollutant_file, climate_weekly_base=climate_weekly_base)
        gc.collect()
    del climate_weekly_base
    gc.collect()
    total_elapsed = time.time() - total_start
    pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
