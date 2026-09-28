from __future__ import annotations
import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIRECTORY = PROJECT_ROOT / 'data' / 'processed'
DEFAULT_SOURCE = DEFAULT_DIRECTORY / 'total_data_with_city_basic_features.csv'
DEFAULT_MODEL = DEFAULT_DIRECTORY / 'model_data_total.csv'
KEY_COLUMNS = ['Code', 'Year']
SOURCE_COLUMNS = ['O3', 'PM10']
OUTPUT_COLUMNS = {'O3': 'O3_Index', 'PM10': 'PM10_Index'}
BASE_YEAR = 2013

def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ('utf-8-sig', 'utf-8', 'gb18030'):
        try:
            return pd.read_csv(path, encoding=encoding, dtype={'Code': 'string'}, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError('unknown', b'', 0, 1, f'Unable to identify CSV encoding: {path}; last error={last_error}')

def require_columns(frame: pd.DataFrame, required: Iterable[str], file_label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f'{file_label}Missing required columns: {missing}')

def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out['Code'] = out['Code'].astype('string').str.strip().str.replace('\\.0+$', '', regex=True)
    out['Year'] = pd.to_numeric(out['Year'], errors='raise').astype(int)
    return out

def assert_unique_keys(frame: pd.DataFrame, file_label: str) -> None:
    duplicate_mask = frame.duplicated(KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        examples = frame.loc[duplicate_mask, KEY_COLUMNS].drop_duplicates().head(10).to_dict('records')
        raise ValueError(f'{file_label} contains duplicate Code-Year keys; duplicate rows={int(duplicate_mask.sum())}; examples={examples}')

def geometric_mean_strictly_positive(values: pd.Series, pollutant: str) -> tuple[float, int]:
    numeric = pd.to_numeric(values, errors='coerce')
    non_missing = numeric.dropna()
    if non_missing.empty:
        raise ValueError(f'{pollutant} in {BASE_YEAR} has no valid data in the baseline year.')
    non_positive = non_missing <= 0
    if non_positive.any():
        raise ValueError(f'{pollutant} in {BASE_YEAR} contains {int(non_positive.sum())} values less than or equal to zero; a geometric mean cannot be calculated.')
    geometric_mean = float(np.exp(np.log(non_missing.to_numpy(float)).mean()))
    return (geometric_mean, int(non_missing.size))

def build_indices(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(source, KEY_COLUMNS + SOURCE_COLUMNS, 'Total_Data_with_City_Basic_Features.csv')
    source = normalize_keys(source)
    assert_unique_keys(source, 'Total_Data_with_City_Basic_Features.csv')
    base_rows = source[source['Year'].eq(BASE_YEAR)].copy()
    if base_rows.empty:
        raise ValueError(f'The source data do not contain baseline year {BASE_YEAR} records.')
    output = source[KEY_COLUMNS].copy()
    audit_rows: list[dict[str, object]] = []
    for source_column, output_column in OUTPUT_COLUMNS.items():
        values = pd.to_numeric(source[source_column], errors='coerce')
        invalid_all_years = values.notna() & values.le(0)
        if invalid_all_years.any():
            examples = source.loc[invalid_all_years, KEY_COLUMNS + [source_column]].head(10)
            raise ValueError(f'{source_column} contains {int(invalid_all_years.sum())} nonpositive concentrations across the study period; a strictly positive index cannot be constructed. Examples:\n{examples.to_string(index=False)}')
        baseline, base_n = geometric_mean_strictly_positive(base_rows[source_column], source_column)
        output[output_column] = values / baseline
        base_index = output.loc[source['Year'].eq(BASE_YEAR), output_column].dropna()
        reconstructed_geometric_mean = float(np.exp(np.log(base_index.to_numpy(float)).mean()))
        audit_rows.append({'Pollutant': source_column, 'Base_Year': BASE_YEAR, 'National_Geometric_Mean': baseline, 'N_Base_Year_Valid': base_n, 'Index_Column': output_column, 'Index_2013_Geometric_Mean_Check': reconstructed_geometric_mean, 'N_All_Years_Missing': int(values.isna().sum())})
    return (output, pd.DataFrame(audit_rows))

def merge_at_end(model: pd.DataFrame, indices: pd.DataFrame) -> pd.DataFrame:
    require_columns(model, KEY_COLUMNS, 'Model_Data_Total.csv')
    model = normalize_keys(model)
    assert_unique_keys(model, 'Model_Data_Total.csv')
    target_columns = list(OUTPUT_COLUMNS.values())
    model = model.drop(columns=target_columns, errors='ignore')
    merged = model.merge(indices, on=KEY_COLUMNS, how='left', validate='one_to_one', indicator=True, sort=False)
    unmatched = merged['_merge'].ne('both')
    if unmatched.any():
        examples = merged.loc[unmatched, KEY_COLUMNS].head(10).to_dict('records')
        raise ValueError(f'The model data contain {int(unmatched.sum())} rows without matching O3/PM10 values; examples={examples}')
    merged = merged.drop(columns='_merge')
    if len(merged) != len(model):
        raise RuntimeError(f'Row count changed after merge: {len(model)} -> {len(merged)}')
    if merged.columns[-2:].tolist() != target_columns:
        raise RuntimeError('O3_Index and PM10_Index are not the final two columns.')
    return merged

def save_safely(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.stem + '.__temporary__' + output_path.suffix)
    try:
        frame.to_csv(temporary_path, index=False, encoding='utf-8-sig')
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build 2013 national geometric-mean indices for O3 and PM10 and append them by Code-Year.')
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path, default=None, help='By default, save a separate model_data_total_with_o3_pm10.csv file.')
    parser.add_argument('--overwrite', action='store_true', help='Back up and overwrite the file specified by --model; cannot be used with --output.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.overwrite and args.output is not None:
        raise ValueError('--overwrite and --output cannot be used together.')
    source_path = args.source.resolve()
    model_path = args.model.resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if args.overwrite:
        output_path = model_path
        backup_path = model_path.with_name(model_path.stem + '_before_O3_PM10' + model_path.suffix)
        if not backup_path.exists():
            shutil.copy2(model_path, backup_path)
            pass
        else:
            pass
    else:
        output_path = args.output.resolve() if args.output is not None else model_path.with_name(model_path.stem + '_with_O3_PM10' + model_path.suffix)
    source = read_csv_flexible(source_path)
    model = read_csv_flexible(model_path)
    indices, audit = build_indices(source)
    merged = merge_at_end(model, indices)
    save_safely(merged, output_path)
    audit_path = output_path.with_name(output_path.stem + '_O3_PM10_baseline_audit.csv')
    audit.to_csv(audit_path, index=False, encoding='utf-8-sig')
    pass
    pass
    pass
    pass
    pass
    pass
if __name__ == '__main__':
    main()
