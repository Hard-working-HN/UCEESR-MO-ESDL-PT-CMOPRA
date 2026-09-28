from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
STAGE_DIR = PROJECT_ROOT / "src" / "hn_pipeline" / "stages"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
SPATIAL_DIR = PROJECT_ROOT / "data" / "spatial"
MODULE2_DIR = DATA_DIR / "module2_time_space_results"

STAGES = [
    "weather_normalization",
    "cpsi_spatiotemporal",
    "o3_pm10_indices",
    "multitask_model",
    "single_policy_did",
    "multi_policy_did",
    "neighbor_spillover",
    "policy_mechanisms",
    "policy_boundaries",
    "community_assessment",
    "q_learning_recommendation",
]

def latest_run(directory: Path) -> Path:
    runs = sorted([p for p in directory.glob("run_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run directory found under {directory}")
    return runs[-1]

def require(paths: list[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        lines = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Required pipeline inputs are missing:\n{lines}")

def run_stage(script: str, args: list[str] | None = None) -> None:
    command = [sys.executable, str(STAGE_DIR / f"{script}.py")]
    if args:
        command.extend(args)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def build_community_config() -> Path:
    mechanism_run = latest_run(OUTPUT_DIR / "policy_mechanisms")
    boundary_run = latest_run(OUTPUT_DIR / "policy_boundaries")
    config = {
        "panel": str(DATA_DIR / "model_data_total.csv"),
        "policy": str(DATA_DIR / "policy_conditions.csv"),
        "edge": str(MODULE2_DIR / "10a_spatial_weights_edge_list.csv"),
        "dual_model": str(MODEL_DIR / "formal_multitask" / "Model5"),
        "o3_model": str(MODEL_DIR / "o3" / "Model5"),
        "pm10_model": str(MODEL_DIR / "pm10" / "Model5"),
        "shape": str(SPATIAL_DIR / "county_2023.shp"),
        "mechanism_dir": str(mechanism_run),
        "boundary_dir": str(boundary_run),
        "output": str(OUTPUT_DIR / "community_assessment"),
    }
    path = PROJECT_ROOT / "config" / "community_runtime.json"
    write_json(path, config)
    return path

def build_q_config() -> Path:
    mechanism_run = latest_run(OUTPUT_DIR / "policy_mechanisms")
    boundary_run = latest_run(OUTPUT_DIR / "policy_boundaries")
    section46_run = latest_run(OUTPUT_DIR / "community_assessment")
    config = {
        "panel": str(DATA_DIR / "model_data_total.csv"),
        "policy": str(DATA_DIR / "policy_conditions.csv"),
        "edge": str(MODULE2_DIR / "10a_spatial_weights_edge_list.csv"),
        "dual_model": str(MODEL_DIR / "formal_multitask" / "Model5"),
        "o3_model": str(MODEL_DIR / "o3" / "Model5"),
        "pm10_model": str(MODEL_DIR / "pm10" / "Model5"),
        "shape": str(SPATIAL_DIR / "county_2023.shp"),
        "mechanism_dir": str(mechanism_run),
        "boundary_dir": str(boundary_run),
        "section46_run": str(section46_run),
        "output": str(OUTPUT_DIR / "q_learning"),
    }
    path = PROJECT_ROOT / "config" / "q_learning_runtime.json"
    write_json(path, config)
    return path

def execute(stage: str) -> None:
    policy_file = DATA_DIR / "policy_conditions.csv"
    model_file = DATA_DIR / "model_data_total.csv"
    edge_file = MODULE2_DIR / "10a_spatial_weights_edge_list.csv"
    shape_file = SPATIAL_DIR / "county_2023.shp"
    if stage == "weather_normalization":
        weather_dir = PROJECT_ROOT / "data" / "raw" / "weather"
        require([
            weather_dir / "Day_CO.csv",
            weather_dir / "Day_NO2.csv",
            weather_dir / "Day_O3.csv",
            weather_dir / "Day_PM10.csv",
            weather_dir / "Day_PM25.csv",
            weather_dir / "Day_SO2.csv",
            weather_dir / "A_2m_temperature_C.csv",
            weather_dir / "relative_humidity.csv",
            weather_dir / "wind_speed.csv",
            weather_dir / "wind_direction_sin.csv",
            weather_dir / "wind_direction_cos.csv",
            weather_dir / "surface_pressure_hPa.csv",
            weather_dir / "surface_solar_radiation_downwards_dailymean_Wm2.csv",
            weather_dir / "total_precipitation_mm.csv",
        ])
        run_stage(stage)
    elif stage == "cpsi_spatiotemporal":
        require([
            MODULE2_DIR / "01_county_annual_panel.csv",
            MODULE2_DIR / "02_city_annual_panel.csv",
            MODULE2_DIR / "03_province_annual_panel.csv",
            MODULE2_DIR / "04_national_annual_panel.csv",
            edge_file,
            shape_file,
        ])
        run_stage(stage)
    elif stage == "o3_pm10_indices":
        require([DATA_DIR / "total_data_with_city_basic_features.csv", model_file])
        run_stage(stage, ["--source", str(DATA_DIR / "total_data_with_city_basic_features.csv"), "--model", str(model_file), "--overwrite"])
    elif stage == "multitask_model":
        require([model_file, edge_file])
        run_stage(stage, ["--data", str(model_file), "--edges", str(edge_file), "--output", str(MODEL_DIR / "formal_multitask")])
    elif stage == "single_policy_did":
        require([policy_file, model_file])
        run_stage(stage, ["--policy-file", str(policy_file), "--model-file", str(model_file), "--out", str(OUTPUT_DIR / "single_policy_did")])
    elif stage == "multi_policy_did":
        require([policy_file, model_file])
        run_stage(stage, ["--policy-file", str(policy_file), "--model-file", str(model_file), "--out", str(OUTPUT_DIR / "multi_policy_did")])
    elif stage == "neighbor_spillover":
        require([policy_file, model_file, shape_file])
        run_stage(stage, ["--policy-file", str(policy_file), "--model-file", str(model_file), "--shp-file", str(shape_file), "--output-root", str(OUTPUT_DIR / "neighbor_spillover")])
    elif stage == "policy_mechanisms":
        require([policy_file, model_file])
        run_stage(stage, ["--policy-file", str(policy_file), "--model-file", str(model_file), "--output-root", str(OUTPUT_DIR / "policy_mechanisms")])
    elif stage == "policy_boundaries":
        require([policy_file, model_file])
        run_stage(stage, ["--policy-file", str(policy_file), "--model-file", str(model_file), "--output-root", str(OUTPUT_DIR / "policy_boundaries")])
    elif stage == "community_assessment":
        require([
            policy_file,
            model_file,
            edge_file,
            MODEL_DIR / "formal_multitask" / "Model5" / "best_model.pt",
            MODEL_DIR / "o3" / "Model5" / "best_model.pt",
            MODEL_DIR / "pm10" / "Model5" / "best_model.pt",
        ])
        config = build_community_config()
        run_stage(stage, ["--config", str(config), "--stage", "full"])
    elif stage == "q_learning_recommendation":
        require([
            policy_file,
            model_file,
            edge_file,
            MODEL_DIR / "formal_multitask" / "Model5" / "best_model.pt",
            MODEL_DIR / "o3" / "Model5" / "best_model.pt",
            MODEL_DIR / "pm10" / "Model5" / "best_model.pt",
        ])
        config = build_q_config()
        run_stage(stage, ["--config", str(config), "--stage", "full"])
    else:
        raise ValueError(f"Unknown stage: {stage}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-at", choices=STAGES, default=STAGES[0])
    parser.add_argument("--stop-after", choices=STAGES, default=STAGES[-1])
    parser.add_argument("--skip", nargs="*", choices=STAGES, default=[])
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    start = STAGES.index(args.start_at)
    stop = STAGES.index(args.stop_after)
    if start > stop:
        raise ValueError("--start-at must not come after --stop-after")
    for stage in STAGES[start:stop + 1]:
        if stage not in args.skip:
            execute(stage)

if __name__ == "__main__":
    main()
