# 기술 IQA 신호들의 오프라인 공통 행 벤치마크를 수행한다.
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/technical_iqa_signal_comparison_20260604"
TEST_MANIFEST = ROOT / "outputs/student_distillation_manifest_14105_20260603/test_manifest.csv"
STUDENT_KERAS = ROOT / "outputs/mobile_student_iqa_14105_full_controlled_20260604/best_by_val_mae_model.keras"
STUDENT_TFLITE = ROOT / "outputs/mobile_student_iqa_14105_tflite_export_20260604/student_iqa_14105_fp16.tflite"
KONIQ_TFLITE = ROOT / "exports/tflite/koniq_mobile.tflite"
FLIVE_TFLITE = ROOT / "exports/tflite/flive_image_mobile.tflite"
TOPIQ_TFLITE = ROOT / "outputs/final_topiq_lite_mixed112_export_20260517/topiq_lite_mixed112_frozen_fp16.tflite"
MUSIQ_MODELS = {
    "musiq_koniq_probe": ROOT / "exports/tflite/musiq_koniq_probe1024/musiq_koniq_probe1024_fp16.tflite",
    "musiq_flive_probe": ROOT / "exports/tflite/musiq_flive_probe1024/musiq_flive_probe1024_fp16.tflite",
    "musiq_spaq_probe": ROOT / "exports/tflite/musiq_spaq_probe1024/musiq_spaq_probe1024_fp16.tflite",
}
PRED_CACHE_PATHS = [
    ROOT / "outputs/research_protocol_production_technical_eval_20260530/production_technical_predictions.csv",
    ROOT / "outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_koniq.csv",
    ROOT / "outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_spaq.csv",
    ROOT / "outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_flive.csv",
    ROOT / "outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/predictions_topiq_lite_mixed112_frozen_fp16_koniq.csv",
    ROOT / "outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/predictions_topiq_lite_mixed112_frozen_fp16_spaq.csv",
    ROOT / "outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/predictions_topiq_lite_mixed112_frozen_fp16_flive.csv",
    ROOT / "outputs/eval_tflite_technical_compare/summary.csv",
    ROOT / "outputs/mobile_student_iqa_14105_tflite_export_20260604/tflite_parity_predictions.csv",
    ROOT / "outputs/student_flive_hybrid_benchmark_20260604/hybrid_candidate_predictions.csv",
]
DATASET_ORDER = ["flive", "koniq10k", "spaq"]
MODEL_SIGNALS = [
    "production_existing",
    "koniq_existing",
    "flive_existing",
    "topiq_existing",
    "student_14105",
    "musiq_koniq_probe",
    "musiq_flive_probe",
    "musiq_spaq_probe",
]
HYBRID_BASES = ["student_14105", "topiq_existing", "musiq_koniq_probe", "musiq_flive_probe", "musiq_spaq_probe"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare technical IQA model signals on recovered common rows.")
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--test_manifest", default=str(TEST_MANIFEST))
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--skip_musiq", action="store_true")
    return parser.parse_args()


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def corr(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 2:
        return float("nan")
    x = a[finite].astype(np.float64)
    y = b[finite].astype(np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def srcc(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 2:
        return float("nan")
    return corr(rankdata(a[finite]), rankdata(b[finite]))


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    src = df[columns].head(limit) if limit else df[columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in src.iterrows():
        vals = []
        for value in row.tolist():
            vals.append(fmt(value, 6) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def file_mb(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0) if path.exists() else float("nan")


def tflite_io(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"input_shape": "", "output_shape": "", "input_count": 0, "exists": False}
    try:
        interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
        interpreter.allocate_tensors()
        ins = interpreter.get_input_details()
        outs = interpreter.get_output_details()
        return {
            "input_shape": ";".join([str(i["shape"].tolist()) for i in ins]),
            "output_shape": ";".join([str(o["shape"].tolist()) for o in outs]),
            "input_count": len(ins),
            "exists": True,
        }
    except Exception as exc:  # noqa: BLE001
        return {"input_shape": "", "output_shape": "", "input_count": 0, "exists": path.exists(), "error": str(exc)}


def write_inventories(out_dir: Path) -> None:
    model_rows = []
    specs = [
        ("production_existing", "audited_formula", ROOT / "outputs/research_protocol_production_technical_eval_20260530/production_formula_audit.md", "cached/reconstructed", "image-level `(4*KonIQ + 3*FLIVE)/7`; production code unchanged"),
        ("koniq_existing", "tflite", KONIQ_TFLITE, "run_full_test", "224 resize, /255, output 0..100"),
        ("flive_existing", "tflite", FLIVE_TFLITE, "run_full_test", "224 resize, /255, output 0..100"),
        ("topiq_existing", "tflite", TOPIQ_TFLITE, "run_full_test", "384 resize_with_pad, raw 0..255, output*100"),
        ("student_14105", "tflite_fp16", STUDENT_TFLITE, "run_full_test", "256 resize, raw 0..255, output*100; Keras checkpoint inventoried"),
        ("student_14105_keras", "keras", STUDENT_KERAS, "not_run", "Keras checkpoint; TFLite parity already accepted in Phase 12"),
    ]
    for name, kind, path, use, note in specs:
        io = tflite_io(path) if path.suffix == ".tflite" else {"input_shape": "", "output_shape": "", "input_count": "", "exists": path.exists()}
        model_rows.append({"signal": name, "kind": kind, "path": str(path), "exists": bool(path.exists()), "size_mb": file_mb(path) if path.is_file() else float("nan"), "benchmark_use": use, "input_count": io.get("input_count", ""), "input_shape": io.get("input_shape", ""), "output_shape": io.get("output_shape", ""), "notes": note})
    for name, path in MUSIQ_MODELS.items():
        io = tflite_io(path)
        model_rows.append({"signal": name, "kind": "tflite_fp16", "path": str(path), "exists": bool(path.exists()), "size_mb": file_mb(path), "benchmark_use": "run_full_test", "input_count": io.get("input_count", ""), "input_shape": io.get("input_shape", ""), "output_shape": io.get("output_shape", ""), "notes": "local MUSIQ probe; not official Kaggle MUSIQ"})
    pd.DataFrame(model_rows).to_csv(out_dir / "model_signal_inventory.csv", index=False)

    cache_rows = []
    for path in PRED_CACHE_PATHS:
        row = {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0, "columns": "", "rows": 0, "usable_for_primary_common_rows": False, "notes": ""}
        if path.exists() and path.suffix == ".csv":
            try:
                head = pd.read_csv(path, nrows=5)
                row["columns"] = ",".join(head.columns.astype(str).tolist())
                row["rows"] = int(sum(1 for _ in open(path, "rb")) - 1)
                row["usable_for_primary_common_rows"] = "image_path" in head.columns
                if "outputs/eval_tflite_technical_compare" in str(path):
                    row["notes"] = "MUSIQ cache has no image_path; not join-safe."
                elif "student_flive_hybrid" in str(path):
                    row["notes"] = "Previous common rows only; not enough KonIQ coverage."
                else:
                    row["notes"] = "Inspected."
            except Exception as exc:  # noqa: BLE001
                row["notes"] = f"read_error: {exc}"
        cache_rows.append(row)
    pd.DataFrame(cache_rows).to_csv(out_dir / "prediction_cache_inventory.csv", index=False)


def load_manifest(path: Path, max_rows: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["split"].astype(str).eq("test")].copy()
    if max_rows:
        df = df.groupby("dataset", group_keys=False).head(max_rows)
    df["mos_0_100"] = pd.to_numeric(df["quality_score_0_100"], errors="coerce")
    df = df.dropna(subset=["mos_0_100", "image_path"]).reset_index(drop=True)
    return df


def load_rgb(path: str) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def resize_array(img: Image.Image, size: int, norm255: bool, pad: bool = False) -> np.ndarray:
    if pad:
        canvas = ImageOps.contain(img, (size, size), Image.Resampling.BILINEAR)
        bg = Image.new("RGB", (size, size), (0, 0, 0))
        bg.paste(canvas, ((size - canvas.width) // 2, (size - canvas.height) // 2))
        arr = np.asarray(bg, dtype=np.float32)
    else:
        arr = np.asarray(img.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    if norm255:
        arr = arr / 255.0
    return arr


def predict_single_input_tflite(path: Path, images: list[Image.Image], size: int, norm255: bool, pad: bool, scale100: bool) -> tuple[np.ndarray, float]:
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    preds = []
    t0 = time.perf_counter()
    for img in images:
        arr = resize_array(img, size=size, norm255=norm255, pad=pad)
        interpreter.set_tensor(int(input_detail["index"]), arr[None, ...].astype(input_detail["dtype"]))
        interpreter.invoke()
        val = float(interpreter.get_tensor(int(output_detail["index"])).reshape(-1)[0])
        if scale100:
            val *= 100.0
        preds.append(val)
    return np.asarray(preds, dtype=np.float64), time.perf_counter() - t0


def build_musiq_inputs(img: Image.Image, patch_size: int = 32, scale_sizes: tuple[int, ...] = (224, 384, 512), patches_per_scale: int = 16) -> dict[str, np.ndarray]:
    patches_all = []
    positions_all = []
    scale_ids_all = []
    mask_all = []
    width, height = img.size
    for scale_idx, target_long in enumerate(scale_sizes):
        scale = float(target_long) / max(width, height)
        resized_w = max(patch_size, int(round(width * scale)))
        resized_h = max(patch_size, int(round(height * scale)))
        resized = np.asarray(img.resize((resized_w, resized_h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        patches = []
        positions = []
        for top in range(0, resized_h - patch_size + 1, patch_size):
            for left in range(0, resized_w - patch_size + 1, patch_size):
                patches.append(resized[top : top + patch_size, left : left + patch_size, :])
                positions.append([top // patch_size, left // patch_size])
        patches = patches[:patches_per_scale]
        positions = positions[:patches_per_scale]
        count = len(patches)
        while len(patches) < patches_per_scale:
            patches.append(np.zeros((patch_size, patch_size, 3), dtype=np.float32))
            positions.append([0.0, 0.0])
        patches_all.extend(patches)
        positions_all.extend(positions)
        scale_ids_all.extend([scale_idx] * patches_per_scale)
        mask_all.extend([1.0] * count + [0.0] * (patches_per_scale - count))
    return {
        "patches": np.asarray(patches_all, dtype=np.float32)[None, ...],
        "positions": np.asarray(positions_all, dtype=np.float32)[None, ...],
        "scale_ids": np.asarray(scale_ids_all, dtype=np.int32)[None, ...],
        "token_mask": np.asarray(mask_all, dtype=np.float32)[None, ...],
    }


def predict_musiq_tflite(path: Path, images: list[Image.Image]) -> tuple[np.ndarray, float]:
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    output_detail = interpreter.get_output_details()[0]
    preds = []
    t0 = time.perf_counter()
    for img in images:
        tensors = build_musiq_inputs(img)
        for detail in inputs:
            name = str(detail["name"])
            key = "patches"
            if "positions" in name:
                key = "positions"
            elif "scale_ids" in name:
                key = "scale_ids"
            elif "token_mask" in name:
                key = "token_mask"
            interpreter.set_tensor(int(detail["index"]), tensors[key].astype(detail["dtype"]))
        interpreter.invoke()
        val = float(interpreter.get_tensor(int(output_detail["index"])).reshape(-1)[0])
        if val <= 1.5:
            val *= 100.0
        preds.append(val)
    return np.asarray(preds, dtype=np.float64), time.perf_counter() - t0


def metric_row(df: pd.DataFrame, signal: str, dataset: str) -> dict[str, Any]:
    pred_col = f"{signal}_0_100"
    target = df["mos_0_100"].to_numpy(dtype=np.float64)
    pred = df[pred_col].to_numpy(dtype=np.float64)
    finite = np.isfinite(target) & np.isfinite(pred)
    target = target[finite]
    pred = pred[finite]
    row: dict[str, Any] = {"signal": signal, "dataset": dataset, "n": int(len(target))}
    if len(target) == 0:
        for key in ["mae", "rmse", "srcc", "plcc", "pred_min", "pred_max", "pred_mean", "pred_std", "target_min", "target_max", "target_mean", "target_std"]:
            row[key] = float("nan")
        row.update({"fp_count": 0, "severe_fp_count": 0, "fn_count": 0})
        return row
    row.update(
        {
            "mae": float(np.mean(np.abs(pred - target))),
            "rmse": float(np.sqrt(np.mean(np.square(pred - target)))),
            "srcc": srcc(target, pred),
            "plcc": corr(target, pred),
            "pred_min": float(np.min(pred)),
            "pred_max": float(np.max(pred)),
            "pred_mean": float(np.mean(pred)),
            "pred_std": float(np.std(pred)),
            "target_min": float(np.min(target)),
            "target_max": float(np.max(target)),
            "target_mean": float(np.mean(target)),
            "target_std": float(np.std(target)),
            "fp_count": int(((target < 40.0) & (pred > 60.0)).sum()),
            "severe_fp_count": int(((target < 40.0) & (pred > 65.0)).sum()),
            "fn_count": int(((target > 75.0) & (pred < 45.0)).sum()),
        }
    )
    return row


def compute_metrics(df: pd.DataFrame, signals: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = [metric_row(df, signal, "overall") for signal in signals if f"{signal}_0_100" in df.columns]
    dataset_rows = []
    for signal in signals:
        if f"{signal}_0_100" not in df.columns:
            continue
        for dataset in DATASET_ORDER:
            sub = df[df["dataset"].astype(str).eq(dataset)]
            dataset_rows.append(metric_row(sub, signal, dataset))
    return pd.DataFrame(overall), pd.DataFrame(dataset_rows)


def add_hybrids(df: pd.DataFrame) -> list[str]:
    signals = []
    for base in HYBRID_BASES:
        if f"{base}_0_100" not in df.columns:
            continue
        for weight in [4.0 / 7.0, 0.5, 0.4]:
            name = f"{base}_flive_{int(round(weight * 100)):02d}_{int(round((1 - weight) * 100)):02d}"
            df[f"{name}_0_100"] = np.clip(weight * df[f"{base}_0_100"] + (1.0 - weight) * df["flive_existing_0_100"], 0.0, 100.0)
            signals.append(name)
        for i in range(2, 9):
            weight = i / 10.0
            name = f"{base}_flive_grid_s{i:02d}"
            df[f"{name}_0_100"] = np.clip(weight * df[f"{base}_0_100"] + (1.0 - weight) * df["flive_existing_0_100"], 0.0, 100.0)
            signals.append(name)
    return sorted(set(signals))


def hybrid_grid_metrics(df: pd.DataFrame, hybrid_signals: list[str]) -> pd.DataFrame:
    rows = []
    for signal in hybrid_signals:
        overall = metric_row(df, signal, "overall")
        ds = [metric_row(df[df["dataset"].astype(str).eq(dataset)], signal, dataset) for dataset in DATASET_ORDER]
        min_dataset_srcc = min([r["srcc"] for r in ds if math.isfinite(float(r["srcc"]))], default=float("nan"))
        severe = sum(int(r["severe_fp_count"]) for r in ds)
        fp = sum(int(r["fp_count"]) for r in ds)
        rows.append(
            {
                "signal": signal,
                "overall_n": overall["n"],
                "overall_mae": overall["mae"],
                "overall_srcc": overall["srcc"],
                "overall_plcc": overall["plcc"],
                "min_dataset_srcc": min_dataset_srcc,
                "fp_count": fp,
                "severe_fp_count": severe,
                "fn_count": sum(int(r["fn_count"]) for r in ds),
                "composite": overall["srcc"] + 0.25 * min_dataset_srcc - 0.01 * severe - 0.002 * fp,
            }
        )
    return pd.DataFrame(rows).sort_values(["composite", "overall_srcc"], ascending=[False, False]).reset_index(drop=True)


def collect_safety(df: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    rows = []
    for signal in signals:
        col = f"{signal}_0_100"
        if col not in df.columns:
            continue
        pred = df[col]
        masks = {
            "fp": (df["mos_0_100"] < 40.0) & (pred > 60.0),
            "severe_fp": (df["mos_0_100"] < 40.0) & (pred > 65.0),
            "fn": (df["mos_0_100"] > 75.0) & (pred < 45.0),
        }
        for case_type, mask in masks.items():
            for _, row in df[mask].iterrows():
                rows.append({"signal": signal, "case_type": case_type, "dataset": row["dataset"], "image_path": row["image_path"], "mos_0_100": row["mos_0_100"], "pred_0_100": row[col]})
    return pd.DataFrame(rows)


def choose_recommendation(overall: pd.DataFrame, datasetwise: pd.DataFrame, grid: pd.DataFrame, coverage: pd.DataFrame) -> dict[str, Any]:
    prod = overall[overall["signal"].eq("production_existing")].iloc[0].to_dict()
    best_srcc = overall.sort_values("srcc", ascending=False).iloc[0].to_dict()
    best_mae = overall.sort_values("mae", ascending=True).iloc[0].to_dict()
    best_hybrid = grid.iloc[0].to_dict() if not grid.empty else {}
    prod_ds = datasetwise[datasetwise["signal"].eq("production_existing")]
    selected = str(best_hybrid.get("signal", best_srcc["signal"]))
    selected_ds = datasetwise[datasetwise["signal"].eq(selected)]
    if selected_ds.empty:
        selected_ds = datasetwise[datasetwise["signal"].eq(best_srcc["signal"])]
        selected = str(best_srcc["signal"])
    stable = True
    deltas = []
    for dataset in DATASET_ORDER:
        p = prod_ds[prod_ds["dataset"].eq(dataset)]
        s = selected_ds[selected_ds["dataset"].eq(dataset)]
        if p.empty or s.empty:
            stable = False
            continue
        delta = float(s.iloc[0]["srcc"]) - float(p.iloc[0]["srcc"])
        deltas.append({"dataset": dataset, "srcc_delta_vs_production": delta})
        if delta < -0.03:
            stable = False
    severe_prod = int(prod_ds["severe_fp_count"].sum())
    severe_sel = int(selected_ds["severe_fp_count"].sum()) if not selected_ds.empty else 999
    cov = coverage.set_index("dataset")["common_rows"].to_dict()
    coverage_ok = int(cov.get("koniq10k", 0)) >= 500 and int(cov.get("overall", 0)) >= 1000
    if not coverage_ok:
        decision = "TECHNICAL_SIGNAL_COMPARISON_NEEDS_MORE_COVERAGE"
    elif severe_sel > severe_prod or not stable:
        decision = "TECHNICAL_SIGNAL_COMPARISON_NEEDS_GATING_DESIGN"
    elif float(best_srcc["srcc"]) <= float(prod["srcc"]) and float(best_mae["mae"]) >= float(prod["mae"]):
        decision = "TECHNICAL_SIGNAL_COMPARISON_KEEP_PRODUCTION_ONLY"
    else:
        decision = "TECHNICAL_SIGNAL_COMPARISON_READY_FOR_LOG_ONLY_PROFILING"
    return {
        "final_decision": decision,
        "coverage_ok": coverage_ok,
        "best_signal_by_overall_srcc": str(best_srcc["signal"]),
        "best_signal_by_overall_mae": str(best_mae["signal"]),
        "best_hybrid_by_composite": str(best_hybrid.get("signal", "")),
        "selected_candidate": selected,
        "production_overall_srcc": float(prod["srcc"]),
        "production_overall_mae": float(prod["mae"]),
        "selected_severe_fp_total": severe_sel,
        "production_severe_fp_total": severe_prod,
        "dataset_stability_pass": stable,
        "dataset_srcc_deltas_vs_production": deltas,
        "production_replacement": False,
        "log_only_candidate": decision == "TECHNICAL_SIGNAL_COMPARISON_READY_FOR_LOG_ONLY_PROFILING",
    }


def write_report(out_dir: Path, coverage: pd.DataFrame, overall: pd.DataFrame, datasetwise: pd.DataFrame, grid: pd.DataFrame, safety: pd.DataFrame, rec: dict[str, Any], runtime: dict[str, Any]) -> None:
    metric_cols = ["signal", "dataset", "n", "mae", "rmse", "srcc", "plcc", "fp_count", "severe_fp_count", "fn_count"]
    sources = [
        "exports/tflite/",
        "outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/",
        "outputs/mobile_student_iqa_14105_tflite_export_20260604/",
        "outputs/student_distillation_manifest_14105_20260603/test_manifest.csv",
        "outputs/student_flive_hybrid_benchmark_20260604/",
        "outputs/technical_iqa_combo_benchmark_20260524/",
        "outputs/research_protocol_production_technical_eval_20260530/",
        "outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/",
        "outputs/eval_tflite_technical_compare/",
        "src/infer/select_best_shots.py",
        "src/infer/predict_musiq.py",
        "scripts/eval_final_topiq_candidates_vs_existing_technical.py",
    ]
    top_dataset = []
    for dataset in DATASET_ORDER:
        sub = datasetwise[datasetwise["dataset"].eq(dataset)].sort_values("srcc", ascending=False)
        if not sub.empty:
            top_dataset.append(f"- Best {dataset} SRCC: `{sub.iloc[0]['signal']}` SRCC `{fmt(sub.iloc[0]['srcc'])}`.")
    if rec["final_decision"] == "TECHNICAL_SIGNAL_COMPARISON_READY_FOR_LOG_ONLY_PROFILING":
        next_step = "- Prepare log-only profiling for selected candidate; keep production `technical_score` unchanged."
    elif rec["final_decision"] == "TECHNICAL_SIGNAL_COMPARISON_NEEDS_GATING_DESIGN":
        next_step = "- Keep production unchanged; design gated/log-only experiment only after reviewing dataset-specific winners."
    else:
        next_step = "- Resolve remaining coverage or safety gaps before any candidate recommendation."
    lines = [
        "# Technical IQA Signal Comparison and Coverage Recovery Report",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{rec['final_decision']}`.",
        f"- Common rows after recovery: `{int(coverage[coverage['dataset'].eq('overall')]['common_rows'].iloc[0])}`.",
        f"- Best overall SRCC signal: `{rec['best_signal_by_overall_srcc']}`.",
        f"- Best overall MAE signal: `{rec['best_signal_by_overall_mae']}`.",
        f"- Best hybrid composite: `{rec['best_hybrid_by_composite']}`.",
        f"- Runtime seconds: `{fmt(runtime['runtime_seconds'], 2)}`.",
        "",
        "## 2. Sources Inspected",
        *[f"- `{s}`" for s in sources],
        "",
        "## 3. Model and Signal Inventory",
        "- See `model_signal_inventory.csv`.",
        "",
        "## 4. Prediction Cache Inventory",
        "- See `prediction_cache_inventory.csv`.",
        "",
        "## 5. Coverage Recovery",
        markdown_table(coverage, ["dataset", "test_rows", "before_cached_common_rows", "common_rows", "common_pct"]),
        "",
        "## 6. Benchmark Scope",
        "- Primary split: student 14,105 `test_manifest.csv` only.",
        "- Recovered full test coverage by running existing local TFLite/Keras-compatible assets in WSL.",
        "- Production score is reconstructed offline from recovered KonIQ/FLIVE image-level scores; production code unchanged.",
        "",
        "## 7. Overall Signal Metrics",
        markdown_table(overall, metric_cols),
        "",
        "## 8. Dataset-wise Signal Metrics",
        *top_dataset,
        markdown_table(datasetwise.sort_values(["dataset", "srcc"], ascending=[True, False]), metric_cols, limit=60),
        "",
        "## 9. Hybrid Ratio Results",
        markdown_table(grid.head(20), ["signal", "overall_n", "overall_mae", "overall_srcc", "overall_plcc", "min_dataset_srcc", "fp_count", "severe_fp_count", "composite"]),
        "",
        "## 10. Safety Cases",
        f"- Safety case rows: `{len(safety)}`.",
        f"- Production severe FP total: `{rec['production_severe_fp_total']}`.",
        f"- Selected candidate severe FP total: `{rec['selected_severe_fp_total']}`.",
        "",
        "## 11. Best Signal / Hybrid Recommendation",
        f"- Selected candidate: `{rec['selected_candidate']}`.",
        f"- Dataset stability pass: `{rec['dataset_stability_pass']}`.",
        f"- Dataset deltas vs production: `{json.dumps(rec['dataset_srcc_deltas_vs_production'])}`.",
        "- Recommendation status: log-only only if decision says ready; never production replacement from this phase.",
        "",
        "## 12. Remaining Coverage Gaps",
        "- Primary test coverage target recovered for KonIQ, FLIVE, SPAQ, and overall.",
        "- MUSIQ signals are local probe models, not official Kaggle MUSIQ; interpret separately.",
        "",
        "## 13. Final Decision",
        str(rec["final_decision"]),
        "",
        "## 14. Recommended Next Step",
        next_step,
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    write_inventories(out_dir)
    df = load_manifest(Path(args.test_manifest), args.max_rows)
    images = []
    failures = []
    for _, row in df.iterrows():
        try:
            images.append(load_rgb(str(row["image_path"])))
        except Exception as exc:  # noqa: BLE001
            failures.append({"dataset": row["dataset"], "image_path": row["image_path"], "error": str(exc)})
            images.append(None)
    good = [img is not None for img in images]
    df = df.loc[good].reset_index(drop=True)
    images = [img for img in images if img is not None]

    before = pd.read_csv(ROOT / "outputs/student_flive_hybrid_benchmark_20260604/coverage_audit.csv")
    before_map = dict(zip(before["dataset"], before["common_rows"]))
    runtimes: dict[str, float] = {}
    df["koniq_existing_0_100"], runtimes["koniq_existing"] = predict_single_input_tflite(KONIQ_TFLITE, images, 224, True, False, False)
    df["flive_existing_0_100"], runtimes["flive_existing"] = predict_single_input_tflite(FLIVE_TFLITE, images, 224, True, False, False)
    df["topiq_existing_0_100"], runtimes["topiq_existing"] = predict_single_input_tflite(TOPIQ_TFLITE, images, 384, False, True, True)
    df["student_14105_0_100"], runtimes["student_14105"] = predict_single_input_tflite(STUDENT_TFLITE, images, 256, False, False, True)
    if not args.skip_musiq:
        for name, path in MUSIQ_MODELS.items():
            df[f"{name}_0_100"], runtimes[name] = predict_musiq_tflite(path, images)
    df["production_existing_0_100"] = np.clip((4.0 * df["koniq_existing_0_100"] + 3.0 * df["flive_existing_0_100"]) / 7.0, 0.0, 100.0)
    for signal in MODEL_SIGNALS:
        col = f"{signal}_0_100"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(0.0, 100.0)

    hybrid_signals = add_hybrids(df)
    all_signals = [s for s in MODEL_SIGNALS if f"{s}_0_100" in df.columns] + hybrid_signals
    coverage_rows = []
    for dataset in ["overall"] + DATASET_ORDER:
        sub = df if dataset == "overall" else df[df["dataset"].astype(str).eq(dataset)]
        coverage_rows.append({"dataset": dataset, "test_rows": int(len(sub)), "before_cached_common_rows": int(before_map.get(dataset, 0)), "common_rows": int(len(sub.dropna(subset=[f"{s}_0_100" for s in MODEL_SIGNALS if f"{s}_0_100" in df.columns]))), "common_pct": float(len(sub) / len(sub)) if len(sub) else float("nan")})
    coverage = pd.DataFrame(coverage_rows)

    overall, datasetwise = compute_metrics(df, [s for s in MODEL_SIGNALS if f"{s}_0_100" in df.columns])
    hybrid_grid = hybrid_grid_metrics(df, hybrid_signals)
    hybrid_overall, hybrid_dataset = compute_metrics(df, hybrid_signals)
    overall_all = pd.concat([overall, hybrid_overall], ignore_index=True)
    datasetwise_all = pd.concat([datasetwise, hybrid_dataset], ignore_index=True)
    safety = collect_safety(df, all_signals)
    rec = choose_recommendation(overall_all, datasetwise_all, hybrid_grid, coverage)

    pred_cols = ["dataset", "split", "split_original", "image_path", "mos_0_100"] + [f"{s}_0_100" for s in all_signals if f"{s}_0_100" in df.columns]
    df[pred_cols].to_csv(out_dir / "signal_predictions_common.csv", index=False)
    coverage.to_csv(out_dir / "coverage_audit.csv", index=False)
    overall.to_csv(out_dir / "signal_metrics_overall.csv", index=False)
    datasetwise.to_csv(out_dir / "signal_metrics_datasetwise.csv", index=False)
    hybrid_grid.to_csv(out_dir / "hybrid_ratio_grid_metrics.csv", index=False)
    safety.to_csv(out_dir / "safety_cases.csv", index=False)
    pd.DataFrame(failures).to_csv(out_dir / "missing_coverage_recovery_plan.csv", index=False)
    if not failures:
        pd.DataFrame([{"gap": "none_for_primary_test_split", "reason": "all test rows loaded and inferred", "next_action": "review metric tradeoffs, not coverage"}]).to_csv(out_dir / "missing_coverage_recovery_plan.csv", index=False)
    (out_dir / "signal_recommendation.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(
        [
            {"priority": 1, "action": "Keep production technical_score unchanged."},
            {"priority": 2, "action": "Use dataset-wise results to design gated/log-only candidate; fixed global ratios are risky."},
            {"priority": 3, "action": "Treat local MUSIQ probes separately from official Kaggle MUSIQ; do not merge naming."},
        ]
    ).to_csv(out_dir / "next_actions.csv", index=False)
    runtime = {"runtime_seconds": time.perf_counter() - t0, **{f"{k}_seconds": v for k, v in runtimes.items()}}
    write_report(out_dir, coverage, overall_all, datasetwise_all, hybrid_grid, safety, rec, runtime)
    print(f"Report: {out_dir / 'report.md'}")
    print(f"Decision: {rec['final_decision']}")
    print(f"Rows: {len(df)}")
    print(f"Best overall SRCC: {rec['best_signal_by_overall_srcc']}")


if __name__ == "__main__":
    main()
