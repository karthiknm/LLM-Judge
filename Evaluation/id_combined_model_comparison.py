from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr


DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]

DEFAULT_BATCH_FILES = {
    "1.5b": [
        Path("Evaluation/qwen25_1.5b_summeval_batch1_preds.jsonl"),
        Path("Evaluation/qwen25_1.5b_summeval_batch2_preds.jsonl"),
        Path("Evaluation/qwen25_1.5b_summeval_batch3_preds.jsonl"),
    ],
    "3b": [
        Path("Evaluation/qwen25_3b_batch1_summeval_preds.jsonl"),
        Path("Evaluation/qwen25_3b_batch2_summeval_preds.jsonl"),
        Path("Evaluation/qwen25_3b_batch3_summeval_preds.jsonl"),
    ],
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_dimension_from_text(raw: str, dim: str, next_dim: str | None) -> dict | None:
    start_marker = f'"{dim}": {{'
    start = raw.find(start_marker)
    if start == -1:
        return None

    block = raw[start : raw.find(f'"{next_dim}": {{', start + 1)] if next_dim else raw[start :]
    try:
        score = int(block.split('"score":')[1].split(",")[0].strip().replace("}", ""))
        return {"score": score}
    except Exception:
        return None


def try_parse_json(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None

    try:
        return json.loads(value)
    except Exception:
        parsed = {}
        for index, dim in enumerate(DIMENSIONS):
            next_dim = DIMENSIONS[index + 1] if index + 1 < len(DIMENSIONS) else None
            extracted = _extract_dimension_from_text(value, dim, next_dim)
            if extracted:
                parsed[dim] = extracted
        return parsed if parsed else None


def extract_scores(row: dict) -> dict[str, float | None]:
    source = row.get("prediction_json") or row.get("prediction_text")
    parsed = try_parse_json(source)
    scores = {}
    for dim in DIMENSIONS:
        score = None
        if isinstance(parsed, dict) and isinstance(parsed.get(dim), dict):
            score = parsed[dim].get("score")
        scores[dim] = score
    return scores


def safe_corr(fn, x, y) -> float:
    if len(x) < 2:
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = fn(x, y)[0]
        return float(value) if value is not None else np.nan
    except Exception:
        return np.nan


def summarize_rows(paths: list[Path]) -> tuple[dict[str, dict], dict[str, int]]:
    combined = {}
    coverage = {
        "source_files": len(paths),
        "rows": 0,
        "unique_ids": 0,
        "duplicate_ids": 0,
        "parse_ok_rows": 0,
    }

    for path in paths:
        rows = load_jsonl(path)
        coverage["rows"] += len(rows)
        coverage["parse_ok_rows"] += sum(bool(row.get("parse_ok")) for row in rows)
        for row in rows:
            row_id = str(row["id"])
            if row_id in combined:
                coverage["duplicate_ids"] += 1
            combined[row_id] = row

    coverage["unique_ids"] = len(combined)
    return combined, coverage


def build_eval_frame(rows_by_id: dict[str, dict]) -> pd.DataFrame:
    records = []
    for row_id, row in rows_by_id.items():
        scores = extract_scores(row)
        record = {
            "id": row_id,
            "parse_ok": bool(row.get("parse_ok")),
        }
        for dim in DIMENSIONS:
            record[f"pred_{dim}"] = scores[dim]
            record[f"gold_{dim}"] = row.get(f"gold_{dim}")
        records.append(record)
    return pd.DataFrame(records)


def evaluate_frame(df: pd.DataFrame, model_name: str, scope: str) -> pd.DataFrame:
    rows = []
    for dim in DIMENSIONS:
        pred = pd.to_numeric(df[f"pred_{dim}"], errors="coerce")
        gold = pd.to_numeric(df[f"gold_{dim}"], errors="coerce")
        mask = ~(pred.isna() | gold.isna())

        pred_values = pred[mask].to_numpy(dtype=float)
        gold_values = gold[mask].to_numpy(dtype=float)
        abs_error = np.abs(pred_values - gold_values)
        sq_error = (pred_values - gold_values) ** 2
        signed_error = pred_values - gold_values

        row = {
            "model": model_name,
            "scope": scope,
            "dimension": dim,
            "n": int(len(pred_values)),
            "spearman": safe_corr(spearmanr, pred_values, gold_values),
            "pearson": safe_corr(pearsonr, pred_values, gold_values),
            "kendall": safe_corr(kendalltau, pred_values, gold_values),
            "mae": np.mean(abs_error) if len(abs_error) else np.nan,
            "rmse": np.sqrt(np.mean(sq_error)) if len(sq_error) else np.nan,
            "exact_match": np.mean(pred_values == gold_values) if len(pred_values) else np.nan,
            "off_by_1_acc": np.mean(abs_error <= 1) if len(abs_error) else np.nan,
            "mean_signed_error": np.mean(signed_error) if len(signed_error) else np.nan,
            "pred_mean": np.mean(pred_values) if len(pred_values) else np.nan,
            "gold_mean": np.mean(gold_values) if len(gold_values) else np.nan,
        }
        rows.append(row)

    eval_df = pd.DataFrame(rows)
    metric_cols = [
        "spearman",
        "pearson",
        "kendall",
        "mae",
        "rmse",
        "exact_match",
        "off_by_1_acc",
        "mean_signed_error",
        "pred_mean",
        "gold_mean",
    ]

    macro = {
        "model": model_name,
        "scope": scope,
        "dimension": "macro_avg",
        "n": int(eval_df["n"].sum()),
    }
    for column in metric_cols:
        macro[column] = float(eval_df[column].mean())

    eval_df = pd.concat([eval_df, pd.DataFrame([macro])], ignore_index=True)
    eval_df[metric_cols] = eval_df[metric_cols].round(4)
    return eval_df


def aligned_frames(df_15b: pd.DataFrame, df_3b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = df_15b.merge(df_3b, on="id", suffixes=("_15b", "_3b"), how="inner")

    keep_15b = ["id"] + [f"{prefix}_15b" for prefix in ["parse_ok"] + [f"pred_{d}" for d in DIMENSIONS] + [f"gold_{d}" for d in DIMENSIONS]]
    keep_3b = ["id"] + [f"{prefix}_3b" for prefix in ["parse_ok"] + [f"pred_{d}" for d in DIMENSIONS] + [f"gold_{d}" for d in DIMENSIONS]]

    aligned_15b = aligned[keep_15b].rename(columns={c: c[:-4] for c in keep_15b if c.endswith("_15b")})
    aligned_3b = aligned[keep_3b].rename(columns={c: c[:-3] for c in keep_3b if c.endswith("_3b")})
    return aligned_15b, aligned_3b


def winner_for_metric(metric_name: str, model15_value: float, model3_value: float) -> str:
    if pd.isna(model15_value) or pd.isna(model3_value):
        return "n/a"
    lower_is_better = {"mae", "rmse", "mean_signed_error_abs"}
    if np.isclose(model15_value, model3_value, equal_nan=True):
        return "tie"
    if metric_name in lower_is_better:
        return "1.5b" if model15_value < model3_value else "3b"
    return "1.5b" if model15_value > model3_value else "3b"


def compare_models(aligned_15b: pd.DataFrame, aligned_3b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    merged = aligned_15b.merge(aligned_3b, on="id", suffixes=("_15b", "_3b"), how="inner")

    for dim in DIMENSIONS:
        pred_15 = pd.to_numeric(merged[f"pred_{dim}_15b"], errors="coerce")
        pred_3 = pd.to_numeric(merged[f"pred_{dim}_3b"], errors="coerce")
        gold_15 = pd.to_numeric(merged[f"gold_{dim}_15b"], errors="coerce")
        gold_3 = pd.to_numeric(merged[f"gold_{dim}_3b"], errors="coerce")

        # Gold labels should match; use both columns in the mask and keep one copy after filtering.
        mask = ~(pred_15.isna() | pred_3.isna() | gold_15.isna() | gold_3.isna())
        pred_15_values = pred_15[mask].to_numpy(dtype=float)
        pred_3_values = pred_3[mask].to_numpy(dtype=float)
        gold_values = gold_15[mask].to_numpy(dtype=float)

        abs_err_15 = np.abs(pred_15_values - gold_values)
        abs_err_3 = np.abs(pred_3_values - gold_values)
        signed_gap = pred_15_values - pred_3_values

        metrics = {
            "dimension": dim,
            "n": int(len(gold_values)),
            "model15_spearman": safe_corr(spearmanr, pred_15_values, gold_values),
            "model3_spearman": safe_corr(spearmanr, pred_3_values, gold_values),
            "model15_pearson": safe_corr(pearsonr, pred_15_values, gold_values),
            "model3_pearson": safe_corr(pearsonr, pred_3_values, gold_values),
            "model15_kendall": safe_corr(kendalltau, pred_15_values, gold_values),
            "model3_kendall": safe_corr(kendalltau, pred_3_values, gold_values),
            "model15_mae": np.mean(abs_err_15) if len(abs_err_15) else np.nan,
            "model3_mae": np.mean(abs_err_3) if len(abs_err_3) else np.nan,
            "model15_rmse": np.sqrt(np.mean((pred_15_values - gold_values) ** 2)) if len(gold_values) else np.nan,
            "model3_rmse": np.sqrt(np.mean((pred_3_values - gold_values) ** 2)) if len(gold_values) else np.nan,
            "model15_exact_match": np.mean(pred_15_values == gold_values) if len(gold_values) else np.nan,
            "model3_exact_match": np.mean(pred_3_values == gold_values) if len(gold_values) else np.nan,
            "model15_off_by_1_acc": np.mean(abs_err_15 <= 1) if len(gold_values) else np.nan,
            "model3_off_by_1_acc": np.mean(abs_err_3 <= 1) if len(gold_values) else np.nan,
            "model15_mean_signed_error": np.mean(pred_15_values - gold_values) if len(gold_values) else np.nan,
            "model3_mean_signed_error": np.mean(pred_3_values - gold_values) if len(gold_values) else np.nan,
            "model_prediction_spearman": safe_corr(spearmanr, pred_15_values, pred_3_values),
            "model_prediction_pearson": safe_corr(pearsonr, pred_15_values, pred_3_values),
            "model_exact_agreement": np.mean(pred_15_values == pred_3_values) if len(gold_values) else np.nan,
            "model_off_by_1_agreement": np.mean(np.abs(signed_gap) <= 1) if len(gold_values) else np.nan,
            "mean_prediction_gap_15_minus_3": np.mean(signed_gap) if len(gold_values) else np.nan,
            "mean_abs_prediction_gap": np.mean(np.abs(signed_gap)) if len(gold_values) else np.nan,
            "win_rate_15b_lower_abs_error": np.mean(abs_err_15 < abs_err_3) if len(gold_values) else np.nan,
            "win_rate_3b_lower_abs_error": np.mean(abs_err_3 < abs_err_15) if len(gold_values) else np.nan,
            "tie_rate_abs_error": np.mean(abs_err_15 == abs_err_3) if len(gold_values) else np.nan,
            "mae_advantage_3b_over_15b": np.mean(abs_err_15 - abs_err_3) if len(gold_values) else np.nan,
        }

        metrics["winner_spearman"] = winner_for_metric("spearman", metrics["model15_spearman"], metrics["model3_spearman"])
        metrics["winner_pearson"] = winner_for_metric("pearson", metrics["model15_pearson"], metrics["model3_pearson"])
        metrics["winner_kendall"] = winner_for_metric("kendall", metrics["model15_kendall"], metrics["model3_kendall"])
        metrics["winner_mae"] = winner_for_metric("mae", metrics["model15_mae"], metrics["model3_mae"])
        metrics["winner_rmse"] = winner_for_metric("rmse", metrics["model15_rmse"], metrics["model3_rmse"])
        metrics["winner_exact_match"] = winner_for_metric("exact_match", metrics["model15_exact_match"], metrics["model3_exact_match"])
        metrics["winner_off_by_1_acc"] = winner_for_metric("off_by_1_acc", metrics["model15_off_by_1_acc"], metrics["model3_off_by_1_acc"])

        rows.append(metrics)

    comparison_df = pd.DataFrame(rows)
    round_cols = [
        col
        for col in comparison_df.columns
        if col not in {"dimension", "n", "winner_spearman", "winner_pearson", "winner_kendall", "winner_mae", "winner_rmse", "winner_exact_match", "winner_off_by_1_acc"}
    ]
    comparison_df[round_cols] = comparison_df[round_cols].round(4)

    macro = {"dimension": "macro_avg", "n": int(comparison_df["n"].max()) if not comparison_df.empty else 0}
    for column in round_cols:
        macro[column] = float(comparison_df[column].mean())
    macro["winner_spearman"] = winner_for_metric("spearman", macro["model15_spearman"], macro["model3_spearman"])
    macro["winner_pearson"] = winner_for_metric("pearson", macro["model15_pearson"], macro["model3_pearson"])
    macro["winner_kendall"] = winner_for_metric("kendall", macro["model15_kendall"], macro["model3_kendall"])
    macro["winner_mae"] = winner_for_metric("mae", macro["model15_mae"], macro["model3_mae"])
    macro["winner_rmse"] = winner_for_metric("rmse", macro["model15_rmse"], macro["model3_rmse"])
    macro["winner_exact_match"] = winner_for_metric("exact_match", macro["model15_exact_match"], macro["model3_exact_match"])
    macro["winner_off_by_1_acc"] = winner_for_metric("off_by_1_acc", macro["model15_off_by_1_acc"], macro["model3_off_by_1_acc"])

    comparison_df = pd.concat([comparison_df, pd.DataFrame([macro])], ignore_index=True)
    comparison_df[round_cols] = comparison_df[round_cols].round(4)
    return comparison_df


def build_coverage_table(coverage_15b: dict[str, int], coverage_3b: dict[str, int], common_ids: int) -> pd.DataFrame:
    rows = [
        {"model": "1.5b", **coverage_15b},
        {"model": "3b", **coverage_3b},
    ]
    df = pd.DataFrame(rows)
    df["parse_ok_rate"] = (df["parse_ok_rows"] / df["rows"]).round(4)
    df["common_ids_with_other_model"] = common_ids
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine batches 1-3 and compare Qwen 1.5B vs 3B against Summeval gold."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Evaluation"),
        help="Directory for output CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_15b, coverage_15b = summarize_rows(DEFAULT_BATCH_FILES["1.5b"])
    rows_3b, coverage_3b = summarize_rows(DEFAULT_BATCH_FILES["3b"])

    df_15b_all = build_eval_frame(rows_15b)
    df_3b_all = build_eval_frame(rows_3b)

    eval_15b_all = evaluate_frame(df_15b_all, "1.5b", "all_available")
    eval_3b_all = evaluate_frame(df_3b_all, "3b", "all_available")
    all_available_eval = pd.concat([eval_15b_all, eval_3b_all], ignore_index=True)

    aligned_15b, aligned_3b = aligned_frames(df_15b_all, df_3b_all)
    eval_15b_common = evaluate_frame(aligned_15b, "1.5b", "common_ids_only")
    eval_3b_common = evaluate_frame(aligned_3b, "3b", "common_ids_only")
    common_eval = pd.concat([eval_15b_common, eval_3b_common], ignore_index=True)

    model_comparison = compare_models(aligned_15b, aligned_3b)
    coverage = build_coverage_table(coverage_15b, coverage_3b, len(set(rows_15b) & set(rows_3b)))

    all_available_path = output_dir / "combined_summeval_eval_all_available.csv"
    common_eval_path = output_dir / "combined_summeval_eval_common_ids.csv"
    comparison_path = output_dir / "combined_model_comparison_common_ids.csv"
    coverage_path = output_dir / "combined_model_coverage.csv"

    all_available_eval.to_csv(all_available_path, index=False)
    common_eval.to_csv(common_eval_path, index=False)
    model_comparison.to_csv(comparison_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    print("Saved:")
    print(f"  {all_available_path}")
    print(f"  {common_eval_path}")
    print(f"  {comparison_path}")
    print(f"  {coverage_path}")
    print()
    print("Coverage summary:")
    print(coverage.to_string(index=False))
    print()
    print("All available rows vs Summeval gold:")
    print(all_available_eval.to_string(index=False))
    print()
    print("Common ids only vs Summeval gold:")
    print(common_eval.to_string(index=False))
    print()
    print("Direct 1.5B vs 3B comparison on common ids:")
    print(model_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
