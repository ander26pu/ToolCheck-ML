#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera graficos para el reporte Top-5: HOG vs HOG+PCA."
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("artifacts/top5_hog_only_benchmark_v1/logs/comparison_hog_vs_hog_pca_top5.csv"),
        help="CSV con la comparativa HOG+PCA vs HOG.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets/top5_hog_vs_hog_pca"),
        help="Directorio de salida para imagenes PNG.",
    )
    parser.add_argument(
        "--hog-only-metrics-csv",
        type=Path,
        default=Path("artifacts/top5_hog_only_benchmark_v1/logs/top5_hog_only_metrics_test.csv"),
        help="CSV con metricas de test para HOG puro.",
    )
    parser.add_argument(
        "--hog-pca-metrics-csv",
        type=Path,
        default=Path("artifacts/top5_hog_pca_benchmark_v1/logs/top5_hog_pca_metrics_test.csv"),
        help="CSV con metricas de test para HOG+PCA.",
    )
    parser.add_argument(
        "--hog-summary-json",
        type=Path,
        default=Path("artifacts/top5_hog_only_benchmark_v1/logs/summary.json"),
        help="Summary JSON para extraer n_test si no se especifica --n-test.",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=None,
        help="Numero de muestras de test. Si se omite, se intenta leer desde --hog-summary-json.",
    )
    return parser.parse_args()


def fmt_secs(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}m"
    return f"{sec / 3600:.2f}h"


def label_bars(ax: plt.Axes, values: np.ndarray, offset: float = 0.01) -> None:
    max_v = float(np.max(values)) if len(values) else 1.0
    for i, val in enumerate(values):
        ax.text(
            i,
            float(val) + offset * max_v,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_metric_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.36

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    f1_pca = df["hog_pca_f1_macro_test"].to_numpy(dtype=float)
    f1_hog = df["hog_only_f1_macro_test"].to_numpy(dtype=float)
    axes[0].bar(x - width / 2, f1_pca, width=width, label="HOG+PCA", color="#4c78a8")
    axes[0].bar(x + width / 2, f1_hog, width=width, label="HOG", color="#f58518")
    axes[0].set_ylabel("F1 macro (test)")
    axes[0].set_ylim(0.75, 0.90)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].legend()
    axes[0].set_title("Comparacion de F1 macro en test")

    acc_pca = df["hog_pca_accuracy_test"].to_numpy(dtype=float)
    acc_hog = df["hog_only_accuracy_test"].to_numpy(dtype=float)
    axes[1].bar(x - width / 2, acc_pca, width=width, label="HOG+PCA", color="#4c78a8")
    axes[1].bar(x + width / 2, acc_hog, width=width, label="HOG", color="#f58518")
    axes[1].set_ylabel("Accuracy (test)")
    axes[1].set_ylim(0.75, 0.90)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].set_title("Comparacion de Accuracy en test")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)

    fig.suptitle("Top-5 modelos: calidad predictiva HOG vs HOG+PCA", fontsize=13, y=0.98)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_f1_accuracy.png", dpi=180)
    plt.close(fig)


def plot_time_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.36

    fit_pca = df["hog_pca_fit_sec_trainval"].to_numpy(dtype=float)
    fit_hog = df["hog_only_fit_sec_trainval"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width / 2, fit_pca, width=width, label="HOG+PCA", color="#4c78a8")
    ax.bar(x + width / 2, fit_hog, width=width, label="HOG", color="#f58518")

    ax.set_yscale("log")
    ax.set_ylabel("Tiempo de entrenamiento (seg, escala log)")
    ax.set_title("Comparacion de costo de entrenamiento")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    ax.set_xticks(x)
    ax.set_xticklabels(models)

    for i, v in enumerate(fit_hog):
        ax.text(i + width / 2, v * 1.05, fmt_secs(float(v)), ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(fit_pca):
        ax.text(i - width / 2, v * 1.05, fmt_secs(float(v)), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "comparison_fit_time_log.png", dpi=180)
    plt.close(fig)


def plot_delta_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.36

    d_f1 = df["delta_f1_hog_only_minus_hog_pca"].to_numpy(dtype=float)
    d_acc = df["delta_accuracy_hog_only_minus_hog_pca"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.bar(x - width / 2, d_f1, width=width, label="Delta F1", color="#54a24b")
    ax.bar(x + width / 2, d_acc, width=width, label="Delta Accuracy", color="#e45756")
    ax.set_title("Ganancia de HOG sobre HOG+PCA (HOG - HOG+PCA)")
    ax.set_ylabel("Delta")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(models)

    label_bars(ax, d_f1, offset=0.05)
    label_bars(ax, d_acc, offset=0.05)

    fig.tight_layout()
    fig.savefig(output_dir / "comparison_delta_metrics.png", dpi=180)
    plt.close(fig)


def plot_rankings(df: pd.DataFrame, output_dir: Path) -> None:
    rank_pca = df.sort_values("hog_pca_f1_macro_test", ascending=True)
    rank_hog = df.sort_values("hog_only_f1_macro_test", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=False)

    axes[0].barh(rank_pca["model"], rank_pca["hog_pca_f1_macro_test"], color="#4c78a8")
    axes[0].set_title("Ranking F1 (HOG+PCA)")
    axes[0].set_xlabel("F1 macro")
    axes[0].set_xlim(0.78, 0.87)
    axes[0].grid(axis="x", linestyle="--", alpha=0.35)

    axes[1].barh(rank_hog["model"], rank_hog["hog_only_f1_macro_test"], color="#f58518")
    axes[1].set_title("Ranking F1 (HOG)")
    axes[1].set_xlabel("F1 macro")
    axes[1].set_xlim(0.78, 0.87)
    axes[1].grid(axis="x", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_dir / "comparison_rankings_f1.png", dpi=180)
    plt.close(fig)


def plot_individual_cards(df: pd.DataFrame, output_dir: Path) -> None:
    for _, row in df.iterrows():
        model = str(row["model"])

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

        axes[0].bar(["HOG+PCA", "HOG"], [row["hog_pca_accuracy_test"], row["hog_only_accuracy_test"]], color=["#4c78a8", "#f58518"])
        axes[0].set_ylim(0.75, 0.90)
        axes[0].set_title("Accuracy")
        axes[0].grid(axis="y", linestyle="--", alpha=0.35)

        axes[1].bar(["HOG+PCA", "HOG"], [row["hog_pca_f1_macro_test"], row["hog_only_f1_macro_test"]], color=["#4c78a8", "#f58518"])
        axes[1].set_ylim(0.75, 0.90)
        axes[1].set_title("F1 macro")
        axes[1].grid(axis="y", linestyle="--", alpha=0.35)

        axes[2].bar(["HOG+PCA", "HOG"], [row["hog_pca_fit_sec_trainval"], row["hog_only_fit_sec_trainval"]], color=["#4c78a8", "#f58518"])
        axes[2].set_yscale("log")
        axes[2].set_title("Tiempo fit (log)")
        axes[2].grid(axis="y", linestyle="--", alpha=0.35)

        fig.suptitle(f"{model}: vista individual HOG+PCA vs HOG", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / f"individual_{model}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def resolve_n_test(args: argparse.Namespace) -> int:
    if args.n_test is not None:
        return int(args.n_test)
    if args.hog_summary_json.exists():
        payload = json.loads(args.hog_summary_json.read_text(encoding="utf-8"))
        n_test = payload.get("n_test")
        if n_test is not None:
            return int(n_test)
    return 252


def build_inference_summary(
    ordered_models: list[str],
    hog_only_metrics: pd.DataFrame,
    hog_pca_metrics: pd.DataFrame,
    n_test: int,
) -> pd.DataFrame:
    hog_only = hog_only_metrics.set_index("model")
    hog_pca = hog_pca_metrics.set_index("model")

    rows: list[dict[str, object]] = []
    for model in ordered_models:
        pca_row = hog_pca.loc[model]
        hog_row = hog_only.loc[model]

        pca_latency_ms = float(pca_row["predict_sec_test"]) * 1000.0 / n_test
        hog_latency_ms = float(hog_row["predict_sec_test"]) * 1000.0 / n_test

        rows.append(
            {
                "model": model,
                "hog_pca_predict_sec_test": float(pca_row["predict_sec_test"]),
                "hog_only_predict_sec_test": float(hog_row["predict_sec_test"]),
                "hog_pca_latency_ms_per_sample": pca_latency_ms,
                "hog_only_latency_ms_per_sample": hog_latency_ms,
                "hog_pca_throughput_samples_per_sec": 1000.0 / pca_latency_ms,
                "hog_only_throughput_samples_per_sec": 1000.0 / hog_latency_ms,
                "hog_pca_f1_macro_test": float(pca_row["f1_macro"]),
                "hog_only_f1_macro_test": float(hog_row["f1_macro"]),
                "hog_pca_eff_f1_over_latency": float(pca_row["f1_macro"]) / pca_latency_ms,
                "hog_only_eff_f1_over_latency": float(hog_row["f1_macro"]) / hog_latency_ms,
            }
        )
    return pd.DataFrame(rows)


def plot_inference_latency(df: pd.DataFrame, output_dir: Path) -> None:
    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.36

    pca_latency = df["hog_pca_latency_ms_per_sample"].to_numpy(dtype=float)
    hog_latency = df["hog_only_latency_ms_per_sample"].to_numpy(dtype=float)

    pca_thr = df["hog_pca_throughput_samples_per_sec"].to_numpy(dtype=float)
    hog_thr = df["hog_only_throughput_samples_per_sec"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    axes[0].bar(x - width / 2, pca_latency, width=width, label="HOG+PCA", color="#4c78a8")
    axes[0].bar(x + width / 2, hog_latency, width=width, label="HOG", color="#f58518")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Latencia por muestra (ms, log)")
    axes[0].set_title("Comparacion de latencia de inferencia")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].bar(x - width / 2, pca_thr, width=width, label="HOG+PCA", color="#4c78a8")
    axes[1].bar(x + width / 2, hog_thr, width=width, label="HOG", color="#f58518")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Throughput (muestras/s, log)")
    axes[1].set_title("Comparacion de throughput de inferencia")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)

    fig.tight_layout()
    fig.savefig(output_dir / "comparison_inference_latency_throughput.png", dpi=180)
    plt.close(fig)


def plot_inference_efficiency(df: pd.DataFrame, output_dir: Path) -> None:
    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.36

    pca_eff = df["hog_pca_eff_f1_over_latency"].to_numpy(dtype=float)
    hog_eff = df["hog_only_eff_f1_over_latency"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width / 2, pca_eff, width=width, label="HOG+PCA", color="#4c78a8")
    ax.bar(x + width / 2, hog_eff, width=width, label="HOG", color="#f58518")
    ax.set_yscale("log")
    ax.set_ylabel("Eficiencia = F1 / latencia_ms (log)")
    ax.set_title("Eficiencia de inferencia por modelo")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(models)

    fig.tight_layout()
    fig.savefig(output_dir / "comparison_inference_efficiency.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.comparison_csv)
    desired_order = ["logreg", "lightgbm", "svm", "rf", "xgboost"]
    df["order"] = df["model"].map({m: i for i, m in enumerate(desired_order)})
    df = df.sort_values("order").drop(columns=["order"]).reset_index(drop=True)

    plot_metric_comparison(df, args.output_dir)
    plot_time_comparison(df, args.output_dir)
    plot_delta_metrics(df, args.output_dir)
    plot_rankings(df, args.output_dir)
    plot_individual_cards(df, args.output_dir)

    n_test = resolve_n_test(args)
    hog_only_metrics = pd.read_csv(args.hog_only_metrics_csv)
    hog_pca_metrics = pd.read_csv(args.hog_pca_metrics_csv)
    inference_df = build_inference_summary(
        ordered_models=df["model"].tolist(),
        hog_only_metrics=hog_only_metrics,
        hog_pca_metrics=hog_pca_metrics,
        n_test=n_test,
    )
    inference_df.to_csv(args.output_dir / "inference_efficiency_table.csv", index=False)
    plot_inference_latency(inference_df, args.output_dir)
    plot_inference_efficiency(inference_df, args.output_dir)

    print(f"Graficos generados en: {args.output_dir}")


if __name__ == "__main__":
    main()
