#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bloque 3 ejecutivo: ranking final ponderado, recomendaciones por escenario "
            "y reporte HTML unico."
        )
    )
    parser.add_argument(
        "--block1-root",
        type=Path,
        default=Path("artifacts/model_dashboard_v1"),
        help="Raiz de artefactos de bloque 1.",
    )
    parser.add_argument(
        "--block2-root",
        type=Path,
        default=Path("artifacts/model_dashboard_v2_probcal"),
        help="Raiz de artefactos de bloque 2.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/model_dashboard_v3_executive"),
        help="Carpeta de salida del bloque 3.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Cantidad de modelos a mostrar en tablas resumen.",
    )
    parser.add_argument(
        "--calib-f1-gap",
        type=float,
        default=0.02,
        help=(
            "Gap maximo permitido respecto al mejor f1_macro para seleccionar "
            "campeon de calibracion-practica."
        ),
    )
    return parser.parse_args()


def ensure_dirs(output_root: Path) -> Dict[str, Path]:
    paths = {
        "logs": output_root / "logs",
        "plots": output_root / "plots",
        "report": output_root / "report",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def write_plotly_html(fig: go.Figure, output_path: Path) -> None:
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)


def normalize_series_benefit(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    s_min = float(s.min())
    s_max = float(s.max())
    if np.isclose(s_min, s_max):
        return pd.Series(np.ones(len(s)), index=s.index, dtype=float)
    return (s - s_min) / (s_max - s_min)


def normalize_series_cost(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    s_min = float(s.min())
    s_max = float(s.max())
    if np.isclose(s_min, s_max):
        return pd.Series(np.ones(len(s)), index=s.index, dtype=float)
    return (s_max - s) / (s_max - s_min)


def load_inputs(block1_root: Path, block2_root: Path) -> Dict[str, pd.DataFrame]:
    p1 = block1_root / "comparison" / "logs"
    p2 = block2_root / "comparison" / "logs"

    required = {
        "perf": p1 / "model_metrics_test.csv",
        "bootstrap": p1 / "bootstrap_ci_metrics.csv",
        "mcnemar": p1 / "mcnemar_vs_baseline.csv",
        "class_f1": p1 / "class_f1_by_model.csv",
        "prob": p2 / "probability_metrics_summary.csv",
        "delta_prob": p2 / "delta_vs_svm_calibrated.csv",
        "roc_class": p2 / "roc_auc_by_class_by_model.csv",
        "pr_class": p2 / "pr_auc_by_class_by_model.csv",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Falta archivo requerido ({name}): {path}")

    data = {
        "perf": pd.read_csv(required["perf"]),
        "bootstrap": pd.read_csv(required["bootstrap"]),
        "mcnemar": pd.read_csv(required["mcnemar"]),
        "class_f1": pd.read_csv(required["class_f1"], index_col=0),
        "prob": pd.read_csv(required["prob"]),
        "delta_prob": pd.read_csv(required["delta_prob"]),
        "roc_class": pd.read_csv(required["roc_class"], index_col=0),
        "pr_class": pd.read_csv(required["pr_class"], index_col=0),
    }
    return data


def build_executive_table(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    prob = data["prob"].copy()
    perf = data["perf"].copy()
    bootstrap = data["bootstrap"].copy()
    mcn = data["mcnemar"].copy()
    class_f1 = data["class_f1"].copy()
    roc_class = data["roc_class"].copy()
    pr_class = data["pr_class"].copy()

    merged = prob.merge(
        perf[
            [
                "model",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
            ]
        ],
        on="model",
        how="left",
    )

    merged = merged.merge(
        bootstrap[
            [
                "model",
                "acc_ci_low",
                "acc_ci_high",
                "f1_macro_ci_low",
                "f1_macro_ci_high",
            ]
        ],
        on="model",
        how="left",
    )

    # Para svm_calibrated tomamos CI de svm como aproximacion del mismo clasificador base.
    if "svm_calibrated" in merged["model"].values and "svm" in bootstrap["model"].values:
        svm_ci = bootstrap[bootstrap["model"] == "svm"].iloc[0]
        mask = merged["model"] == "svm_calibrated"
        for col in ["acc_ci_low", "acc_ci_high", "f1_macro_ci_low", "f1_macro_ci_high"]:
            merged.loc[mask, col] = float(svm_ci[col])

    merged["f1_ci_width"] = merged["f1_macro_ci_high"] - merged["f1_macro_ci_low"]
    merged["acc_ci_width"] = merged["acc_ci_high"] - merged["acc_ci_low"]

    # Robustez por clase (std menor -> mejor estabilidad entre clases)
    def class_std_from_matrix(matrix: pd.DataFrame, model_name: str) -> float:
        if model_name in matrix.columns:
            return float(matrix[model_name].std(ddof=0))
        return float("nan")

    merged["class_f1_std"] = merged["model"].apply(lambda m: class_std_from_matrix(class_f1, m))
    merged["class_roc_auc_std"] = merged["model"].apply(lambda m: class_std_from_matrix(roc_class, m))
    merged["class_pr_auc_std"] = merged["model"].apply(lambda m: class_std_from_matrix(pr_class, m))

    # Señal estadistica vs baseline svm no calibrado (McNemar del bloque 1)
    mcn_map = mcn.set_index("model")[
        ["p_exact", "b_base_correct_other_wrong", "c_base_wrong_other_correct"]
    ]
    merged["p_mcnemar_vs_svm"] = merged["model"].map(mcn_map["p_exact"])
    merged["mcnemar_delta_correct_vs_svm"] = merged["model"].map(
        mcn_map["c_base_wrong_other_correct"] - mcn_map["b_base_correct_other_wrong"]
    )
    merged["significant_vs_svm_0p05"] = merged["p_mcnemar_vs_svm"].fillna(1.0) < 0.05

    return merged


def score_profiles(executive_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    df = executive_df.copy()

    # Normalizaciones
    benefit_cols = [
        "accuracy",
        "f1_macro",
        "precision_macro",
        "recall_macro",
        "roc_auc_macro_ovr",
        "pr_auc_macro",
    ]
    cost_cols = ["log_loss", "ece", "brier_multiclass", "f1_ci_width", "class_f1_std"]

    for col in benefit_cols:
        df[f"n_{col}"] = normalize_series_benefit(df[col])
    for col in cost_cols:
        df[f"ninv_{col}"] = normalize_series_cost(df[col])

    # Perfiles de ponderacion
    weights = {
        "performance_first": {
            "n_f1_macro": 0.40,
            "n_accuracy": 0.20,
            "n_precision_macro": 0.10,
            "n_recall_macro": 0.10,
            "n_roc_auc_macro_ovr": 0.10,
            "n_pr_auc_macro": 0.10,
        },
        "calibration_first": {
            "ninv_ece": 0.35,
            "ninv_log_loss": 0.25,
            "ninv_brier_multiclass": 0.20,
            "n_f1_macro": 0.10,
            "n_roc_auc_macro_ovr": 0.05,
            "n_pr_auc_macro": 0.05,
        },
        "balanced_reliable": {
            "n_f1_macro": 0.23,
            "n_accuracy": 0.14,
            "ninv_ece": 0.14,
            "ninv_log_loss": 0.10,
            "ninv_brier_multiclass": 0.08,
            "n_roc_auc_macro_ovr": 0.10,
            "n_pr_auc_macro": 0.10,
            "ninv_f1_ci_width": 0.06,
            "ninv_class_f1_std": 0.05,
        },
    }

    for profile_name, w in weights.items():
        score = np.zeros(len(df), dtype=float)
        for metric_col, weight in w.items():
            score += df[metric_col].fillna(0.0).values * weight
        df[f"score_{profile_name}"] = score

    return df, weights


def pick_recommendations(df: pd.DataFrame, calib_f1_gap: float) -> Dict[str, str | float]:
    # Campeon de performance puro
    perf_row = df.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]

    # Campeon de calibracion puro
    calib_row = df.sort_values(["ece", "log_loss", "f1_macro"], ascending=[True, True, False]).iloc[0]

    # Campeon de calibracion practica (similar performance y mejor calibracion)
    max_f1 = float(df["f1_macro"].max())
    candidate = df[df["f1_macro"] >= max_f1 - calib_f1_gap].copy()
    calib_practical_row = candidate.sort_values(["ece", "log_loss"], ascending=[True, True]).iloc[0]

    # Campeon balanceado por score
    bal_row = df.sort_values("score_balanced_reliable", ascending=False).iloc[0]

    return {
        "performance_champion": str(perf_row["model"]),
        "performance_champion_f1_macro": float(perf_row["f1_macro"]),
        "calibration_champion": str(calib_row["model"]),
        "calibration_champion_ece": float(calib_row["ece"]),
        "calibration_practical_champion": str(calib_practical_row["model"]),
        "calibration_practical_f1_macro": float(calib_practical_row["f1_macro"]),
        "calibration_practical_ece": float(calib_practical_row["ece"]),
        "balanced_champion": str(bal_row["model"]),
        "balanced_champion_score": float(bal_row["score_balanced_reliable"]),
        "calibration_gap_used": float(calib_f1_gap),
    }


def plot_pareto_f1_ece(df: pd.DataFrame, png_path: Path, html_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df["ece"], df["f1_macro"], s=100, c="#4e79a7", alpha=0.85)
    for _, row in df.iterrows():
        ax.text(row["ece"], row["f1_macro"], str(row["model"]), fontsize=8)
    ax.set_title("Pareto F1-macro vs ECE")
    ax.set_xlabel("ECE (menor mejor)")
    ax.set_ylabel("F1-macro (mayor mejor)")
    ax.grid(alpha=0.35)
    plt.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    fig_html = px.scatter(
        df,
        x="ece",
        y="f1_macro",
        size="accuracy",
        color="model",
        title="Pareto F1-macro vs ECE",
        hover_data=["log_loss", "roc_auc_macro_ovr", "pr_auc_macro"],
    )
    write_plotly_html(fig_html, html_path)


def plot_profile_scores(df: pd.DataFrame, png_path: Path, html_path: Path) -> None:
    cols = ["score_performance_first", "score_calibration_first", "score_balanced_reliable"]
    plot_df = df[["model"] + cols].copy()

    x = np.arange(len(plot_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width, plot_df[cols[0]], width=width, label="performance_first")
    ax.bar(x, plot_df[cols[1]], width=width, label="calibration_first")
    ax.bar(x + width, plot_df[cols[2]], width=width, label="balanced_reliable")
    ax.set_title("Scores por perfil de decision")
    ax.set_ylabel("Score normalizado ponderado")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    long_df = plot_df.melt(id_vars="model", var_name="profile", value_name="score")
    fig_html = px.bar(
        long_df,
        x="model",
        y="score",
        color="profile",
        barmode="group",
        title="Scores por perfil de decision",
    )
    write_plotly_html(fig_html, html_path)


def plot_heatmap_rank(df: pd.DataFrame, png_path: Path, html_path: Path) -> None:
    ranking = pd.DataFrame(
        {
            "performance_first": df["score_performance_first"]
            .rank(ascending=False, method="min")
            .to_numpy(),
            "calibration_first": df["score_calibration_first"]
            .rank(ascending=False, method="min")
            .to_numpy(),
            "balanced_reliable": df["score_balanced_reliable"]
            .rank(ascending=False, method="min")
            .to_numpy(),
        },
        index=df["model"].astype(str).to_numpy(),
    )
    ranking = ranking.astype(int)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(ranking.values, cmap="YlGn_r", aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title("Ranking por perfil (1 = mejor)")
    ax.set_xticks(np.arange(len(ranking.columns)))
    ax.set_xticklabels(ranking.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(ranking.index)))
    ax.set_yticklabels(ranking.index)
    for i in range(ranking.shape[0]):
        for j in range(ranking.shape[1]):
            ax.text(j, i, str(ranking.values[i, j]), ha="center", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(png_path, dpi=170)
    plt.close(fig)

    fig_html = go.Figure(
        data=go.Heatmap(
            z=ranking.values,
            x=ranking.columns.tolist(),
            y=ranking.index.tolist(),
            colorscale="YlGn_r",
            hovertemplate="Modelo=%{y}<br>Perfil=%{x}<br>Rank=%{z}<extra></extra>",
        )
    )
    fig_html.update_layout(title="Ranking por perfil (1 = mejor)")
    write_plotly_html(fig_html, html_path)


def plot_top_models_radar(df: pd.DataFrame, png_path: Path, html_path: Path) -> None:
    top_models = (
        df.sort_values("score_balanced_reliable", ascending=False)
        .head(4)["model"]
        .tolist()
    )
    metrics = [
        "n_f1_macro",
        "n_accuracy",
        "n_roc_auc_macro_ovr",
        "n_pr_auc_macro",
        "ninv_ece",
        "ninv_log_loss",
    ]
    labels = ["f1", "acc", "roc_auc", "pr_auc", "1-ece", "1-logloss"]

    # Matplotlib radar
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    for model in top_models:
        row = df[df["model"] == model].iloc[0]
        vals = [float(row[m]) for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, label=model)
        ax.fill(angles, vals, alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Top modelos (score balanceado) - radar normalizado")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))
    plt.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    # Plotly radar
    fig_html = go.Figure()
    for model in top_models:
        row = df[df["model"] == model].iloc[0]
        vals = [float(row[m]) for m in metrics]
        fig_html.add_trace(
            go.Scatterpolar(
                r=vals,
                theta=labels,
                fill="toself",
                name=model,
            )
        )
    fig_html.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Top modelos (score balanceado) - radar normalizado",
    )
    write_plotly_html(fig_html, html_path)


def render_executive_html(
    output_path: Path,
    rec: Dict[str, str | float],
    top_table: pd.DataFrame,
    calib_table: pd.DataFrame,
    balance_table: pd.DataFrame,
) -> None:
    def fmt_table(df: pd.DataFrame) -> str:
        return df.to_html(index=False, classes="tbl", border=0, justify="left")

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Reporte Ejecutivo de Modelos</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #222; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 20px; margin-top: 26px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px 14px; background: #fafafa; }}
    .k {{ font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: .3px; }}
    .v {{ font-size: 18px; font-weight: 600; margin-top: 4px; }}
    .tbl {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    .tbl th, .tbl td {{ border: 1px solid #e6e6e6; padding: 6px 8px; text-align: left; }}
    .tbl th {{ background: #f3f5f7; }}
    .hint {{ color: #666; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>Reporte Ejecutivo de Modelos</h1>
  <p class="hint">Resumen consolidado de rendimiento, calibración y estabilidad para decisión final.</p>

  <div class="cards">
    <div class="card"><div class="k">Campeon performance</div><div class="v">{rec['performance_champion']}</div></div>
    <div class="card"><div class="k">Campeon calibración</div><div class="v">{rec['calibration_champion']}</div></div>
    <div class="card"><div class="k">Campeon calibración práctica</div><div class="v">{rec['calibration_practical_champion']}</div></div>
    <div class="card"><div class="k">Campeon balanceado</div><div class="v">{rec['balanced_champion']}</div></div>
  </div>

  <h2>Top Por F1-Macro</h2>
  {fmt_table(top_table)}

  <h2>Top Perfil Calibración</h2>
  {fmt_table(calib_table)}

  <h2>Top Perfil Balanceado</h2>
  {fmt_table(balance_table)}

  <h2>Sugerencia de Uso</h2>
  <ul>
    <li><b>Si priorizas exactitud global:</b> {rec['performance_champion']}.</li>
    <li><b>Si priorizas confiabilidad probabilística:</b> {rec['calibration_champion']}.</li>
    <li><b>Si quieres equilibrio para producción:</b> {rec['balanced_champion']}.</li>
  </ul>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = ensure_dirs(args.output_root)

    data = load_inputs(args.block1_root, args.block2_root)
    exec_df = build_executive_table(data)
    exec_scored, weights = score_profiles(exec_df)
    rec = pick_recommendations(exec_scored, args.calib_f1_gap)

    exec_scored.sort_values("score_balanced_reliable", ascending=False).to_csv(
        out["logs"] / "executive_model_table.csv", index=False
    )

    ranking_df = exec_scored[
        [
            "model",
            "score_performance_first",
            "score_calibration_first",
            "score_balanced_reliable",
        ]
    ].copy()
    ranking_df.to_csv(out["logs"] / "ranking_profiles.csv", index=False)

    with open(out["logs"] / "profile_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)
    with open(out["logs"] / "recommendations.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    plot_pareto_f1_ece(
        exec_scored,
        png_path=out["plots"] / "pareto_f1_vs_ece.png",
        html_path=out["plots"] / "pareto_f1_vs_ece.html",
    )
    plot_profile_scores(
        exec_scored.sort_values("score_balanced_reliable", ascending=False),
        png_path=out["plots"] / "profile_scores.png",
        html_path=out["plots"] / "profile_scores.html",
    )
    plot_heatmap_rank(
        exec_scored.sort_values("score_balanced_reliable", ascending=False),
        png_path=out["plots"] / "rank_heatmap_profiles.png",
        html_path=out["plots"] / "rank_heatmap_profiles.html",
    )
    plot_top_models_radar(
        exec_scored,
        png_path=out["plots"] / "top_models_radar.png",
        html_path=out["plots"] / "top_models_radar.html",
    )

    top_perf = exec_scored.sort_values("f1_macro", ascending=False).head(args.top_k)[
        [
            "model",
            "accuracy",
            "f1_macro",
            "roc_auc_macro_ovr",
            "pr_auc_macro",
            "ece",
            "log_loss",
        ]
    ]
    top_cal = exec_scored.sort_values(["ece", "log_loss"], ascending=[True, True]).head(args.top_k)[
        [
            "model",
            "ece",
            "log_loss",
            "f1_macro",
            "accuracy",
            "roc_auc_macro_ovr",
        ]
    ]
    top_bal = exec_scored.sort_values("score_balanced_reliable", ascending=False).head(args.top_k)[
        [
            "model",
            "score_balanced_reliable",
            "f1_macro",
            "accuracy",
            "ece",
            "log_loss",
            "f1_ci_width",
        ]
    ]

    render_executive_html(
        output_path=out["report"] / "executive_report.html",
        rec=rec,
        top_table=top_perf.round(4),
        calib_table=top_cal.round(4),
        balance_table=top_bal.round(4),
    )

    summary = {
        "output_root": str(args.output_root),
        "recommendations": rec,
        "n_models": int(len(exec_scored)),
        "top_by_f1": top_perf["model"].head(3).tolist(),
        "top_by_calibration": top_cal["model"].head(3).tolist(),
        "top_by_balanced": top_bal["model"].head(3).tolist(),
    }
    with open(out["logs"] / "executive_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Bloque 3 ejecutivo generado")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 72)


if __name__ == "__main__":
    main()
