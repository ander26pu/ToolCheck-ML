#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler


MODEL_REGISTRY: List[Tuple[str, str]] = [
    ("svm", "svm_hog_pca_v1/models/svm_best_model.joblib"),
    ("rf", "rf_hog_pca_v1/models/rf_best_model.joblib"),
    ("logreg", "logreg_hog_pca_v1/models/logreg_best_model.joblib"),
    ("lightgbm", "lightgbm_hog_pca_v1/models/lightgbm_best_model.joblib"),
    ("extratrees", "extratrees_hog_pca_v1/models/extratrees_best_model.joblib"),
    ("xgboost", "xgboost_hog_pca_v1/models/xgboost_best_model.joblib"),
    ("catboost", "catboost_hog_pca_v1/models/catboost_best_model.joblib"),
    (
        "gradientboosting",
        "gradientboosting_hog_pca_v1/models/gradientboosting_best_model.joblib",
    ),
    ("adaboost", "adaboost_hog_pca_v1/models/adaboost_best_model.joblib"),
    ("knn", "knn_hog_pca_v1/models/knn_best_model.joblib"),
    ("naivebayes", "naivebayes_hog_pca_v1/models/naivebayes_best_model.joblib"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera dashboard de diagnostico por modelo y comparativo "
            "(PNG + HTML + CSV) sobre HOG+PCA."
        )
    )
    parser.add_argument(
        "--features-split",
        type=Path,
        default=Path("artifacts/preprocess_hog_pca_v3/features/hog_pca_features_split.npz"),
        help="NPZ con train/val/test de HOG+PCA.",
    )
    parser.add_argument(
        "--features-all",
        type=Path,
        default=Path("artifacts/preprocess_hog_pca_v3/features/hog_pca_features.npz"),
        help="NPZ con X_pca completo para embeddings.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts"),
        help="Carpeta raiz de artefactos donde estan los modelos.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/model_dashboard_v1"),
        help="Carpeta de salida del dashboard.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=800,
        help="Iteraciones bootstrap para IC de metricas.",
    )
    parser.add_argument(
        "--max-embed-samples",
        type=int,
        default=1800,
        help="Maximo de muestras para embeddings (None/0 usa todas).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad.",
    )
    return parser.parse_args()


def ensure_dirs(output_root: Path) -> Dict[str, Path]:
    paths = {
        "models": output_root / "models",
        "comparison_logs": output_root / "comparison" / "logs",
        "comparison_plots": output_root / "comparison" / "plots",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_split_features(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = {"X_test", "y_test", "classes"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Faltan claves en NPZ split: {sorted(missing)}")
    return {k: data[k] for k in data.files}


def load_all_features(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = {"X_pca", "y", "split", "classes"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Faltan claves en NPZ all: {sorted(missing)}")
    return {k: data[k] for k in data.files}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
    }


def discover_models(artifacts_root: Path) -> Dict[str, Path]:
    discovered: Dict[str, Path] = {}
    for model_name, rel_path in MODEL_REGISTRY:
        model_path = artifacts_root / rel_path
        if model_path.exists():
            discovered[model_name] = model_path
    return discovered


def normalize_predictions(pred: np.ndarray, classes: List[str]) -> np.ndarray:
    pred_arr = np.asarray(pred).reshape(-1)
    if np.issubdtype(pred_arr.dtype, np.integer):
        labels: List[str] = []
        for idx in pred_arr.astype(int).tolist():
            if 0 <= idx < len(classes):
                labels.append(classes[idx])
            else:
                labels.append(str(idx))
        return np.array(labels, dtype=str)
    return pred_arr.astype(str)



def write_plotly_html(fig: go.Figure, output_path: Path) -> None:
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)

def to_class_report_df(report: Dict[str, Dict[str, float]], classes: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, float | str | int]] = []
    for class_name in classes:
        row = report.get(class_name, {})
        rows.append(
            {
                "class_name": class_name,
                "precision": float(row.get("precision", 0.0)),
                "recall": float(row.get("recall", 0.0)),
                "f1-score": float(row.get("f1-score", 0.0)),
                "support": int(row.get("support", 0)),
            }
        )
    for agg_name in ["macro avg", "weighted avg"]:
        row = report.get(agg_name, {})
        rows.append(
            {
                "class_name": agg_name,
                "precision": float(row.get("precision", 0.0)),
                "recall": float(row.get("recall", 0.0)),
                "f1-score": float(row.get("f1-score", 0.0)),
                "support": int(row.get("support", 0)),
            }
        )
    return pd.DataFrame(rows)

def plot_confusion_static(
    cm: np.ndarray,
    classes: List[str],
    title: str,
    output_path: Path,
    normalized: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 8.0))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_yticklabels(classes)

    fmt = ".2f" if normalized else "d"
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_html(
    cm: np.ndarray,
    classes: List[str],
    title: str,
    output_path: Path,
) -> None:
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale="Blues",
            hovertemplate="Real=%{y}<br>Pred=%{x}<br>Valor=%{z}<extra></extra>",
        )
    )
    fig.update_layout(title=title, xaxis_title="Prediccion", yaxis_title="Real", height=720)
    write_plotly_html(fig, output_path)


def plot_per_class_metrics(
    class_df: pd.DataFrame,
    model_name: str,
    png_path: Path,
    html_path: Path,
) -> None:
    classes_df = class_df[~class_df["class_name"].isin(["macro avg", "weighted avg"])].copy()
    x = np.arange(len(classes_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, classes_df["precision"], width=width, label="precision", color="#4e79a7")
    ax.bar(x, classes_df["recall"], width=width, label="recall", color="#f28e2b")
    ax.bar(x + width, classes_df["f1-score"], width=width, label="f1", color="#59a14f")
    ax.set_title(f"Metricas por clase - {model_name}")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(classes_df["class_name"], rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    long_df = classes_df.melt(
        id_vars=["class_name"],
        value_vars=["precision", "recall", "f1-score"],
        var_name="metric",
        value_name="score",
    )
    fig_html = px.bar(
        long_df,
        x="class_name",
        y="score",
        color="metric",
        barmode="group",
        title=f"Metricas por clase - {model_name}",
    )
    fig_html.update_layout(xaxis_title="Clase", yaxis_title="Score", yaxis=dict(range=[0.0, 1.0]))
    write_plotly_html(fig_html, html_path)


def extract_final_estimator(model):
    if hasattr(model, "named_steps") and isinstance(model.named_steps, dict):
        last_key = list(model.named_steps.keys())[-1]
        return model.named_steps[last_key]
    return model


def extract_feature_importance(model, n_features: int) -> np.ndarray | None:
    estimator = extract_final_estimator(model)

    importances: np.ndarray | None = None
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        importances = np.mean(np.abs(coef), axis=0)
    elif hasattr(estimator, "get_feature_importance"):
        try:
            importances = np.asarray(estimator.get_feature_importance(), dtype=float)
        except Exception:
            importances = None

    if importances is None:
        return None

    if importances.ndim > 1:
        importances = np.mean(np.abs(importances), axis=0)

    if importances.shape[0] != n_features:
        return None

    return importances


def plot_feature_importance(
    feature_df: pd.DataFrame,
    model_name: str,
    png_path: Path,
    html_path: Path,
) -> None:
    top_df = feature_df.sort_values("importance", ascending=False).head(25).copy()
    top_df = top_df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_df["feature"], top_df["importance"], color="#4e79a7")
    ax.set_title(f"Top feature importance - {model_name}")
    ax.set_xlabel("Importancia")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    fig_html = px.bar(
        top_df,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top feature importance - {model_name}",
    )
    write_plotly_html(fig_html, html_path)


def get_top_confusions(cm: np.ndarray, classes: List[str], top_k: int = 15) -> pd.DataFrame:
    rows: List[Dict[str, int | str]] = []
    for i, real in enumerate(classes):
        for j, pred in enumerate(classes):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                rows.append({"real": real, "pred": pred, "count": count})
    if not rows:
        return pd.DataFrame(columns=["real", "pred", "count"])
    df = pd.DataFrame(rows).sort_values("count", ascending=False).head(top_k).reset_index(drop=True)
    return df


def bootstrap_metrics_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iters: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc_vals: List[float] = []
    f1_vals: List[float] = []

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    for _ in range(n_iters):
        idx = rng.integers(0, n, size=n)
        yt = y_true_arr[idx]
        yp = y_pred_arr[idx]
        acc_vals.append(float(accuracy_score(yt, yp)))
        f1_vals.append(
            float(
                precision_recall_fscore_support(
                    yt,
                    yp,
                    average="macro",
                    zero_division=0,
                )[2]
            )
        )

    return {
        "acc_mean": float(np.mean(acc_vals)),
        "acc_ci_low": float(np.percentile(acc_vals, 2.5)),
        "acc_ci_high": float(np.percentile(acc_vals, 97.5)),
        "f1_macro_mean": float(np.mean(f1_vals)),
        "f1_macro_ci_low": float(np.percentile(f1_vals, 2.5)),
        "f1_macro_ci_high": float(np.percentile(f1_vals, 97.5)),
    }


def mcnemar_vs_baseline(
    y_true: np.ndarray,
    pred_base: np.ndarray,
    pred_other: np.ndarray,
) -> Dict[str, float | int]:
    base_correct = pred_base == y_true
    other_correct = pred_other == y_true

    b = int(np.sum(base_correct & ~other_correct))
    c = int(np.sum(~base_correct & other_correct))
    n = b + c
    if n == 0:
        return {
            "b_base_correct_other_wrong": b,
            "c_base_wrong_other_correct": c,
            "n_discordant": n,
            "chi2_cc": 0.0,
            "p_exact": 1.0,
        }

    chi2_cc = float(((abs(b - c) - 1) ** 2) / n)
    p_exact = float(binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
    return {
        "b_base_correct_other_wrong": b,
        "c_base_wrong_other_correct": c,
        "n_discordant": n,
        "chi2_cc": chi2_cc,
        "p_exact": p_exact,
    }

def class_color_map(classes: List[str]) -> Dict[str, str]:
    cmap = plt.get_cmap("tab20", len(classes))
    colors = {}
    for idx, cls in enumerate(classes):
        r, g, b, _ = cmap(idx)
        colors[cls] = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
    return colors


def sample_for_embedding(
    x: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    max_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_samples <= 0 or len(x) <= max_samples:
        return x, y, split

    rng = np.random.default_rng(seed)
    unique_classes = sorted(np.unique(y).tolist())
    n_per_class = max(1, max_samples // len(unique_classes))
    selected_idx: List[int] = []

    y_arr = np.asarray(y)
    for cls in unique_classes:
        idx_cls = np.where(y_arr == cls)[0]
        take = min(len(idx_cls), n_per_class)
        chosen = rng.choice(idx_cls, size=take, replace=False)
        selected_idx.extend(chosen.tolist())

    selected = np.array(sorted(selected_idx), dtype=int)
    if len(selected) > max_samples:
        selected = rng.choice(selected, size=max_samples, replace=False)
        selected = np.array(sorted(selected.tolist()), dtype=int)

    return x[selected], y[selected], split[selected]


def plot_embedding_2d(
    emb: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    classes: List[str],
    title: str,
    png_path: Path,
    html_path: Path,
) -> None:
    colors = class_color_map(classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls in classes:
        mask = labels == cls
        if np.any(mask):
            ax.scatter(emb[mask, 0], emb[mask, 1], s=10, alpha=0.7, label=cls)
    ax.set_title(title)
    ax.set_xlabel("dim_1")
    ax.set_ylabel("dim_2")
    ax.legend(loc="best", fontsize=8, markerscale=1.2)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    df = pd.DataFrame(
        {
            "dim_1": emb[:, 0],
            "dim_2": emb[:, 1],
            "class_name": labels,
            "split": splits,
        }
    )
    fig_html = px.scatter(
        df,
        x="dim_1",
        y="dim_2",
        color="class_name",
        symbol="split",
        title=title,
        color_discrete_map=colors,
        opacity=0.8,
    )
    write_plotly_html(fig_html, html_path)


def plot_embedding_3d(
    emb: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    classes: List[str],
    title: str,
    png_path: Path,
    html_path: Path,
) -> None:
    colors = class_color_map(classes)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for cls in classes:
        mask = labels == cls
        if np.any(mask):
            ax.scatter(emb[mask, 0], emb[mask, 1], emb[mask, 2], s=10, alpha=0.7, label=cls)
    ax.set_title(title)
    ax.set_xlabel("dim_1")
    ax.set_ylabel("dim_2")
    ax.set_zlabel("dim_3")
    ax.legend(loc="best", fontsize=7)
    plt.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    df = pd.DataFrame(
        {
            "dim_1": emb[:, 0],
            "dim_2": emb[:, 1],
            "dim_3": emb[:, 2],
            "class_name": labels,
            "split": splits,
        }
    )
    fig_html = px.scatter_3d(
        df,
        x="dim_1",
        y="dim_2",
        z="dim_3",
        color="class_name",
        symbol="split",
        title=title,
        color_discrete_map=colors,
        opacity=0.8,
    )
    fig_html.update_traces(marker=dict(size=3))
    write_plotly_html(fig_html, html_path)


def plot_comparison_bar(
    metrics_df: pd.DataFrame,
    png_path: Path,
    html_path: Path,
) -> None:
    show_cols = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    plot_df = metrics_df[["model"] + show_cols].copy()

    x = np.arange(len(plot_df))
    width = 0.2
    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, col in enumerate(show_cols):
        ax.bar(x + (idx - 1.5) * width, plot_df[col], width=width, label=col)
    ax.set_title("Comparacion global de modelos")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    long_df = plot_df.melt(id_vars=["model"], var_name="metric", value_name="score")
    fig_html = px.bar(
        long_df,
        x="model",
        y="score",
        color="metric",
        barmode="group",
        title="Comparacion global de modelos",
    )
    fig_html.update_layout(yaxis=dict(range=[0.0, 1.0]))
    write_plotly_html(fig_html, html_path)


def plot_heatmap_static(
    matrix: pd.DataFrame,
    title: str,
    png_path: Path,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(matrix.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=7)

    plt.tight_layout()
    fig.savefig(png_path, dpi=170)
    plt.close(fig)


def plot_heatmap_html(
    matrix: pd.DataFrame,
    title: str,
    html_path: Path,
    colorscale: str,
) -> None:
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale=colorscale,
            hovertemplate="Fila=%{y}<br>Col=%{x}<br>Valor=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, xaxis_title="Modelo", yaxis_title="Clase")
    write_plotly_html(fig, html_path)

def plot_split_distribution(
    y: np.ndarray,
    split: np.ndarray,
    classes: List[str],
    png_path: Path,
    html_path: Path,
) -> pd.DataFrame:
    df = pd.DataFrame({"class_name": y, "split": split})
    grouped = (
        df.groupby(["split", "class_name"]).size().reset_index(name="count")
    )

    split_order = ["train", "val", "test", "failed"]
    plot_df = (
        grouped.pivot(index="class_name", columns="split", values="count")
        .fillna(0)
        .reindex(index=classes, columns=[c for c in split_order if c in grouped["split"].unique()])
        .fillna(0)
    )

    x = np.arange(len(plot_df.index))
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(plot_df.index))
    for split_name in plot_df.columns:
        vals = plot_df[split_name].values
        ax.bar(x, vals, bottom=bottom, label=split_name)
        bottom += vals
    ax.set_title("Distribucion de clases por split")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=30, ha="right")
    ax.set_ylabel("Cantidad")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    fig_html = px.bar(
        grouped,
        x="class_name",
        y="count",
        color="split",
        title="Distribucion de clases por split",
        barmode="stack",
    )
    write_plotly_html(fig_html, html_path)

    return grouped


def run_embeddings(
    x_all: np.ndarray,
    y_all: np.ndarray,
    split_all: np.ndarray,
    classes: List[str],
    comparison_logs: Path,
    comparison_plots: Path,
    seed: int,
    max_samples: int,
) -> None:
    x_embed, y_embed, split_embed = sample_for_embedding(
        x_all,
        y_all,
        split_all,
        max_samples=max_samples,
        seed=seed,
    )

    pd.DataFrame(
        {
            "n_samples_total": [int(len(x_all))],
            "n_samples_embedding": [int(len(x_embed))],
            "max_samples": [int(max_samples)],
        }
    ).to_csv(comparison_logs / "embedding_metadata.csv", index=False)

    print(f"[embeddings] muestras usadas: {len(x_embed)}")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_embed)

    print("[embeddings] PCA 2D/3D...")
    pca_model = PCA(n_components=3, random_state=seed)
    pca_coords = pca_model.fit_transform(x_scaled)

    pd.DataFrame(
        {
            "dim_1": pca_coords[:, 0],
            "dim_2": pca_coords[:, 1],
            "dim_3": pca_coords[:, 2],
            "class_name": y_embed,
            "split": split_embed,
        }
    ).to_csv(comparison_logs / "embedding_pca_points.csv", index=False)

    plot_embedding_2d(
        emb=pca_coords[:, :2],
        labels=y_embed,
        splits=split_embed,
        classes=classes,
        title="PCA 2D sobre HOG+PCA",
        png_path=comparison_plots / "embedding_pca_2d.png",
        html_path=comparison_plots / "embedding_pca_2d.html",
    )
    plot_embedding_3d(
        emb=pca_coords,
        labels=y_embed,
        splits=split_embed,
        classes=classes,
        title="PCA 3D sobre HOG+PCA",
        png_path=comparison_plots / "embedding_pca_3d.png",
        html_path=comparison_plots / "embedding_pca_3d.html",
    )

    reduce_dim = min(50, x_scaled.shape[1])
    reducer = PCA(n_components=reduce_dim, random_state=seed)
    x_for_manifold = reducer.fit_transform(x_scaled)

    perplexity = min(35, max(8, (len(x_for_manifold) - 1) // 3))

    print("[embeddings] t-SNE 2D...")
    tsne_2d = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1200,
        random_state=seed,
    ).fit_transform(x_for_manifold)
    pd.DataFrame(
        {
            "dim_1": tsne_2d[:, 0],
            "dim_2": tsne_2d[:, 1],
            "class_name": y_embed,
            "split": split_embed,
        }
    ).to_csv(comparison_logs / "embedding_tsne_2d_points.csv", index=False)
    plot_embedding_2d(
        emb=tsne_2d,
        labels=y_embed,
        splits=split_embed,
        classes=classes,
        title="t-SNE 2D",
        png_path=comparison_plots / "embedding_tsne_2d.png",
        html_path=comparison_plots / "embedding_tsne_2d.html",
    )

    print("[embeddings] t-SNE 3D...")
    tsne_3d = TSNE(
        n_components=3,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1200,
        random_state=seed,
    ).fit_transform(x_for_manifold)
    pd.DataFrame(
        {
            "dim_1": tsne_3d[:, 0],
            "dim_2": tsne_3d[:, 1],
            "dim_3": tsne_3d[:, 2],
            "class_name": y_embed,
            "split": split_embed,
        }
    ).to_csv(comparison_logs / "embedding_tsne_3d_points.csv", index=False)
    plot_embedding_3d(
        emb=tsne_3d,
        labels=y_embed,
        splits=split_embed,
        classes=classes,
        title="t-SNE 3D",
        png_path=comparison_plots / "embedding_tsne_3d.png",
        html_path=comparison_plots / "embedding_tsne_3d.html",
    )

    print("[embeddings] UMAP 2D...")
    umap_2d = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.08,
        metric="euclidean",
        random_state=seed,
    ).fit_transform(x_for_manifold)
    pd.DataFrame(
        {
            "dim_1": umap_2d[:, 0],
            "dim_2": umap_2d[:, 1],
            "class_name": y_embed,
            "split": split_embed,
        }
    ).to_csv(comparison_logs / "embedding_umap_2d_points.csv", index=False)
    plot_embedding_2d(
        emb=umap_2d,
        labels=y_embed,
        splits=split_embed,
        classes=classes,
        title="UMAP 2D",
        png_path=comparison_plots / "embedding_umap_2d.png",
        html_path=comparison_plots / "embedding_umap_2d.html",
    )

    print("[embeddings] UMAP 3D...")
    umap_3d = umap.UMAP(
        n_components=3,
        n_neighbors=30,
        min_dist=0.08,
        metric="euclidean",
        random_state=seed,
    ).fit_transform(x_for_manifold)
    pd.DataFrame(
        {
            "dim_1": umap_3d[:, 0],
            "dim_2": umap_3d[:, 1],
            "dim_3": umap_3d[:, 2],
            "class_name": y_embed,
            "split": split_embed,
        }
    ).to_csv(comparison_logs / "embedding_umap_3d_points.csv", index=False)
    plot_embedding_3d(
        emb=umap_3d,
        labels=y_embed,
        splits=split_embed,
        classes=classes,
        title="UMAP 3D",
        png_path=comparison_plots / "embedding_umap_3d.png",
        html_path=comparison_plots / "embedding_umap_3d.html",
    )

def main() -> None:
    args = parse_args()
    out = ensure_dirs(args.output_root)

    split_data = load_split_features(args.features_split)
    all_data = load_all_features(args.features_all)

    x_test = split_data["X_test"].astype(np.float32)
    y_test = split_data["y_test"].astype(str)
    classes = [str(c) for c in split_data["classes"].tolist()]

    x_all = all_data["X_pca"].astype(np.float32)
    y_all = all_data["y"].astype(str)
    split_all = all_data["split"].astype(str)

    models = discover_models(args.artifacts_root)
    if not models:
        raise RuntimeError("No se encontraron modelos entrenados en artifacts/.")

    print(f"Modelos encontrados: {', '.join(models.keys())}")
    print(f"X_test shape: {x_test.shape}")

    predictions: Dict[str, np.ndarray] = {}
    metrics_rows: List[Dict[str, float | str]] = []
    class_f1_by_model: Dict[str, Dict[str, float]] = {}

    for model_name, model_path in models.items():
        print(f"[model] {model_name}: cargando {model_path}")
        model = joblib.load(model_path)

        raw_pred = model.predict(x_test)
        y_pred = normalize_predictions(raw_pred, classes)
        predictions[model_name] = y_pred

        metrics = compute_metrics(y_test, y_pred)
        metrics_rows.append({"model": model_name, **metrics})

        report = classification_report(
            y_test,
            y_pred,
            labels=classes,
            output_dict=True,
            zero_division=0,
        )
        class_df = to_class_report_df(report, classes)
        class_f1_by_model[model_name] = {
            cls: float(report.get(cls, {}).get("f1-score", 0.0)) for cls in classes
        }

        cm = confusion_matrix(y_test, y_pred, labels=classes)
        cm_norm = cm.astype(np.float64)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums > 0)

        model_dir = out["models"] / model_name
        model_logs = model_dir / "logs"
        model_plots = model_dir / "plots"
        model_logs.mkdir(parents=True, exist_ok=True)
        model_plots.mkdir(parents=True, exist_ok=True)

        class_df.to_csv(model_logs / "classification_report_test.csv", index=False)
        with open(model_logs / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump({"model": model_name, **metrics}, f, indent=2, ensure_ascii=False)

        top_conf = get_top_confusions(cm, classes, top_k=20)
        top_conf.to_csv(model_logs / "top_confusions.csv", index=False)

        plot_confusion_static(
            cm=cm,
            classes=classes,
            title=f"Confusion matrix counts - {model_name}",
            output_path=model_plots / "confusion_matrix_counts.png",
            normalized=False,
        )
        plot_confusion_static(
            cm=cm_norm,
            classes=classes,
            title=f"Confusion matrix normalized - {model_name}",
            output_path=model_plots / "confusion_matrix_normalized.png",
            normalized=True,
        )
        plot_confusion_html(
            cm=cm,
            classes=classes,
            title=f"Confusion matrix counts - {model_name}",
            output_path=model_plots / "confusion_matrix_counts.html",
        )
        plot_confusion_html(
            cm=cm_norm,
            classes=classes,
            title=f"Confusion matrix normalized - {model_name}",
            output_path=model_plots / "confusion_matrix_normalized.html",
        )

        plot_per_class_metrics(
            class_df=class_df,
            model_name=model_name,
            png_path=model_plots / "per_class_metrics.png",
            html_path=model_plots / "per_class_metrics.html",
        )

        importances = extract_feature_importance(model, n_features=x_test.shape[1])
        if importances is not None:
            fi_df = pd.DataFrame(
                {
                    "feature": [f"PC{i+1}" for i in range(len(importances))],
                    "importance": importances,
                }
            )
            fi_top = fi_df.sort_values("importance", ascending=False).head(60).reset_index(drop=True)
            fi_top.to_csv(model_logs / "feature_importance_top.csv", index=False)
            plot_feature_importance(
                feature_df=fi_df,
                model_name=model_name,
                png_path=model_plots / "feature_importance_top.png",
                html_path=model_plots / "feature_importance_top.html",
            )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(out["comparison_logs"] / "model_metrics_test.csv", index=False)

    predictions_df = pd.DataFrame({"y_true": y_test})
    for model_name in models.keys():
        predictions_df[f"pred_{model_name}"] = predictions[model_name]
    predictions_df.to_csv(out["comparison_logs"] / "predictions_test_by_model.csv", index=False)

    plot_comparison_bar(
        metrics_df=metrics_df,
        png_path=out["comparison_plots"] / "leaderboard_global_metrics.png",
        html_path=out["comparison_plots"] / "leaderboard_global_metrics.html",
    )

    class_f1_df = pd.DataFrame(class_f1_by_model).reindex(index=classes)
    class_f1_df.to_csv(out["comparison_logs"] / "class_f1_by_model.csv")
    plot_heatmap_static(
        matrix=class_f1_df,
        title="F1-score por clase y modelo",
        png_path=out["comparison_plots"] / "class_f1_by_model_heatmap.png",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
    )
    plot_heatmap_html(
        matrix=class_f1_df,
        title="F1-score por clase y modelo",
        html_path=out["comparison_plots"] / "class_f1_by_model_heatmap.html",
        colorscale="YlGnBu",
    )

    baseline_name = "svm" if "svm" in class_f1_df.columns else class_f1_df.columns[0]
    delta_class_df = class_f1_df.subtract(class_f1_df[baseline_name], axis=0)
    delta_class_df.to_csv(out["comparison_logs"] / "delta_vs_baseline_class_f1.csv")
    plot_heatmap_static(
        matrix=delta_class_df,
        title=f"Delta F1 por clase vs baseline ({baseline_name})",
        png_path=out["comparison_plots"] / "delta_vs_baseline_class_f1_heatmap.png",
        cmap="RdBu_r",
        vmin=-0.40,
        vmax=0.40,
    )
    plot_heatmap_html(
        matrix=delta_class_df,
        title=f"Delta F1 por clase vs baseline ({baseline_name})",
        html_path=out["comparison_plots"] / "delta_vs_baseline_class_f1_heatmap.html",
        colorscale="RdBu",
    )

    base_row = metrics_df.set_index("model").loc[baseline_name]
    delta_macro_rows = []
    for _, row in metrics_df.iterrows():
        delta_macro_rows.append(
            {
                "model": row["model"],
                "delta_accuracy": float(row["accuracy"] - base_row["accuracy"]),
                "delta_f1_macro": float(row["f1_macro"] - base_row["f1_macro"]),
                "delta_precision_macro": float(row["precision_macro"] - base_row["precision_macro"]),
                "delta_recall_macro": float(row["recall_macro"] - base_row["recall_macro"]),
            }
        )
    delta_macro_df = pd.DataFrame(delta_macro_rows).sort_values("delta_f1_macro", ascending=False)
    delta_macro_df.to_csv(out["comparison_logs"] / "delta_vs_baseline_macro.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(delta_macro_df["model"], delta_macro_df["delta_f1_macro"], color="#4e79a7")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(f"Delta F1-macro vs baseline ({baseline_name})")
    ax.set_ylabel("Delta F1-macro")
    ax.set_xticks(np.arange(len(delta_macro_df["model"])))
    ax.set_xticklabels(delta_macro_df["model"], rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(out["comparison_plots"] / "delta_vs_baseline_f1_macro.png", dpi=150)
    plt.close(fig)

    fig_html = px.bar(
        delta_macro_df,
        x="model",
        y="delta_f1_macro",
        title=f"Delta F1-macro vs baseline ({baseline_name})",
    )
    write_plotly_html(fig_html, out["comparison_plots"] / "delta_vs_baseline_f1_macro.html")

    agreement_matrix = pd.DataFrame(index=models.keys(), columns=models.keys(), dtype=float)
    for m1 in models.keys():
        for m2 in models.keys():
            agreement_matrix.loc[m1, m2] = float(np.mean(predictions[m1] == predictions[m2]))
    agreement_matrix.to_csv(out["comparison_logs"] / "pairwise_prediction_agreement.csv")
    plot_heatmap_static(
        matrix=agreement_matrix,
        title="Acuerdo de prediccion entre modelos",
        png_path=out["comparison_plots"] / "pairwise_prediction_agreement_heatmap.png",
        cmap="PuBuGn",
        vmin=0.0,
        vmax=1.0,
    )
    plot_heatmap_html(
        matrix=agreement_matrix,
        title="Acuerdo de prediccion entre modelos",
        html_path=out["comparison_plots"] / "pairwise_prediction_agreement_heatmap.html",
        colorscale="PuBuGn",
    )

    print("[stats] bootstrap CI por modelo...")
    ci_rows = []
    for idx, model_name in enumerate(models.keys()):
        ci = bootstrap_metrics_ci(
            y_true=y_test,
            y_pred=predictions[model_name],
            n_iters=args.bootstrap_iters,
            seed=args.seed + idx,
        )
        ci_rows.append({"model": model_name, **ci})
    ci_df = pd.DataFrame(ci_rows).sort_values("f1_macro_mean", ascending=False)
    ci_df.to_csv(out["comparison_logs"] / "bootstrap_ci_metrics.csv", index=False)

    print(f"[stats] McNemar vs baseline={baseline_name}...")
    mcnemar_rows = []
    pred_base = predictions[baseline_name]
    for model_name in models.keys():
        if model_name == baseline_name:
            continue
        mcn = mcnemar_vs_baseline(y_test, pred_base, predictions[model_name])
        mcnemar_rows.append({"baseline": baseline_name, "model": model_name, **mcn})
    mcnemar_df = pd.DataFrame(mcnemar_rows).sort_values("p_exact")
    mcnemar_df.to_csv(out["comparison_logs"] / "mcnemar_vs_baseline.csv", index=False)

    split_counts = plot_split_distribution(
        y=y_all,
        split=split_all,
        classes=classes,
        png_path=out["comparison_plots"] / "split_distribution_by_class.png",
        html_path=out["comparison_plots"] / "split_distribution_by_class.html",
    )
    split_counts.to_csv(out["comparison_logs"] / "split_distribution_by_class.csv", index=False)

    run_embeddings(
        x_all=x_all,
        y_all=y_all,
        split_all=split_all,
        classes=classes,
        comparison_logs=out["comparison_logs"],
        comparison_plots=out["comparison_plots"],
        seed=args.seed,
        max_samples=args.max_embed_samples,
    )

    summary = {
        "models_evaluated": list(models.keys()),
        "baseline_model": baseline_name,
        "n_test_samples": int(len(y_test)),
        "n_classes": int(len(classes)),
        "output_root": str(args.output_root),
    }
    with open(out["comparison_logs"] / "dashboard_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Dashboard de comparacion generado")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 72)


if __name__ == "__main__":
    main()


