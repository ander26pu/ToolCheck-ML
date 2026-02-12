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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC, SVC


PROBA_MODEL_REGISTRY: List[Tuple[str, str]] = [
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
            "Bloque 2 dashboard: probabilidad, calibracion, ROC/PR multiclass y comparativos."
        )
    )
    parser.add_argument(
        "--features-split",
        type=Path,
        default=Path("artifacts/preprocess_hog_pca_v3/features/hog_pca_features_split.npz"),
        help="NPZ con X_train/X_val/X_test y etiquetas.",
    )
    parser.add_argument(
        "--svm-summary",
        type=Path,
        default=Path("artifacts/svm_hog_pca_v1/logs/summary.json"),
        help="Resumen de SVM para reconstruir hiperparametros del modelo base.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts"),
        help="Raiz de artefactos con modelos ya entrenados.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/model_dashboard_v2_probcal"),
        help="Salida de bloque 2.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=15,
        help="Numero de bins para reliability diagram y ECE.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla reproducible.",
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


def write_plotly_html(fig: go.Figure, output_path: Path) -> None:
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)


def load_split(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "classes"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Faltan claves en split NPZ: {sorted(missing)}")
    return {k: data[k] for k in data.files}


def discover_probability_models(artifacts_root: Path) -> Dict[str, Path]:
    models: Dict[str, Path] = {}
    for model_name, rel_path in PROBA_MODEL_REGISTRY:
        p = artifacts_root / rel_path
        if p.exists():
            models[model_name] = p
    return models


def extract_final_estimator(model):
    if hasattr(model, "named_steps") and isinstance(model.named_steps, dict):
        last_key = list(model.named_steps.keys())[-1]
        return model.named_steps[last_key]
    return model


def get_model_classes(model):
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    est = extract_final_estimator(model)
    if hasattr(est, "classes_"):
        return np.asarray(est.classes_)
    return None


def align_proba_columns(
    proba: np.ndarray,
    model_classes: np.ndarray | None,
    target_classes: List[str],
) -> np.ndarray:
    n_samples = proba.shape[0]
    n_target = len(target_classes)

    if model_classes is None:
        if proba.shape[1] != n_target:
            raise ValueError("No se pudo alinear proba: sin classes_ y dimensiones distintas")
        return proba

    aligned = np.zeros((n_samples, n_target), dtype=np.float64)

    for src_idx, cls in enumerate(model_classes.tolist()):
        tgt_idx = None
        if isinstance(cls, (np.integer, int)):
            if 0 <= int(cls) < n_target:
                tgt_idx = int(cls)
        else:
            cls_str = str(cls)
            if cls_str in target_classes:
                tgt_idx = target_classes.index(cls_str)
        if tgt_idx is not None:
            aligned[:, tgt_idx] = proba[:, src_idx]

    row_sum = aligned.sum(axis=1, keepdims=True)
    aligned = np.divide(aligned, row_sum, out=np.zeros_like(aligned), where=row_sum > 0)
    return aligned


def multiclass_brier_score(y_idx: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    y_onehot = np.eye(n_classes)[y_idx]
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))


def reliability_bins(y_idx: np.ndarray, proba: np.ndarray, n_bins: int) -> pd.DataFrame:
    pred_idx = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    correct = (pred_idx == y_idx).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(np.sum(mask))
        if count == 0:
            rows.append(
                {
                    "bin": i,
                    "conf_low": lo,
                    "conf_high": hi,
                    "count": 0,
                    "avg_confidence": 0.0,
                    "accuracy": 0.0,
                    "gap": 0.0,
                    "weight": 0.0,
                }
            )
            continue

        avg_conf = float(np.mean(confidence[mask]))
        acc = float(np.mean(correct[mask]))
        gap = abs(acc - avg_conf)
        weight = float(count / len(confidence))
        rows.append(
            {
                "bin": i,
                "conf_low": lo,
                "conf_high": hi,
                "count": count,
                "avg_confidence": avg_conf,
                "accuracy": acc,
                "gap": gap,
                "weight": weight,
            }
        )

    return pd.DataFrame(rows)


def compute_prob_metrics(y_true: np.ndarray, proba: np.ndarray, classes: List[str]) -> Dict[str, float]:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[c] for c in y_true], dtype=int)
    pred_idx = np.argmax(proba, axis=1)
    y_pred = np.array([classes[i] for i in pred_idx], dtype=str)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    proba_clip = np.clip(proba, 1e-12, 1.0)
    ll = float(log_loss(y_idx, proba_clip, labels=list(range(len(classes)))))
    brier = multiclass_brier_score(y_idx, proba_clip, len(classes))

    bin_df = reliability_bins(y_idx, proba_clip, n_bins=15)
    ece = float((bin_df["gap"] * bin_df["weight"]).sum())
    mce = float(bin_df["gap"].max())

    y_bin = label_binarize(y_idx, classes=list(range(len(classes))))
    try:
        roc_macro = float(roc_auc_score(y_bin, proba_clip, average="macro", multi_class="ovr"))
    except Exception:
        roc_macro = float("nan")
    try:
        pr_macro = float(average_precision_score(y_bin, proba_clip, average="macro"))
    except Exception:
        pr_macro = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "log_loss": ll,
        "brier_multiclass": brier,
        "ece": ece,
        "mce": mce,
        "roc_auc_macro_ovr": roc_macro,
        "pr_auc_macro": pr_macro,
    }


def per_class_auc(y_true: np.ndarray, proba: np.ndarray, classes: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[c] for c in y_true], dtype=int)
    y_bin = label_binarize(y_idx, classes=list(range(len(classes))))

    roc_rows = []
    pr_rows = []
    for cls_idx, cls_name in enumerate(classes):
        y_true_bin = y_bin[:, cls_idx]
        y_score = proba[:, cls_idx]

        try:
            fpr, tpr, _ = roc_curve(y_true_bin, y_score)
            roc_auc = float(auc(fpr, tpr))
        except Exception:
            roc_auc = float("nan")
        try:
            ap = float(average_precision_score(y_true_bin, y_score))
        except Exception:
            ap = float("nan")

        roc_rows.append({"class_name": cls_name, "roc_auc": roc_auc})
        pr_rows.append({"class_name": cls_name, "average_precision": ap})

    return pd.DataFrame(roc_rows), pd.DataFrame(pr_rows)


def plot_reliability(
    bin_df: pd.DataFrame,
    model_name: str,
    png_path: Path,
    html_path: Path,
) -> None:
    centers = (bin_df["conf_low"] + bin_df["conf_high"]) / 2.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfecta")
    ax.plot(centers, bin_df["accuracy"], marker="o", label="Modelo")
    ax.set_title(f"Reliability diagram - {model_name}")
    ax.set_xlabel("Confianza")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    fig_html = go.Figure()
    fig_html.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfecta", line=dict(dash="dash"))
    )
    fig_html.add_trace(
        go.Scatter(
            x=centers,
            y=bin_df["accuracy"],
            mode="lines+markers",
            name="Modelo",
            hovertemplate="Conf=%{x:.2f}<br>Acc=%{y:.2f}<extra></extra>",
        )
    )
    fig_html.update_layout(
        title=f"Reliability diagram - {model_name}",
        xaxis_title="Confianza",
        yaxis_title="Accuracy",
        xaxis=dict(range=[0.0, 1.0]),
        yaxis=dict(range=[0.0, 1.0]),
    )
    write_plotly_html(fig_html, html_path)


def plot_confidence_hist(
    proba: np.ndarray,
    model_name: str,
    png_path: Path,
    html_path: Path,
) -> None:
    confidence = np.max(proba, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(confidence, bins=20, color="#4e79a7", alpha=0.85)
    ax.set_title(f"Distribucion de confianza - {model_name}")
    ax.set_xlabel("Confianza maxima")
    ax.set_ylabel("Frecuencia")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    fig_html = px.histogram(
        x=confidence,
        nbins=20,
        title=f"Distribucion de confianza - {model_name}",
    )
    fig_html.update_layout(xaxis_title="Confianza maxima", yaxis_title="Frecuencia")
    write_plotly_html(fig_html, html_path)


def plot_roc_pr(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: List[str],
    model_name: str,
    roc_png: Path,
    roc_html: Path,
    pr_png: Path,
    pr_html: Path,
) -> None:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[c] for c in y_true], dtype=int)
    y_bin = label_binarize(y_idx, classes=list(range(len(classes))))

    fig_roc, ax_roc = plt.subplots(figsize=(8, 7))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 7))

    roc_html_fig = go.Figure()
    pr_html_fig = go.Figure()

    for cls_idx, cls_name in enumerate(classes):
        y_true_bin = y_bin[:, cls_idx]
        y_score = proba[:, cls_idx]

        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        roc_auc_val = float(auc(fpr, tpr))
        ax_roc.plot(fpr, tpr, label=f"{cls_name} (AUC={roc_auc_val:.2f})")
        roc_html_fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{cls_name} (AUC={roc_auc_val:.2f})",
            )
        )

        precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
        ap = float(average_precision_score(y_true_bin, y_score))
        ax_pr.plot(recall, precision, label=f"{cls_name} (AP={ap:.2f})")
        pr_html_fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{cls_name} (AP={ap:.2f})",
            )
        )

    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax_roc.set_title(f"ROC one-vs-rest - {model_name}")
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.0)
    ax_roc.grid(alpha=0.3)
    ax_roc.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig_roc.savefig(roc_png, dpi=160)
    plt.close(fig_roc)

    roc_html_fig.update_layout(
        title=f"ROC one-vs-rest - {model_name}",
        xaxis_title="FPR",
        yaxis_title="TPR",
        xaxis=dict(range=[0.0, 1.0]),
        yaxis=dict(range=[0.0, 1.0]),
    )
    write_plotly_html(roc_html_fig, roc_html)

    ax_pr.set_title(f"Precision-Recall one-vs-rest - {model_name}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.0)
    ax_pr.grid(alpha=0.3)
    ax_pr.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    fig_pr.savefig(pr_png, dpi=160)
    plt.close(fig_pr)

    pr_html_fig.update_layout(
        title=f"Precision-Recall one-vs-rest - {model_name}",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0.0, 1.0]),
        yaxis=dict(range=[0.0, 1.0]),
    )
    write_plotly_html(pr_html_fig, pr_html)


def build_svm_base_from_summary(summary_path: Path, seed: int) -> Pipeline:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best = summary["best_candidate_train_val"]
    kind = str(best.get("kind", "linear"))

    if kind == "linear":
        estimator = LinearSVC(
            C=float(best["C"]),
            dual=False,
            random_state=seed,
            max_iter=30000,
        )
    elif kind == "rbf":
        gamma = best.get("gamma", "scale")
        estimator = SVC(
            C=float(best["C"]),
            kernel="rbf",
            gamma=gamma,
        )
    else:
        raise ValueError(f"Tipo SVM no soportado en summary: {kind}")

    return Pipeline([("scaler", StandardScaler()), ("svm", estimator)])


def select_svm_calibration_method(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    classes: List[str],
    svm_base: Pipeline,
) -> Tuple[str, pd.DataFrame]:
    rows = []
    for method in ["sigmoid", "isotonic"]:
        cal = CalibratedClassifierCV(estimator=svm_base, method=method, cv=5)
        cal.fit(x_train, y_train)
        proba_val = cal.predict_proba(x_val)
        proba_val = align_proba_columns(proba_val, get_model_classes(cal), classes)
        metrics = compute_prob_metrics(y_val, proba_val, classes)
        rows.append({"method": method, **metrics})

    df = pd.DataFrame(rows).sort_values(["log_loss", "ece", "f1_macro"]).reset_index(drop=True)
    best_method = str(df.iloc[0]["method"])
    return best_method, df


def make_bar_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    y_label: str,
    png_path: Path,
    html_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(df[x_col], df[y_col], color="#4e79a7")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xticks(np.arange(len(df[x_col])))
    ax.set_xticklabels(df[x_col], rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    fig_html = px.bar(df, x=x_col, y=y_col, title=title)
    fig_html.update_layout(yaxis_title=y_label)
    write_plotly_html(fig_html, html_path)


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    png_path: Path,
    html_path: Path,
    cmap: str,
    vmin: float,
    vmax: float,
    colorscale: str,
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
            ax.text(j, i, f"{matrix.values[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(png_path, dpi=170)
    plt.close(fig)

    fig_html = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale=colorscale,
            hovertemplate="Fila=%{y}<br>Col=%{x}<br>Valor=%{z:.4f}<extra></extra>",
        )
    )
    fig_html.update_layout(title=title, xaxis_title="Modelo", yaxis_title="Clase")
    write_plotly_html(fig_html, html_path)


def main() -> None:
    args = parse_args()
    out = ensure_dirs(args.output_root)

    split_data = load_split(args.features_split)
    classes = [str(c) for c in split_data["classes"].tolist()]

    x_train = split_data["X_train"].astype(np.float32)
    y_train = split_data["y_train"].astype(str)
    x_val = split_data["X_val"].astype(np.float32)
    y_val = split_data["y_val"].astype(str)
    x_test = split_data["X_test"].astype(np.float32)
    y_test = split_data["y_test"].astype(str)

    proba_models = discover_probability_models(args.artifacts_root)

    print("Modelos con proba encontrados:", ", ".join(proba_models.keys()))

    print("Calibrando SVM (sigmoid/isotonic)...")
    svm_base = build_svm_base_from_summary(args.svm_summary, seed=args.seed)
    best_method, svm_val_df = select_svm_calibration_method(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        classes=classes,
        svm_base=svm_base,
    )
    svm_val_df.to_csv(out["comparison_logs"] / "svm_calibration_val_metrics.csv", index=False)

    print(f"Metodo elegido para SVM: {best_method}")
    svm_cal = CalibratedClassifierCV(
        estimator=build_svm_base_from_summary(args.svm_summary, seed=args.seed),
        method=best_method,
        cv=5,
    )
    x_trainval = np.vstack([x_train, x_val]).astype(np.float32)
    y_trainval = np.concatenate([y_train, y_val])
    svm_cal.fit(x_trainval, y_trainval)

    svm_model_dir = out["models"] / "svm_calibrated"
    (svm_model_dir / "model").mkdir(parents=True, exist_ok=True)
    joblib.dump(svm_cal, svm_model_dir / "model" / "svm_calibrated.joblib")

    models_eval: Dict[str, object] = {"svm_calibrated": svm_cal}
    for model_name, model_path in proba_models.items():
        models_eval[model_name] = joblib.load(model_path)

    metrics_rows: List[Dict[str, float | str]] = []
    roc_by_model: Dict[str, pd.DataFrame] = {}
    pr_by_model: Dict[str, pd.DataFrame] = {}

    for model_name, model in models_eval.items():
        print(f"[model-proba] {model_name}")
        model_dir = out["models"] / model_name
        model_logs = model_dir / "logs"
        model_plots = model_dir / "plots"
        model_logs.mkdir(parents=True, exist_ok=True)
        model_plots.mkdir(parents=True, exist_ok=True)

        if not hasattr(model, "predict_proba"):
            print(f"  {model_name} no tiene predict_proba, se omite")
            continue

        proba = model.predict_proba(x_test)
        proba = align_proba_columns(proba, get_model_classes(model), classes)

        metrics = compute_prob_metrics(y_test, proba, classes)
        metrics_rows.append({"model": model_name, **metrics})

        pd.DataFrame([metrics]).to_csv(model_logs / "probability_metrics_test.csv", index=False)

        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([class_to_idx[c] for c in y_test], dtype=int)
        bins_df = reliability_bins(y_idx, np.clip(proba, 1e-12, 1.0), args.n_bins)
        bins_df.to_csv(model_logs / "reliability_bins.csv", index=False)

        roc_df, pr_df = per_class_auc(y_test, np.clip(proba, 1e-12, 1.0), classes)
        roc_df.to_csv(model_logs / "roc_auc_by_class.csv", index=False)
        pr_df.to_csv(model_logs / "pr_auc_by_class.csv", index=False)
        roc_by_model[model_name] = roc_df
        pr_by_model[model_name] = pr_df

        plot_reliability(
            bin_df=bins_df,
            model_name=model_name,
            png_path=model_plots / "reliability_diagram.png",
            html_path=model_plots / "reliability_diagram.html",
        )
        plot_confidence_hist(
            proba=proba,
            model_name=model_name,
            png_path=model_plots / "confidence_histogram.png",
            html_path=model_plots / "confidence_histogram.html",
        )
        plot_roc_pr(
            y_true=y_test,
            proba=np.clip(proba, 1e-12, 1.0),
            classes=classes,
            model_name=model_name,
            roc_png=model_plots / "roc_ovr_multiclass.png",
            roc_html=model_plots / "roc_ovr_multiclass.html",
            pr_png=model_plots / "pr_ovr_multiclass.png",
            pr_html=model_plots / "pr_ovr_multiclass.html",
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(out["comparison_logs"] / "probability_metrics_summary.csv", index=False)

    baseline = "svm_calibrated"
    base_row = metrics_df.set_index("model").loc[baseline]
    delta_rows = []
    for _, row in metrics_df.iterrows():
        delta_rows.append(
            {
                "model": row["model"],
                "delta_accuracy": float(row["accuracy"] - base_row["accuracy"]),
                "delta_f1_macro": float(row["f1_macro"] - base_row["f1_macro"]),
                "delta_log_loss": float(row["log_loss"] - base_row["log_loss"]),
                "delta_ece": float(row["ece"] - base_row["ece"]),
                "delta_roc_auc_macro_ovr": float(row["roc_auc_macro_ovr"] - base_row["roc_auc_macro_ovr"]),
                "delta_pr_auc_macro": float(row["pr_auc_macro"] - base_row["pr_auc_macro"]),
            }
        )
    delta_df = pd.DataFrame(delta_rows).sort_values("delta_f1_macro", ascending=False)
    delta_df.to_csv(out["comparison_logs"] / "delta_vs_svm_calibrated.csv", index=False)

    make_bar_plot(
        df=metrics_df,
        x_col="model",
        y_col="f1_macro",
        title="F1-macro (test) - modelos con probabilidad",
        y_label="F1-macro",
        png_path=out["comparison_plots"] / "leaderboard_f1_macro_prob_models.png",
        html_path=out["comparison_plots"] / "leaderboard_f1_macro_prob_models.html",
    )
    make_bar_plot(
        df=metrics_df.sort_values("log_loss", ascending=True),
        x_col="model",
        y_col="log_loss",
        title="Log-loss (test) - menor es mejor",
        y_label="Log-loss",
        png_path=out["comparison_plots"] / "leaderboard_log_loss.png",
        html_path=out["comparison_plots"] / "leaderboard_log_loss.html",
    )
    make_bar_plot(
        df=metrics_df.sort_values("ece", ascending=True),
        x_col="model",
        y_col="ece",
        title="ECE (test) - menor es mejor",
        y_label="ECE",
        png_path=out["comparison_plots"] / "leaderboard_ece.png",
        html_path=out["comparison_plots"] / "leaderboard_ece.html",
    )
    make_bar_plot(
        df=metrics_df.sort_values("roc_auc_macro_ovr", ascending=False),
        x_col="model",
        y_col="roc_auc_macro_ovr",
        title="ROC AUC macro OVR (test)",
        y_label="ROC AUC",
        png_path=out["comparison_plots"] / "leaderboard_roc_auc_macro.png",
        html_path=out["comparison_plots"] / "leaderboard_roc_auc_macro.html",
    )
    make_bar_plot(
        df=metrics_df.sort_values("pr_auc_macro", ascending=False),
        x_col="model",
        y_col="pr_auc_macro",
        title="PR AUC macro (test)",
        y_label="PR AUC",
        png_path=out["comparison_plots"] / "leaderboard_pr_auc_macro.png",
        html_path=out["comparison_plots"] / "leaderboard_pr_auc_macro.html",
    )

    roc_matrix = pd.DataFrame(
        {
            model_name: roc_df.set_index("class_name")["roc_auc"]
            for model_name, roc_df in roc_by_model.items()
        }
    ).reindex(index=classes)
    pr_matrix = pd.DataFrame(
        {
            model_name: pr_df.set_index("class_name")["average_precision"]
            for model_name, pr_df in pr_by_model.items()
        }
    ).reindex(index=classes)

    roc_matrix.to_csv(out["comparison_logs"] / "roc_auc_by_class_by_model.csv")
    pr_matrix.to_csv(out["comparison_logs"] / "pr_auc_by_class_by_model.csv")

    plot_heatmap(
        matrix=roc_matrix,
        title="ROC AUC por clase y modelo",
        png_path=out["comparison_plots"] / "roc_auc_class_model_heatmap.png",
        html_path=out["comparison_plots"] / "roc_auc_class_model_heatmap.html",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        colorscale="YlGnBu",
    )
    plot_heatmap(
        matrix=pr_matrix,
        title="PR AUC por clase y modelo",
        png_path=out["comparison_plots"] / "pr_auc_class_model_heatmap.png",
        html_path=out["comparison_plots"] / "pr_auc_class_model_heatmap.html",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        colorscale="YlOrRd",
    )

    summary = {
        "models_probability_evaluated": metrics_df["model"].tolist(),
        "baseline": baseline,
        "svm_calibration_method": best_method,
        "n_test_samples": int(len(y_test)),
        "n_classes": int(len(classes)),
        "output_root": str(args.output_root),
    }
    with open(out["comparison_logs"] / "probcal_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Bloque 2 (probabilidad/calibracion) generado")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 72)


if __name__ == "__main__":
    main()
