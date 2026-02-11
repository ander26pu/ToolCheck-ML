#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena Random Forest con split train/val/test sobre HOG+PCA."
    )
    parser.add_argument(
        "--features-npz",
        type=Path,
        default=Path("artifacts/preprocess_hog_pca_v3/features/hog_pca_features_split.npz"),
        help="Ruta al archivo hog_pca_features_split.npz.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/rf_hog_pca_v1"),
        help="Directorio de salida para modelo, logs y plots.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="f1_macro",
        choices=["f1_macro", "precision_macro", "accuracy"],
        help="Metrica de validacion para escoger el mejor modelo.",
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
        "logs": output_root / "logs",
        "plots": output_root / "plots",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_split_features(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = {
        "X_train",
        "y_train",
        "X_val",
        "y_val",
        "X_test",
        "y_test",
        "classes",
    }
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Faltan claves en NPZ: {sorted(missing)}")
    payload = {k: data[k] for k in data.files}
    return payload


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


def model_candidates() -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for n_estimators in [300, 500, 800]:
        for max_depth in [None, 25, 40]:
            for max_features in ["sqrt", "log2"]:
                candidates.append(
                    {
                        "name": f"rf_n{n_estimators}_d{max_depth}_f{max_features}",
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "max_features": max_features,
                        "min_samples_leaf": 1,
                        "class_weight": "balanced",
                    }
                )
    return candidates


def build_model(candidate: Dict[str, object], seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(candidate["n_estimators"]),
        max_depth=candidate["max_depth"],
        max_features=candidate["max_features"],
        min_samples_leaf=int(candidate["min_samples_leaf"]),
        class_weight=candidate["class_weight"],
        random_state=seed,
        n_jobs=-1,
    )


def plot_split_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    classes: List[str],
    output_path: Path,
) -> None:
    train_counts = pd.Series(y_train).value_counts().reindex(classes, fill_value=0)
    val_counts = pd.Series(y_val).value_counts().reindex(classes, fill_value=0)
    test_counts = pd.Series(y_test).value_counts().reindex(classes, fill_value=0)

    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, train_counts.values, width=width, label="train", color="#1f77b4")
    ax.bar(x, val_counts.values, width=width, label="val", color="#ff7f0e")
    ax.bar(x + width, test_counts.values, width=width, label="test", color="#2ca02c")
    ax.set_title("Distribucion de clases por split")
    ax.set_ylabel("Cantidad de imagenes")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_selection_scores(selection_df: pd.DataFrame, metric: str, output_path: Path) -> None:
    sorted_df = selection_df.sort_values(metric, ascending=False).reset_index(drop=True)
    top_df = sorted_df.head(12)
    x = np.arange(len(top_df))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, top_df[metric], color="#4e79a7")
    ax.set_title(f"Top modelos RF por {metric} en validation")
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(top_df["name"], rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, classes: List[str], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_precision_by_class(
    report_val: Dict[str, Dict[str, float]],
    report_test: Dict[str, Dict[str, float]],
    classes: List[str],
    output_path: Path,
) -> None:
    val_precision = [float(report_val.get(c, {}).get("precision", 0.0)) for c in classes]
    test_precision = [float(report_test.get(c, {}).get("precision", 0.0)) for c in classes]
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, val_precision, width=width, label="val", color="#f28e2b")
    ax.bar(x + width / 2, test_precision, width=width, label="test", color="#59a14f")
    ax.set_title("Precision por clase")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_classification_report_csv(
    report: Dict[str, Dict[str, float]], classes: List[str], output_path: Path
) -> None:
    rows = []
    for class_name in classes:
        entry = report.get(class_name, {})
        rows.append(
            {
                "class_name": class_name,
                "precision": float(entry.get("precision", 0.0)),
                "recall": float(entry.get("recall", 0.0)),
                "f1-score": float(entry.get("f1-score", 0.0)),
                "support": int(entry.get("support", 0)),
            }
        )
    for key in ["macro avg", "weighted avg"]:
        entry = report.get(key, {})
        rows.append(
            {
                "class_name": key,
                "precision": float(entry.get("precision", 0.0)),
                "recall": float(entry.get("recall", 0.0)),
                "f1-score": float(entry.get("f1-score", 0.0)),
                "support": int(entry.get("support", 0)),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def select_best_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    metric: str,
    seed: int,
) -> Tuple[RandomForestClassifier, Dict[str, object], pd.DataFrame]:
    rows = []
    best_payload: Dict[str, object] | None = None
    best_model: RandomForestClassifier | None = None

    candidates = model_candidates()
    total = len(candidates)
    for idx, candidate in enumerate(candidates, start=1):
        t0 = time.time()
        model = build_model(candidate, seed)
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        scores = compute_metrics(y_val, val_pred)
        elapsed = time.time() - t0

        row = {
            **candidate,
            "elapsed_sec": elapsed,
            **scores,
        }
        rows.append(row)
        print(
            f"[{idx:02d}/{total}] {candidate['name']} -> "
            f"val_{metric}={scores[metric]:.4f} acc={scores['accuracy']:.4f} "
            f"time={elapsed:.2f}s"
        )

        if best_payload is None:
            best_payload = row
            best_model = model
            continue

        better = row[metric] > best_payload[metric]
        tie_break = row[metric] == best_payload[metric] and row["accuracy"] > best_payload["accuracy"]
        if better or tie_break:
            best_payload = row
            best_model = model

    if best_payload is None or best_model is None:
        raise RuntimeError("No se pudo entrenar ningun modelo candidato.")

    return best_model, best_payload, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_paths = ensure_dirs(args.output_root)

    print(f"Cargando features: {args.features_npz}")
    payload = load_split_features(args.features_npz)
    x_train = payload["X_train"].astype(np.float32)
    y_train = payload["y_train"].astype(str)
    x_val = payload["X_val"].astype(np.float32)
    y_val = payload["y_val"].astype(str)
    x_test = payload["X_test"].astype(np.float32)
    y_test = payload["y_test"].astype(str)
    classes = [str(c) for c in payload["classes"].tolist()]

    print(
        f"Shapes -> train={x_train.shape}, val={x_val.shape}, test={x_test.shape}, "
        f"classes={len(classes)}"
    )

    plot_split_distribution(
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        classes=classes,
        output_path=output_paths["plots"] / "class_split_distribution.png",
    )

    print("Entrenando candidatos Random Forest y seleccionando por validation...")
    best_model_train, best_row, selection_df = select_best_model(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        metric=args.selection_metric,
        seed=args.seed,
    )
    selection_df = selection_df.sort_values(
        [args.selection_metric, "accuracy"], ascending=False
    ).reset_index(drop=True)
    selection_df.to_csv(output_paths["logs"] / "model_selection_val.csv", index=False)
    plot_selection_scores(
        selection_df=selection_df,
        metric=args.selection_metric,
        output_path=output_paths["plots"] / "model_selection_validation.png",
    )

    val_pred = best_model_train.predict(x_val)
    val_metrics = compute_metrics(y_val, val_pred)
    report_val = classification_report(
        y_val,
        val_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    print(
        f"Mejor modelo (train->val): {best_row['name']} | "
        f"val_{args.selection_metric}={val_metrics[args.selection_metric]:.4f}"
    )

    print("Reentrenando mejor configuracion con train+val para evaluar en test...")
    best_model_final = build_model(best_row, args.seed)
    x_trainval = np.vstack([x_train, x_val]).astype(np.float32)
    y_trainval = np.concatenate([y_train, y_val])
    best_model_final.fit(x_trainval, y_trainval)

    test_pred = best_model_final.predict(x_test)
    test_metrics = compute_metrics(y_test, test_pred)
    report_test = classification_report(
        y_test,
        test_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )

    cm_val = confusion_matrix(y_val, val_pred, labels=classes)
    cm_test = confusion_matrix(y_test, test_pred, labels=classes)
    plot_confusion(
        cm=cm_val,
        classes=classes,
        title="Matriz de confusion (validation)",
        output_path=output_paths["plots"] / "confusion_matrix_val.png",
    )
    plot_confusion(
        cm=cm_test,
        classes=classes,
        title="Matriz de confusion (test)",
        output_path=output_paths["plots"] / "confusion_matrix_test.png",
    )
    plot_precision_by_class(
        report_val=report_val,
        report_test=report_test,
        classes=classes,
        output_path=output_paths["plots"] / "precision_by_class_val_vs_test.png",
    )

    write_classification_report_csv(
        report=report_val,
        classes=classes,
        output_path=output_paths["logs"] / "classification_report_val.csv",
    )
    write_classification_report_csv(
        report=report_test,
        classes=classes,
        output_path=output_paths["logs"] / "classification_report_test.csv",
    )
    joblib.dump(best_model_final, output_paths["models"] / "rf_best_model.joblib")

    summary = {
        "features_npz": str(args.features_npz),
        "selection_metric": args.selection_metric,
        "best_candidate_train_val": {
            "name": str(best_row["name"]),
            "n_estimators": int(best_row["n_estimators"]),
            "max_depth": best_row["max_depth"],
            "max_features": best_row["max_features"],
            "min_samples_leaf": int(best_row["min_samples_leaf"]),
            "class_weight": best_row["class_weight"],
        },
        "dataset_shapes": {
            "X_train": list(x_train.shape),
            "X_val": list(x_val.shape),
            "X_test": list(x_test.shape),
            "classes": classes,
        },
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
        },
    }
    with open(output_paths["logs"] / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Entrenamiento Random Forest finalizado")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Artefactos en: {args.output_root.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
