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
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena modelo clasico "
            "(ExtraTrees, LogReg, KNN, GradientBoosting, AdaBoost, NaiveBayes) "
            "con split train/val/test."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "extratrees",
            "logreg",
            "knn",
            "gradientboosting",
            "adaboost",
            "naivebayes",
        ],
        help="Modelo clasico a entrenar.",
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
        default=Path("artifacts/classic_ml_hog_pca"),
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


def model_candidates(model_name: str) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []

    if model_name == "extratrees":
        for n_estimators in [300, 500, 800]:
            for max_depth in [None, 25, 40]:
                for max_features in ["sqrt", "log2"]:
                    candidates.append(
                        {
                            "name": f"et_n{n_estimators}_d{max_depth}_f{max_features}",
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "max_features": max_features,
                            "min_samples_leaf": 1,
                            "class_weight": "balanced",
                        }
                    )
        return candidates

    if model_name == "logreg":
        for c_value in [0.1, 0.3, 1.0, 3.0, 10.0]:
            for class_weight in [None, "balanced"]:
                weight_label = "none" if class_weight is None else "balanced"
                candidates.append(
                    {
                        "name": f"logreg_C{c_value:g}_w{weight_label}",
                        "C": c_value,
                        "class_weight": class_weight,
                    }
                )
        return candidates

    if model_name == "knn":
        for n_neighbors in [3, 5, 7, 11, 15]:
            for weights in ["uniform", "distance"]:
                for p_dist in [1, 2]:
                    candidates.append(
                        {
                            "name": f"knn_k{n_neighbors}_{weights}_p{p_dist}",
                            "n_neighbors": n_neighbors,
                            "weights": weights,
                            "p": p_dist,
                        }
                    )
        return candidates

    if model_name == "gradientboosting":
        for n_estimators in [80, 150]:
            for learning_rate in [0.05, 0.1]:
                for subsample in [0.8, 1.0]:
                    candidates.append(
                        {
                            "name": (
                                f"gb_n{n_estimators}_lr{learning_rate}_"
                                f"d2_ss{subsample}_fsqrt"
                            ),
                            "n_estimators": n_estimators,
                            "learning_rate": learning_rate,
                            "max_depth": 2,
                            "subsample": subsample,
                            "max_features": "sqrt",
                        }
                    )
        return candidates

    if model_name == "adaboost":
        for n_estimators in [200, 400, 600]:
            for learning_rate in [0.3, 0.7, 1.0]:
                for base_depth in [1, 2]:
                    candidates.append(
                        {
                            "name": (
                                f"ab_n{n_estimators}_lr{learning_rate}_"
                                f"d{base_depth}"
                            ),
                            "n_estimators": n_estimators,
                            "learning_rate": learning_rate,
                            "base_depth": base_depth,
                        }
                    )
        return candidates

    if model_name == "naivebayes":
        for var_smoothing in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
            for use_scaler in [False, True]:
                scaler_label = "sc" if use_scaler else "nosc"
                candidates.append(
                    {
                        "name": f"gnb_vs{var_smoothing:.0e}_{scaler_label}",
                        "var_smoothing": var_smoothing,
                        "use_scaler": use_scaler,
                    }
                )
        return candidates

    raise ValueError(f"Modelo no soportado: {model_name}")


def build_model(model_name: str, candidate: Dict[str, object], seed: int):
    if model_name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=int(candidate["n_estimators"]),
            max_depth=candidate["max_depth"],
            max_features=candidate["max_features"],
            min_samples_leaf=int(candidate["min_samples_leaf"]),
            class_weight=candidate["class_weight"],
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "logreg":
        estimator = LogisticRegression(
            C=float(candidate["C"]),
            class_weight=candidate["class_weight"],
            solver="lbfgs",
            max_iter=10000,
            random_state=seed,
        )
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("logreg", estimator),
            ]
        )

    if model_name == "knn":
        estimator = KNeighborsClassifier(
            n_neighbors=int(candidate["n_neighbors"]),
            weights=str(candidate["weights"]),
            p=int(candidate["p"]),
            n_jobs=-1,
        )
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("knn", estimator),
            ]
        )

    if model_name == "gradientboosting":
        return GradientBoostingClassifier(
            n_estimators=int(candidate["n_estimators"]),
            learning_rate=float(candidate["learning_rate"]),
            max_depth=int(candidate["max_depth"]),
            subsample=float(candidate["subsample"]),
            max_features=candidate["max_features"],
            random_state=seed,
        )

    if model_name == "adaboost":
        base_estimator = DecisionTreeClassifier(
            max_depth=int(candidate["base_depth"]),
            random_state=seed,
        )
        return AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=int(candidate["n_estimators"]),
            learning_rate=float(candidate["learning_rate"]),
            random_state=seed,
        )

    if model_name == "naivebayes":
        estimator = GaussianNB(var_smoothing=float(candidate["var_smoothing"]))
        if bool(candidate["use_scaler"]):
            return Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("gnb", estimator),
                ]
            )
        return estimator

    raise ValueError(f"Modelo no soportado: {model_name}")


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


def plot_selection_scores(
    selection_df: pd.DataFrame, metric: str, model_name: str, output_path: Path
) -> None:
    sorted_df = selection_df.sort_values(metric, ascending=False).reset_index(drop=True)
    top_df = sorted_df.head(12)
    x = np.arange(len(top_df))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, top_df[metric], color="#4e79a7")
    ax.set_title(f"Top modelos {model_name} por {metric} en validation")
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
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    metric: str,
    seed: int,
):
    rows = []
    best_payload: Dict[str, object] | None = None
    best_model = None

    candidates = model_candidates(model_name)
    total = len(candidates)
    for idx, candidate in enumerate(candidates, start=1):
        t0 = time.time()
        model = build_model(model_name, candidate, seed)
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        scores = compute_metrics(y_val, val_pred)
        elapsed = time.time() - t0

        row = {
            "model": model_name,
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
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    idx_to_class = np.array(classes)

    y_train_idx = np.array([class_to_idx[c] for c in y_train], dtype=np.int32)
    y_val_idx = np.array([class_to_idx[c] for c in y_val], dtype=np.int32)
    y_test_idx = np.array([class_to_idx[c] for c in y_test], dtype=np.int32)

    print(
        f"Model={args.model} | train={x_train.shape}, val={x_val.shape}, "
        f"test={x_test.shape}, classes={len(classes)}"
    )

    plot_split_distribution(
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        classes=classes,
        output_path=output_paths["plots"] / "class_split_distribution.png",
    )

    print(f"Entrenando candidatos {args.model} y seleccionando por validation...")
    best_model_train, best_row, selection_df = select_best_model(
        model_name=args.model,
        x_train=x_train,
        y_train=y_train_idx,
        x_val=x_val,
        y_val=y_val_idx,
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
        model_name=args.model,
        output_path=output_paths["plots"] / "model_selection_validation.png",
    )

    val_pred_idx = best_model_train.predict(x_val)
    val_pred = idx_to_class[val_pred_idx]
    val_metrics = compute_metrics(y_val, val_pred)
    report_val = classification_report(
        y_val, val_pred, labels=classes, output_dict=True, zero_division=0
    )
    print(
        f"Mejor modelo (train->val): {best_row['name']} | "
        f"val_{args.selection_metric}={val_metrics[args.selection_metric]:.4f}"
    )

    print("Reentrenando mejor configuracion con train+val para evaluar en test...")
    best_model_final = build_model(args.model, best_row, args.seed)
    x_trainval = np.vstack([x_train, x_val]).astype(np.float32)
    y_trainval_idx = np.concatenate([y_train_idx, y_val_idx])
    best_model_final.fit(x_trainval, y_trainval_idx)

    test_pred_idx = best_model_final.predict(x_test)
    test_pred = idx_to_class[test_pred_idx]
    test_metrics = compute_metrics(y_test, test_pred)
    report_test = classification_report(
        y_test, test_pred, labels=classes, output_dict=True, zero_division=0
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

    model_name_for_file = f"{args.model}_best_model.joblib"
    joblib.dump(best_model_final, output_paths["models"] / model_name_for_file)

    summary = {
        "model": args.model,
        "features_npz": str(args.features_npz),
        "selection_metric": args.selection_metric,
        "best_candidate_train_val": {
            k: (
                int(v)
                if isinstance(v, np.integer)
                else (float(v) if isinstance(v, np.floating) else v)
            )
            for k, v in best_row.items()
            if k
            not in {
                "elapsed_sec",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
            }
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
    print(f"Entrenamiento {args.model} finalizado")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Artefactos en: {args.output_root.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
