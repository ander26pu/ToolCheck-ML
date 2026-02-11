#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


EXPECTED_MARKER_IDS = {0, 1, 2, 3}
MARKER_LAYOUT_M = {
    0: (0.00, 0.00),
    1: (0.10, 0.00),
    2: (0.00, 0.10),
    3: (0.10, 0.10),
}
POSITION_ORDER = ["tl", "tr", "br", "bl"]
POSITION_TO_EXPECTED_ID = {"tl": 0, "tr": 1, "br": 3, "bl": 2}
EXPECTED_ID_TO_POSITION = {v: k for k, v in POSITION_TO_EXPECTED_ID.items()}


@dataclass
class ImageItem:
    class_name: str
    path: Path


@dataclass
class CalibrationResult:
    success: bool
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    reprojection_error: float
    used_frames: int
    used_markers: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocesa dataset con ArUco + extrae HOG + aplica PCA."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Carpeta con clases e imagenes crudas.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/preprocess_hog_pca"),
        help="Carpeta de salida de artefactos y debug.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=128,
        help="Tamano final cuadrado para HOG (default: 128).",
    )
    parser.add_argument(
        "--rectified-size",
        type=int,
        default=800,
        help="Tamano cuadrado de imagen rectificada.",
    )
    parser.add_argument(
        "--marker-margin",
        type=int,
        default=15,
        help="Margen para enmascarar marcadores ArUco en pixeles.",
    )
    parser.add_argument(
        "--marker-size-cm",
        type=float,
        default=5.0,
        help="Lado real del marcador ArUco en cm.",
    )
    parser.add_argument(
        "--corner-expansion-factor",
        type=float,
        default=0.95,
        help="Factor de interpolacion inner->outer para rectificacion por esquinas.",
    )
    parser.add_argument(
        "--debug-samples-per-class",
        type=int,
        default=5,
        help="Cantidad de muestras por clase para guardar checkpoints visuales.",
    )
    parser.add_argument(
        "--detect-max-dim",
        type=int,
        default=960,
        help="Maximo lado para deteccion de marcadores (se reescala internamente).",
    )
    parser.add_argument(
        "--calibration-max-images",
        type=int,
        default=480,
        help="Maximo de imagenes para calibrar camara.",
    )
    parser.add_argument(
        "--calibration-progress-every",
        type=int,
        default=25,
        help="Cada cuantas imagenes mostrar avance en calibracion.",
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.95,
        help="Varianza acumulada objetivo para PCA.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proporcion para split train.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proporcion para split validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proporcion para split test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para muestreo reproducible.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limite total de imagenes a procesar (debug rapido).",
    )
    return parser.parse_args()


def list_images(data_root: Path) -> List[ImageItem]:
    image_items: List[ImageItem] = []
    for class_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for image_path in sorted(class_dir.glob("*.jpg")):
            image_items.append(ImageItem(class_name=class_dir.name, path=image_path))
    return image_items


def ensure_dirs(output_root: Path) -> Dict[str, Path]:
    paths = {
        "preprocessed": output_root / "preprocessed" / "final_gray",
        "debug_samples": output_root / "debug_samples",
        "debug_failures": output_root / "debug_failures",
        "features": output_root / "features",
        "plots": output_root / "plots",
        "logs": output_root / "logs",
        "calibration": output_root / "calibration",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_detector() -> cv2.aruco.ArucoDetector:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 63
    params.adaptiveThreshConstant = 5
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 5.0
    params.polygonalApproxAccuracyRate = 0.1
    params.minCornerDistanceRate = 0.01
    params.minDistanceToBorder = 0
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def build_board(marker_size_cm: float) -> cv2.aruco.Board:
    marker_size_m = marker_size_cm / 100.0
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    obj_points = []
    ids = []
    for marker_id in [0, 1, 2, 3]:
        x, y = MARKER_LAYOUT_M[marker_id]
        obj_points.append(
            np.array(
                [
                    [x, y, 0.0],
                    [x + marker_size_m, y, 0.0],
                    [x + marker_size_m, y + marker_size_m, 0.0],
                    [x, y + marker_size_m, 0.0],
                ],
                dtype=np.float32,
            )
        )
        ids.append(marker_id)
    return cv2.aruco.Board(obj_points, aruco_dict, np.array(ids, dtype=np.int32))


def compute_qc_metrics(image_bgr: np.ndarray) -> Tuple[float, float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    return lap_var, brightness


def qc_thresholds(metrics: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    lap_values = np.array([m[0] for m in metrics], dtype=np.float64)
    bright_values = np.array([m[1] for m in metrics], dtype=np.float64)
    return {
        "laplacian_blur_p5": float(np.percentile(lap_values, 5)),
        "brightness_low_p1": float(np.percentile(bright_values, 1)),
        "brightness_high_p99": float(np.percentile(bright_values, 99)),
    }


def downscale_gray(gray: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    return gray, scale


def build_detection_variants(gray: np.ndarray) -> Dict[str, np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return {
        "gray": gray,
        "clahe": clahe.apply(gray),
        "bright": cv2.convertScaleAbs(gray, alpha=1.2, beta=15),
        "dark": cv2.convertScaleAbs(gray, alpha=0.85, beta=-10),
    }


def score_detection(
    corners: Optional[List[np.ndarray]], ids: Optional[np.ndarray]
) -> Tuple[int, int, float]:
    if ids is None or corners is None:
        return (-1, -1, -1.0)
    ids_flat = ids.flatten().tolist()
    expected_count = len(EXPECTED_MARKER_IDS.intersection(ids_flat))
    total = len(ids_flat)
    if total == 0:
        return (-1, -1, -1.0)
    perimeters = []
    for c in corners:
        cc = c.reshape(-1, 2).astype(np.float32)
        perimeters.append(float(cv2.arcLength(cc, True)))
    perimeter_mean = float(np.mean(perimeters)) if perimeters else 0.0
    return (expected_count, total, perimeter_mean)


def detect_markers_best(
    image_bgr: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    board: Optional[cv2.aruco.Board],
    camera_matrix: Optional[np.ndarray],
    dist_coeffs: Optional[np.ndarray],
    max_dim: int,
) -> Tuple[List[np.ndarray], Optional[np.ndarray], Dict[str, object]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_small, scale = downscale_gray(gray, max_dim=max_dim)
    variants = build_detection_variants(gray_small)

    best = {
        "corners": [],
        "ids": None,
        "variant": "none",
        "score": (-1, -1, -1.0),
    }

    for variant_name, variant_img in variants.items():
        corners, ids, rejected = detector.detectMarkers(variant_img)
        if board is not None:
            try:
                corners, ids, rejected, _ = cv2.aruco.refineDetectedMarkers(
                    variant_img,
                    board,
                    corners,
                    ids,
                    rejected,
                    cameraMatrix=camera_matrix,
                    distCoeffs=dist_coeffs,
                )
            except cv2.error:
                pass
        score = score_detection(corners, ids)
        if score > best["score"]:
            best = {
                "corners": corners if corners is not None else [],
                "ids": ids,
                "variant": variant_name,
                "score": score,
            }

    scaled_corners: List[np.ndarray] = []
    if best["corners"]:
        for c in best["corners"]:
            scaled_corners.append((c / scale).astype(np.float32))

    ids = best["ids"]
    info = {
        "variant": best["variant"],
        "expected_ids_detected": int(best["score"][0]),
        "total_markers_detected": int(best["score"][1]),
        "scale_for_detection": float(scale),
    }
    return scaled_corners, ids, info


def marker_object_corners_m(marker_id: int, marker_size_cm: float) -> np.ndarray:
    size_m = marker_size_cm / 100.0
    x, y = MARKER_LAYOUT_M[marker_id]
    return np.array(
        [
            [x, y],
            [x + size_m, y],
            [x + size_m, y + size_m],
            [x, y + size_m],
        ],
        dtype=np.float32,
    )


def board_bounds_m(marker_size_cm: float) -> Tuple[float, float, float, float]:
    size_m = marker_size_cm / 100.0
    xs = []
    ys = []
    for x, y in MARKER_LAYOUT_M.values():
        xs.extend([x, x + size_m])
        ys.extend([y, y + size_m])
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def board_points_to_destination(
    board_points: np.ndarray, marker_size_cm: float, rectified_size: int
) -> np.ndarray:
    min_x, min_y, max_x, max_y = board_bounds_m(marker_size_cm)
    width = max_x - min_x
    height = max_y - min_y
    dst = np.zeros_like(board_points, dtype=np.float32)
    dst[:, 0] = (board_points[:, 0] - min_x) * (rectified_size - 1) / width
    dst[:, 1] = (board_points[:, 1] - min_y) * (rectified_size - 1) / height
    return dst


def best_expected_markers(
    corners: List[np.ndarray], ids: Optional[np.ndarray]
) -> Dict[int, np.ndarray]:
    if ids is None:
        return {}
    best_by_id: Dict[int, np.ndarray] = {}
    best_perimeter: Dict[int, float] = {}
    for c, marker_id in zip(corners, ids.flatten().tolist()):
        if marker_id not in EXPECTED_MARKER_IDS:
            continue
        cc = c.reshape(-1, 2).astype(np.float32)
        perimeter = float(cv2.arcLength(cc, True))
        if marker_id not in best_by_id or perimeter > best_perimeter[marker_id]:
            best_by_id[marker_id] = cc
            best_perimeter[marker_id] = perimeter
    return best_by_id


def marker_center_board_points(marker_size_cm: float) -> Dict[str, np.ndarray]:
    size_m = marker_size_cm / 100.0
    out: Dict[str, np.ndarray] = {}
    for pos in POSITION_ORDER:
        marker_id = POSITION_TO_EXPECTED_ID[pos]
        x, y = MARKER_LAYOUT_M[marker_id]
        out[pos] = np.array([x + size_m / 2.0, y + size_m / 2.0], dtype=np.float32)
    return out


def choose_missing_positions_by_corner_distance(
    missing_positions: List[str],
    candidate_pool: List[Dict[str, object]],
    image_targets: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, object]]:
    if not missing_positions or not candidate_pool:
        return {}

    n = len(candidate_pool)
    m = len(missing_positions)
    if n < m:
        return {}

    # Limitamos para evitar explosión combinatoria en permutaciones.
    if n > 10:
        candidate_pool = sorted(
            candidate_pool, key=lambda x: float(x["perimeter"]), reverse=True
        )[:10]
        n = len(candidate_pool)
        if n < m:
            return {}

    best_score = float("inf")
    best_assignment: Optional[Tuple[int, ...]] = None
    for perm in itertools.permutations(range(n), m):
        score = 0.0
        for pos, idx in zip(missing_positions, perm):
            center = candidate_pool[idx]["center"]
            score += float(np.linalg.norm(center - image_targets[pos]))
        if score < best_score:
            best_score = score
            best_assignment = perm

    if best_assignment is None:
        return {}

    out: Dict[str, Dict[str, object]] = {}
    for pos, idx in zip(missing_positions, best_assignment):
        out[pos] = candidate_pool[idx]
    return out


def src_points_from_assigned_markers(
    assigned_by_pos: Dict[str, Dict[str, object]], expansion_factor: float
) -> Optional[np.ndarray]:
    # Corner indices del algoritmo que ya funcionaba en tu flujo original.
    # OpenCV ArUco corners: [0,1,2,3]
    corner_idx = {
        "tl": (2, 0),  # (inner, outer)
        "tr": (3, 1),
        "br": (0, 2),
        "bl": (1, 3),
    }
    pts: List[np.ndarray] = []
    for pos in POSITION_ORDER:
        cand = assigned_by_pos.get(pos)
        if cand is None:
            return None
        corners = np.array(cand["corners"], dtype=np.float32).reshape(4, 2)
        inner_idx, outer_idx = corner_idx[pos]
        inner = corners[inner_idx]
        outer = corners[outer_idx]
        p = inner + expansion_factor * (outer - inner)
        pts.append(p)
    return np.array(pts, dtype=np.float32)


def green_mask_robust(rectified_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2HSV)
    mask_main = cv2.inRange(
        hsv, np.array([35, 30, 30], dtype=np.uint8), np.array([95, 255, 255], dtype=np.uint8)
    )
    mask_wide = cv2.inRange(
        hsv, np.array([25, 20, 20], dtype=np.uint8), np.array([105, 255, 255], dtype=np.uint8)
    )
    ratio_main = float(np.count_nonzero(mask_main)) / mask_main.size
    ratio_wide = float(np.count_nonzero(mask_wide)) / mask_wide.size
    # Queremos capturar la placa verde completa; preferimos máscara con cobertura realista.
    if 0.20 <= ratio_main <= 0.95:
        mask = mask_main
    elif 0.20 <= ratio_wide <= 0.98:
        mask = mask_wide
    else:
        mask = cv2.bitwise_or(mask_main, mask_wide)

    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


def clean_alpha_keep_tool(alpha: np.ndarray) -> np.ndarray:
    binary = (alpha > 20).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return alpha

    h, w = alpha.shape[:2]
    border_margin = max(3, int(min(h, w) * 0.02))
    best_label = -1
    best_area = -1

    for label in range(1, n_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_border = (
            x <= border_margin
            or y <= border_margin
            or (x + ww) >= (w - border_margin)
            or (y + hh) >= (h - border_margin)
        )
        # Priorizamos componentes no pegadas al borde (la herramienta).
        if not touches_border and area > best_area:
            best_area = area
            best_label = label

    # Fallback: si todo toca borde, usar la componente mas grande.
    if best_label < 0:
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_label = label

    keep = (labels == best_label).astype(np.uint8) * 255
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    cleaned = cv2.bitwise_and(alpha, alpha, mask=keep)
    return cleaned

def create_marker_mask(
    image_shape: Tuple[int, int, int], ordered_markers: List[np.ndarray], margin: int
) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for marker in ordered_markers:
        center = marker.mean(axis=0)
        expanded = []
        for corner in marker:
            direction = corner - center
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
                expanded.append(corner + direction * margin)
            else:
                expanded.append(corner)
        poly = np.array(expanded, dtype=np.int32)
        cv2.fillConvexPoly(mask, poly, 255)
    return mask


def dynamic_green_mask(rectified_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    band = max(8, int(min(h, w) * 0.04))
    top = hsv[:band, :, :].reshape(-1, 3)
    bottom = hsv[h - band :, :, :].reshape(-1, 3)
    left = hsv[:, :band, :].reshape(-1, 3)
    right = hsv[:, w - band :, :].reshape(-1, 3)
    border_pixels = np.vstack([top, bottom, left, right])

    valid = (border_pixels[:, 1] > 25) & (border_pixels[:, 2] > 25)
    lower = np.array([35, 25, 25], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    if valid.any():
        border_h = border_pixels[valid, 0]
        border_s = border_pixels[valid, 1]
        border_v = border_pixels[valid, 2]
        hue_med = float(np.median(border_h))
        if 20 <= hue_med <= 110:
            h_low = int(max(20, hue_med - 20))
            h_high = int(min(110, hue_med + 20))
            s_low = int(max(20, np.percentile(border_s, 10) - 10))
            v_low = int(max(20, np.percentile(border_v, 10) - 10))
            lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
            upper = np.array([h_high, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


def auto_roi_from_alpha(alpha: np.ndarray, margin: int = 10) -> Optional[Tuple[int, int, int, int]]:
    _, binary = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x_min, y_min = math.inf, math.inf
    x_max, y_max = 0, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    h_img, w_img = alpha.shape[:2]
    x_min = max(0, int(x_min) - margin)
    y_min = max(0, int(y_min) - margin)
    x_max = min(w_img, int(x_max) + margin)
    y_max = min(h_img, int(y_max) + margin)
    return x_min, y_min, x_max - x_min, y_max - y_min


def resize_letterbox_gray(gray: np.ndarray, target_size: int, pad_value: int = 114) -> np.ndarray:
    h, w = gray.shape[:2]
    scale = min(target_size / float(w), target_size / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(gray, (new_w, new_h), interpolation=interp)
    canvas = np.full((target_size, target_size), pad_value, dtype=np.uint8)
    x_off = (target_size - new_w) // 2
    y_off = (target_size - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def draw_detection_debug(
    image_bgr: np.ndarray,
    corners: List[np.ndarray],
    ids: Optional[np.ndarray],
    ordered_markers: Optional[List[np.ndarray]],
    src_points: Optional[np.ndarray],
) -> np.ndarray:
    vis = image_bgr.copy()
    if corners:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    if ordered_markers is not None:
        colors = [(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0)]
        for idx, marker in enumerate(ordered_markers):
            poly = marker.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [poly], isClosed=True, color=colors[idx], thickness=2)
            center = marker.mean(axis=0).astype(int)
            cv2.putText(
                vis,
                f"C{idx}",
                (int(center[0]), int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                colors[idx],
                2,
                cv2.LINE_AA,
            )
    if src_points is not None:
        for idx, pt in enumerate(src_points):
            p = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(vis, p, 6, (0, 0, 255), -1)
            cv2.putText(
                vis,
                f"P{idx}",
                (p[0] + 5, p[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    return vis


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def choose_debug_samples(items: List[ImageItem], per_class: int) -> Dict[str, set]:
    picked: Dict[str, set] = defaultdict(set)
    by_class: Dict[str, List[ImageItem]] = defaultdict(list)
    for item in items:
        by_class[item.class_name].append(item)
    for class_name, class_items in by_class.items():
        for item in class_items[:per_class]:
            picked[class_name].add(item.path.stem)
    return picked


def stratified_subset(items: List[ImageItem], max_items: int, seed: int) -> List[ImageItem]:
    if len(items) <= max_items:
        return list(items)
    rng = np.random.default_rng(seed)
    by_class: Dict[str, List[ImageItem]] = defaultdict(list)
    for item in items:
        by_class[item.class_name].append(item)
    class_names = sorted(by_class.keys())
    quota = max(1, max_items // len(class_names))
    selected: List[ImageItem] = []
    leftovers: List[ImageItem] = []
    for class_name in class_names:
        class_items = by_class[class_name]
        idx = np.arange(len(class_items))
        rng.shuffle(idx)
        keep = idx[: min(quota, len(class_items))]
        rest = idx[min(quota, len(class_items)) :]
        selected.extend([class_items[i] for i in keep])
        leftovers.extend([class_items[i] for i in rest])
    if len(selected) < max_items and leftovers:
        idx = np.arange(len(leftovers))
        rng.shuffle(idx)
        needed = max_items - len(selected)
        selected.extend([leftovers[i] for i in idx[:needed]])
    return selected[:max_items]

def estimate_calibration(
    items: List[ImageItem],
    detector: cv2.aruco.ArucoDetector,
    board: cv2.aruco.Board,
    max_images: int,
    detect_max_dim: int,
    seed: int,
    progress_every: int,
) -> CalibrationResult:
    if max_images <= 0:
        return CalibrationResult(
            success=False,
            camera_matrix=np.eye(3, dtype=np.float32),
            dist_coeffs=np.zeros((5, 1), dtype=np.float32),
            reprojection_error=float("nan"),
            used_frames=0,
            used_markers=0,
        )

    subset = stratified_subset(items, max_items=max_images, seed=seed)
    all_corners: List[np.ndarray] = []
    all_ids: List[np.ndarray] = []
    marker_counter: List[int] = []
    image_size: Optional[Tuple[int, int]] = None
    total_subset = len(subset)
    used_markers_running = 0
    start_ts = time.perf_counter()

    print(
        f"  [calib] analizando {total_subset} imagenes candidatas...",
        flush=True,
    )

    for idx, item in enumerate(subset, start=1):
        image = cv2.imread(str(item.path))
        if image is None:
            if progress_every > 0 and (idx % progress_every == 0 or idx == total_subset):
                elapsed = time.perf_counter() - start_ts
                print(
                    f"  [calib {idx}/{total_subset}] frames_validos={len(marker_counter)} "
                    f"markers={used_markers_running} elapsed={elapsed/60.0:.1f} min",
                    flush=True,
                )
            continue
        if image_size is None:
            image_size = (image.shape[1], image.shape[0])
        corners, ids, _ = detect_markers_best(
            image,
            detector,
            board=board,
            camera_matrix=None,
            dist_coeffs=None,
            max_dim=detect_max_dim,
        )
        if ids is None or len(corners) == 0:
            continue

        # Usar solo frames que contienen los 4 IDs esperados para evitar
        # correspondencias ambiguas en calibracion.
        best_by_id = best_expected_markers(corners, ids)

        if len(best_by_id) < 4:
            continue

        ordered_ids = [0, 1, 2, 3]
        frame_corners = [best_by_id[mid].reshape(1, 4, 2) for mid in ordered_ids]
        frame_ids = np.array(ordered_ids, dtype=np.int32).reshape(-1, 1)

        all_corners.extend(frame_corners)
        all_ids.extend(frame_ids)
        marker_counter.append(4)
        used_markers_running += 4

        if progress_every > 0 and (idx % progress_every == 0 or idx == total_subset):
            elapsed = time.perf_counter() - start_ts
            print(
                f"  [calib {idx}/{total_subset}] frames_validos={len(marker_counter)} "
                f"markers={used_markers_running} elapsed={elapsed/60.0:.1f} min",
                flush=True,
            )

    if image_size is None or len(marker_counter) < 12:
        elapsed = time.perf_counter() - start_ts
        print(
            f"  [calib] insuficiente para calibrar (frames_validos={len(marker_counter)}, "
            f"elapsed={elapsed/60.0:.1f} min)",
            flush=True,
        )
        return CalibrationResult(
            success=False,
            camera_matrix=np.eye(3, dtype=np.float32),
            dist_coeffs=np.zeros((5, 1), dtype=np.float32),
            reprojection_error=float("nan"),
            used_frames=len(marker_counter),
            used_markers=int(np.sum(marker_counter)) if marker_counter else 0,
        )

    try:
        print(
            f"  [calib] ejecutando calibrateCameraAruco con {len(marker_counter)} frames...",
            flush=True,
        )
        reproj, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraAruco(
            all_corners,
            np.array(all_ids, dtype=np.int32),
            np.array(marker_counter, dtype=np.int32),
            board,
            image_size,
            None,
            None,
        )

        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])
        w, h = image_size
        plausible_intrinsics = (
            np.isfinite(camera_matrix).all()
            and np.isfinite(dist_coeffs).all()
            and 200.0 < fx < 20000.0
            and 200.0 < fy < 20000.0
            and 0.0 < cx < w
            and 0.0 < cy < h
        )
        plausible_reproj = np.isfinite(reproj) and float(reproj) < 8.0
        calibration_ok = bool(plausible_intrinsics and plausible_reproj)
        elapsed = time.perf_counter() - start_ts
        print(
            f"  [calib] terminado en {elapsed/60.0:.1f} min "
            f"(reproj={float(reproj):.4f}, success={calibration_ok})",
            flush=True,
        )

        return CalibrationResult(
            success=calibration_ok,
            camera_matrix=camera_matrix.astype(np.float32),
            dist_coeffs=dist_coeffs.astype(np.float32),
            reprojection_error=float(reproj),
            used_frames=len(marker_counter),
            used_markers=int(np.sum(marker_counter)),
        )
    except cv2.error:
        elapsed = time.perf_counter() - start_ts
        print(
            f"  [calib] fallo de OpenCV tras {elapsed/60.0:.1f} min; se usara fallback sin undistort.",
            flush=True,
        )
        return CalibrationResult(
            success=False,
            camera_matrix=np.eye(3, dtype=np.float32),
            dist_coeffs=np.zeros((5, 1), dtype=np.float32),
            reprojection_error=float("nan"),
            used_frames=len(marker_counter),
            used_markers=int(np.sum(marker_counter)),
        )


def make_hog(target_size: int) -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        (target_size, target_size),
        (16, 16),
        (8, 8),
        (8, 8),
        9,
    )


def preprocess_one(
    image_bgr: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    board: cv2.aruco.Board,
    calibration: CalibrationResult,
    rectified_size: int,
    corner_expansion_factor: float,
    marker_margin: int,
    marker_size_cm: float,
    target_size: int,
    detect_max_dim: int,
) -> Tuple[bool, Dict[str, np.ndarray], Dict[str, object], Optional[np.ndarray]]:
    debug_images: Dict[str, np.ndarray] = {}
    info: Dict[str, object] = {}

    debug_images["00_raw"] = image_bgr

    if calibration.success:
        undistorted = cv2.undistort(image_bgr, calibration.camera_matrix, calibration.dist_coeffs)
    else:
        undistorted = image_bgr.copy()
    debug_images["01_undistort"] = undistorted

    corners, ids, detect_info = detect_markers_best(
        undistorted,
        detector,
        board=board,
        camera_matrix=calibration.camera_matrix if calibration.success else None,
        dist_coeffs=calibration.dist_coeffs if calibration.success else None,
        max_dim=detect_max_dim,
    )
    info.update(detect_info)

    marker_candidates: List[Dict[str, object]] = []
    ids_flat = ids.flatten().tolist() if ids is not None else []
    for idx, c in enumerate(corners):
        cc = c.reshape(4, 2).astype(np.float32)
        marker_id = int(ids_flat[idx]) if idx < len(ids_flat) else -1
        marker_candidates.append(
            {
                "idx": idx,
                "id": marker_id,
                "corners": cc,
                "center": cc.mean(axis=0),
                "perimeter": float(cv2.arcLength(cc, True)),
            }
        )

    assigned_by_pos: Dict[str, Dict[str, object]] = {}
    used_indices = set()
    for pos in POSITION_ORDER:
        expected_id = POSITION_TO_EXPECTED_ID[pos]
        best: Optional[Dict[str, object]] = None
        for cand in marker_candidates:
            if int(cand["id"]) != expected_id:
                continue
            if best is None or float(cand["perimeter"]) > float(best["perimeter"]):
                best = cand
        if best is not None:
            assigned_by_pos[pos] = best
            used_indices.add(int(best["idx"]))

    expected_assigned = len(assigned_by_pos)
    h, w = undistorted.shape[:2]
    image_targets = {
        "tl": np.array([0.0, 0.0], dtype=np.float32),
        "tr": np.array([float(w - 1), 0.0], dtype=np.float32),
        "br": np.array([float(w - 1), float(h - 1)], dtype=np.float32),
        "bl": np.array([0.0, float(h - 1)], dtype=np.float32),
    }
    missing_positions = [p for p in POSITION_ORDER if p not in assigned_by_pos]
    pool = [c for c in marker_candidates if int(c["idx"]) not in used_indices]
    filled = choose_missing_positions_by_corner_distance(
        missing_positions=missing_positions,
        candidate_pool=pool,
        image_targets=image_targets,
    )
    assigned_by_pos.update(filled)

    selected_markers: List[np.ndarray] = []
    source_points: List[np.ndarray] = []
    for pos in POSITION_ORDER:
        cand = assigned_by_pos.get(pos)
        if cand is None:
            continue
        selected_markers.append(np.array(cand["corners"], dtype=np.float32))
        source_points.append(np.array(cand["center"], dtype=np.float32))

    info["markers_expected_assigned"] = int(expected_assigned)
    info["markers_used_for_transform"] = int(len(source_points))
    info["markers_total_candidates"] = int(len(marker_candidates))
    if expected_assigned >= 4:
        info["selection_mode"] = "corners_expected_ids"
    elif expected_assigned >= 2:
        info["selection_mode"] = "center_expected_plus_positional"
    else:
        info["selection_mode"] = "center_positional"

    if len(source_points) < 2:
        debug_images["02_detection"] = draw_detection_debug(
            undistorted,
            corners,
            ids,
            selected_markers if selected_markers else None,
            np.array(source_points, dtype=np.float32) if source_points else None,
        )
        info["fail_stage"] = "marker_selection"
        info["fail_reason"] = "not_enough_markers_for_transform"
        return False, debug_images, info, None

    source_arr = np.array(source_points, dtype=np.float32)
    transform_kind = "affine"
    inliers_count = 0
    matrix: Optional[np.ndarray] = None

    # Caso preferido: los 4 IDs esperados detectados -> misma logica que tu flujo original.
    if expected_assigned >= 4:
        src_pts = src_points_from_assigned_markers(
            assigned_by_pos, expansion_factor=corner_expansion_factor
        )
        if src_pts is not None and src_pts.shape == (4, 2):
            dst_pts = np.array(
                [
                    [0.0, 0.0],
                    [float(rectified_size - 1), 0.0],
                    [float(rectified_size - 1), float(rectified_size - 1)],
                    [0.0, float(rectified_size - 1)],
                ],
                dtype=np.float32,
            )
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            transform_kind = "perspective_corners"
            inliers_count = 4

    # Fallback: centros (perspective/affine), para cuando faltan IDs esperados.
    if matrix is None:
        center_board_map = marker_center_board_points(marker_size_cm=marker_size_cm)
        center_board_arr = np.array([center_board_map[p] for p in POSITION_ORDER], dtype=np.float32)
        center_dst_arr = board_points_to_destination(
            center_board_arr, marker_size_cm=marker_size_cm, rectified_size=rectified_size
        )
        center_dst_map = {p: center_dst_arr[i] for i, p in enumerate(POSITION_ORDER)}
        dest_points: List[np.ndarray] = []
        src_points_fb: List[np.ndarray] = []
        for pos in POSITION_ORDER:
            cand = assigned_by_pos.get(pos)
            if cand is None:
                continue
            src_points_fb.append(np.array(cand["center"], dtype=np.float32))
            dest_points.append(np.array(center_dst_map[pos], dtype=np.float32))
        src_fb = np.array(src_points_fb, dtype=np.float32)
        dst_fb = np.array(dest_points, dtype=np.float32)

        if len(src_points_fb) >= 4:
            matrix = cv2.getPerspectiveTransform(src_fb[:4], dst_fb[:4])
            transform_kind = "perspective_centers"
            inliers_count = 4
        else:
            matrix_aff, inliers = cv2.estimateAffinePartial2D(
                src_fb, dst_fb, method=cv2.LMEDS
            )
            if matrix_aff is None and len(src_points_fb) == 3:
                matrix_aff, inliers = cv2.estimateAffine2D(
                    src_fb, dst_fb, method=cv2.LMEDS
                )
            matrix = matrix_aff
            transform_kind = "affine_centers"
            inliers_count = int(inliers.sum()) if inliers is not None else 0

    debug_images["02_detection"] = draw_detection_debug(
        undistorted, corners, ids, selected_markers, source_arr
    )
    if matrix is None:
        info["fail_stage"] = "transform"
        info["fail_reason"] = "cv2_transform_failed"
        return False, debug_images, info, None
    info["transform_kind"] = transform_kind
    info["transform_inliers"] = inliers_count

    if transform_kind.startswith("perspective"):
        rectified = cv2.warpPerspective(undistorted, matrix, (rectified_size, rectified_size))
    else:
        rectified = cv2.warpAffine(undistorted, matrix, (rectified_size, rectified_size))
    debug_images["03_rectified"] = rectified

    marker_mask = create_marker_mask(undistorted.shape, selected_markers, margin=marker_margin)
    if transform_kind.startswith("perspective"):
        marker_mask_rectified = cv2.warpPerspective(
            marker_mask, matrix, (rectified_size, rectified_size)
        )
    else:
        marker_mask_rectified = cv2.warpAffine(
            marker_mask, matrix, (rectified_size, rectified_size)
        )
    green_mask = green_mask_robust(rectified)
    debug_images["04_green_mask"] = green_mask

    combined_mask = cv2.bitwise_or(green_mask, marker_mask_rectified)
    alpha = cv2.bitwise_not(combined_mask)
    alpha = clean_alpha_keep_tool(alpha)
    object_bgr = rectified.copy()
    object_bgr[alpha < 8] = 0
    object_rgba = cv2.cvtColor(object_bgr, cv2.COLOR_BGR2BGRA)
    object_rgba[:, :, 3] = alpha
    debug_images["05_object_rgba"] = object_rgba

    roi = auto_roi_from_alpha(alpha, margin=10)
    if roi is None:
        info["fail_stage"] = "roi"
        info["fail_reason"] = "empty_alpha_roi"
        return False, debug_images, info, None

    x, y, w, h = roi
    # Si el ROI casi ocupa toda la placa, recorta por dentro del marco de marcadores
    # para centrar la herramienta y reducir distracciones de bordes/soportes.
    roi_area = float(w * h)
    rect_area = float(rectified_size * rectified_size)
    if roi_area > 0.75 * rect_area:
        margin_px = int(max(12, rectified_size * 0.09))
        x = max(x, margin_px)
        y = max(y, margin_px)
        x2 = min(x + w, rectified_size - margin_px)
        y2 = min(y + h, rectified_size - margin_px)
        w = max(1, x2 - x)
        h = max(1, y2 - y)

    roi_rgba = object_rgba[y : y + h, x : x + w]
    debug_images["06_roi_rgba"] = roi_rgba

    roi_bgr = cv2.cvtColor(roi_rgba, cv2.COLOR_BGRA2BGR)
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray_denoised = cv2.bilateralFilter(roi_gray, d=7, sigmaColor=40, sigmaSpace=40)
    debug_images["07_gray_denoised"] = gray_denoised

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_denoised)
    debug_images["08_gray_clahe"] = gray_clahe

    final_gray = resize_letterbox_gray(gray_clahe, target_size=target_size, pad_value=114)
    debug_images["09_final_gray"] = final_gray

    return True, debug_images, info, final_gray


def save_debug_bundle(base_dir: Path, debug_images: Dict[str, np.ndarray]) -> None:
    for key, image in debug_images.items():
        ext = ".png"
        if image.ndim == 3 and image.shape[2] == 3:
            ext = ".jpg"
        save_image(base_dir / f"{key}{ext}", image)


def plot_pca_scatter(
    x_pca: np.ndarray, labels: Sequence[str], classes: Sequence[str], output_path: Path
) -> None:
    if x_pca.shape[1] < 2:
        return
    plt.figure(figsize=(11, 8))
    cmap = plt.get_cmap("tab10", len(classes))
    labels_array = np.array(labels)
    for idx, class_name in enumerate(classes):
        mask = labels_array == class_name
        plt.scatter(
            x_pca[mask, 0],
            x_pca[mask, 1],
            s=18,
            alpha=0.75,
            label=class_name,
            color=cmap(idx),
        )
    plt.title("PCA sobre HOG (PC1 vs PC2)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.25)
    plt.legend(markerscale=1.3, fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_pca_variance(pca: PCA, output_path: Path) -> None:
    cum = np.cumsum(pca.explained_variance_ratio_)
    xs = np.arange(1, len(cum) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(xs, cum, marker="o", markersize=3, linewidth=1.2)
    plt.axhline(0.95, color="red", linestyle="--", linewidth=1.2, label="95%")
    plt.title("Varianza acumulada de PCA")
    plt.xlabel("Numero de componentes")
    plt.ylabel("Varianza acumulada")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_hog_norms(hog_features: np.ndarray, output_path: Path) -> None:
    norms = np.linalg.norm(hog_features, axis=1)
    plt.figure(figsize=(10, 5))
    plt.hist(norms, bins=30, color="steelblue", edgecolor="black", alpha=0.75)
    plt.title("Distribucion de norma L2 de descriptores HOG")
    plt.xlabel("Norma L2")
    plt.ylabel("Frecuencia")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_stratified_split(
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> np.ndarray:
    if labels.ndim != 1:
        raise ValueError("labels debe ser un vector 1D")
    if len(labels) < 3:
        return np.array(["train"] * len(labels))

    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Los ratios train/val/test deben ser >= 0")
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("La suma de ratios train/val/test debe ser > 0")

    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    test_ratio /= ratio_sum

    indices = np.arange(len(labels))
    split = np.empty(len(labels), dtype="<U10")

    if test_ratio > 0:
        try:
            idx_trainval, idx_test = train_test_split(
                indices,
                test_size=test_ratio,
                stratify=labels,
                random_state=seed,
            )
        except ValueError:
            idx_trainval, idx_test = train_test_split(
                indices,
                test_size=test_ratio,
                stratify=None,
                random_state=seed,
            )
    else:
        idx_trainval = indices
        idx_test = np.array([], dtype=int)

    remain_ratio = train_ratio + val_ratio
    if val_ratio > 0 and len(idx_trainval) > 1 and remain_ratio > 0:
        val_share_in_trainval = val_ratio / remain_ratio
        try:
            idx_train, idx_val = train_test_split(
                idx_trainval,
                test_size=val_share_in_trainval,
                stratify=labels[idx_trainval],
                random_state=seed,
            )
        except ValueError:
            idx_train, idx_val = train_test_split(
                idx_trainval,
                test_size=val_share_in_trainval,
                stratify=None,
                random_state=seed,
            )
    else:
        idx_train = idx_trainval
        idx_val = np.array([], dtype=int)

    split[idx_train] = "train"
    split[idx_val] = "val"
    split[idx_test] = "test"
    return split

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    if not args.data_root.exists():
        raise FileNotFoundError(f"No existe data-root: {args.data_root}")

    output_paths = ensure_dirs(args.output_root)
    image_items = list_images(args.data_root)
    if args.max_images is not None:
        image_items = image_items[: args.max_images]

    if not image_items:
        raise RuntimeError("No se encontraron imagenes .jpg en data-root.")

    print(f"Imagenes encontradas: {len(image_items)}")

    detector = build_detector()
    board = build_board(marker_size_cm=args.marker_size_cm)

    qc_rows = []
    valid_items: List[ImageItem] = []
    corrupted_items = []
    for item in image_items:
        img = cv2.imread(str(item.path))
        if img is None:
            corrupted_items.append(item)
            continue
        lap, bright = compute_qc_metrics(img)
        qc_rows.append((item, lap, bright))
        valid_items.append(item)

    if not qc_rows:
        raise RuntimeError("Todas las imagenes fallaron al cargar.")

    thresholds = qc_thresholds([(lap, bright) for _, lap, bright in qc_rows])
    print("Umbrales QC (auto por percentiles):")
    print(json.dumps(thresholds, indent=2))

    print("Estimando calibracion de camara con ArUco...")
    calibration = estimate_calibration(
        valid_items,
        detector=detector,
        board=board,
        max_images=args.calibration_max_images,
        detect_max_dim=args.detect_max_dim,
        seed=args.seed,
        progress_every=args.calibration_progress_every,
    )
    print(
        f"Calibracion: success={calibration.success}, "
        f"frames={calibration.used_frames}, markers={calibration.used_markers}, "
        f"reproj_error={calibration.reprojection_error:.4f}"
    )

    np.savez_compressed(
        output_paths["calibration"] / "camera_calibration.npz",
        success=np.array([int(calibration.success)], dtype=np.int32),
        camera_matrix=calibration.camera_matrix,
        dist_coeffs=calibration.dist_coeffs,
        reprojection_error=np.array([calibration.reprojection_error], dtype=np.float32),
        used_frames=np.array([calibration.used_frames], dtype=np.int32),
        used_markers=np.array([calibration.used_markers], dtype=np.int32),
    )

    debug_samples = choose_debug_samples(image_items, per_class=args.debug_samples_per_class)
    hog = make_hog(args.target_size)

    features: List[np.ndarray] = []
    labels: List[str] = []
    rel_paths: List[str] = []
    metadata_rows: List[Dict[str, object]] = []
    fail_counter = Counter()

    qc_lookup = {row[0].path: (row[1], row[2]) for row in qc_rows}

    for idx, item in enumerate(image_items, start=1):
        row = {
            "class_name": item.class_name,
            "file_name": item.path.name,
            "relative_path": str(item.path.relative_to(args.data_root)),
            "status": "failed",
            "fail_stage": "",
            "fail_reason": "",
            "laplacian_var": None,
            "brightness_mean": None,
            "flag_blur": False,
            "flag_dark": False,
            "flag_bright": False,
            "markers_expected_detected": 0,
            "markers_total_detected": 0,
            "markers_expected_assigned": 0,
            "markers_used_for_transform": 0,
            "detect_variant": "",
            "selection_mode": "",
            "transform_kind": "",
            "transform_inliers": 0,
            "final_gray_path": "",
        }

        print(f"[{idx:04d}/{len(image_items)}] {item.class_name}/{item.path.name}")

        image_bgr = cv2.imread(str(item.path))
        if image_bgr is None:
            row["fail_stage"] = "load"
            row["fail_reason"] = "corrupt_or_unreadable"
            fail_counter["load:corrupt_or_unreadable"] += 1
            metadata_rows.append(row)
            continue

        lap, bright = qc_lookup[item.path]
        row["laplacian_var"] = lap
        row["brightness_mean"] = bright
        row["flag_blur"] = lap < thresholds["laplacian_blur_p5"]
        row["flag_dark"] = bright < thresholds["brightness_low_p1"]
        row["flag_bright"] = bright > thresholds["brightness_high_p99"]

        ok, debug_images, info, final_gray = preprocess_one(
            image_bgr=image_bgr,
            detector=detector,
            board=board,
            calibration=calibration,
            rectified_size=args.rectified_size,
            corner_expansion_factor=args.corner_expansion_factor,
            marker_margin=args.marker_margin,
            marker_size_cm=args.marker_size_cm,
            target_size=args.target_size,
            detect_max_dim=args.detect_max_dim,
        )

        row["markers_expected_detected"] = int(info.get("expected_ids_detected", 0))
        row["markers_total_detected"] = int(info.get("total_markers_detected", 0))
        row["markers_expected_assigned"] = int(info.get("markers_expected_assigned", 0))
        row["markers_used_for_transform"] = int(info.get("markers_used_for_transform", 0))
        row["detect_variant"] = str(info.get("variant", ""))
        row["selection_mode"] = str(info.get("selection_mode", ""))
        row["transform_kind"] = str(info.get("transform_kind", ""))
        row["transform_inliers"] = int(info.get("transform_inliers", 0))

        should_save_sample = item.path.stem in debug_samples[item.class_name]

        if not ok or final_gray is None:
            row["fail_stage"] = str(info.get("fail_stage", "unknown"))
            row["fail_reason"] = str(info.get("fail_reason", "unknown"))
            fail_counter[f"{row['fail_stage']}:{row['fail_reason']}"] += 1
            fail_dir = output_paths["debug_failures"] / item.class_name / item.path.stem
            save_debug_bundle(fail_dir, debug_images)
            metadata_rows.append(row)
            continue

        final_out = (
            output_paths["preprocessed"] / item.class_name / f"{item.path.stem}.png"
        )
        save_image(final_out, final_gray)
        row["final_gray_path"] = str(final_out.relative_to(args.output_root))
        row["status"] = "ok"

        if should_save_sample:
            sample_dir = output_paths["debug_samples"] / item.class_name / item.path.stem
            save_debug_bundle(sample_dir, debug_images)

        hog_vec = hog.compute(final_gray).reshape(-1).astype(np.float32)
        features.append(hog_vec)
        labels.append(item.class_name)
        rel_paths.append(str(item.path.relative_to(args.data_root)))
        metadata_rows.append(row)

    if not features:
        raise RuntimeError("No hubo imagenes procesadas exitosamente para HOG/PCA.")

    x_hog = np.vstack(features).astype(np.float32)
    labels_arr = np.array(labels)
    paths_arr = np.array(rel_paths)
    classes = sorted(set(labels))

    split_arr = build_stratified_split(
        labels=labels_arr,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    mask_train = split_arr == "train"
    mask_val = split_arr == "val"
    mask_test = split_arr == "test"

    x_hog_train = x_hog[mask_train]
    x_hog_val = x_hog[mask_val]
    x_hog_test = x_hog[mask_test]

    np.savez_compressed(
        output_paths["features"] / "hog_features.npz",
        X_hog=x_hog,
        y=labels_arr,
        paths=paths_arr,
        split=split_arr,
        classes=np.array(classes),
    )
    np.savez_compressed(
        output_paths["features"] / "hog_features_split.npz",
        X_train=x_hog_train,
        y_train=labels_arr[mask_train],
        paths_train=paths_arr[mask_train],
        X_val=x_hog_val,
        y_val=labels_arr[mask_val],
        paths_val=paths_arr[mask_val],
        X_test=x_hog_test,
        y_test=labels_arr[mask_test],
        paths_test=paths_arr[mask_test],
        classes=np.array(classes),
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_hog_train)
    x_val_scaled = scaler.transform(x_hog_val) if len(x_hog_val) else np.empty((0, x_hog.shape[1]), dtype=np.float32)
    x_test_scaled = scaler.transform(x_hog_test) if len(x_hog_test) else np.empty((0, x_hog.shape[1]), dtype=np.float32)

    pca = PCA(n_components=args.pca_variance, random_state=args.seed, svd_solver="full")
    x_train_pca = pca.fit_transform(x_train_scaled).astype(np.float32)
    x_val_pca = pca.transform(x_val_scaled).astype(np.float32) if len(x_val_scaled) else np.empty((0, x_train_pca.shape[1]), dtype=np.float32)
    x_test_pca = pca.transform(x_test_scaled).astype(np.float32) if len(x_test_scaled) else np.empty((0, x_train_pca.shape[1]), dtype=np.float32)

    x_pca_all = np.zeros((len(x_hog), x_train_pca.shape[1]), dtype=np.float32)
    x_pca_all[mask_train] = x_train_pca
    if len(x_val_pca):
        x_pca_all[mask_val] = x_val_pca
    if len(x_test_pca):
        x_pca_all[mask_test] = x_test_pca

    np.savez_compressed(
        output_paths["features"] / "hog_pca_features.npz",
        X_pca=x_pca_all,
        y=labels_arr,
        paths=paths_arr,
        split=split_arr,
        classes=np.array(classes),
    )
    np.savez_compressed(
        output_paths["features"] / "hog_pca_features_split.npz",
        X_train=x_train_pca,
        y_train=labels_arr[mask_train],
        paths_train=paths_arr[mask_train],
        X_val=x_val_pca,
        y_val=labels_arr[mask_val],
        paths_val=paths_arr[mask_val],
        X_test=x_test_pca,
        y_test=labels_arr[mask_test],
        paths_test=paths_arr[mask_test],
        classes=np.array(classes),
    )
    joblib.dump(scaler, output_paths["features"] / "hog_scaler.joblib")
    joblib.dump(pca, output_paths["features"] / "hog_pca_model.joblib")

    split_df = pd.DataFrame(
        {"relative_path": paths_arr, "class_name": labels_arr, "split": split_arr}
    )
    split_df.to_csv(output_paths["logs"] / "dataset_split.csv", index=False)

    plot_pca_scatter(
        x_pca=x_pca_all,
        labels=labels,
        classes=classes,
        output_path=output_paths["plots"] / "pca_scatter_pc1_pc2.png",
    )
    plot_pca_variance(
        pca=pca,
        output_path=output_paths["plots"] / "pca_cumulative_variance.png",
    )
    plot_hog_norms(
        hog_features=x_hog,
        output_path=output_paths["plots"] / "hog_norm_distribution.png",
    )

    metadata_df = pd.DataFrame(metadata_rows)
    split_lookup = {p: s for p, s in zip(paths_arr.tolist(), split_arr.tolist())}
    metadata_df["split"] = metadata_df["relative_path"].map(split_lookup).fillna("failed")
    metadata_df.to_csv(output_paths["logs"] / "preprocess_metadata.csv", index=False)

    split_counts = {
        "train": int(np.sum(mask_train)),
        "val": int(np.sum(mask_val)),
        "test": int(np.sum(mask_test)),
    }
    split_class_counts = (
        split_df.groupby(["split", "class_name"]).size().reset_index(name="count")
    )
    split_class_counts.to_csv(
        output_paths["logs"] / "dataset_split_by_class.csv", index=False
    )

    summary = {
        "total_images": int(len(image_items)),
        "success_images": int((metadata_df["status"] == "ok").sum()),
        "failed_images": int((metadata_df["status"] != "ok").sum()),
        "corrupted_images": int(len(corrupted_items)),
        "classes": classes,
        "qc_thresholds": thresholds,
        "calibration": {
            "success": calibration.success,
            "used_frames": calibration.used_frames,
            "used_markers": calibration.used_markers,
            "reprojection_error": calibration.reprojection_error,
        },
        "hog": {
            "target_size": args.target_size,
            "descriptor_size": int(hog.getDescriptorSize()),
            "samples": int(x_hog.shape[0]),
            "dims": int(x_hog.shape[1]),
        },
        "split": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "counts": split_counts,
        },
        "pca": {
            "n_components": int(x_pca_all.shape[1]),
            "explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
            "fit_on": "train_only",
        },
        "top_fail_reasons": fail_counter.most_common(15),
    }
    with open(output_paths["logs"] / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Pipeline terminado")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Artefactos en: {args.output_root.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
