#!/usr/bin/env python
from __future__ import annotations

import argparse
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
from sklearn.preprocessing import StandardScaler


EXPECTED_MARKER_IDS = {0, 1, 2, 3}
MARKER_LAYOUT_M = {
    0: (0.00, 0.00),
    1: (0.10, 0.00),
    2: (0.00, 0.10),
    3: (0.10, 0.10),
}


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

    marker_map = best_expected_markers(corners, ids)
    info["selection_mode"] = "homography_expected_ids"
    info["markers_used_for_h"] = int(len(marker_map))
    debug_images["02_detection"] = draw_detection_debug(
        undistorted, corners, ids, list(marker_map.values()) if marker_map else None, None
    )

    if len(marker_map) < 2:
        info["fail_stage"] = "marker_selection"
        info["fail_reason"] = "not_enough_expected_ids_for_homography"
        return False, debug_images, info, None

    image_points: List[np.ndarray] = []
    board_points: List[np.ndarray] = []
    for marker_id in sorted(marker_map.keys()):
        image_points.append(marker_map[marker_id])
        board_points.append(marker_object_corners_m(marker_id, marker_size_cm=marker_size_cm))
    image_points_arr = np.vstack(image_points).astype(np.float32)
    board_points_arr = np.vstack(board_points).astype(np.float32)
    dest_points_arr = board_points_to_destination(
        board_points_arr, marker_size_cm=marker_size_cm, rectified_size=rectified_size
    )

    matrix, inliers = cv2.findHomography(
        image_points_arr, dest_points_arr, method=cv2.RANSAC, ransacReprojThreshold=4.0
    )
    if matrix is None:
        info["fail_stage"] = "homography"
        info["fail_reason"] = "cv2_findhomography_failed"
        return False, debug_images, info, None
    info["homography_inliers"] = int(inliers.sum()) if inliers is not None else 0

    rectified = cv2.warpPerspective(undistorted, matrix, (rectified_size, rectified_size))
    debug_images["03_rectified"] = rectified

    marker_mask = create_marker_mask(
        undistorted.shape, list(marker_map.values()), margin=marker_margin
    )
    marker_mask_rectified = cv2.warpPerspective(
        marker_mask, matrix, (rectified_size, rectified_size)
    )
    green_mask = dynamic_green_mask(rectified)
    debug_images["04_green_mask"] = green_mask

    combined_mask = cv2.bitwise_or(green_mask, marker_mask_rectified)
    alpha = cv2.bitwise_not(combined_mask)
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
            "detect_variant": "",
            "selection_mode": "",
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
            marker_margin=args.marker_margin,
            marker_size_cm=args.marker_size_cm,
            target_size=args.target_size,
            detect_max_dim=args.detect_max_dim,
        )

        row["markers_expected_detected"] = int(info.get("expected_ids_detected", 0))
        row["markers_total_detected"] = int(info.get("total_markers_detected", 0))
        row["detect_variant"] = str(info.get("variant", ""))
        row["selection_mode"] = str(info.get("selection_mode", ""))

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

    np.savez_compressed(
        output_paths["features"] / "hog_features.npz",
        X_hog=x_hog,
        y=labels_arr,
        paths=paths_arr,
        classes=np.array(classes),
    )

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_hog)
    pca = PCA(n_components=args.pca_variance, random_state=args.seed, svd_solver="full")
    x_pca = pca.fit_transform(x_scaled).astype(np.float32)

    np.savez_compressed(
        output_paths["features"] / "hog_pca_features.npz",
        X_pca=x_pca,
        y=labels_arr,
        paths=paths_arr,
        classes=np.array(classes),
    )
    joblib.dump(scaler, output_paths["features"] / "hog_scaler.joblib")
    joblib.dump(pca, output_paths["features"] / "hog_pca_model.joblib")

    plot_pca_scatter(
        x_pca=x_pca,
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
    metadata_df.to_csv(output_paths["logs"] / "preprocess_metadata.csv", index=False)

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
        "pca": {
            "n_components": int(x_pca.shape[1]),
            "explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
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
