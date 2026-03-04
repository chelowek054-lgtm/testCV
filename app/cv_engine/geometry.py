"""Гомография, полигон автомобиля из bbox, площадь пересечения полигонов (IoU)."""
from typing import List, Optional, Tuple

import cv2
import numpy as np

_homography: Optional[np.ndarray] = None


def set_homography(H: Optional[np.ndarray]) -> None:
    """Матрица 3x3 (image -> map). None — координаты изображения."""
    global _homography
    _homography = H


def get_homography() -> Optional[np.ndarray]:
    return _homography


def bbox_to_vehicle_polygon(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Четыре угла bbox в порядке обхода: (x1,y1), (x2,y1), (x2,y2), (x1,y2)."""
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )


def apply_homography_to_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Точки (N, 2) -> координаты карты через H (3x3)."""
    if points.size == 0:
        return points
    pts = points.reshape(-1, 2)
    ones = np.ones((len(pts), 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])  # (N, 3)
    out = (H @ pts_h.T).T  # (N, 3)
    w = out[:, 2:3]
    w[w == 0] = 1e-10
    out = out[:, :2] / w
    return out.astype(np.float32)


def vehicle_polygon_in_map_space(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Полигон из 4 углов bbox в координатах карты (или изображения, если H не задана)."""
    poly = bbox_to_vehicle_polygon(x1, y1, x2, y2)
    H = get_homography()
    if H is not None:
        poly = apply_homography_to_points(poly, H)
    return poly


def _polygon_intersection_area(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Площадь пересечения двух полигонов (N, 2), через маски OpenCV."""
    if len(poly1) < 3 or len(poly2) < 3:
        return 0.0
    p1 = np.array(poly1, dtype=np.float32).reshape(-1, 2)
    p2 = np.array(poly2, dtype=np.float32).reshape(-1, 2)
    x_min = min(p1[:, 0].min(), p2[:, 0].min())
    x_max = max(p1[:, 0].max(), p2[:, 0].max())
    y_min = min(p1[:, 1].min(), p2[:, 1].min())
    y_max = max(p1[:, 1].max(), p2[:, 1].max())
    w = max(1, int(x_max - x_min) + 2)
    h = max(1, int(y_max - y_min) + 2)
    shift = np.array([[x_min, y_min]], dtype=np.float32)
    m1 = np.zeros((h, w), dtype=np.uint8)
    m2 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m1, [(p1 - shift).astype(np.int32)], 255)
    cv2.fillPoly(m2, [(p2 - shift).astype(np.int32)], 255)
    inter = cv2.bitwise_and(m1, m2)
    return float(cv2.countNonZero(inter))


def spot_with_max_intersection(
    vehicle_poly: np.ndarray,
    spots: List[Tuple[int, np.ndarray]],
) -> Optional[int]:
    """Id места с максимальной площадью пересечения с полигоном автомобиля."""
    if not spots or vehicle_poly.size < 6:
        return None
    best_id = None
    best_area = 0.0
    for spot_id, spot_poly in spots:
        area = _polygon_intersection_area(vehicle_poly, spot_poly)
        if area > best_area:
            best_area = area
            best_id = spot_id
    return best_id
