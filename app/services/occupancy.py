"""Хранение векторной карты и расчёт занятости мест (IoU полигонов, сглаживание)."""
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.models import VectorMap, SpotStatus

_vector_map: Optional[VectorMap] = None
_spot_status: Dict[int, float] = {}
_lock = threading.Lock()
_alpha = 0.7  # сглаживание: new = _alpha * old + (1 - _alpha) * raw
IOU_THRESHOLD = 0.25


def set_vector_map(vm: VectorMap) -> VectorMap:
    global _vector_map
    _vector_map = vm
    with _lock:
        for spot in vm.spots:
            _spot_status.setdefault(spot.id, 0.0)
    return vm


def get_vector_map() -> Optional[VectorMap]:
    return _vector_map


def is_vector_map_loaded() -> bool:
    return _vector_map is not None


def update_spot_occupancy_from_bboxes(
    bboxes: Sequence[Tuple[float, float, float, float]],
) -> None:
    """
    Полигон авто (4 угла bbox, опционально через гомографию) → IoU с каждым местом.
    Место с max IoU >= IOU_THRESHOLD считается занятым. Занятость сглаживается.
    """
    import cv2
    from app.cv_engine.geometry import vehicle_polygon_in_map_space, _polygon_intersection_area

    global _vector_map
    if _vector_map is None:
        return

    spots_with_poly: List[Tuple[int, np.ndarray]] = []
    for spot in _vector_map.spots:
        if len(spot.polygon) >= 3:
            spots_with_poly.append((spot.id, np.array(spot.polygon, dtype=np.float32)))

    if not spots_with_poly:
        return

    occupied_ids: set = set()
    for x1, y1, x2, y2 in bboxes:
        vehicle_poly = vehicle_polygon_in_map_space(x1, y1, x2, y2)
        if len(vehicle_poly) < 3:
            continue

        area_car = cv2.contourArea(vehicle_poly.astype(np.float32))
        if area_car <= 0:
            continue

        best_spot_id = None
        best_iou = 0.0

        for spot_id, spot_poly in spots_with_poly:
            area_spot = cv2.contourArea(spot_poly.astype(np.float32))
            if area_spot <= 0:
                continue

            inter = _polygon_intersection_area(vehicle_poly, spot_poly)
            if inter <= 0:
                continue

            union = area_car + area_spot - inter
            if union <= 0:
                continue

            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_spot_id = spot_id

        if best_spot_id is not None and best_iou >= IOU_THRESHOLD:
            occupied_ids.add(best_spot_id)

    with _lock:
        for spot in _vector_map.spots:
            raw = 100.0 if spot.id in occupied_ids else 0.0
            prev = _spot_status.get(spot.id, 0.0)
            new_val = _alpha * prev + (1.0 - _alpha) * raw
            _spot_status[spot.id] = float(new_val)


def update_spot_occupancy_from_points(points: Sequence[Tuple[float, float]]) -> None:
    """Устаревший вариант: точка внутри полигона. Используйте update_spot_occupancy_from_bboxes."""
    global _vector_map
    if _vector_map is None:
        return

    import cv2

    pts_np = np.array(points, dtype=np.float32) if points else None

    with _lock:
        for spot in _vector_map.spots:
            poly = np.array(spot.polygon, dtype=np.float32)
            if poly.shape[0] < 3:
                continue

            raw = 0.0
            if pts_np is not None and len(pts_np) > 0:
                for p in pts_np:
                    inside = cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False)
                    if inside >= 0:
                        raw = 100.0
                        break

            prev = _spot_status.get(spot.id, 0.0)
            new_val = _alpha * prev + (1.0 - _alpha) * raw
            _spot_status[spot.id] = float(new_val)


def get_spot_status_list() -> List[SpotStatus]:
    with _lock:
        return [SpotStatus(id=i, occupancy=occ) for i, occ in _spot_status.items()]
