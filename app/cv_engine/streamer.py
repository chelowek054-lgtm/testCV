"""Фоновый воркер: видеопоток, YOLO, отрисовка векторной карты."""
import cv2
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.cv_engine.detector import run_inference
from app.services import (
    get_vector_map,
    get_spot_status_list,
    update_spot_occupancy_from_bboxes,
)

FRAME_SKIP = 3  # YOLO раз в N кадров для ускорения потока


class CVWorker:
    """Читает поток с камеры, запускает детекцию, обновляет занятость мест, отдаёт кадры в /video."""

    def __init__(self) -> None:
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.stream_url: Optional[str] = None
        self.last_frame_jpeg: Optional[bytes] = None
        self.preview_frame_jpeg: Optional[bytes] = None
        self._frame_index: int = 0
        self._last_detections: List[Tuple[float, ...]] = []

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _get_capture(self) -> Optional[cv2.VideoCapture]:
        if not self.stream_url:
            return None
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.stream_url)
        return self.cap

    def set_stream_url(self, url: str) -> None:
        """Установить источник видеопотока (MP4/RTSP URL)."""
        self.stream_url = url
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _loop(self) -> None:
        while self.running:
            cap = self._get_capture()
            if cap is None:
                # Источник ещё не задан — ждём и пробуем снова
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            self._frame_index += 1
            if self._frame_index % FRAME_SKIP == 0 or not self._last_detections:
                detections = run_inference(frame)
                self._last_detections = detections
                bboxes = [(x1, y1, x2, y2) for (x1, y1, x2, y2, _, _) in detections]
                update_spot_occupancy_from_bboxes(bboxes)
            else:
                detections = self._last_detections

            frame_to_draw = frame.copy()
            vm = get_vector_map()
            spot_polys: list = []
            occ_by_id: Dict[int, float] = {}

            if vm and vm.spots:
                status_list = get_spot_status_list()
                occ_by_id = {s.id: s.occupancy for s in status_list}
                overlay = frame_to_draw.copy()

                for spot in vm.spots:
                    if len(spot.polygon) >= 3:
                        pts = np.array(spot.polygon, dtype=np.int32).reshape((-1, 1, 2))
                        spot_polys.append((spot.id, pts))

                for spot_id, pts in spot_polys:
                    occ = occ_by_id.get(spot_id, 0.0)
                    color = (0, 0, 255) if occ > 50.0 else (0, 255, 0)
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.polylines(overlay, [pts], isClosed=True, color=(255, 255, 255), thickness=1)

                frame_to_draw = cv2.addWeighted(overlay, 0.45, frame_to_draw, 0.55, 0)
                for spot_id, pts in spot_polys:
                    occ = occ_by_id.get(spot_id, 0.0)
                    edge = (0, 0, 255) if occ > 50.0 else (0, 255, 0)
                    cv2.polylines(frame_to_draw, [pts], isClosed=True, color=edge, thickness=2)

            success, buffer = cv2.imencode(".jpg", frame_to_draw)
            if success:
                self.last_frame_jpeg = buffer.tobytes()

        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.running = False


_cv_worker: Optional[CVWorker] = None


def get_cv_worker() -> CVWorker:
    global _cv_worker
    if _cv_worker is None:
        _cv_worker = CVWorker()
    return _cv_worker

