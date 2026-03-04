"""HTTP-маршруты: главная с канвасом, видео, векторная карта, старт/стоп."""
import time
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse

from app.cv_engine import get_cv_worker
from app.models import HealthStatus, SpotStatus, VectorMap
from app.services import get_vector_map, get_spot_status_list, set_vector_map

router = APIRouter()


@router.get("/health", response_model=HealthStatus)
def health():
    cv_worker = get_cv_worker()
    return HealthStatus(
        status="ok",
        cv_worker_running=cv_worker.running,
    )


@router.get("/", response_class=HTMLResponse)
def index():
    """Главная: ввод URL, превью, рисование полигонов, сохранение карты, старт/стоп."""
    return """
    <html>
        <head>
            <title>Parking Occupancy</title>
            <style>
                body { margin: 0; background: #000; display: flex; flex-direction: column;
                       justify-content: flex-start; align-items: center; height: 100vh; color: #fff; font-family: sans-serif; padding-top: 20px; }
                #preview-container { position: relative; display: inline-block; }
                #previewCanvas { border: 1px solid #444; max-width: 100vw; max-height: 70vh; }
                .status { margin-top: 10px; }
                form { margin-bottom: 10px; }
                input[type="text"] { width: 400px; padding: 4px 8px; }
                #preview-img { display: none; }
                button { margin-right: 4px; }
            </style>
        </head>
        <body>
            <form method="post" action="/source-mp4">
                <input type="text" name="mp4_url" placeholder="MP4 / RTSP URL" required />
                <button type="submit">Задать источник</button>
            </form>
            <form method="post" action="/start">
                <button type="submit">Старт системы</button>
            </form>
            <form method="post" action="/stop">
                <button type="submit">Остановить обработку</button>
            </form>

            <div id="preview-container">
                <img id="preview-img" src="/preview" alt="Preview frame">
                <canvas id="previewCanvas"></canvas>
            </div>

            <div class="status">
                <button id="finishPolygonBtn">Завершить полигон</button>
                <button id="saveMapBtn">Сохранить векторную карту</button>
            </div>

            <div class="status">
                <a href="/live" target="_blank" style="color: #4af;">Открыть поток</a>
                &nbsp;|&nbsp;
                <a href="/spots" target="_blank" style="color: #4af;">/spots</a> (JSON занятости мест)
            </div>

            <script>
                const img = document.getElementById('preview-img');
                const canvas = document.getElementById('previewCanvas');
                const ctx = canvas.getContext('2d');
                const finishBtn = document.getElementById('finishPolygonBtn');
                const saveBtn = document.getElementById('saveMapBtn');

                let polygons = [];        // [[{x, y}, ...], ...]
                let currentPolygon = [];  // [{x, y}, ...]

                function redraw() {
                    if (!img.complete || img.naturalWidth === 0) {
                        return;
                    }
                    // Подгоняем размер canvas под размер изображения
                    canvas.width = img.naturalWidth;
                    canvas.height = img.naturalHeight;
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                    ctx.lineWidth = 2;

                    // Уже сохранённые полигоны
                    ctx.strokeStyle = 'rgba(0, 255, 0, 0.9)';
                    ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
                    polygons.forEach(poly => {
                        if (poly.length < 2) return;
                        ctx.beginPath();
                        ctx.moveTo(poly[0].x, poly[0].y);
                        for (let i = 1; i < poly.length; i++) {
                            ctx.lineTo(poly[i].x, poly[i].y);
                        }
                        ctx.closePath();
                        ctx.stroke();
                        ctx.fill();
                    });

                    // Текущий редактируемый полигон
                    ctx.strokeStyle = 'rgba(255, 255, 0, 0.9)';
                    ctx.fillStyle = 'rgba(255, 255, 0, 0.2)';
                    if (currentPolygon.length > 0) {
                        ctx.beginPath();
                        ctx.moveTo(currentPolygon[0].x, currentPolygon[0].y);
                        for (let i = 1; i < currentPolygon.length; i++) {
                            ctx.lineTo(currentPolygon[i].x, currentPolygon[i].y);
                        }
                        ctx.stroke();
                    }

                    // Точки
                    ctx.fillStyle = '#ff0';
                    [...polygons, currentPolygon].forEach(poly => {
                        poly.forEach(p => {
                            ctx.beginPath();
                            ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
                            ctx.fill();
                        });
                    });
                }

                img.onload = function () {
                    redraw();
                };

                canvas.addEventListener('click', function (e) {
                    const rect = canvas.getBoundingClientRect();
                    const scaleX = canvas.width / rect.width;
                    const scaleY = canvas.height / rect.height;
                    const x = (e.clientX - rect.left) * scaleX;
                    const y = (e.clientY - rect.top) * scaleY;

                    currentPolygon.push({ x, y });
                    redraw();
                });

                finishBtn.addEventListener('click', function () {
                    if (currentPolygon.length >= 3) {
                        polygons.push(currentPolygon);
                        currentPolygon = [];
                        redraw();
                    } else {
                        alert('Нужно минимум 3 точки для полигона');
                    }
                });

                saveBtn.addEventListener('click', async function () {
                    if (polygons.length === 0) {
                        alert('Нет ни одного полигона для сохранения');
                        return;
                    }
                    const spots = polygons.map((poly, idx) => ({
                        id: idx + 1,
                        polygon: poly.map(p => [p.x, p.y]),
                    }));
                    try {
                        const resp = await fetch('/vector-map', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ spots }),
                        });
                        if (!resp.ok) {
                            const text = await resp.text();
                            alert('Ошибка сохранения векторной карты: ' + text);
                        } else {
                            alert('Векторная карта сохранена');
                        }
                    } catch (e) {
                        alert('Ошибка сети при сохранении векторной карты: ' + e);
                    }
                });
            </script>
        </body>
    </html>
    """


@router.post("/source-mp4")
def set_mp4_source(mp4_url: str = Form(...)):
    """Установить источник видео (MP4/RTSP), обновить превью."""
    cv_worker = get_cv_worker()
    cv_worker.set_stream_url(mp4_url)
    cap = cv2.VideoCapture(mp4_url)
    ret, frame = cap.read()
    if ret:
        vm = get_vector_map()
        if vm is not None:
            for spot in vm.spots:
                if len(spot.polygon) >= 4:
                    pts = np.array(spot.polygon, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        success, buffer = cv2.imencode(".jpg", frame)
        if success:
            cv_worker.preview_frame_jpeg = buffer.tobytes()
    cap.release()
    return RedirectResponse(url="/", status_code=303)


@router.post("/stop")
def stop_processing():
    cv_worker = get_cv_worker()
    cv_worker.stop()
    return RedirectResponse(url="/", status_code=303)


@router.post("/start")
def start_processing():
    cv_worker = get_cv_worker()
    cv_worker.start()
    return RedirectResponse(url="/live", status_code=303)


@router.get("/preview")
def preview_frame():
    cv_worker = get_cv_worker()
    if not cv_worker.preview_frame_jpeg:
        return Response(status_code=204)
    return Response(content=cv_worker.preview_frame_jpeg, media_type="image/jpeg")


@router.get("/live", response_class=HTMLResponse)
def live_page():
    return """
    <html>
        <head>
            <title>Parking Occupancy - Live</title>
            <style>
                body { margin: 0; background: #000; display: flex; justify-content: center;
                       align-items: center; height: 100vh; }
                img { max-width: 100vw; max-height: 100vh; }
            </style>
        </head>
        <body>
            <img src="/video" alt="YOLO Stream">
        </body>
    </html>
    """


@router.get("/video")
def video_stream():
    """MJPEG-поток с векторной картой из CV worker."""
    cv_worker = get_cv_worker()

    def frame_generator():
        boundary = b"--frame\r\n"
        start_wait = time.time()
        while True:
            if not cv_worker.running:
                break
            frame = cv_worker.last_frame_jpeg
            if frame is None:
                if time.time() - start_wait > 5:
                    break
                time.sleep(0.1)
                continue
            yield (
                boundary
                + b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/vector-map", response_model=VectorMap)
def upload_vector_map(vm: VectorMap):
    return set_vector_map(vm)


@router.get("/vector-map", response_model=VectorMap)
def read_vector_map():
    vm = get_vector_map()
    if vm is None:
        return VectorMap(spots=[])
    return vm


@router.get("/spots", response_model=List[SpotStatus])
def get_spots():
    return get_spot_status_list()

