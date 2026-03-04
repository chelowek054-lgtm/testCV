"""Точка входа FastAPI: Parking Occupancy API."""
from fastapi import FastAPI

from app.api import router as api_router
from app.cv_engine import get_cv_worker


def create_app() -> FastAPI:
    app = FastAPI(title="Parking Occupancy API")
    app.include_router(api_router)

    @app.on_event("shutdown")
    def _shutdown():
        get_cv_worker().stop()

    return app


app = create_app()

