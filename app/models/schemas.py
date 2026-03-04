"""Pydantic-схемы для API и векторной карты."""
from typing import List

from pydantic import BaseModel


class Spot(BaseModel):
    """Парковочное место: id и полигон в координатах изображения [[x, y], ...]."""
    id: int
    polygon: List[List[float]]


class VectorMap(BaseModel):
    """Векторная карта: список парковочных мест."""
    spots: List[Spot]


class SpotStatus(BaseModel):
    """Статус места: занятость 0–100%."""
    id: int
    occupancy: float


class HealthStatus(BaseModel):
    """Статус сервиса и фонового CV worker."""
    status: str
    cv_worker_running: bool

