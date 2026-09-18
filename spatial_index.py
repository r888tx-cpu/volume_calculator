# -*- coding: utf-8 -*-
"""
Модуль пространственной индексации и геометрических расчетов.
Использует scipy.spatial.cKDTree для сверхбыстрого O(log N) поиска точек и NumPy для векторизации.
"""
from typing import Any, List, Optional, Tuple
import numpy as np


class SpatialIndexService:
    """Сервис для быстрого геометрического поиска и индексации точек."""

    def __init__(self, points: Optional[List[Any]] = None):
        self._points = points or []
        self._kdtree = None
        self._cached_len = 0
        self._coords = None
        self._coords_len = 0

    def set_points(self, points: List[Any]) -> None:
        """Устанавливает новый список точек и сбрасывает пространственный индекс."""
        self._points = points
        self.invalidate()

    def invalidate(self) -> None:
        """Инвалидирует кеш cKDTree и координат."""
        self._kdtree = None
        self._cached_len = 0
        self._coords = None
        self._coords_len = 0

    def _ensure_coords(self) -> np.ndarray:
        """Возвращает кешированный массив координат (N, 2) в формате [Y_восток, X_север]."""
        pts_len = len(self._points)
        if self._coords is None or self._coords_len != pts_len:
            if not self._points:
                self._coords = np.empty((0, 2), dtype=np.float64)
            else:
                self._coords = np.array([[p.y, p.x] for p in self._points], dtype=np.float64)
            self._coords_len = pts_len
        return self._coords

    def get_points_in_bbox(self, y_min: float, y_max: float, x_min: float, x_max: float) -> np.ndarray:
        """
        Быстро возвращает 1D-массив целочисленных индексов точек внутри прямоугольника [y_min, y_max] x [x_min, x_max].
        Работает через векторные булевы маски NumPy за доли миллисекунды.
        """
        if not self._points:
            return np.empty(0, dtype=int)
        coords = self._ensure_coords()
        ys = coords[:, 0]
        xs = coords[:, 1]
        mask = (ys >= y_min) & (ys <= y_max) & (xs >= x_min) & (xs <= x_max)
        return np.flatnonzero(mask)

    def get_kdtree(self):
        """Возвращает актуальный экземпляр cKDTree."""
        if not self._points:
            return None
        pts_len = len(self._points)
        if self._kdtree is None or self._cached_len != pts_len:
            try:
                from scipy.spatial import cKDTree
                coords = self._ensure_coords()
                self._kdtree = cKDTree(coords)
                self._cached_len = pts_len
            except Exception:
                self._kdtree = None
        return self._kdtree

    def find_nearest_point(self, click_y: float, click_x: float) -> Tuple[int, float]:
        """
        Ищет индекс ближайшей точки и евклидово расстояние до неё.
        Возвращает (min_idx, min_dist). Если точек нет, возвращает (-1, inf).
        """
        if not self._points:
            return -1, float("inf")

        tree = self.get_kdtree()
        if tree is not None:
            try:
                dist, idx = tree.query([click_y, click_x])
                return int(idx), float(dist)
            except Exception:
                pass

        # Fallback на цикл в случае непредвиденных сбоев
        min_dist = float("inf")
        min_idx = -1
        for i, p in enumerate(self._points):
            d = (p.y - click_y) ** 2 + (p.x - click_x) ** 2
            if d < min_dist:
                min_dist = d
                min_idx = i
        return min_idx, float(np.sqrt(min_dist))

    @staticmethod
    def get_click_tolerance(xlim: Tuple[float, float], ylim: Tuple[float, float], factor: float = 0.035) -> float:
        """Вычисляет допуск клика на основе текущих границ видимой области осей."""
        span_x = abs(xlim[1] - xlim[0])
        span_y = abs(ylim[1] - ylim[0])
        return max(span_x, span_y) * factor

    def find_nearest_boundary_edge(self, pt_idx: int, boundary_indices: List[int]) -> int:
        """
        Находит индекс ребра контура k, к которому точка pt_idx ближе всего.
        Ребро k соединяет boundary_indices[k] и boundary_indices[(k + 1) % N].
        Возвращает k (от 0 до N - 1).
        """
        N = len(boundary_indices)
        if N < 2:
            return max(0, N - 1)
        if pt_idx < 0 or pt_idx >= len(self._points):
            return 0

        p_target = self._points[pt_idx]
        P = np.array([p_target.y, p_target.x], dtype=float)

        try:
            idx_a = boundary_indices
            idx_b = boundary_indices[1:] + boundary_indices[:1]
            A = np.array([[self._points[i].y, self._points[i].x] for i in idx_a], dtype=float)
            B = np.array([[self._points[i].y, self._points[i].x] for i in idx_b], dtype=float)
            AB = B - A
            AP = P - A
            L2 = np.sum(AB ** 2, axis=1)
            zero_mask = L2 < 1e-12
            L2_safe = np.where(zero_mask, 1.0, L2)
            t = np.clip(np.sum(AP * AB, axis=1) / L2_safe, 0.0, 1.0)
            t[zero_mask] = 0.0
            proj = A + t[:, np.newaxis] * AB
            dists = np.linalg.norm(P - proj, axis=1)
            return int(np.argmin(dists))
        except Exception:
            best_dist = float("inf")
            best_k = 0
            for k in range(N):
                idx_a = boundary_indices[k]
                idx_b = boundary_indices[(k + 1) % N]
                pa = self._points[idx_a]
                pb = self._points[idx_b]
                A = np.array([pa.y, pa.x], dtype=float)
                B = np.array([pb.y, pb.x], dtype=float)
                AB = B - A
                L2 = float(np.dot(AB, AB))
                if L2 < 1e-12:
                    t = 0.0
                else:
                    t = float(np.clip(np.dot(P - A, AB) / L2, 0.0, 1.0))
                proj = A + t * AB
                dist = float(np.linalg.norm(P - proj))
                if dist < best_dist:
                    best_dist = dist
                    best_k = k
            return best_k
