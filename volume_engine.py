# -*- coding: utf-8 -*-
"""
Модуль для расчета объемов между двумя поверхностями (TIN / Регулярная сетка),
сшитыми по точкам внешнего контура.
"""

import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import matplotlib.path as mpl_path
import matplotlib.tri as mtri
from typing import List, Tuple, Dict, Any, Optional

def polygon_area_2d(polygon: np.ndarray) -> float:
    """Вычисление 2D площади полигона по формуле Гаусса (Shoelace formula)."""
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

def calculate_tri_surface_area_3d(points: np.ndarray, simplices: np.ndarray) -> float:
    """Вычисление 3D площади триангулированной поверхности (векторизованный вариант)."""
    if len(simplices) == 0 or len(points) < 3:
        return 0.0
    v1 = points[simplices[:, 1]] - points[simplices[:, 0]]
    v2 = points[simplices[:, 2]] - points[simplices[:, 0]]
    return float(0.5 * np.sum(np.linalg.norm(np.cross(v1, v2), axis=1)))

def calculate_grid_surface_area_3d(z_grid: np.ndarray, res: float, inside_mask: np.ndarray, area_2d: float) -> float:
    """Вычисление 3D площади поверхности, заданной на регулярной сетке строго внутри полигонального контура."""
    if z_grid is None or len(z_grid) == 0 or area_2d <= 0:
        return float(area_2d)

    if inside_mask is not None and inside_mask.ndim == 1 and z_grid is not None:
        inside_mask = inside_mask.reshape(z_grid.shape)

    # Маска ячеек сетки, где все 4 вершины находятся внутри полигона контура
    quad_mask = (inside_mask[:-1, :-1] & inside_mask[:-1, 1:] &
                 inside_mask[1:, :-1] & inside_mask[1:, 1:])

    z00 = z_grid[:-1, :-1]
    z01 = z_grid[:-1, 1:]
    z10 = z_grid[1:, :-1]
    z11 = z_grid[1:, 1:]

    valid = (quad_mask & ~np.isnan(z00) & ~np.isnan(z01) &
             ~np.isnan(z10) & ~np.isnan(z11))

    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return float(area_2d)

    dz_x1 = z01[valid] - z00[valid]
    dz_y1 = z11[valid] - z01[valid]
    tri1_area = 0.5 * res * np.sqrt(dz_x1**2 + dz_y1**2 + res**2)

    dz_x2 = z11[valid] - z10[valid]
    dz_y2 = z10[valid] - z00[valid]
    tri2_area = 0.5 * res * np.sqrt(dz_x2**2 + dz_y2**2 + res**2)

    raw_3d_sum = float(np.sum(tri1_area + tri2_area))
    effective_2d = float(n_valid * (res ** 2))
    scale = (area_2d / effective_2d) if effective_2d > 0 else 1.0
    return max(float(area_2d), raw_3d_sum * scale)

def remove_duplicate_points_2d(pts: np.ndarray, tol: float = 0.001) -> np.ndarray:
    """Удаляет близкие дубликаты точек в плане (XY)."""
    if len(pts) == 0:
        return pts
    rounded = np.round(pts[:, :2] / tol) * tol
    _, u_indices = np.unique(rounded, axis=0, return_index=True)
    return pts[np.sort(u_indices)]

class VolumeCalculator:
    def __init__(self,
                 top_points: np.ndarray,
                 bottom_points: np.ndarray,
                 boundary_points: Optional[np.ndarray] = None,
                 grid_resolution: float = 0.2,
                 work_type: str = "auto",
                 stitching: bool = True):
        """
        top_points: Nx3 массив точек верхней поверхности (X, Y, H)
        bottom_points: Mx3 массив точек нижней поверхности (X, Y, H)
        boundary_points: Kx3 массив точек контура в порядке обхода (X, Y, H)
        grid_resolution: шаг расчетной сетки (в метрах)
        work_type: 'auto', 'cut' (выемка), 'fill' (насыпь), или 'grading' (две поверхности)
        stitching: True для сшивания рельефа по контуру (однофайловый режим),
                   False для двух независимых съемок/файлов (точки верха и низа строго изолированы).
        """
        self.stitching = stitching
        self.raw_top = np.asarray(top_points, dtype=np.float64) if len(top_points) > 0 else np.empty((0, 3))
        self.raw_bottom = np.asarray(bottom_points, dtype=np.float64) if len(bottom_points) > 0 else np.empty((0, 3))
        self.grid_resolution = max(0.01, float(grid_resolution))

        all_pts = []
        if len(self.raw_top) > 0:
            all_pts.append(self.raw_top)
        if len(self.raw_bottom) > 0:
            all_pts.append(self.raw_bottom)
        if boundary_points is not None and len(boundary_points) > 0:
            all_pts.append(np.asarray(boundary_points, dtype=np.float64))

        if not all_pts:
            raise ValueError("Нет точек для расчета объема")

        all_pts_arr = np.vstack(all_pts)

        # Локальный сдвиг координат для защиты от потери точности float
        self.x0 = float(np.min(all_pts_arr[:, 0]))
        self.y0 = float(np.min(all_pts_arr[:, 1]))

        # Определение внешнего контура
        if boundary_points is not None and len(boundary_points) >= 3:
            self.boundary = np.asarray(boundary_points, dtype=np.float64)
        else:
            # Автоматическая выпуклая оболочка
            hull = ConvexHull(all_pts_arr[:, :2] - [self.x0, self.y0])
            self.boundary = all_pts_arr[hull.vertices]

        mean_bound = float(np.mean(self.boundary[:, 2]))

        # Формирование поверхностей и определение типа земляных работ
        if not self.stitching:
            # Режим двух независимых поверхностей (из раздельных файлов Верх/Низ):
            # Точки верха и низа гарантированно и строго остаются в своих массивах без искажения контуром!
            mean_top = float(np.mean(self.raw_top[:, 2])) if len(self.raw_top) > 0 else 0.0
            mean_bot = float(np.mean(self.raw_bottom[:, 2])) if len(self.raw_bottom) > 0 else 0.0
            if work_type == "auto":
                self.work_type = "fill" if mean_top >= mean_bot else "cut"
            else:
                self.work_type = work_type

            self.top_surface_pts = self.raw_top.copy() if len(self.raw_top) > 0 else self.boundary.copy()
            self.bottom_surface_pts = self.raw_bottom.copy() if len(self.raw_bottom) > 0 else self.boundary.copy()

        else:
            # Режим со сшиванием по внешнему контуру (однофайловый режим)
            if work_type == "auto":
                if len(self.raw_bottom) > 0 and len(self.raw_top) == 0:
                    mean_bot = float(np.mean(self.raw_bottom[:, 2]))
                    work_type = "cut" if mean_bot < mean_bound - 1e-4 else "fill"
                elif len(self.raw_top) > 0 and len(self.raw_bottom) == 0:
                    mean_top = float(np.mean(self.raw_top[:, 2]))
                    work_type = "cut" if mean_top < mean_bound - 1e-4 else "fill"
                elif len(self.raw_top) > 0 and len(self.raw_bottom) > 0:
                    mean_top = float(np.mean(self.raw_top[:, 2]))
                    mean_bot = float(np.mean(self.raw_bottom[:, 2]))
                    if mean_top < mean_bound and mean_bot < mean_bound:
                        work_type = "cut"
                    elif mean_top > mean_bound and mean_bot > mean_bound:
                        work_type = "fill"
                    else:
                        work_type = "grading"
                else:
                    work_type = "fill"

            self.work_type = work_type

            # Формирование верхней и нижней поверхностей при сшивании по контуру:
            if self.work_type == "cut":
                if len(self.raw_top) > 0:
                    self.top_surface_pts = np.vstack([self.raw_top, self.boundary])
                else:
                    self.top_surface_pts = self.boundary.copy()

                if len(self.raw_bottom) > 0:
                    self.bottom_surface_pts = np.vstack([self.raw_bottom, self.boundary])
                elif len(self.raw_top) > 0 and float(np.mean(self.raw_top[:, 2])) < mean_bound - 1e-4:
                    self.bottom_surface_pts = np.vstack([self.raw_top, self.boundary])
                    self.top_surface_pts = self.boundary.copy()
                else:
                    self.bottom_surface_pts = self.boundary.copy()

            elif self.work_type == "fill":
                if len(self.raw_top) > 0:
                    self.top_surface_pts = np.vstack([self.raw_top, self.boundary])
                elif len(self.raw_bottom) > 0 and float(np.mean(self.raw_bottom[:, 2])) > mean_bound + 1e-4:
                    self.top_surface_pts = np.vstack([self.raw_bottom, self.boundary])
                    self.bottom_surface_pts = self.boundary.copy()
                else:
                    self.top_surface_pts = self.boundary.copy()

                if len(self.raw_bottom) > 0:
                    self.bottom_surface_pts = np.vstack([self.raw_bottom, self.boundary])
                else:
                    self.bottom_surface_pts = self.boundary.copy()

            else:
                self.top_surface_pts = np.vstack([self.raw_top, self.boundary]) if len(self.raw_top) > 0 else self.boundary.copy()
                self.bottom_surface_pts = np.vstack([self.raw_bottom, self.boundary]) if len(self.raw_bottom) > 0 else self.boundary.copy()

        self.top_surface_pts = remove_duplicate_points_2d(self.top_surface_pts)
        self.bottom_surface_pts = remove_duplicate_points_2d(self.bottom_surface_pts)
    # ── Вспомогательные методы (вынесены из дублирующегося кода) ──────────

    def _build_grid(self, poly_2d: np.ndarray):
        """Строит регулярную сетку внутри полигона контура.

        Returns:
            (grid_x, grid_y, inside_mask, inside_pts, res) или None при ошибке.
        """
        min_x, min_y = np.min(poly_2d, axis=0)
        max_x, max_y = np.max(poly_2d, axis=0)
        span_max = max(max_x - min_x, max_y - min_y)

        res = self.grid_resolution
        if span_max / res > 500:
            res = span_max / 500.0
        if span_max / res < 20:
            res = max(0.01, span_max / 50.0)

        gx = np.arange(min_x, max_x + res, res)
        gy = np.arange(min_y, max_y + res, res)
        if len(gx) < 2:
            gx = np.linspace(min_x, max_x, 20)
        if len(gy) < 2:
            gy = np.linspace(min_y, max_y, 20)

        grid_x, grid_y = np.meshgrid(gx, gy)
        grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))

        path = mpl_path.Path(poly_2d)
        inside_mask = path.contains_points(grid_pts)
        inside_pts = grid_pts[inside_mask]

        return grid_x, grid_y, inside_mask, inside_pts, res

    @staticmethod
    def _interpolate_surface(pts_2d: np.ndarray, pts_z: np.ndarray,
                              query_pts: np.ndarray) -> np.ndarray:
        """Интерполирует поверхность: LinearND с fallback на NearestND для NaN."""
        z = LinearNDInterpolator(pts_2d, pts_z)(query_pts)
        nan_mask = np.isnan(z)
        if np.any(nan_mask):
            z[nan_mask] = NearestNDInterpolator(pts_2d, pts_z)(query_pts[nan_mask])
        return z

    def _integrate_volume(self, dh: np.ndarray, effective_cell_area: float):
        """Интегрирует объём из разности высот.

        Returns:
            (v_fill, v_cut, v_net)
        """
        if self.work_type == "cut":
            v_cut = float(np.sum(np.maximum(dh, 0.0)) * effective_cell_area)
            return 0.0, v_cut, -v_cut
        elif self.work_type == "fill":
            v_fill = float(np.sum(np.maximum(dh, 0.0)) * effective_cell_area)
            return v_fill, 0.0, v_fill
        else:
            fill_m = dh > 0
            cut_m = dh < 0
            v_fill = float(np.sum(dh[fill_m]) * effective_cell_area)
            v_cut = float(np.sum(-dh[cut_m]) * effective_cell_area)
            return v_fill, v_cut, v_fill - v_cut

    @staticmethod
    def _fill_cartogram_grids(grid_shape, inside_mask, z_top, z_bot, dh):
        """Заполняет 2D-матрицы для визуализации картограммы."""
        z_top_full = np.full(grid_shape, np.nan)
        z_bot_full = np.full(grid_shape, np.nan)
        dh_full = np.full(grid_shape, np.nan)

        inside_indices = np.where(inside_mask)[0]
        rows = inside_indices // grid_shape[1]
        cols = inside_indices % grid_shape[1]

        z_top_full[rows, cols] = z_top
        z_bot_full[rows, cols] = z_bot
        dh_full[rows, cols] = dh
        return z_top_full, z_bot_full, dh_full

    # ── Основные методы расчёта ────────────────────────────────────────

    def calculate(self) -> Dict[str, Any]:
        """
        Вычисляет объем методом численного интегрирования по регулярной сетке и TIN.
        """
        poly_2d = self.boundary[:, :2]
        area_2d = polygon_area_2d(poly_2d)

        grid_x, grid_y, inside_mask, inside_pts, res = self._build_grid(poly_2d)

        if len(inside_pts) == 0:
            return {"error": "Внутри контура не найдено точек расчетной сетки"}

        top_local_2d = self.top_surface_pts[:, :2] - [self.x0, self.y0]
        bot_local_2d = self.bottom_surface_pts[:, :2] - [self.x0, self.y0]
        grid_local_inside = inside_pts - [self.x0, self.y0]

        # Интерполяция обеих поверхностей
        z_top = self._interpolate_surface(top_local_2d, self.top_surface_pts[:, 2], grid_local_inside)
        z_bot = self._interpolate_surface(bot_local_2d, self.bottom_surface_pts[:, 2], grid_local_inside)

        dh = z_top - z_bot
        effective_cell_area = area_2d / len(inside_pts)

        v_fill, v_cut, v_net = self._integrate_volume(dh, effective_cell_area)

        avg_thickness = float(np.mean(dh))
        max_thickness = float(np.max(dh)) if len(dh) > 0 else 0.0
        min_thickness = float(np.min(dh)) if len(dh) > 0 else 0.0

        z_top_full, z_bot_full, dh_full = self._fill_cartogram_grids(
            grid_x.shape, inside_mask, z_top, z_bot, dh)

        top_area_3d = calculate_grid_surface_area_3d(z_top_full, res, inside_mask, area_2d)
        bot_area_3d = calculate_grid_surface_area_3d(z_bot_full, res, inside_mask, area_2d)

        return {
            "v_net": v_net,
            "v_fill": v_fill,
            "v_cut": v_cut,
            "area_2d": area_2d,
            "top_area_3d": top_area_3d,
            "bot_area_3d": bot_area_3d,
            "avg_thickness": avg_thickness,
            "max_thickness": max_thickness,
            "min_thickness": min_thickness,
            "grid_resolution": res,
            "num_grid_points": len(inside_pts),
            "grid_x": grid_x,
            "grid_y": grid_y,
            "inside_mask": inside_mask.reshape(grid_x.shape),
            "z_top_grid": z_top_full,
            "z_bot_grid": z_bot_full,
            "dh_grid": dh_full,
            "top_surface_pts": self.top_surface_pts,
            "bottom_surface_pts": self.bottom_surface_pts,
            "boundary": self.boundary,
            "work_type": self.work_type,
        }

    def calculate_with_custom_tin(self,
                                   pts_2d: np.ndarray,
                                   simplices: np.ndarray,
                                   excluded_simplex_indices: set = None,
                                   pts_3d: np.ndarray = None,
                                   tin_surface: str = "auto") -> Dict[str, Any]:
        """
        Вычисляет объём методом TIN-интеграции и строит регулярную сетку для картограммы
        и 3D визуализации с точным учётом пользовательской триангуляции (в т.ч. переброшенных рёбер).

        pts_2d: Nx2 массив XY-координат всех точек
        simplices: Mx3 массив индексов вершин треугольников
        excluded_simplex_indices: набор индексов треугольников, которые исключаются из расчёта
        pts_3d: Nx3 массив XYZ-координат всех точек
        tin_surface: 'auto', 'top' (насыпь), 'bottom' (выемка)
        """
        if excluded_simplex_indices is None:
            excluded_simplex_indices = set()

        if pts_3d is None:
            if pts_2d.shape[1] >= 3:
                pts_3d = pts_2d
                pts_2d = pts_3d[:, :2]
            else:
                pts_3d = self.top_surface_pts

        if tin_surface == "auto":
            tin_surface = "bottom" if self.work_type == "cut" else "top"

        poly_2d = self.boundary[:, :2]
        area_2d = polygon_area_2d(poly_2d)
        path = mpl_path.Path(poly_2d)

        # Отбираем активные треугольники внутри контура (векторизованный вариант)
        n_simplices = len(simplices)
        if n_simplices == 0:
            return {"error": "Нет активных треугольников внутри контура для расчета"}

        # Маска исключённых треугольников
        excluded_mask = np.array([i in excluded_simplex_indices for i in range(n_simplices)], dtype=bool)

        # Центроиды всех треугольников
        centroids = np.mean(pts_2d[simplices], axis=1)  # (M, 2)
        inside_centroids = path.contains_points(centroids)

        # Площади треугольников (фильтрация вырожденных)
        p0 = pts_2d[simplices[:, 0]]
        p1 = pts_2d[simplices[:, 1]]
        p2 = pts_2d[simplices[:, 2]]
        areas = 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) -
                              (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))

        active_mask = ~excluded_mask & inside_centroids & (areas >= 1e-10)
        active_simplices = simplices[active_mask]

        active_simplices = np.asarray(active_simplices, dtype=int)

        if len(active_simplices) == 0:
            return {"error": "Нет активных треугольников внутри контура для расчета"}

        grid_x, grid_y, inside_mask, inside_pts, res = self._build_grid(poly_2d)

        if len(inside_pts) == 0:
            return {"error": "Внутри контура не найдено точек расчетной сетки"}

        grid_local_inside = inside_pts - [self.x0, self.y0]

        if tin_surface == "bottom":
            # Выемка: пользовательский TIN описывает НИЖНЮЮ поверхность (дно и откосы котлована)
            try:
                triang_bot = mtri.Triangulation(pts_2d[:, 0], pts_2d[:, 1], active_simplices)
                interp_bot = mtri.LinearTriInterpolator(triang_bot, pts_3d[:, 2])
                z_bot = interp_bot(inside_pts[:, 0], inside_pts[:, 1])
                if hasattr(z_bot, "filled"):
                    z_bot = z_bot.filled(np.nan)
            except Exception:
                z_bot = LinearNDInterpolator(pts_2d[np.unique(active_simplices)],
                                             pts_3d[np.unique(active_simplices), 2])(inside_pts)
                if np.any(np.isnan(z_bot)):
                    z_bot_near = NearestNDInterpolator(pts_2d[np.unique(active_simplices)],
                                                       pts_3d[np.unique(active_simplices), 2])(inside_pts)
                    z_bot = np.where(np.isnan(z_bot), z_bot_near, z_bot)

            nan_bot = np.isnan(z_bot)
            if np.any(nan_bot) and len(active_simplices) > 0:
                is_in_excluded = np.zeros(len(inside_pts), dtype=bool)
                if excluded_simplex_indices:
                    valid_ex = [idx for idx in excluded_simplex_indices if 0 <= idx < len(simplices)]
                    for ex_idx in valid_ex:
                        ex_path = mpl_path.Path(pts_2d[simplices[ex_idx]])
                        is_in_excluded |= ex_path.contains_points(inside_pts)
                fill_mask = nan_bot & (~is_in_excluded)
                if np.any(fill_mask):
                    active_u = np.unique(active_simplices)
                    near_bot = NearestNDInterpolator(pts_2d[active_u], pts_3d[active_u, 2])
                    z_bot[fill_mask] = near_bot(inside_pts[fill_mask, 0], inside_pts[fill_mask, 1])

            # Верхняя поверхность интерполируется по дневной поверхности (self.top_surface_pts)
            top_local_2d = self.top_surface_pts[:, :2] - [self.x0, self.y0]
            z_top = self._interpolate_surface(top_local_2d, self.top_surface_pts[:, 2], grid_local_inside)

            bot_area_3d = max(float(area_2d), calculate_tri_surface_area_3d(pts_3d, active_simplices))

        else:
            # Насыпь: пользовательский TIN описывает ВЕРХНЮЮ поверхность (гребень и откосы насыпи)
            try:
                triang_top = mtri.Triangulation(pts_2d[:, 0], pts_2d[:, 1], active_simplices)
                interp_top = mtri.LinearTriInterpolator(triang_top, pts_3d[:, 2])
                z_top = interp_top(inside_pts[:, 0], inside_pts[:, 1])
                if hasattr(z_top, "filled"):
                    z_top = z_top.filled(np.nan)
            except Exception:
                z_top = LinearNDInterpolator(pts_2d[np.unique(active_simplices)],
                                             pts_3d[np.unique(active_simplices), 2])(inside_pts)
                if np.any(np.isnan(z_top)):
                    z_top_near = NearestNDInterpolator(pts_2d[np.unique(active_simplices)],
                                                       pts_3d[np.unique(active_simplices), 2])(inside_pts)
                    z_top = np.where(np.isnan(z_top), z_top_near, z_top)

            nan_top = np.isnan(z_top)
            if np.any(nan_top) and len(active_simplices) > 0:
                is_in_excluded = np.zeros(len(inside_pts), dtype=bool)
                if excluded_simplex_indices:
                    valid_ex = [idx for idx in excluded_simplex_indices if 0 <= idx < len(simplices)]
                    for ex_idx in valid_ex:
                        ex_path = mpl_path.Path(pts_2d[simplices[ex_idx]])
                        is_in_excluded |= ex_path.contains_points(inside_pts)
                fill_mask = nan_top & (~is_in_excluded)
                if np.any(fill_mask):
                    active_u = np.unique(active_simplices)
                    near_top = NearestNDInterpolator(pts_2d[active_u], pts_3d[active_u, 2])
                    z_top[fill_mask] = near_top(inside_pts[fill_mask, 0], inside_pts[fill_mask, 1])

            # Нижняя поверхность интерполируется по основанию (self.bottom_surface_pts)
            bot_local_2d = self.bottom_surface_pts[:, :2] - [self.x0, self.y0]
            z_bot = self._interpolate_surface(bot_local_2d, self.bottom_surface_pts[:, 2], grid_local_inside)

            top_area_3d = max(float(area_2d), calculate_tri_surface_area_3d(pts_3d, active_simplices))

        # Разность высот мощности слоя ΔH
        dh = z_top - z_bot
        valid_mask = ~np.isnan(dh)

        if not np.any(valid_mask):
            return {"error": "Не удалось вычислить мощность слоя в точках сетки"}

        effective_cell_area = area_2d / len(inside_pts)
        dh_valid = dh[valid_mask]

        v_fill, v_cut, v_net = self._integrate_volume(dh_valid, effective_cell_area)

        avg_thickness = float(np.mean(dh_valid))
        max_thickness = float(np.max(dh_valid)) if len(dh_valid) > 0 else 0.0
        min_thickness = float(np.min(dh_valid)) if len(dh_valid) > 0 else 0.0

        z_top_full, z_bot_full, dh_full = self._fill_cartogram_grids(
            grid_x.shape, inside_mask, z_top, z_bot, dh)

        if tin_surface == "bottom":
            top_area_3d = calculate_grid_surface_area_3d(z_top_full, res, inside_mask, area_2d)
        else:
            bot_area_3d = calculate_grid_surface_area_3d(z_bot_full, res, inside_mask, area_2d)

        return {
            "v_net": v_net,
            "v_fill": v_fill,
            "v_cut": v_cut,
            "area_2d": area_2d,
            "top_area_3d": top_area_3d,
            "bot_area_3d": bot_area_3d,
            "avg_thickness": avg_thickness,
            "max_thickness": max_thickness,
            "min_thickness": min_thickness,
            "grid_resolution": res,
            "num_grid_points": int(np.sum(valid_mask)),
            "grid_x": grid_x,
            "grid_y": grid_y,
            "inside_mask": inside_mask.reshape(grid_x.shape),
            "z_top_grid": z_top_full,
            "z_bot_grid": z_bot_full,
            "dh_grid": dh_full,
            "top_surface_pts": self.top_surface_pts,
            "bottom_surface_pts": self.bottom_surface_pts,
            "boundary": self.boundary,
            "num_triangles": len(active_simplices),
            "active_simplices": active_simplices,
            "pts_3d": pts_3d,
            "custom_tin": True,
            "work_type": self.work_type,
            "tin_surface": tin_surface,
        }
