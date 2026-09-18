# -*- coding: utf-8 -*-
"""
Тесты для вычислительного ядра volume_engine.py.
Фиксируют текущее поведение перед оптимизацией.
"""
import sys
import os
import pytest
import numpy as np

# Обеспечиваем импорт из корня проекта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from volume_engine import VolumeCalculator, polygon_area_2d, calculate_tri_surface_area_3d, remove_duplicate_points_2d


# ═══════════════════════════════════════════════════════════════════
# Утилитарные функции
# ═══════════════════════════════════════════════════════════════════

class TestPolygonArea2d:
    """Тесты для polygon_area_2d (формула Гаусса / Shoelace)."""

    def test_unit_square(self):
        """Единичный квадрат: площадь = 1."""
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        assert np.isclose(polygon_area_2d(square), 1.0, atol=1e-10)

    def test_rectangle_10x5(self):
        """Прямоугольник 10x5: площадь = 50."""
        rect = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=float)
        assert np.isclose(polygon_area_2d(rect), 50.0, atol=1e-10)

    def test_triangle(self):
        """Треугольник: площадь = 0.5 * base * height."""
        tri = np.array([[0, 0], [4, 0], [0, 3]], dtype=float)
        assert np.isclose(polygon_area_2d(tri), 6.0, atol=1e-10)

    def test_reverse_winding(self):
        """Обратный порядок обхода не влияет на абсолютное значение площади."""
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        square_rev = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=float)
        assert np.isclose(polygon_area_2d(square), polygon_area_2d(square_rev), atol=1e-10)

    def test_degenerate_line(self):
        """Вырожденный полигон (линия): площадь = 0."""
        line = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        assert np.isclose(polygon_area_2d(line), 0.0, atol=1e-10)

    def test_with_3d_points_xy_only(self):
        """Работает с Nx3 массивом, используя только первые 2 колонки."""
        square_3d = np.array([[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]], dtype=float)
        assert np.isclose(polygon_area_2d(square_3d[:, :2]), 100.0, atol=1e-10)


class TestTriSurfaceArea3d:
    """Тесты для calculate_tri_surface_area_3d."""

    def test_flat_triangle(self):
        """Один плоский треугольник: площадь = 0.5 * 1 * 1 = 0.5."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        simplices = np.array([[0, 1, 2]])
        assert np.isclose(calculate_tri_surface_area_3d(points, simplices), 0.5, atol=1e-10)

    def test_two_triangles_square(self):
        """Два треугольника = квадрат 1x1: площадь = 1."""
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        simplices = np.array([[0, 1, 2], [0, 2, 3]])
        assert np.isclose(calculate_tri_surface_area_3d(points, simplices), 1.0, atol=1e-10)

    def test_tilted_triangle(self):
        """Наклонный треугольник в 3D: площадь больше проекции."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=float)
        simplices = np.array([[0, 1, 2]])
        # Площадь проекции = 0.5, 3D площадь должна быть больше
        area = calculate_tri_surface_area_3d(points, simplices)
        assert area > 0.5

    def test_empty_simplices(self):
        """Пустой массив треугольников → площадь = 0."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        simplices = np.empty((0, 3), dtype=int)
        assert calculate_tri_surface_area_3d(points, simplices) == 0.0

    def test_too_few_points(self):
        """Менее 3 точек → площадь = 0."""
        points = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        simplices = np.array([[0, 1, 0]])
        assert calculate_tri_surface_area_3d(points, simplices) == 0.0


class TestRemoveDuplicatePoints2d:
    """Тесты для remove_duplicate_points_2d."""

    def test_no_duplicates(self):
        """Без дубликатов: массив не меняется."""
        pts = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 3]], dtype=float)
        result = remove_duplicate_points_2d(pts)
        assert len(result) == 3

    def test_exact_duplicates(self):
        """Точные дубликаты по XY удаляются."""
        pts = np.array([[0, 0, 1], [0, 0, 2], [1, 1, 3]], dtype=float)
        result = remove_duplicate_points_2d(pts)
        assert len(result) == 2

    def test_near_duplicates_within_tol(self):
        """Близкие точки (< tol) удаляются."""
        pts = np.array([[0, 0, 1], [0.0005, 0.0005, 2], [1, 1, 3]], dtype=float)
        result = remove_duplicate_points_2d(pts, tol=0.001)
        assert len(result) == 2

    def test_empty_array(self):
        """Пустой массив → возвращает пустой."""
        pts = np.empty((0, 3), dtype=float)
        result = remove_duplicate_points_2d(pts)
        assert len(result) == 0


# ═══════════════════════════════════════════════════════════════════
# Класс VolumeCalculator — создание и инициализация
# ═══════════════════════════════════════════════════════════════════

class TestVolumeCalculatorInit:
    """Тесты инициализации VolumeCalculator."""

    def test_raises_on_empty_points(self):
        """Пустые массивы точек → ValueError."""
        with pytest.raises(ValueError, match="Нет точек"):
            VolumeCalculator(
                top_points=np.empty((0, 3)),
                bottom_points=np.empty((0, 3)),
            )

    def test_auto_boundary_from_convex_hull(self):
        """Без boundary_points строится ConvexHull."""
        top = np.array([[0, 0, 10], [10, 0, 10], [10, 10, 10], [0, 10, 10]], dtype=float)
        bot = np.array([[5, 5, 5]], dtype=float)
        calc = VolumeCalculator(top_points=top, bottom_points=bot)
        assert calc.boundary is not None
        assert len(calc.boundary) >= 3

    def test_explicit_boundary(self):
        """Явный контур используется как есть."""
        boundary = np.array([[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]], dtype=float)
        top = np.array([[5, 5, 8]], dtype=float)
        bot = np.array([[5, 5, 5]], dtype=float)
        calc = VolumeCalculator(top_points=top, bottom_points=bot, boundary_points=boundary)
        assert len(calc.boundary) == 4

    def test_coordinate_offset_applied(self):
        """Координатный сдвиг x0, y0 вычисляется корректно."""
        boundary = np.array([
            [439000, 2281300, 198],
            [439010, 2281300, 198],
            [439010, 2281310, 198],
            [439000, 2281310, 198],
        ], dtype=float)
        top = np.array([[439005, 2281305, 201]], dtype=float)
        bot = np.array([[439005, 2281305, 198]], dtype=float)
        calc = VolumeCalculator(top_points=top, bottom_points=bot, boundary_points=boundary)
        assert calc.x0 == 439000.0
        assert calc.y0 == 2281300.0


# ═══════════════════════════════════════════════════════════════════
# Класс VolumeCalculator — расчёт объёмов
# ═══════════════════════════════════════════════════════════════════

class TestVolumeCalculatorCalculate:
    """Тесты метода VolumeCalculator.calculate()."""

    @pytest.fixture
    def square_boundary(self):
        """10x10 квадратный контур на высоте 198.0."""
        return np.array([
            [439000.0, 2281300.0, 198.0],
            [439010.0, 2281300.0, 198.0],
            [439010.0, 2281310.0, 198.0],
            [439000.0, 2281310.0, 198.0],
        ])

    def test_pyramid_volume(self, square_boundary):
        """
        Тест пирамиды (из оригинального test_calc.py).
        Основание 10x10, вершина в центре на +3м.
        Ожидаемый объём = 1/3 * 100 * 3 = 100 м³.
        Допуск 5% из-за дискретизации.
        """
        top = np.array([[439005.0, 2281305.0, 201.0]])
        bot = np.array([[439005.0, 2281305.0, 198.0]])
        calc = VolumeCalculator(
            top_points=top,
            bottom_points=bot,
            boundary_points=square_boundary,
            grid_resolution=0.05,
        )
        res = calc.calculate()

        assert "error" not in res
        # Объём пирамиды ≈ 100 м³ с допуском дискретизации
        assert np.isclose(res["v_fill"], 100.0, rtol=0.05), \
            f"Ожидали ~100, получили {res['v_fill']:.3f}"

    def test_flat_surfaces_zero_volume(self, square_boundary):
        """
        Две плоские поверхности на одной высоте → объём = 0.
        """
        top = np.array([[439005.0, 2281305.0, 198.0]])
        bot = np.array([[439005.0, 2281305.0, 198.0]])
        calc = VolumeCalculator(
            top_points=top,
            bottom_points=bot,
            boundary_points=square_boundary,
            grid_resolution=0.2,
        )
        res = calc.calculate()

        assert "error" not in res
        assert np.isclose(res["v_net"], 0.0, atol=0.1), \
            f"Ожидали ~0, получили {res['v_net']:.3f}"

    def test_flat_slab_volume(self, square_boundary):
        """
        Две плоские поверхности с ΔH = 2м на площади 100 м².
        Ожидаемый объём = 200 м³.
        Используем несколько точек верхней поверхности для формирования плоскости.
        """
        top = np.array([
            [439000.0, 2281300.0, 200.0],
            [439010.0, 2281300.0, 200.0],
            [439010.0, 2281310.0, 200.0],
            [439000.0, 2281310.0, 200.0],
            [439005.0, 2281305.0, 200.0],
        ])
        bot = np.array([
            [439000.0, 2281300.0, 198.0],
            [439010.0, 2281300.0, 198.0],
            [439010.0, 2281310.0, 198.0],
            [439000.0, 2281310.0, 198.0],
            [439005.0, 2281305.0, 198.0],
        ])
        calc = VolumeCalculator(
            top_points=top,
            bottom_points=bot,
            boundary_points=square_boundary,
            grid_resolution=0.2,
        )
        res = calc.calculate()

        assert "error" not in res
        assert np.isclose(res["v_fill"], 200.0, rtol=0.05), \
            f"Ожидали ~200, получили {res['v_fill']:.3f}"

    def test_triangular_boundary(self):
        """Треугольный контур: площадь = 0.5 * 10 * 10 = 50 м²."""
        boundary = np.array([
            [439000.0, 2281300.0, 198.0],
            [439010.0, 2281300.0, 198.0],
            [439000.0, 2281310.0, 198.0],
        ])
        top = np.array([[439003.0, 2281303.0, 200.0]])
        bot = np.array([[439003.0, 2281303.0, 198.0]])
        calc = VolumeCalculator(
            top_points=top,
            bottom_points=bot,
            boundary_points=boundary,
            grid_resolution=0.2,
        )
        res = calc.calculate()

        assert "error" not in res
        assert np.isclose(res["area_2d"], 50.0, rtol=0.05), \
            f"Ожидали ~50, получили {res['area_2d']:.3f}"

    def test_result_keys_present(self, square_boundary):
        """Результат содержит все ожидаемые ключи."""
        top = np.array([[439005.0, 2281305.0, 201.0]])
        bot = np.array([[439005.0, 2281305.0, 198.0]])
        calc = VolumeCalculator(
            top_points=top, bottom_points=bot,
            boundary_points=square_boundary, grid_resolution=0.5,
        )
        res = calc.calculate()

        expected_keys = {
            "v_net", "v_fill", "v_cut", "area_2d",
            "top_area_3d", "bot_area_3d",
            "avg_thickness", "max_thickness", "min_thickness",
            "grid_resolution", "num_grid_points",
            "grid_x", "grid_y", "inside_mask",
            "z_top_grid", "z_bot_grid", "dh_grid",
            "top_surface_pts", "bottom_surface_pts",
            "boundary", "work_type",
        }
        assert expected_keys.issubset(res.keys()), \
            f"Отсутствуют ключи: {expected_keys - res.keys()}"

    def test_grid_resolution_auto_clamp(self, square_boundary):
        """Слишком мелкая сетка автоматически укрупняется."""
        top = np.array([[439005.0, 2281305.0, 200.0]])
        bot = np.array([[439005.0, 2281305.0, 198.0]])
        calc = VolumeCalculator(
            top_points=top, bottom_points=bot,
            boundary_points=square_boundary,
            grid_resolution=0.001,  # слишком мелко для 10м объекта
        )
        res = calc.calculate()
        # Должно быть не более 500 шагов по максимальному спану
        assert "error" not in res
        assert res["grid_resolution"] >= 0.001

    def test_work_type_fill_detected(self, square_boundary):
        """Автоматическое определение типа 'fill' для насыпи."""
        top = np.array([[439005.0, 2281305.0, 202.0]])
        bot = np.empty((0, 3))
        calc = VolumeCalculator(
            top_points=top,
            bottom_points=bot,
            boundary_points=square_boundary,
            work_type="auto",
        )
        assert calc.work_type == "fill"

    def test_work_type_cut_detected(self, square_boundary):
        """Автоматическое определение типа 'cut' для выемки."""
        top = np.empty((0, 3))
        bot = np.array([[439005.0, 2281305.0, 195.0]])
        calc = VolumeCalculator(
            top_points=top,
            bottom_points=bot,
            boundary_points=square_boundary,
            work_type="auto",
        )
        assert calc.work_type == "cut"

    def test_explicit_work_type(self, square_boundary):
        """Явно заданный work_type не переопределяется."""
        top = np.array([[439005.0, 2281305.0, 202.0]])
        bot = np.array([[439005.0, 2281305.0, 198.0]])
        calc = VolumeCalculator(
            top_points=top,
            bottom_points=bot,
            boundary_points=square_boundary,
            work_type="grading",
        )
        assert calc.work_type == "grading"

    def test_cut_surface_areas_invariants_with_external_points(self, square_boundary):
        """
        Проверка физического инварианта для выемки:
        Площадь дна котлована (3D) строго больше площади верха (3D),
        а площадь верха ограничена контуром (не раздувается точками за пределами контура).
        """
        # Дно котлована с уклоном к центру
        bot = np.array([
            [439005.0, 2281305.0, 195.0],  # глубина 3м по центру
        ])
        # Точки дневной поверхности далеко за пределами контура (как при реальной съемке)
        far_top = np.array([
            [438900.0, 2281200.0, 198.0],
            [439100.0, 2281200.0, 198.0],
            [439100.0, 2281400.0, 198.0],
            [438900.0, 2281400.0, 198.0],
        ])
        calc = VolumeCalculator(
            top_points=far_top,
            bottom_points=bot,
            boundary_points=square_boundary,  # 10x10 = 100 м²
            grid_resolution=0.2,
            work_type="cut",
        )
        res = calc.calculate()
        assert res["area_2d"] == pytest.approx(100.0, rel=1e-3)
        # Верх плоский (198.0 м), площадь 3D должна быть ~100 м², а не 40000 м² внешней съемки!
        assert res["top_area_3d"] == pytest.approx(100.0, abs=0.5)
        # Низ — котлован со склонами, площадь 3D должна быть строго больше верха!
        assert res["bot_area_3d"] > res["top_area_3d"]
        assert res["bot_area_3d"] > res["area_2d"]

    def test_custom_tin_cut_surface_areas_invariants(self, square_boundary):
        """Проверка инвариантов площадей при выемке с пользовательским TIN."""
        from scipy.spatial import Delaunay
        bot = np.array([
            [439005.0, 2281305.0, 195.0],
        ])
        bot_all = np.vstack([square_boundary, bot])
        tri = Delaunay(bot_all[:, :2])

        far_top = np.array([
            [438900.0, 2281200.0, 198.0],
            [439100.0, 2281200.0, 198.0],
            [439100.0, 2281400.0, 198.0],
            [438900.0, 2281400.0, 198.0],
        ])
        calc = VolumeCalculator(
            top_points=far_top,
            bottom_points=bot,
            boundary_points=square_boundary,
            grid_resolution=0.2,
            work_type="cut",
        )
        res = calc.calculate_with_custom_tin(
            pts_2d=bot_all[:, :2],
            simplices=tri.simplices,
            pts_3d=bot_all,
            tin_surface="bottom",
        )
        assert res["area_2d"] == pytest.approx(100.0, rel=1e-3)
        assert res["top_area_3d"] == pytest.approx(100.0, abs=0.5)
        assert res["bot_area_3d"] > res["top_area_3d"]
        assert res["bot_area_3d"] > res["area_2d"]

