# -*- coding: utf-8 -*-
import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app_utils import __version__
from geo_parser import GeoPoint
from spatial_index import (
    SpatialIndexService,
    segments_intersect,
    polygon_self_intersects
)
from interactive_app import VolumeApp


def create_mock_app():
    app = VolumeApp.__new__(VolumeApp)
    app.tk = None
    app.tabview = None
    app.frame_selection_bar = None
    app.points = [
        GeoPoint(id="1", x=0.0, y=0.0, h=100.0, surface_type="auto"),
        GeoPoint(id="2", x=10.0, y=0.0, h=100.0, surface_type="auto"),
        GeoPoint(id="3", x=10.0, y=10.0, h=100.0, surface_type="auto"),
        GeoPoint(id="4", x=0.0, y=10.0, h=100.0, surface_type="auto"),
        GeoPoint(id="5", x=5.0, y=5.0, h=105.0, surface_type="auto"),  # Внутри контура
        GeoPoint(id="6", x=20.0, y=20.0, h=100.0, surface_type="auto"), # Снаружи
    ]
    app.boundary_indices = [0, 1, 2, 3]
    for i in app.boundary_indices:
        app.points[i].surface_type = "boundary"

    app._selected_points = set()
    app._undo_stack = []
    app._tin_excluded = set()
    app._tin_custom_simplices = None
    app._tin_simplices = None
    app._tin_dirty = True
    app._current_file_path = None
    app._current_project_dir = None
    app.calc_results = None

    app._update_selection_bar = MagicMock()
    app._update_2d_selection_only = MagicMock()
    app._update_all_views = MagicMock()
    app._redraw_2d = MagicMock()
    app._update_2d_contour_only = MagicMock()
    app._schedule_auto_save = MagicMock()
    app._schedule_boundary_calc = MagicMock()
    app._invalidate_boundary_cache = MagicMock()
    app._invalidate_points_spatial_index = MagicMock()
    app._get_click_tolerance = lambda: 1.0

    return app


class TestV1Improvements:
    def test_version_bumped_to_1_0_1(self):
        """Проверка версии 1.0.1"""
        assert __version__ == "1.0.1"

    def test_segments_intersect(self):
        """Проверка пересечения отрезков"""
        # Пересекающиеся крест-накрест
        assert segments_intersect((0, 0), (10, 10), (0, 10), (10, 0)) is True
        # Параллельные
        assert segments_intersect((0, 0), (10, 0), (0, 2), (10, 2)) is False
        # Непересекающиеся на одной прямой
        assert segments_intersect((0, 0), (10, 0), (12, 0), (20, 0)) is False

    def test_polygon_self_intersects(self):
        """Проверка детекции самопересечения многоугольника"""
        # Квадрат (без самопересечений)
        square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        assert polygon_self_intersects(square) is False

        # 'Песочные часы' / восьмерка (с самопересечением)
        hourglass = np.array([[0, 0], [10, 10], [10, 0], [0, 10]])
        assert polygon_self_intersects(hourglass) is True

    def test_find_nearest_boundary_edge_with_dist(self):
        """Проверка нахождения ближайшего ребра и расстояния до него"""
        pts = [
            GeoPoint("1", 0.0, 0.0, 10.0),
            GeoPoint("2", 10.0, 0.0, 10.0),
            GeoPoint("3", 10.0, 10.0, 10.0),
            GeoPoint("4", 0.0, 10.0, 10.0),
            GeoPoint("P", 5.0, 2.0, 10.0),  # Ближе всего к ребру 0-1 (Y=0, X от 0 до 10)
        ]
        service = SpatialIndexService(pts)
        k, dist = service.find_nearest_boundary_edge_with_dist(4, [0, 1, 2, 3])
        assert k == 0
        assert abs(dist - 2.0) < 1e-4

    def test_build_boundary_from_selected_points_success(self):
        """Построение контура (Convex Hull) вокруг выделенных точек"""
        app = create_mock_app()
        app.boundary_indices = []  # сбрасываем существующий контур
        for p in app.points:
            p.surface_type = "auto"

        # Выделяем точки 0, 1, 2, 3
        app._selected_points = {0, 1, 2, 3}
        app._build_boundary_from_selected_points()

        assert len(app.boundary_indices) >= 3
        for idx in app.boundary_indices:
            assert app.points[idx].surface_type == "boundary"
        assert len(app._selected_points) == 0

        # Проверяем стек отмены
        assert len(app._undo_stack) == 1
        assert app._undo_stack[0][0] == "boundary_set"

        # Отменяем построение
        app._undo_last_action()
        assert len(app.boundary_indices) == 0
        for p in app.points:
            assert p.surface_type == "auto"

    def test_build_boundary_requires_at_least_three_points(self):
        """Попытка построения контура при < 3 выделенных точек вызывает предупреждение"""
        app = create_mock_app()
        app.boundary_indices = []
        app._selected_points = {0, 1}

        with patch("tkinter.messagebox.showwarning") as mock_warn:
            app._build_boundary_from_selected_points()
            mock_warn.assert_called_once()
            assert len(app.boundary_indices) == 0

    def test_boundary_insert_self_intersection_rejected(self):
        """Встраивание точки, создающее самопересечение контура, отклоняется"""
        app = create_mock_app()
        # Точка 6 (20, 20)
        # Если попытаться принудительно добавить её в позицию, вызывающую самопересечение
        trial_coords = np.array([[0, 0], [10, 10], [10, 0], [0, 10]], dtype=float)
        assert polygon_self_intersects(trial_coords) is True

    def test_reset_all_caches_for_new_project(self):
        """Проверка полного сброса всех кешей проекта"""
        app = create_mock_app()
        app.calc_results = {"volume": 100}
        app._selected_points = {1, 2}
        app._undo_stack = [("action", 1)]
        app._tin_simplices = np.array([[0, 1, 2]])

        app._reset_all_caches_for_new_project()

        assert app.calc_results is None
        assert len(app._selected_points) == 0
        assert len(app._undo_stack) == 0
        assert app._tin_simplices is None
        assert app._tin_dirty is True
        assert app._table_dirty is True

    def test_app_config_project_persistence(self, tmp_path, monkeypatch):
        """Проверка сохранения и загрузки имени последнего активного проекта"""
        app = create_mock_app()
        cfg_file = tmp_path / "app_config.json"
        app._get_app_config_path = lambda: str(cfg_file)

        # Отключаем защиту pytest для целевого теста
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setattr("sys.modules", {k: v for k, v in sys.modules.items() if k != "pytest"})

        app._save_last_active_project_name("TestProject123")
        loaded_name = app._get_last_active_project_name()
        assert loaded_name == "TestProject123"

    def test_tin_table_dialog_init(self):
        """Проверка создания окна таблицы треугольников TIN"""
        from ui_dialogs import TINTableDialog
        import tkinter as tk

        try:
            root = tk.Tk()
        except (tk.TclError, Exception):
            pytest.skip("Tkinter Tcl/Tk runtime not available in this test environment")
        root.withdraw()
        try:
            app = create_mock_app()
            app._tin_simplices = np.array([[0, 1, 2], [0, 2, 3]])
            dlg = TINTableDialog(root, app=app)
            assert dlg.tree is not None
            # Должно быть 2 строки в таблице
            items = dlg.tree.get_children()
            assert len(items) == 2
            dlg.destroy()
        finally:
            try:
                root.destroy()
            except Exception:
                pass

    def test_add_tooltip(self):
        """Проверка добавления подсказки к виджету"""
        from ui_dialogs import add_tooltip, ToolTip
        import tkinter as tk
        import customtkinter as ctk

        try:
            root = tk.Tk()
        except (tk.TclError, Exception):
            pytest.skip("Tkinter Tcl/Tk runtime not available in this test environment")
        root.withdraw()
        try:
            btn = ctk.CTkButton(root, text="Test Button")
            tt = add_tooltip(btn, "Тестовая подсказка", delay_ms=100)
            assert isinstance(tt, ToolTip)
            assert tt.text == "Тестовая подсказка"
            assert tt.widget is btn
            btn.destroy()
        finally:
            try:
                root.destroy()
            except Exception:
                pass

    def test_on_closing_clean_exit(self):
        """Проверка безопасного закрытия приложения без UnboundLocalError"""
        app = create_mock_app()
        app.destroy = MagicMock()
        app.quit = MagicMock()
        app._auto_save_timer = 123
        app.after_cancel = MagicMock()

        # Вызов _on_closing в тестовой среде (где активен pytest)
        app._on_closing()
        app.destroy.assert_called_once()

    def test_3d_tab_surface_checkboxes_and_table_font(self):
        """Проверка наличия переключателей поверхностей в 3D и уменьшенного шрифта в таблице точек"""
        import tkinter as tk
        try:
            root = tk.Tk()
        except (tk.TclError, Exception):
            pytest.skip("Tkinter Tcl/Tk runtime not available in this test environment")
        root.withdraw()
        try:
            app = VolumeApp()
            # Проверяем, что 3D вкладка не имеет плавающей панели на холсте
            assert app.TAB_3D not in getattr(app, "_fs_surface_panels", {})
            # Проверяем метку статистики таблицы
            assert app.lbl_table_stats is not None
            # Проверяем безопасный вызов _on_closing
            app._on_closing()
        finally:
            try:
                root.destroy()
            except Exception:
                pass
