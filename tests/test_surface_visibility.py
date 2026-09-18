# -*- coding: utf-8 -*-
"""
Тесты видимости точек на 2D-схеме при переключении поверхностей «▲ Верхняя» / «▼ Нижняя».
Проверяет, что при выборе «только нижняя» точки за пределами контура гарантированно скрываются.
"""
import sys
import os
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
from unittest.mock import MagicMock
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geo_parser import GeoPoint
import interactive_app


def create_fill_app():
    """Создает приложение с насыпью (Fill) и точками внутри и снаружи контура."""
    app = interactive_app.VolumeApp.__new__(interactive_app.VolumeApp)
    app.points = [
        GeoPoint(id="б1", x=0.0, y=0.0, h=10.0, surface_type="auto"),
        GeoPoint(id="б2", x=10.0, y=0.0, h=10.0, surface_type="auto"),
        GeoPoint(id="б3", x=10.0, y=10.0, h=10.0, surface_type="auto"),
        GeoPoint(id="б4", x=0.0, y=10.0, h=10.0, surface_type="auto"),
        GeoPoint(id="тело", x=5.0, y=5.0, h=15.0, surface_type="auto"),
        GeoPoint(id="земля_вне", x=20.0, y=20.0, h=10.0, surface_type="auto"),
    ]
    app.boundary_indices = [0, 1, 2, 3]
    app._undo_stack = []
    app.calc_results = None
    app.TAB_2D = "2D Схема в плане"

    app._show_top = MagicMock()
    app._show_bottom = MagicMock()
    app._selected_points = set()

    fig = Figure()
    ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)
    fig.canvas = canvas
    app.fig_2d = fig
    app.ax_2d = ax
    app.canvas_2d = canvas

    app._position_reset_contour_button = MagicMock()
    app._update_selection_bar = MagicMock()
    app._apply_fig_layout = MagicMock()
    app._is_separate_surfaces = False
    app._boundary_interpolator_cache = None
    app._boundary_path_cache = None
    app._local_surface_threshold = None
    return app


def create_cut_app():
    """Создает приложение с выемкой (Cut) и точками внутри и снаружи контура."""
    app = interactive_app.VolumeApp.__new__(interactive_app.VolumeApp)
    app.points = [
        GeoPoint(id="б1", x=0.0, y=0.0, h=20.0, surface_type="auto"),
        GeoPoint(id="б2", x=10.0, y=0.0, h=20.0, surface_type="auto"),
        GeoPoint(id="б3", x=10.0, y=10.0, h=20.0, surface_type="auto"),
        GeoPoint(id="б4", x=0.0, y=10.0, h=20.0, surface_type="auto"),
        GeoPoint(id="дно", x=5.0, y=5.0, h=15.0, surface_type="auto"),
        GeoPoint(id="земля_вне", x=20.0, y=20.0, h=20.0, surface_type="auto"),
    ]
    app.boundary_indices = [0, 1, 2, 3]
    app._undo_stack = []
    app.calc_results = None
    app.TAB_2D = "2D Схема в плане"

    app._show_top = MagicMock()
    app._show_bottom = MagicMock()
    app._selected_points = set()

    fig = Figure()
    ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)
    fig.canvas = canvas
    app.fig_2d = fig
    app.ax_2d = ax
    app.canvas_2d = canvas

    app._position_reset_contour_button = MagicMock()
    app._update_selection_bar = MagicMock()
    app._apply_fig_layout = MagicMock()
    app._is_separate_surfaces = False
    app._boundary_interpolator_cache = None
    app._boundary_path_cache = None
    app._local_surface_threshold = None
    return app


class TestFillSurfaceVisibility:
    def test_fill_only_bottom_hides_outside_points(self):
        app = create_fill_app()
        app._show_top.get.return_value = False
        app._show_bottom.get.return_value = True

        # Подошва насыпи (контур) видна
        for b_idx in [0, 1, 2, 3]:
            assert app._is_point_visible_on_2d(b_idx) is True
        # Тело насыпи скрыто
        assert app._is_point_visible_on_2d(4) is False
        # Точка за пределами контура СКРЫТА
        assert app._is_point_visible_on_2d(5) is False

    def test_fill_only_top_shows_pile_only(self):
        app = create_fill_app()
        app._show_top.get.return_value = True
        app._show_bottom.get.return_value = False

        # Подошва насыпи скрыта
        for b_idx in [0, 1, 2, 3]:
            assert app._is_point_visible_on_2d(b_idx) is False
        # Тело насыпи видно
        assert app._is_point_visible_on_2d(4) is True
        # Точка за пределами контура скрыта
        assert app._is_point_visible_on_2d(5) is False

    def test_fill_both_surfaces_shows_all(self):
        app = create_fill_app()
        app._show_top.get.return_value = True
        app._show_bottom.get.return_value = True

        for idx in range(6):
            assert app._is_point_visible_on_2d(idx) is True


class TestCutSurfaceVisibility:
    def test_cut_only_bottom_hides_outside_points(self):
        app = create_cut_app()
        app._show_top.get.return_value = False
        app._show_bottom.get.return_value = True

        # Бровка скрыта
        for b_idx in [0, 1, 2, 3]:
            assert app._is_point_visible_on_2d(b_idx) is False
        # Дно котлована видно
        assert app._is_point_visible_on_2d(4) is True
        # Точка за пределами контура СКРЫТА
        assert app._is_point_visible_on_2d(5) is False

    def test_cut_only_top_shows_daylight_surface(self):
        app = create_cut_app()
        app._show_top.get.return_value = True
        app._show_bottom.get.return_value = False

        # Бровка видна
        for b_idx in [0, 1, 2, 3]:
            assert app._is_point_visible_on_2d(b_idx) is True
        # Дно котлована скрыто
        assert app._is_point_visible_on_2d(4) is False
        # Дневная поверхность за пределами контура видна
        assert app._is_point_visible_on_2d(5) is True

    def test_cut_both_surfaces_shows_all(self):
        app = create_cut_app()
        app._show_top.get.return_value = True
        app._show_bottom.get.return_value = True

        for idx in range(6):
            assert app._is_point_visible_on_2d(idx) is True
