# -*- coding: utf-8 -*-
"""
Тесты цветовой схемы картограммы масс (Швейцарская светотеневая модель: SwissFill / SwissCut, Hillshade, горизонтали).
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


def create_mock_app(is_cut: bool, show_top=True, show_bottom=True):
    app = interactive_app.VolumeApp.__new__(interactive_app.VolumeApp)
    app._is_excavation = MagicMock(return_value=is_cut)
    app.boundary_indices = [0, 1, 2, 3]

    app._show_top = MagicMock()
    app._show_top.get.return_value = show_top
    app._show_bottom = MagicMock()
    app._show_bottom.get.return_value = show_bottom

    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    fig.canvas = canvas
    app.fig_diff = fig
    app.ax_diff = fig.add_subplot(111)
    app.canvas_diff = canvas

    app._format_coord_display = MagicMock(return_value="")
    app._apply_axes_theme = MagicMock()
    app._apply_fig_layout = MagicMock()
    app.tk = MagicMock()
    app._toolbar_diff = None
    return app


def test_fill_cartogram_swiss_relief():
    """Для насыпей картограмма мощности использует Swiss Hillshade и горизонтали с плюсом."""
    app = create_mock_app(is_cut=False, show_top=True, show_bottom=True)

    gx, gy = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 10))
    dh = np.linspace(0.5, 4.0, 100).reshape(10, 10)
    z_top = 100.0 + dh
    z_bot = np.full((10, 10), 100.0)

    res = {
        "grid_x": gx,
        "grid_y": gy,
        "dh_grid": dh,
        "z_top_grid": z_top,
        "z_bot_grid": z_bot,
        "boundary": np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
    }

    app._redraw_diff(res)

    # Проверяем наличие слоя Hillshade (imshow) и его обрезку контуром
    assert len(app.ax_diff.images) >= 1
    im = app.ax_diff.images[0]
    assert im.get_clip_path() is not None

    # Проверяем наличие подписей отметок горизонталей
    assert len(app.ax_diff.texts) > 0
    assert any("+" in t.get_text() for t in app.ax_diff.texts)

    # Проверяем наличие шкалы colorbar
    cb_axes = [ax for ax in app.fig_diff.axes if ax is not app.ax_diff]
    assert len(cb_axes) == 1


def test_cut_cartogram_swiss_relief():
    """Для выемок картограмма мощности использует Swiss Hillshade и горизонтали с минусом."""
    app = create_mock_app(is_cut=True, show_top=True, show_bottom=True)

    gx, gy = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 10))
    dh = np.linspace(0.5, 4.0, 100).reshape(10, 10)
    z_top = np.full((10, 10), 100.0)
    z_bot = 100.0 - dh

    res = {
        "grid_x": gx,
        "grid_y": gy,
        "dh_grid": dh,
        "z_top_grid": z_top,
        "z_bot_grid": z_bot,
        "boundary": np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
    }

    app._redraw_diff(res)

    # Проверяем наличие слоя Hillshade (imshow) и его обрезку контуром
    assert len(app.ax_diff.images) >= 1
    im = app.ax_diff.images[0]
    assert im.get_clip_path() is not None

    # Подписи горизонталей содержат знак "-" для выемки
    assert len(app.ax_diff.texts) > 0
    assert any("-" in t.get_text() for t in app.ax_diff.texts)

    # Проверяем наличие шкалы colorbar
    cb_axes = [ax for ax in app.fig_diff.axes if ax is not app.ax_diff]
    assert len(cb_axes) == 1


def test_single_surface_top_copper():
    """При отображении только верхней поверхности используется copper и нет лишних теней."""
    app = create_mock_app(is_cut=False, show_top=True, show_bottom=False)

    gx, gy = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 10))
    dh = np.linspace(0.5, 4.0, 100).reshape(10, 10)
    z_top = 100.0 + dh
    z_bot = np.full((10, 10), 100.0)

    res = {
        "grid_x": gx,
        "grid_y": gy,
        "dh_grid": dh,
        "z_top_grid": z_top,
        "z_bot_grid": z_bot,
        "boundary": np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
    }

    app._redraw_diff(res)

    contour_set = None
    for c in app.ax_diff.collections:
        if hasattr(c, "cmap"):
            contour_set = c
            break

    assert contour_set is not None
    assert contour_set.cmap.name == "copper"
    assert len(app.ax_diff.images) == 0


def test_colorbar_cut_inverted():
    """Шкала colorbar для выемок перевернута (0 у бровки вверху, глубина внизу)."""
    app = create_mock_app(is_cut=True, show_top=True, show_bottom=True)

    gx, gy = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 10))
    dh = np.linspace(0.5, 4.0, 100).reshape(10, 10)
    z_top = np.full((10, 10), 100.0)
    z_bot = 100.0 - dh

    res = {
        "grid_x": gx,
        "grid_y": gy,
        "dh_grid": dh,
        "z_top_grid": z_top,
        "z_bot_grid": z_bot,
        "boundary": np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
    }

    app._redraw_diff(res)

    cb_axes = [ax for ax in app.fig_diff.axes if ax is not app.ax_diff]
    assert len(cb_axes) == 1
    cb_ax = cb_axes[0]
    ylim = cb_ax.get_ylim()
    assert ylim[0] > ylim[1]  # Перевернутая шкала: 0 вверху, глубина внизу


def test_colorbar_fill_standard():
    """Шкала colorbar для насыпей имеет стандартную ориентацию (снизу вверх)."""
    app = create_mock_app(is_cut=False, show_top=True, show_bottom=True)

    gx, gy = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 10))
    dh = np.linspace(0.5, 4.0, 100).reshape(10, 10)
    z_top = 100.0 + dh
    z_bot = np.full((10, 10), 100.0)

    res = {
        "grid_x": gx,
        "grid_y": gy,
        "dh_grid": dh,
        "z_top_grid": z_top,
        "z_bot_grid": z_bot,
        "boundary": np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
    }

    app._redraw_diff(res)

    cb_axes = [ax for ax in app.fig_diff.axes if ax is not app.ax_diff]
    assert len(cb_axes) == 1
    cb_ax = cb_axes[0]
    ylim = cb_ax.get_ylim()
    assert ylim[0] < ylim[1]  # Стандартная шкала: 0 внизу, гребень вверху


def test_cartogram_hillshade_transposition_alignment():
    """Тест проверяет, что форма растра hillshade (imshow) согласуется с осями Y (Northing) и X (Easting)."""
    app = create_mock_app(is_cut=False, show_top=True, show_bottom=True)

    # Несимметричная прямоугольная сетка: 20 точек по X (Northing, вертикаль), 50 точек по Y (Easting, горизонталь)
    gx_vec = np.linspace(100, 200, 20)
    gy_vec = np.linspace(500, 600, 50)
    # np.meshgrid(gx_vec, gy_vec): shape is (len(gy_vec), len(gx_vec)) = (50, 20)
    gx, gy = np.meshgrid(gx_vec, gy_vec)
    dh = np.linspace(1.0, 3.0, 50 * 20).reshape(50, 20)
    z_top = 100.0 + dh
    z_bot = np.full((50, 20), 100.0)

    res = {
        "grid_x": gx,
        "grid_y": gy,
        "dh_grid": dh,
        "z_top_grid": z_top,
        "z_bot_grid": z_bot,
        "boundary": np.array([[100, 500], [100, 600], [200, 600], [200, 500]]),
    }

    app._redraw_diff(res)

    assert len(app.ax_diff.images) == 1
    im = app.ax_diff.images[0]
    im_data = im.get_array()
    # im_data строки (axis 0) должны соответствовать gx (20 точек), столбцы (axis 1) - gy (50 точек)
    assert im_data.shape == (20, 50)



