# -*- coding: utf-8 -*-
"""
Тесты для Этапа 2: Дифференциальное обновление 2D-схемы (без ax.clear) и векторизованный scatter.
"""
import sys
import os
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geo_parser import GeoPoint
import interactive_app


def create_test_app():
    """Создает тестовый экземпляр VolumeApp с реальными осями Matplotlib."""
    app = interactive_app.VolumeApp.__new__(interactive_app.VolumeApp)
    app.points = [
        GeoPoint(id="т1", x=10.0, y=20.0, h=15.0, surface_type="top"),
        GeoPoint(id="т2", x=30.0, y=20.0, h=15.0, surface_type="top"),
        GeoPoint(id="т3", x=30.0, y=50.0, h=15.0, surface_type="bottom"),
        GeoPoint(id="т4", x=10.0, y=50.0, h=15.0, surface_type="bottom"),
    ]
    app.boundary_indices = [0, 1, 2]
    app._undo_stack = []
    app.calc_results = None
    app.TAB_2D = "2D Схема в плане"

    app._show_top = MagicMock()
    app._show_top.get.return_value = True
    app._show_bottom = MagicMock()
    app._show_bottom.get.return_value = True

    app._selected_points = set()

    # Инициализация Matplotlib холста (чистый Agg без Tcl/Tk)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure()
    ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)
    fig.canvas = canvas
    app.fig_2d = fig
    app.ax_2d = ax
    app.canvas_2d = canvas

    # Моки вспомогательных UI методов
    app._position_reset_contour_button = MagicMock()
    app._update_selection_bar = MagicMock()
    app._apply_fig_layout = MagicMock()

    app._is_separate_surfaces = False
    app._boundary_interpolator_cache = None
    app._local_surface_threshold = None
    app._format_coord_display = lambda y, x: ""

    # Ссылки на художники
    app._contour_line_artist = None
    app._contour_fill_artist = None
    app._contour_order_artists = []
    app._selected_scatter_artist = None
    app._scatter_top_artist = None
    app._scatter_bot_artist = None
    app._point_annotations = []
    app._points_kdtree = None
    app._points_coords_len = 0
    app._blit_bg = None
    app._box_select_rect_artist = None
    app._box_select_start = None
    app._box_select_moved = False
    app._contour_drag_source_idx = None
    app._contour_drag_pos = None
    app._contour_drag_moved = False
    app._contour_drag_target_idx = None
    app._contour_drag_start_coord = None
    app._contour_drag_artists = []
    app._contour_drag_line1 = None
    app._contour_drag_line2 = None
    app._contour_drag_guide = None
    app._contour_drag_ring = None

    return app


def test_redraw_2d_creates_vectorized_artists():
    """_redraw_2d создает векторизованные художники для верхних/нижних точек и контура."""
    app = create_test_app()
    app._redraw_2d()

    assert app._contour_line_artist is not None
    assert app._contour_fill_artist is not None
    assert app._selected_scatter_artist is not None
    assert app._scatter_top_artist is not None or app._scatter_bot_artist is not None

    # Линия контура содержит координаты исходных 3 точек контура + замыкание
    line_x, line_y = app._contour_line_artist.get_data()
    assert len(line_x) == 4  # 3 точки + 1 замыкающая
    plt.close(app.fig_2d)


def test_update_2d_contour_only_does_not_clear_axes():
    """_update_2d_contour_only обновляет данные линии без вызова ax.clear()."""
    app = create_test_app()
    app._redraw_2d()

    # Добавляем 4-ю точку в контур
    app.boundary_indices = [0, 1, 2, 3]

    with patch.object(app.ax_2d, "clear") as mock_clear:
        app._update_2d_contour_only()
        mock_clear.assert_not_called()

    # Проверяем, что линия контура теперь содержит 4 точки + замыкание = 5 точек
    line_y, line_x = app._contour_line_artist.get_data()
    assert len(line_y) == 5
    assert len(line_x) == 5
    assert line_y[0] == line_y[-1]  # замкнутый полигон
    plt.close(app.fig_2d)


def test_update_2d_selection_only_updates_offsets():
    """_update_2d_selection_only обновляет offsets без ax.clear()."""
    app = create_test_app()
    app._redraw_2d()

    app._selected_points = {0, 2}

    with patch.object(app.ax_2d, "clear") as mock_clear:
        app._update_2d_selection_only()
        mock_clear.assert_not_called()

    offsets = app._selected_scatter_artist.get_offsets()
    assert len(offsets) == 2
    # Точки 0 и 2: Y = [20.0, 50.0], X = [10.0, 30.0]
    np.testing.assert_allclose(offsets[0], [20.0, 10.0])
    np.testing.assert_allclose(offsets[1], [50.0, 30.0])
    plt.close(app.fig_2d)


def test_clear_selection_clears_offsets():
    """Сброс выделения обнуляет offsets без перерисовки всей схемы."""
    app = create_test_app()
    app._redraw_2d()

    app._selected_points = {1}
    app._update_2d_selection_only()
    assert len(app._selected_scatter_artist.get_offsets()) == 1

    app._clear_selected_points()
    assert len(app._selected_points) == 0
    assert len(app._selected_scatter_artist.get_offsets()) == 0
    plt.close(app.fig_2d)


def test_kdtree_spatial_index():
    """cKDTree корректно строится и ускоряет поиск ближайшей точки."""
    app = create_test_app()
    tree = app._get_points_kdtree()
    assert tree is not None

    # Ищем точку вблизи (Y=20.1, X=10.1) -> должна найтись точка 0 (Y=20.0, X=10.0)
    idx, dist = app._find_nearest_point(20.1, 10.1)
    assert idx == 0
    assert abs(dist - np.hypot(0.1, 0.1)) < 1e-6

    # Инвалидация кеша
    app._invalidate_points_spatial_index()
    assert app._points_kdtree is None
    plt.close(app.fig_2d)


def test_vectorized_nearest_boundary_edge():
    """Векторизованный расчет ближайшего ребра контура дает верный индекс."""
    app = create_test_app()
    # Контур из вершин [0, 1, 2]: (20,10) -> (20,30) -> (50,30) -> замкнут к (20,10)
    # Точка 3: (50, 10). Проверяем ближайшее ребро.
    k = app._find_nearest_boundary_edge(3)
    assert 0 <= k < len(app.boundary_indices)
    plt.close(app.fig_2d)


def test_box_select_blitting():
    """Перемещение рамки выделения использует аппаратный blitting без перерисовки всего холста."""
    app = create_test_app()
    app._redraw_2d()

    # Начало рамки
    app._box_select_start = (20.0, 10.0, 100, 100)
    app._box_select_moved = False
    app._capture_blit_bg()
    assert app._blit_bg is not None

    event_motion = MagicMock()
    event_motion.x = 200
    event_motion.y = 200
    event_motion.xdata = 40.0
    event_motion.ydata = 30.0

    with patch.object(app.canvas_2d, "restore_region", wraps=app.canvas_2d.restore_region) as mock_restore, \
         patch.object(app.ax_2d, "draw_artist", wraps=app.ax_2d.draw_artist) as mock_draw_art, \
         patch.object(app.canvas_2d, "blit", wraps=app.canvas_2d.blit) as mock_blit:
        app._handle_box_select_motion(event_motion)
        assert mock_restore.called
        assert mock_draw_art.called
        assert mock_blit.called
        assert app._box_select_rect_artist.get_visible() is True

    # Завершение рамки
    event_release = MagicMock()
    event_release.xdata = 40.0
    event_release.ydata = 30.0
    app._is_ctrl_down = MagicMock(return_value=False)
    app._handle_box_select_release(event_release)

    assert app._box_select_rect_artist.get_visible() is False
    assert app._blit_bg is None
    plt.close(app.fig_2d)


def test_contour_drag_blitting():
    """Перетаскивание вершины контура использует аппаратный blitting."""
    app = create_test_app()
    app._redraw_2d()

    app._contour_drag_source_idx = 0
    app._contour_drag_pos = 0
    app._contour_drag_start_coord = (100, 100)
    app._contour_drag_moved = False
    app._capture_blit_bg()
    assert app._blit_bg is not None

    event_motion = MagicMock()
    event_motion.x = 150
    event_motion.y = 150
    event_motion.xdata = 25.0
    event_motion.ydata = 15.0

    with patch.object(app.canvas_2d, "restore_region", wraps=app.canvas_2d.restore_region) as mock_restore, \
         patch.object(app.ax_2d, "draw_artist", wraps=app.ax_2d.draw_artist) as mock_draw_art, \
         patch.object(app.canvas_2d, "blit", wraps=app.canvas_2d.blit) as mock_blit:
        app._handle_contour_drag_motion(event_motion)
        assert mock_restore.called
        assert mock_draw_art.called
        assert mock_blit.called
        assert app._contour_drag_line1.get_visible() is True

    # Отпускание кнопки мыши
    event_release = MagicMock()
    event_release.button = 3
    event_release.xdata = 25.0
    event_release.ydata = 15.0
    app._on_canvas_release(event_release)

    assert app._contour_drag_line1.get_visible() is False
    assert app._blit_bg is None
    plt.close(app.fig_2d)

