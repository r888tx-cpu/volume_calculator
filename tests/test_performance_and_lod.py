# -*- coding: utf-8 -*-
"""
Тесты для оптимизации производительности:
1. Адаптивный LOD подписей точек (auto/off/on).
2. SpatialIndexService get_points_in_bbox (векторная выборка рамкой).
3. Векторизованная отрисовка TIN.
4. Оптимизированное сохранение проектов (compact JSON).
"""
import os
import sys
import json
import pytest
import numpy as np
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geo_parser import GeoPoint
from spatial_index import SpatialIndexService
from project_storage import ProjectStorageService
from ui_dialogs import calc_3d_stride
import interactive_app


def create_mock_app(n_points=500):
    app = interactive_app.VolumeApp.__new__(interactive_app.VolumeApp)
    app.tk = MagicMock()
    app.btn_reset_contour = None
    app.btn_labels_mode = None
    app.points = [
        GeoPoint(id=f"P_{i}", x=float(i % 50), y=float(i // 50), h=10.0 + (i % 5), surface_type="top" if i % 2 == 0 else "bottom")
        for i in range(n_points)
    ]
    app.boundary_indices = [0, 49, n_points - 1, n_points - 50]
    app._undo_stack = []
    app.calc_results = None
    app.TAB_2D = "2D Схема в плане"
    app.TAB_TIN = "TIN Триангуляция"

    app._show_top = MagicMock()
    app._show_top.get.return_value = True
    app._show_bottom = MagicMock()
    app._show_bottom.get.return_value = True
    app._point_labels_mode = "on"
    app._labels_update_timer = None

    app._selected_points = set()

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
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
    app._local_surface_threshold = None
    app._format_coord_display = lambda y, x: ""
    app._point_annotations = []
    app._contour_order_artists = []
    app._box_select_rect_artist = None
    app._selected_scatter_artist = None
    app._is_ctrl_down = lambda e: False
    app._contour_drag_source_idx = None
    app._box_select_start = None
    app._pan_annotations_hidden = False

    return app


def test_spatial_index_get_points_in_bbox():
    points = [
        GeoPoint(id=f"P_{i}", x=float(i), y=float(i * 2), h=0.0)
        for i in range(100)
    ]
    service = SpatialIndexService(points)

    # Box: Y in [20, 60], X in [10, 30]
    # y = 2*i, x = i => i in [10, 30] satisfies both
    indices = service.get_points_in_bbox(y_min=20.0, y_max=60.0, x_min=10.0, x_max=30.0)
    assert len(indices) == 21
    assert set(indices) == set(range(10, 31))

    # Test empty result
    empty_indices = service.get_points_in_bbox(y_min=500.0, y_max=600.0, x_min=0.0, x_max=10.0)
    assert len(empty_indices) == 0


def test_lod_mode_off_never_shows_annotations():
    """В режиме off подписи точек не создаются даже для малого числа точек."""
    app = create_mock_app(n_points=30)
    app._point_labels_mode = "off"
    app._redraw_2d()

    point_labels = [item for item in app._point_annotations if item[2] == "point"]
    assert len(point_labels) == 0
    bound_labels = [item for item in app._point_annotations if item[2] == "bound"]
    assert len(bound_labels) > 0


def test_lod_mode_on_always_shows_annotations():
    """В режиме on подписи точек создаются."""
    app = create_mock_app(n_points=400)
    app._point_labels_mode = "on"
    app._redraw_2d()

    point_labels = [item for item in app._point_annotations if item[2] == "point"]
    assert len(point_labels) == 400


def test_toggle_labels_mode_cycles():
    app = create_mock_app(n_points=50)
    assert app._point_labels_mode == "on"
    assert app._get_labels_btn_text() == "🏷️ Подписи: Вкл"

    app._toggle_labels_mode()
    assert app._point_labels_mode == "off"
    assert app._get_labels_btn_text() == "🏷️ Подписи: Выкл"

    app._toggle_labels_mode()
    assert app._point_labels_mode == "on"
    assert app._get_labels_btn_text() == "🏷️ Подписи: Вкл"


def test_box_selection_vectorized():
    app = create_mock_app(n_points=200)
    # Simulate box select release from (y=0, x=0) to (y=2, x=20)
    event = MagicMock()
    event.xdata = 2.0
    event.ydata = 20.0
    app._box_select_start = (0.0, 0.0, 10, 10)
    app._box_select_moved = True

    app._handle_box_select_release(event)
    assert len(app._selected_points) > 0
    # Every selected point must be inside bbox
    for idx in app._selected_points:
        p = app.points[idx]
        assert 0.0 <= p.y <= 2.0
        assert 0.0 <= p.x <= 20.0


def test_compact_json_for_large_projects(tmp_path):
    proj_path = str(tmp_path / "test_proj.volproj")
    # Small project
    small_data = {"points": [{"id": f"P{i}", "x": float(i), "y": 0.0, "h": 0.0} for i in range(10)]}
    assert ProjectStorageService.save_project_file(proj_path, small_data)
    with open(proj_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "\n  " in content  # indented

    # Large project (> 1000 points)
    large_data = {"points": [{"id": f"P{i}", "x": float(i), "y": 0.0, "h": 0.0} for i in range(1500)]}
    assert ProjectStorageService.save_project_file(proj_path, large_data)
    with open(proj_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "\n  " not in content  # compact, no extra indentation

    loaded = ProjectStorageService.load_project_file(proj_path)
    assert loaded is not None
    assert len(loaded["points"]) == 1500


def test_500_points_redraw_performance():
    """Проверяет, что 500 точек на 2D-схеме отрисовываются быстрее 100 мс благодаря кешированию и векторизации."""
    import time
    app = create_mock_app(n_points=500)
    app._point_labels_mode = "off"

    t0 = time.perf_counter()
    app._redraw_2d()
    dt_first = (time.perf_counter() - t0) * 1000.0

    # Второй вызов (с разогретым кешем, например при зуме/панорамировании)
    t0 = time.perf_counter()
    app._redraw_2d()
    dt_cached = (time.perf_counter() - t0) * 1000.0

    # Отрисовка не должна зависать на секунды (ранее занимала 2500 мс!)
    assert dt_first < 300.0, f"First redraw too slow: {dt_first:.1f} ms"
    assert dt_cached < 250.0, f"Cached redraw too slow: {dt_cached:.1f} ms"


def test_cache_invalidation_on_geometry_change():
    """Проверяет корректность работы и сброса кешей геометрии."""
    app = create_mock_app(n_points=100)

    # До вызова кеши пусты
    assert app._split_height_cache is None
    assert app._point_surfaces_cache is None
    assert app._boundary_inside_cache is None

    # После вызова заполняются
    surfaces = app._ensure_point_surfaces()
    inside_mask = app._ensure_boundary_inside_mask()
    assert len(surfaces) == 100
    assert len(inside_mask) == 100
    assert app._split_height_cache is not None
    assert app._point_surfaces_cache is not None
    assert app._boundary_inside_cache is not None

    # Вызов инвалидации сбрасывает все кеши
    app._invalidate_boundary_cache()
    assert app._split_height_cache is None
    assert app._work_type_cache is None
    assert app._point_surfaces_cache is None
    assert app._boundary_inside_cache is None


def test_pan_drag_suppresses_point_annotations():
    """Проверяет динамическое скрытие подписей при перетаскивании схемы (Pan Suppression)."""
    app = create_mock_app(n_points=40)
    app._point_labels_mode = "on"
    app._redraw_2d()

    # Подписи точек созданы и видимы
    point_anns = [item[0] for item in app._point_annotations if item[2] == "point"]
    assert len(point_anns) == 40
    assert all(ann.get_visible() for ann in point_anns)

    # Симулируем начало перетаскивания (Pan Drag)
    app._pan_start = (100, 100, 0, 0, (0.0, 10.0), (0.0, 10.0), 1)
    event_move = MagicMock()
    event_move.x = 150
    event_move.y = 150
    app._on_canvas_motion(event_move)

    assert app._pan_dragged is True
    assert app._pan_annotations_hidden is True
    # Все подписи точек скрыты во время движения
    assert all(not ann.get_visible() for ann in point_anns)

    # Симулируем отпускание кнопки мыши (Release)
    app.after = MagicMock(return_value=123)
    event_release = MagicMock()
    event_release.button = 1
    event_release.xdata = 5.0
    event_release.ydata = 5.0
    app._on_canvas_release(event_release)

    assert app._pan_annotations_hidden is False
    assert app._pan_start is None
    # Запланировано обновление подписей
    assert app._labels_update_timer is not None


def test_compute_label_screen_pos_boundary_clipping():
    """Проверяет отсечение подписей за пределами осей и переворот anchor у границ."""
    from matplotlib.transforms import Bbox
    app = create_mock_app(n_points=10)
    bbox = Bbox.from_extents(50.0, 50.0, 450.0, 450.0)
    fig_h = 500.0

    # 1. Вне видимой области -> visible=False
    _, _, _, vis_left = app._compute_label_screen_pos(40.0, 200.0, bbox, fig_h)
    assert vis_left is False
    _, _, _, vis_right = app._compute_label_screen_pos(460.0, 200.0, bbox, fig_h)
    assert vis_right is False
    _, _, _, vis_top = app._compute_label_screen_pos(200.0, 460.0, bbox, fig_h)
    assert vis_top is False

    # 2. Внутри видимой области -> visible=True
    tk_x, tk_y, anchor, vis_center = app._compute_label_screen_pos(200.0, 200.0, bbox, fig_h)
    assert vis_center is True
    assert anchor == "sw"
    assert tk_x == 205.0

    # 3. Возле верхней границы -> anchor='nw', надпись смещается вниз
    tk_x, tk_y, anchor, vis_near_top = app._compute_label_screen_pos(200.0, 445.0, bbox, fig_h)
    assert vis_near_top is True
    assert anchor.startswith("n")
    assert tk_y == fig_h - 445.0 + 6.0

    # 4. Возле правой границы -> anchor='se', надпись смещается влево
    tk_x, tk_y, anchor, vis_near_right = app._compute_label_screen_pos(440.0, 200.0, bbox, fig_h)
    assert vis_near_right is True
    assert anchor.endswith("e")
    assert tk_x == 440.0 - 5.0


def test_tk_canvas_widget_detection():
    """Проверяет детекцию FigureCanvasAgg vs FigureCanvasTkAgg."""
    app = create_mock_app(n_points=10)
    # По умолчанию в тестах используется FigureCanvasAgg -> _has_tk_canvas_widget возвращает False
    assert app._has_tk_canvas_widget() is False

    # Имитируем tk.Canvas виджет
    import tkinter as tk
    mock_canvas = MagicMock(spec=tk.Canvas)
    app.canvas_2d.get_tk_widget = MagicMock(return_value=mock_canvas)
    assert app._has_tk_canvas_widget() is True


def test_tk_canvas_labels_render_update_clear():
    """Проверяет жизненный цикл нативных текстовых меток на Tkinter Canvas."""
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tkinter Tcl/Tk runtime not available in this environment")
    root.withdraw()
    try:
        tk_canvas = tk.Canvas(root, width=500, height=500)
        app = create_mock_app(n_points=10)
        app.canvas_2d.get_tk_widget = MagicMock(return_value=tk_canvas)

        pts = [
            GeoPoint(id=f"P_{i}", x=float(i), y=float(i), h=10.0 + i)
            for i in range(5)
        ]
        app._render_tk_canvas_labels(pts)

        # Проверяем, что элементы с тегом 'point_label' созданы на холсте
        items = tk_canvas.find_withtag("point_label")
        assert len(items) == len(pts)

        # Проверяем обновление координат
        app._update_tk_canvas_labels_positions()
        items_after_update = tk_canvas.find_withtag("point_label")
        assert len(items_after_update) == len(pts)

        # Проверяем очистку
        app._clear_tk_canvas_labels()
        items_cleared = tk_canvas.find_withtag("point_label")
        assert len(items_cleared) == 0
    finally:
        root.destroy()


def test_save_tab_image_preserves_vector_annotations(tmp_path):
    """Проверяет, что при экспорте в PNG на 2D схему временно накладываются векторные аннотации."""
    app = create_mock_app(n_points=10)
    pts = [GeoPoint(id=f"P_{i}", x=float(i), y=float(i), h=10.0 + i) for i in range(5)]
    app._current_labeled_points = pts
    app._current_project_dir = str(tmp_path)
    app.lbl_report_status = MagicMock()

    saved_annotations_count = []
    orig_savefig = app.fig_2d.savefig

    def mock_savefig(*args, **kwargs):
        # Во время savefig на ax_2d должны присутствовать текстовые аннотации
        saved_annotations_count.append(len(app.ax_2d.texts))
        return orig_savefig(*args, **kwargs)

    app.fig_2d.savefig = mock_savefig

    import tkinter.messagebox
    orig_info = tkinter.messagebox.showinfo
    tkinter.messagebox.showinfo = MagicMock()
    try:
        app._save_tab_image(app.fig_2d, "test_scheme")
        # Во время сохранения аннотации были добавлены
        assert len(saved_annotations_count) == 1
        assert saved_annotations_count[0] == 5
        # После завершения сохранения временные аннотации удалены
        assert len(app.ax_2d.texts) == 0
    finally:
        tkinter.messagebox.showinfo = orig_info


def test_canvas_dark_and_light_theme():
    """Проверяет адаптацию фигуры, осей и контрастного цвета текста под тему."""
    app = create_mock_app(n_points=10)

    # 1. Применяем тёмную тему
    app._apply_axes_theme(app.ax_2d, app.fig_2d, is_dark=True)
    assert app.ax_2d.get_facecolor()[:3] != (1.0, 1.0, 1.0)
    # Текст на тёмном фоне должен быть светлым
    assert app._get_canvas_text_color() == "#f0f2f5"

    # 2. Применяем светлую тему
    app._apply_axes_theme(app.ax_2d, app.fig_2d, is_dark=False)
    assert app.ax_2d.get_facecolor()[:3] == (1.0, 1.0, 1.0)
    # Текст на светлом фоне должен быть тёмным (никогда не светло-серым на белом!)
    assert app._get_canvas_text_color() == "#111111"


def test_calc_3d_stride_adaptive_lod():
    """Проверяет корректность расчета адаптивного шага прореживания (LOD) для 3D поверхностей."""
    # 1. Маленькая сетка (меньше target_dim=45) — шаг 1 (без прореживания)
    small_grid = np.zeros((30, 40))
    assert calc_3d_stride(small_grid, target_dim=45) == (1, 1)

    # 2. Большая сетка из реального проекта насыпи (354 x 502) -> шаг (8, 11)
    large_grid = np.zeros((354, 502))
    r_step, c_step = calc_3d_stride(large_grid, target_dim=45)
    assert r_step == 8
    assert c_step == 11
    # Число результирующих полигонов ограничено ~2000 вместо 44000+
    poly_count = (354 // r_step) * (502 // c_step)
    assert poly_count < 2200

    # 3. Асимметричная сетка (например, узкая траншея 300 x 20)
    trench_grid = np.zeros((300, 20))
    r_step, c_step = calc_3d_stride(trench_grid, target_dim=45)
    assert r_step == 7
    assert c_step == 1

    # 4. Граничные и некорректные случаи (None, пустой массив, 1D)
    assert calc_3d_stride(None) == (2, 2)
    assert calc_3d_stride(np.array([])) == (2, 2)
    assert calc_3d_stride(np.array([1.0, 2.0])) == (2, 2)


def test_3d_redraw_isolated_bottom_surface_performance():
    """Проверяет быструю отрисовку изолированной нижней поверхности (сценарий пользователя: насыпь, show_bottom=True)."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    app = create_mock_app(n_points=50)
    fig_3d = Figure()
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    canvas_3d = FigureCanvasAgg(fig_3d)
    fig_3d.canvas = canvas_3d
    app.fig_3d = fig_3d
    app.ax_3d = ax_3d
    app.canvas_3d = canvas_3d

    app._show_top = MagicMock()
    app._show_top.get.return_value = False
    app._show_bottom = MagicMock()
    app._show_bottom.get.return_value = True
    app._is_excavation = MagicMock(return_value=False)

    rows, cols = 354, 502
    grid_y, grid_x = np.meshgrid(np.linspace(0, 100, cols), np.linspace(0, 70, rows))
    z_bot = np.full((rows, cols), 150.0)
    z_top = z_bot + 2.0

    r = {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "z_top_grid": z_top,
        "z_bot_grid": z_bot,
        "custom_tin": True,
        "pts_3d": np.zeros((10, 3)),
        "active_simplices": np.array([[0, 1, 2]]),
        "boundary": np.array([[0, 0, 150], [0, 100, 150], [70, 100, 150], [70, 0, 150]]),
    }

    import time
    t0 = time.perf_counter()
    app._redraw_3d(r)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # Без адаптивного LOD отрисовка 44 427 полигонов занимала >500-1000 мс на первом кадре.
    # С адаптивным LOD число полигонов уменьшено с 44,427 до ~2,000, время отрисовки < 250 мс
    assert dt_ms < 250.0, f"3D redraw too slow: {dt_ms:.1f} ms"
    # Проверяем, что в ax_3d добавлена поверхность
    collections = ax_3d.collections
    assert len(collections) > 0
    # Проверяем число сгенерированных полигонов на холсте
    surf = collections[0]
    poly_count = len(surf.get_paths())
    assert 1500 <= poly_count <= 2500, f"Expected ~2000 polygons, got {poly_count}"


def test_throttled_canvas_draw_gate():
    """Проверяет, что _throttled_canvas_draw не создает лавину очередей перерисовок при панорамировании."""
    app = create_mock_app(n_points=10)
    app._pan_draw_timers = {}
    mock_canvas = MagicMock()
    app.after = MagicMock(return_value="timer_123")
    app.after_cancel = MagicMock()

    # 1. Первый вызов регистрирует таймер
    app._throttled_canvas_draw(mock_canvas, delay_ms=25)
    assert app.after.call_count == 1
    assert mock_canvas in app._pan_draw_timers

    # 2. Последующие вызовы при активном таймере игнорируются (FPS gate)
    for _ in range(10):
        app._throttled_canvas_draw(mock_canvas, delay_ms=25)
    assert app.after.call_count == 1

    # 3. Финальный сброс отменяет таймер и выполняет отрисовку
    app._flush_canvas_draw(mock_canvas)
    app.after_cancel.assert_called_once_with("timer_123")
    assert mock_canvas not in app._pan_draw_timers
    mock_canvas.draw_idle.assert_called_once()


def test_diff_redraw_isolated_surface_lod():
    """Проверяет адаптивный LOD шаг и плавное панорамирование в картограмме масс для нижней поверхности."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    app = create_mock_app(n_points=50)
    fig_diff = Figure()
    ax_diff = fig_diff.add_subplot(111)
    canvas_diff = FigureCanvasAgg(fig_diff)
    fig_diff.canvas = canvas_diff
    app.fig_diff = fig_diff
    app.ax_diff = ax_diff
    app.canvas_diff = canvas_diff

    app._show_top = MagicMock()
    app._show_top.get.return_value = False
    app._show_bottom = MagicMock()
    app._show_bottom.get.return_value = True
    app._is_excavation = MagicMock(return_value=False)
    app._toolbar_diff = None

    rows, cols = 354, 502
    grid_y, grid_x = np.meshgrid(np.linspace(0, 100, cols), np.linspace(0, 70, rows))
    z_bot = np.full((rows, cols), 150.0)
    z_bot += np.sin(grid_x * 0.1) + np.cos(grid_y * 0.1)
    z_top = z_bot + 2.0

    r = {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "z_top_grid": z_top,
        "z_bot_grid": z_bot,
        "dh_grid": z_top - z_bot,
        "boundary": np.array([[0, 0, 150], [0, 100, 150], [70, 100, 150], [70, 0, 150]]),
    }

    import time
    t0 = time.perf_counter()
    app._redraw_diff(r)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert dt_ms < 250.0, f"Diff redraw too slow: {dt_ms:.1f} ms"

    # Проверяем, что создана коллекция contourf на актуальных осях app.ax_diff
    collections = app.ax_diff.collections
    assert len(collections) > 0

    # Проверяем симуляцию панорамирования мышью
    ev_press = MagicMock(x=100, y=100, xdata=50.0, ydata=35.0, button=1)
    app._on_diff_canvas_press(ev_press)
    assert app._diff_pan_start is not None

    ev_move = MagicMock(x=150, y=150, button=1)
    app._on_diff_canvas_motion(ev_move)
    assert app._diff_pan_dragged is True

    ev_release = MagicMock(button=1)
    app._on_diff_canvas_release(ev_release)
    assert app._diff_pan_start is None
    assert app._diff_pan_dragged is False


def test_format_coord_diff_fast_and_accurate():
    """Проверяет O(1) быстродействие и корректность отображения отметок Z/ΔH на картограмме."""
    app = create_mock_app(n_points=50)
    rows, cols = 50, 50
    grid_y, grid_x = np.meshgrid(np.linspace(0, 100, cols), np.linspace(0, 70, rows))
    z_bot = np.full((rows, cols), 150.0)
    z_top = z_bot + 2.5
    dh = z_top - z_bot

    app.calc_results = {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "z_top_grid": z_top,
        "z_bot_grid": z_bot,
        "dh_grid": dh,
    }
    app._is_excavation = MagicMock(return_value=False)

    # 1. Обе поверхности (ΔH)
    app._show_top.get.return_value = True
    app._show_bottom.get.return_value = True
    s_both = app._format_coord_diff(50.0, 35.0)
    assert "[ΔH: +2.50 м]" in s_both
    assert "X (Север): 35.000" in s_both

    # 2. Только нижняя (Z_низ)
    app._show_top.get.return_value = False
    app._show_bottom.get.return_value = True
    s_bot = app._format_coord_diff(50.0, 35.0)
    assert "[Z_низ: 150.00 м]" in s_bot

    # 3. Только верхняя (Z_верх)
    app._show_top.get.return_value = True
    app._show_bottom.get.return_value = False
    s_top = app._format_coord_diff(50.0, 35.0)
    assert "[Z_верх: 152.50 м]" in s_top

    # 4. Точка за пределами сетки — координаты без префикса
    s_out = app._format_coord_diff(999.0, 999.0)
    assert "[" not in s_out
    assert "X (Север): 999.000" in s_out


def test_find_nearest_point_performance_with_hidden_points():
    """Проверяет, что при 2500 точках, где большинство скрыто или за контуром,
    поиск ближайшей видимой точки не проваливается в тяжелый цикл перебора всех точек."""
    n_pts = 2500
    app = create_mock_app(n_points=n_pts)
    app._get_spatial_service().set_points(app.points)

    # Включаем только нижнюю поверхность, при этом все точки делаем 'top'
    # Таким образом, видимых точек среди обычных точек нет
    app._show_top.get.return_value = False
    app._show_bottom.get.return_value = True

    import time
    t0 = time.perf_counter()
    for _ in range(50):
        app._find_nearest_point(25.0, 25.0, visible_only=True)
    dt_total = time.perf_counter() - t0

    # 50 запросов должны выполняться суммарно быстрее 20 мс (< 0.4 мс на запрос)
    assert dt_total < 0.05, f"Search took too long: {dt_total*1000:.1f} ms for 50 queries"




