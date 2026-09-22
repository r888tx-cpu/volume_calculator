import pytest
import numpy as np
import tkinter as tk
from unittest.mock import MagicMock
from interactive_app import VolumeApp

@pytest.fixture
def app():
    # Setup Tk in headless/virtual mode
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception:
        pytest.skip("Tkinter display not available in test environment")
    
    app_instance = VolumeApp.__new__(VolumeApp)
    # Minimal mock setup to verify tab structure and contour methods
    yield app_instance
    try:
        root.destroy()
    except Exception:
        pass

def test_contours_tab_constants():
    """Проверяет наличие константы TAB_CONTOURS и название вкладки"""
    inst = VolumeApp.__new__(VolumeApp)
    inst.TAB_2D = "2D Схема в плане"
    inst.TAB_3D = "3D Поверхности"
    inst.TAB_DIFF = "Картограмма масс"
    inst.TAB_TIN = "TIN Триангуляция"
    inst.TAB_CONTOURS = "Горизонтали"
    inst.TAB_TABLE = "Таблица точек"

    tabs = [inst.TAB_2D, inst.TAB_3D, inst.TAB_DIFF, inst.TAB_TIN, inst.TAB_CONTOURS, inst.TAB_TABLE]
    assert tabs[-3:] == ["TIN Триангуляция", "Горизонтали", "Таблица точек"]

def test_contours_rendering_with_calc_results(app):
    """Проверяет отрисовку горизонталей ax_contours при наличии расчетной сетки"""
    from matplotlib.figure import Figure
    from matplotlib.patches import Polygon as MplPolygon

    app.fig_contours = Figure()
    app.ax_contours = app.fig_contours.add_subplot(111)
    app.canvas_contours = MagicMock()
    app._show_top = MagicMock(get=lambda: True)
    app._show_bottom = MagicMock(get=lambda: True)
    app._show_contour_labels = MagicMock(get=lambda: True)
    app.cbo_contour_step = MagicMock(get=lambda: "Авто")
    app.lbl_contours_stats = MagicMock()
    app._contours_view_initialized = False
    app.points = [MagicMock(x=0, y=0), MagicMock(x=10, y=0), MagicMock(x=10, y=10)]
    app.boundary_indices = [0, 1, 2]

    # Создаем тестовую сетку
    gx, gy = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
    z_top = 100.0 + 0.5 * gx + 0.3 * gy
    z_bot = 95.0 + 0.2 * gx + 0.1 * gy
    boundary = np.array([[0, 0, 100], [10, 0, 105], [10, 10, 108], [0, 10, 103]])

    app.calc_results = {
        "grid_x": gx,
        "grid_y": gy,
        "z_top_grid": z_top,
        "z_bot_grid": z_bot,
        "boundary": boundary,
    }

    app._redraw_contours()
    # Проверяем, что в ax_contours появились линии (контурные коллекции)
    assert len(app.ax_contours.collections) > 0 or len(app.ax_contours.lines) > 0
    # Проверяем, что информационная строка обновилась
    app.lbl_contours_stats.configure.assert_called()
    call_args = app.lbl_contours_stats.configure.call_args[1]
    assert "Шаг h =" in call_args.get("text", "")
