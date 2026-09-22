# -*- coding: utf-8 -*-
"""
Тесты для Этапа 1: Ленивый рендеринг (Dirty Flags), устранение каскадных вызовов и дебаунсинг.
"""
import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geo_parser import GeoPoint
import interactive_app


def create_mock_app():
    """Создает экземпляр VolumeApp в обход Tkinter окна для безопасного юнит-тестирования."""
    app = interactive_app.VolumeApp.__new__(interactive_app.VolumeApp)
    app.points = []
    app.boundary_indices = []
    app._undo_stack = []
    app.calc_results = None
    app.TAB_2D = "2D Схема в плане"
    app.TAB_3D = "3D Поверхности"
    app.TAB_DIFF = "Картограмма масс"
    app.TAB_TIN = "TIN Триангуляция"
    app.TAB_CONTOURS = "Горизонтали"
    app.TAB_TABLE = "Таблица точек"

    # Инициализация dirty-флагов и таймеров из __init__
    app._tin_dirty = False
    app._table_dirty = False
    app._3d_dirty = False
    app._diff_dirty = False
    app._2d_dirty = False
    app._contours_dirty = False
    app._auto_save_timer = None
    app._boundary_calc_timer = None
    app._current_file_path = "test.txt"

    # Моки UI методов
    app.ax_2d = MagicMock()
    app.canvas_2d = MagicMock()
    app._redraw_2d = MagicMock()
    app._update_2d_contour_only = MagicMock()
    app._update_2d_selection_only = MagicMock()
    app._update_table = MagicMock()
    app._redraw_tin = MagicMock()
    app._redraw_contours = MagicMock()
    app._redraw_3d = MagicMock()
    app._redraw_diff = MagicMock()
    app._lift_canvas_overlays = MagicMock()
    app._update_toolbar_volume_labels = MagicMock()
    app.save_project_state = MagicMock()
    app._calc_boundary_silent = MagicMock()
    app.after = MagicMock()
    app.after_cancel = MagicMock()

    return app


def test_insert_point_into_boundary_sets_dirty_flags():
    """При встраивании точки в контур выставляются dirty-флаги без синхронного вызова скрытых вкладок."""
    app = create_mock_app()
    app.points = [
        GeoPoint(id="1", x=0.0, y=0.0, h=10.0),
        GeoPoint(id="2", x=10.0, y=0.0, h=10.0),
        GeoPoint(id="3", x=10.0, y=10.0, h=10.0),
        GeoPoint(id="4", x=0.0, y=10.0, h=10.0),
    ]
    app.boundary_indices = [0, 1, 2]
    app._schedule_auto_save = MagicMock()
    app._schedule_boundary_calc = MagicMock()

    # Встраиваем 4-ю точку (индекс 3)
    app._insert_point_into_boundary(3)

    # Проверяем: вызывается быстрое обновление контура, а Table и TIN не вызывались синхронно!
    app._update_2d_contour_only.assert_called_once()
    app._update_table.assert_not_called()
    app._redraw_tin.assert_not_called()

    # Проверяем, что dirty-флаги выставлены
    assert app._table_dirty is True
    assert app._tin_dirty is True

    # Проверяем, что автосохранение и расчет запланированы через дебаунс
    app._schedule_auto_save.assert_called_once()
    app._schedule_boundary_calc.assert_called_once_with(400)


def test_tabview_change_clears_dirty_flags():
    """При переключении на неактивную вкладку с dirty-флагом происходит ее обновление и сброс флага."""
    app = create_mock_app()
    app._table_dirty = True
    app._tin_dirty = True

    # Переключаемся на вкладку таблицы
    app.tabview = MagicMock()
    app.tabview.get.return_value = app.TAB_TABLE

    app._on_tabview_change()
    app._update_table.assert_called_once()
    assert app._table_dirty is False

    # Переключаемся на вкладку TIN
    app.tabview.get.return_value = app.TAB_TIN
    app._on_tabview_change()
    app._redraw_tin.assert_called_once_with(reset_view=False)
    assert app._tin_dirty is False


def test_debounce_boundary_calc_cancels_previous_timer():
    """Быстрая серия кликов отменяет предыдущий таймер авто-расчёта."""
    app = create_mock_app()
    app.after.side_effect = ["timer_1", "timer_2"]

    # Первый клик
    app._schedule_boundary_calc(400)
    assert app._boundary_calc_timer == "timer_1"
    app.after_cancel.assert_not_called()

    # Второй быстрый клик
    app._schedule_boundary_calc(400)
    app.after_cancel.assert_called_once_with("timer_1")
    assert app._boundary_calc_timer == "timer_2"


def test_debounce_auto_save_cancels_previous_timer():
    """Быстрая серия действий отменяет предыдущий таймер сохранения на диск."""
    app = create_mock_app()
    app.after.side_effect = ["save_timer_1", "save_timer_2"]

    app._schedule_auto_save(1500)
    assert app._auto_save_timer == "save_timer_1"
    app.after_cancel.assert_not_called()

    app._schedule_auto_save(1500)
    app.after_cancel.assert_called_once_with("save_timer_1")
    assert app._auto_save_timer == "save_timer_2"


def test_update_all_views_marks_inactive_tabs_dirty():
    """_update_all_views обновляет активную вкладку и маркирует неактивные как dirty."""
    app = create_mock_app()
    app.tabview = MagicMock()
    app.tabview.get.return_value = app.TAB_2D

    app._update_all_views()

    # 2D перерисована
    app._redraw_2d.assert_called_once()
    # Скрытые вкладки не перерисованы, а помечены как dirty
    app._update_table.assert_not_called()
    app._redraw_tin.assert_not_called()
    assert app._table_dirty is True
    assert app._tin_dirty is True
    assert app._3d_dirty is True
    assert app._diff_dirty is True
