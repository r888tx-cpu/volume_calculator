# -*- coding: utf-8 -*-
"""
    Тесты изоляции классификации поверхностей и типа земляных работ контуром сшивания.
    Проверяет, что при большом количестве точек на участке (например, 2300 точек)
    поверхности и тип работ определяются исключительно по точкам внутри контура,
    а не по удалённым внешним точкам.
"""
import os
import sys
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
from unittest.mock import MagicMock
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geo_parser import parse_line, GeoPoint
import interactive_app


def create_app_with_mound():
    "Constructs app instance with mound data"
    app = interactive_app.VolumeApp.__new__(interactive_app.VolumeApp)
    cord_file = os.path.join(os.path.dirname(__file__), "..", "Projects", "насыпи 2300 точек.txt")
    if not os.path.exists(cord_file):
        pytest.skip("Coordinate file not found")

    with open(cord_file, "r", encoding="utf-8") as f:
        pts = [parse_line(line) for line in f]
    app.points = [p for p in pts if p is not None]

    p_map = {int(p.id.split('_')[0]): i for i, p in enumerate(app.points)}
    seq = [2074] + list(range(2033, 2074))
    app.boundary_indices = [p_map[pid] for pid in seq if pid in p_map]

    app._undo_stack = []
    app._tin_excluded = set()
    app._tin_custom_simplices = None
    app._tin_simplices = None
    app._tin_pts_2d = None
    app.calc_results = None
    app.TAB_2D = "2D Схема в плане"
    app.TAB_3D = "3D Поверхности"
    app.TAB_DIFF = "Картограмма масс"
    app.TAB_TIN = "TIN Триангуляциэ"
    app.TAB_TABLE = "Таблица точек"

    app._show_top = MagicMock()
    app._show_top.get.return_value = True
    app._show_bottom = MagicMock()
    app._show_bottom.get.return_value = True
    app._selected_points = set()

    fig = Figure()
    ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)
    fig.canvas = canvas
    app.fig_2d = fig
    app.ax_2d = ax
    app.canvas_2d = canvas

    app.fig_tin = Figure()
    app.ax_tin = app.fig_tin.add_subplot(111)
    app.canvas_tin = FigureCanvasAgg(app.fig_tin)

    app.ent_res = MagicMock()
    app.ent_res.get.return_value = "0.2"
    app.txt_results = MagicMock()
    app.btn_reset_contour = MagicMock()
    app.lbl_table_stats = MagicMock()
    app.bl_tin_stats = MagicMock()
    app.lbl_table_stats = MagicMock()
    app.lbl_tin_stats = MagicMock()
    app.lbl_report_status = MagicMock()
    app.tree_tin = MagicMock()
    app.tree_tin.get_children.return_value = []
    app._tin_selected_idx = None
    app._tin_patches = []
    app._tin_updating_selection = False
    app._tin_view_initialized = False
    app._tin_pan_dragged = False
    app._tin_pan_start = None
    app._tin_dirty = True
    app._current_file_path = None
    app._current_project_dir = None

    app._position_reset_contour_button = MagicMock()
    app._update_selection_bar = MagicMock()
    app._apply_fig_layout = MagicMock()
    app._update_toolbar_volume_labels = MagicMock()
    app._schedule_auto_save = MagicMock()
    app.tabview = MagicMock()
    app.tabview.get.return_value = app.TAB_2D

    app._is_separate_surfaces = False
    app._boundary_interpolator_cache = None
    app._boundary_path_cache = None
    app._local_surface_threshold = None
    app._split_height_cache = None
    app._work_type_cache = None
    app._point_surfaces_cache = None
    app._boundary_inside_cache = None
    app._boundary_indices_set = None

    return app


def test_work_type_detected_as_fill_for_mound_inside_contour():
    app = create_app_with_mound()
    assert len(app.boundary_indices) == 42
    work_type = app._detect_work_type_from_points()
    assert work_type == "fill", f"Expected fill for mound, got {work_type}"


def test_split_height_calculated_from_inside_points():
    app = create_app_with_mound()
    mean_bound, split_h, work_type = app._get_split_height()
    assert work_type == "fill"
    assert mean_bound > 181.0
    assert split_h > 182.0, f"split_h={split_h} too low"


def test_points_inside_contour_classified_as_top():
    app = create_app_with_mound()
    surfaces = app._ensure_point_surfaces()
    inside_mask = app._ensure_boundary_inside_mask()
    bound_set = set(app.boundary_indices)

    inside_non_bound = [i for i in range(len(app.points)) if inside_mask[i] and i not in bound_set]
    assert len(inside_non_bound) > 100

    top_count = sum(1 for i in inside_non_bound if surfaces[i] == "top")
    assert top_count / len(inside_non_bound) > 0.85, f"top_count={top_count} of {len(inside_non_bound)}"


def test_volume_calculation_fill_positive_cut_zero():
    app = create_app_with_mound()
    app.calculate_volume(silent=True)
    assert app.calc_results is not None
    r = app.calc_results
    assert r["v_fill"] > 1200.0, f"v_fill={r['v_fill']} must be > 1200"
    assert r["v_cut"] == 0.0, f"v_cut={r['v_cut']} must be 0"


def test_empty_contour_fallback():
    app = create_app_with_mound()
    app.boundary_indices = app.boundary_indices[:2]
    app._invalidate_boundary_cache()
    work_type = app._detect_work_type_from_points()
    assert work_type in ("cut", "fill")
    surfaces = app._ensure_point_surfaces()
    assert len(surfaces) == len(app.points)
