# -*- coding: utf-8 -*-
import os
import pytest
import numpy as np
from unittest.mock import MagicMock

from geo_parser import GeoPoint, load_points_from_file
from interactive_app import VolumeApp


def get_foundation_points():
    pts_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Projects", "Фундамент 4.txt")
    if os.path.exists(pts_file):
        return load_points_from_file(pts_file)
    # Fallback to hardcoded 8 points from "Фундамент 4.txt"
    raw = [
        ("9", 2281322.9057, 439011.9227, 197.2723),
        ("10", 2281322.0151, 439011.6384, 197.2682),
        ("11", 2281322.3478, 439010.7580, 197.2727),
        ("12", 2281323.2123, 439011.0605, 197.2725),
        ("13", 2281322.8666, 439011.8907, 199.1997),
        ("14", 2281322.0321, 439011.6136, 199.2093),
        ("15", 2281322.3439, 439010.7954, 199.2062),
        ("16", 2281323.1697, 439011.0851, 199.2016),
    ]
    return [GeoPoint(id=id_v, x=x, y=y, h=h, surface_type="auto") for id_v, x, y, h in raw]


def create_foundation_mock_app():
    app = VolumeApp.__new__(VolumeApp)
    app.points = get_foundation_points()
    app.boundary_indices = []
    app._selected_points = set()
    app._undo_stack = []
    app._tin_excluded = set()
    app._tin_custom_simplices = None
    app._tin_simplices = None
    app._tin_dirty = True
    app._tin_selected_idx = None
    app._is_separate_surfaces = False

    app.ent_res = MagicMock()
    app.ent_res.get.return_value = "0.05"
    app.lbl_tin_stats = MagicMock()
    app.canvas_tin = MagicMock()
    app.ax_tin = MagicMock()
    app.fig_tin = MagicMock()
    app.tree_tin = MagicMock()
    app.tree_tin.get_children.return_value = []

    app._show_top = MagicMock()
    app._show_top.get.return_value = True
    app._show_bottom = MagicMock()
    app._show_bottom.get.return_value = True

    app.calc_results = None
    app._display_results = MagicMock()
    app._update_all_views = MagicMock()
    app._schedule_auto_save = MagicMock()
    app._auto_save_report = MagicMock()
    app._update_toolbar_volume_labels = MagicMock()
    app._schedule_boundary_calc = MagicMock()
    app._tin_view_initialized = False
    app._suppress_tab_switch = True
    app._fit_tin_view = MagicMock()
    app._current_file_path = None

    app._invalidate_boundary_cache()
    return app


def test_heuristic_detects_foundation():
    app = create_foundation_mock_app()
    assert app._detect_two_surfaces_heuristic() is True


def test_heuristic_rejects_single_surface_mound():
    app = VolumeApp.__new__(VolumeApp)
    # 8 points representing an embankment mound
    app.points = [
        GeoPoint(id="1", x=0.0, y=0.0, h=100.0, surface_type="auto"),
        GeoPoint(id="2", x=10.0, y=0.0, h=100.0, surface_type="auto"),
        GeoPoint(id="3", x=10.0, y=10.0, h=100.0, surface_type="auto"),
        GeoPoint(id="4", x=0.0, y=10.0, h=100.0, surface_type="auto"),
        GeoPoint(id="5", x=5.0, y=5.0, h=104.0, surface_type="auto"),
        GeoPoint(id="6", x=4.0, y=5.0, h=103.5, surface_type="auto"),
        GeoPoint(id="7", x=6.0, y=5.0, h=103.8, surface_type="auto"),
        GeoPoint(id="8", x=5.0, y=6.0, h=103.2, surface_type="auto"),
    ]
    app._invalidate_boundary_cache()
    assert app._detect_two_surfaces_heuristic() is False


def test_heuristic_detects_thin_slab_5cm():
    app = VolumeApp.__new__(VolumeApp)
    app.points = [
        GeoPoint(id="1", x=0.0, y=0.0, h=100.00, surface_type="auto"),
        GeoPoint(id="2", x=2.0, y=0.0, h=100.00, surface_type="auto"),
        GeoPoint(id="3", x=2.0, y=2.0, h=100.00, surface_type="auto"),
        GeoPoint(id="4", x=0.0, y=2.0, h=100.00, surface_type="auto"),
        GeoPoint(id="5", x=0.01, y=0.01, h=100.05, surface_type="auto"),
        GeoPoint(id="6", x=1.99, y=0.02, h=100.05, surface_type="auto"),
        GeoPoint(id="7", x=2.01, y=1.99, h=100.05, surface_type="auto"),
        GeoPoint(id="8", x=0.02, y=2.01, h=100.05, surface_type="auto"),
    ]
    app._invalidate_boundary_cache()
    assert app._detect_two_surfaces_heuristic() is True


def test_heuristic_rejects_large_single_surface_with_sparse_slopes():
    # Large survey (e.g. 100 points) where a few points happen to have close horizontal distance and > 5cm dH
    # but they do NOT make up >= 25% of total points.
    app = VolumeApp.__new__(VolumeApp)
    pts = []
    # Grid 10x10
    idx = 1
    for r in range(10):
        for c in range(10):
            # smooth slope with continuous elevations
            pts.append(GeoPoint(id=str(idx), x=float(c * 2.0), y=float(r * 2.0), h=100.0 + (r + c) * 0.1, surface_type="auto"))
            idx += 1
    app.points = pts
    app._invalidate_boundary_cache()
    assert app._detect_two_surfaces_heuristic() is False


def test_auto_classify_foundation():
    app = create_foundation_mock_app()
    app._auto_classify_initial()

    assert app._is_separate_surfaces is True
    assert app._is_two_surfaces() is True

    # 4 top points and 4 bottom points
    top_pts = [p for p in app.points if p.surface_type == "top"]
    bot_pts = [p for p in app.points if p.surface_type == "bottom"]
    assert len(top_pts) == 4
    assert len(bot_pts) == 4
    assert all(p.h > 198.0 for p in top_pts)
    assert all(p.h < 198.0 for p in bot_pts)


def test_foundation_volume_calculation():
    app = create_foundation_mock_app()
    app._auto_classify_initial()

    # Calculate volume
    app.calculate_volume(silent=True)
    res = app.calc_results
    assert res is not None

    # Volume must be ~1.658 m3, NOT 0.552 m3 (pyramid collapse)
    assert pytest.approx(res["v_net"], rel=0.03) == 1.658
    assert res["v_net"] > 1.50
    assert pytest.approx(res["area_2d"], rel=0.05) == 0.858
    assert pytest.approx(res["avg_thickness"], rel=0.05) == 1.932


def test_foundation_tin_top_surface():
    app = create_foundation_mock_app()
    app._auto_classify_initial()

    # Mock ax_tin add_patch to capture polygons
    patches = []
    app.ax_tin.add_patch = lambda patch: patches.append(patch)
    app.ax_tin.plot = MagicMock()
    app.ax_tin.text = MagicMock()

    app._redraw_tin(reset_view=True)

    # In two surfaces mode, top TIN consists of exactly 2 triangles covering the top plane of the foundation
    assert app._tin_simplices is not None
    assert len(app._tin_simplices) == 2

    # Check vertices of triangles are all from the top surface
    for tri in app._tin_simplices:
        for v in tri:
            assert app.points[v].surface_type == "top"
            assert app.points[v].h > 198.0
