# -*- coding: utf-8 -*-
import pytest
from geo_parser import GeoPoint
from interactive_app import VolumeApp


def create_test_app():
    app = VolumeApp.__new__(VolumeApp)
    app.points = [
        GeoPoint(id="1", x=10.0, y=10.0, h=100.0, surface_type="auto"),
        GeoPoint(id="2", x=20.0, y=10.0, h=101.0, surface_type="top"),
        GeoPoint(id="3", x=20.0, y=20.0, h=99.0, surface_type="bottom"),
        GeoPoint(id="4", x=10.0, y=20.0, h=100.5, surface_type="boundary"),
    ]
    app.boundary_indices = [0, 1, 2, 3]
    app._selected_points = set()
    app._undo_stack = []
    app._tin_excluded = set()
    app._tin_custom_simplices = None
    app._current_file_path = None
    app.calc_results = None

    app._update_selection_bar = lambda: None
    app._update_2d_selection_only = lambda: None
    app._update_all_views = lambda reset_view=False: None
    app._schedule_auto_save = lambda: None
    app._schedule_boundary_calc = lambda delay: None
    app._invalidate_boundary_cache = lambda: None
    app._invalidate_points_spatial_index = lambda: None

    return app


class TestPointContextMenuActions:
    def test_assign_point_surface_top(self):
        app = create_test_app()
        assert app.points[0].surface_type == "auto"
        assert 0 in app.boundary_indices

        app._assign_point_surface(0, "top")

        assert app.points[0].surface_type == "top"
        assert 0 not in app.boundary_indices
        assert len(app._undo_stack) == 1
        assert app._undo_stack[0][0] == "batch_assign"

    def test_assign_point_surface_bottom(self):
        app = create_test_app()
        app._assign_point_surface(1, "bottom")

        assert app.points[1].surface_type == "bottom"
        assert 1 not in app.boundary_indices

    def test_batch_assign_surface_via_context_menu(self):
        app = create_test_app()
        app._selected_points = {0, 1}

        app._assign_point_surface(0, "bottom")

        assert app.points[0].surface_type == "bottom"
        assert app.points[1].surface_type == "bottom"
        assert len(app._selected_points) == 0

    def test_delete_point_single(self, monkeypatch):
        app = create_test_app()
        import tkinter.messagebox as mb
        monkeypatch.setattr(mb, "askyesno", lambda *args, **kwargs: True)

        assert len(app.points) == 4
        app._delete_point(1)

        assert len(app.points) == 3
        assert app.boundary_indices == [0, 1, 2]

    def test_delete_point_cancelled(self, monkeypatch):
        app = create_test_app()
        import tkinter.messagebox as mb
        monkeypatch.setattr(mb, "askyesno", lambda *args, **kwargs: False)

        app._delete_point(1)
        assert len(app.points) == 4
        assert len(app.boundary_indices) == 4

    def test_delete_selected_points_batch(self, monkeypatch):
        app = create_test_app()
        import tkinter.messagebox as mb
        monkeypatch.setattr(mb, "askyesno", lambda *args, **kwargs: True)

        app._selected_points = {0, 2}
        app._delete_point(-1)

        assert len(app.points) == 2
        assert len(app._selected_points) == 0

    def test_delete_selected_points_cancelled(self, monkeypatch):
        app = create_test_app()
        import tkinter.messagebox as mb
        monkeypatch.setattr(mb, "askyesno", lambda *args, **kwargs: False)

        app._selected_points = {0, 2}
        app._delete_point(-1)

        assert len(app.points) == 4
        assert len(app._selected_points) == 2


def test_theme_toggle_logic():
    app = create_test_app()
    import customtkinter as ctk
    old_mode = ctk.get_appearance_mode()
    app._apply_ttk_dark_style = lambda: None
    app.after = lambda ms, func: func()
    app._toggle_app_theme()
    new_mode = ctk.get_appearance_mode()
    assert new_mode != old_mode
    app._toggle_app_theme()
    assert ctk.get_appearance_mode() == old_mode


class TestSurfaceAssignmentModeAndSafety:
    def test_assign_point_surface_toggle(self):
        app = create_test_app()
        # Initial: point 0 is auto
        assert app.points[0].surface_type == "auto"

        # 1. Assign to top with toggle_same=True
        app._assign_point_surface(0, "top", toggle_same=True)
        assert app.points[0].surface_type == "top"

        # 2. Assign again to top with toggle_same=True -> should toggle back to auto
        app._assign_point_surface(0, "top", toggle_same=True)
        assert app.points[0].surface_type == "auto"

        # 3. Assign to auto directly (via context menu)
        app.points[0].surface_type = "bottom"
        app._assign_point_surface(0, "auto", toggle_same=False)
        assert app.points[0].surface_type == "auto"

    def test_active_session_surface_assignment_and_volume(self):
        app = VolumeApp()
        app.withdraw()
        try:
            # Setup simple project with 4 boundary points and 1 inner point
            app.points = [
                GeoPoint(id="1", x=0.0, y=0.0, h=10.0, surface_type="auto"),
                GeoPoint(id="2", x=10.0, y=0.0, h=10.0, surface_type="auto"),
                GeoPoint(id="3", x=10.0, y=10.0, h=10.0, surface_type="auto"),
                GeoPoint(id="4", x=0.0, y=10.0, h=10.0, surface_type="auto"),
                GeoPoint(id="5", x=5.0, y=5.0, h=5.0, surface_type="auto"),
            ]
            app.boundary_indices = [0, 1, 2, 3]
            app._invalidate_boundary_cache()
            app._update_all_views()

            # Initially point 5 (index 4) is auto (detected as bottom in excavation)
            surfs = app._ensure_point_surfaces()
            assert surfs[4] == "bottom"
            assert app.points[4].surface_type == "auto"

            # Switch mode to 3. Назначение: Верхняя поверхность
            app.cbo_mode.set("3. Назначение: Верхняя поверхность")
            app._on_cbo_mode_selected_cmd("3. Назначение: Верхняя поверхность")
            assert app.current_mode.get() == "assign_top"

            class MockEv:
                def __init__(self, x, y, xdata, ydata, button=1, dblclick=False):
                    self.x, self.y = x, y
                    self.xdata, self.ydata = xdata, ydata
                    self.button = button
                    self.dblclick = dblclick

            # 1. Double-click in assign_top mode -> MUST NOT add point 5 to boundary
            dbl_ev = MockEv(200, 200, 5.0, 5.0, button=1, dblclick=True)
            app._on_canvas_press(dbl_ev)
            assert app.boundary_indices == [0, 1, 2, 3]
            assert 4 not in app.boundary_indices

            # 2. Single click on point 5 in assign_top mode -> MUST immediately update surface in current session
            press_ev = MockEv(200, 200, 5.0, 5.0, button=1, dblclick=False)
            app._on_canvas_press(press_ev)
            release_ev = MockEv(200, 200, 5.0, 5.0, button=1, dblclick=False)
            app._on_canvas_release(release_ev)

            assert app.points[4].surface_type == "top"
            # Crucial: verify that cached surfaces are invalidated and updated immediately
            surfs_after = app._ensure_point_surfaces()
            assert surfs_after[4] == "top"

            # 3. Verify _is_two_surfaces remains False (since project has contour)
            assert not app._is_two_surfaces()

            # 4. Verify volume calculation succeeds
            app.calculate_volume(silent=True)
            assert app.calc_results is not None
            assert "error" not in app.calc_results or app.calc_results.get("error") is None

            # 5. Undo action -> point 5 restored to auto
            app._undo_last_action()
            assert app.points[4].surface_type == "auto"
            surfs_undone = app._ensure_point_surfaces()
            assert surfs_undone[4] == "bottom"

        finally:
            app.destroy()


