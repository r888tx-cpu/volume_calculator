# -*- coding: utf-8 -*-
"""
Тесты для автоматического открытия окна переопределения координат
при импорте файлов с нестандартными координатами (не соответствующих правилу 6/7).
"""
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
import tkinter.messagebox as mb

from geo_parser import GeoPoint
from interactive_app import VolumeApp


def create_mock_app():
    app = VolumeApp.__new__(VolumeApp)
    app._current_file_path = None
    app._current_project_dir = None
    app._column_mapping = None
    app.points = []
    app.calc_results = None
    app._is_separate_surfaces = False
    app.boundary_indices = []
    app._selected_points = set()
    app._undo_stack = []
    app._tin_excluded = set()
    app._tin_custom_simplices = None

    app.lbl_file_info = MagicMock()
    app._invalidate_boundary_cache = MagicMock()
    app._auto_classify_initial = MagicMock()
    app._update_all_views = MagicMock()
    app.save_project_state = MagicMock()
    app._get_project_file_path = lambda fp: fp + ".volproj"
    app.after = MagicMock()
    app.attributes = MagicMock()
    app.tk = MagicMock()

    return app


class TestFileHasDataLines:
    def test_empty_file(self):
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
            f.write("   \n\n\t\n")
            path = f.name
        try:
            assert VolumeApp._file_has_data_lines(path) is False
        finally:
            os.remove(path)

    def test_comments_only_file(self):
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
            f.write("# comment 1\n// comment 2\n")
            path = f.name
        try:
            assert VolumeApp._file_has_data_lines(path) is False
        finally:
            os.remove(path)

    def test_data_file(self):
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
            f.write("# Header\n1, 3299.1, 1616.2, 120.5\n")
            path = f.name
        try:
            assert VolumeApp._file_has_data_lines(path) is True
        finally:
            os.remove(path)

    def test_nonexistent_file(self):
        assert VolumeApp._file_has_data_lines("nonexistent_file_xyz_123.txt") is False


class TestLoadFileAutoRemap:
    def test_standard_6_7_file_loads_without_dialog(self, monkeypatch):
        app = create_mock_app()
        dialog_mock = MagicMock()
        monkeypatch.setattr("interactive_app.CoordinateRemapDialog", dialog_mock)
        warn_mock = MagicMock()
        monkeypatch.setattr(mb, "showwarning", warn_mock)

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".txt") as f:
            f.write("1, 439007.646, 2281305.861, 198.123\n2, 439010.000, 2281310.000, 198.500\n")
            path = f.name

        try:
            app.load_file(path)
            assert len(app.points) == 2
            dialog_mock.assert_not_called()
            warn_mock.assert_not_called()
        finally:
            os.remove(path)

    def test_non_6_7_file_opens_remap_dialog_immediately(self, monkeypatch):
        app = create_mock_app()

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".txt") as f:
            # Координаты не 6/7 знаков (например 3299, 1616)
            f.write("1, 3299.100, 1616.200, 120.500\n2, 3305.000, 1620.000, 121.000\n")
            path = f.name

        def fake_dialog(parent, filepath, initial_mapping=None):
            dlg = MagicMock()
            dlg.result_mapping = {"id": 0, "x": 1, "y": 2, "z": 3}
            return dlg

        monkeypatch.setattr("interactive_app.CoordinateRemapDialog", fake_dialog)
        warn_mock = MagicMock()
        monkeypatch.setattr(mb, "showwarning", warn_mock)

        try:
            app.load_file(path)
            # Должно загрузиться после применения маппинга без предварительного предупреждения!
            assert len(app.points) == 2
            assert app.points[0].x == 3299.1
            assert app.points[0].y == 1616.2
            assert app.points[0].h == 120.5
            warn_mock.assert_not_called()
        finally:
            os.remove(path)

    def test_non_6_7_file_cancelled_dialog_clean_abort(self, monkeypatch):
        app = create_mock_app()

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".txt") as f:
            f.write("1, 3299.100, 1616.200, 120.500\n")
            path = f.name

        def fake_dialog_cancel(parent, filepath, initial_mapping=None):
            dlg = MagicMock()
            dlg.result_mapping = None  # Пользователь нажал "Отмена"
            return dlg

        monkeypatch.setattr("interactive_app.CoordinateRemapDialog", fake_dialog_cancel)
        warn_mock = MagicMock()
        monkeypatch.setattr(mb, "showwarning", warn_mock)

        try:
            app.load_file(path)
            assert len(app.points) == 0
            # Не должно быть сообщения об ошибке при отмене диалога
            warn_mock.assert_not_called()
        finally:
            os.remove(path)

    def test_empty_file_warns_empty(self, monkeypatch):
        app = create_mock_app()
        dialog_mock = MagicMock()
        monkeypatch.setattr("interactive_app.CoordinateRemapDialog", dialog_mock)
        warn_mock = MagicMock()
        monkeypatch.setattr(mb, "showwarning", warn_mock)

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".txt") as f:
            f.write("   \n\n")
            path = f.name

        try:
            app.load_file(path)
            assert len(app.points) == 0
            dialog_mock.assert_not_called()
            warn_mock.assert_called_once()
            assert "пуст или не содержит данных" in warn_mock.call_args[0][1]
        finally:
            os.remove(path)


class TestImportTwoSurfacesAutoRemap:
    def test_import_two_surfaces_with_remap(self, monkeypatch, tmp_path):
        app = create_mock_app()
        app.cbo_projects = MagicMock()
        app._create_project_folder = lambda name: (name, str(tmp_path / "proj"))
        app._scan_saved_projects = MagicMock()

        f_top = tmp_path / "top.txt"
        f_top.write_text("1, 3299.1, 1616.2, 125.0\n2, 3300.0, 1620.0, 125.5\n", encoding="utf-8")

        f_bot = tmp_path / "bot.txt"
        f_bot.write_text("1, 3299.1, 1616.2, 120.0\n2, 3300.0, 1620.0, 120.5\n", encoding="utf-8")

        from tkinter import filedialog
        file_choices = [str(f_top), str(f_bot)]
        monkeypatch.setattr(filedialog, "askopenfilename", lambda **kwargs: file_choices.pop(0))

        def fake_dialog(parent, filepath, initial_mapping=None):
            dlg = MagicMock()
            dlg.result_mapping = {"id": 0, "x": 1, "y": 2, "z": 3}
            return dlg

        monkeypatch.setattr("interactive_app.CoordinateRemapDialog", fake_dialog)
        err_mock = MagicMock()
        monkeypatch.setattr(mb, "showerror", err_mock)

        app._open_separate_files_dialog()

        err_mock.assert_not_called()
        assert len(app.points) == 4
        assert any(p.surface_type == "top" for p in app.points)
        assert any(p.surface_type == "bottom" for p in app.points)

