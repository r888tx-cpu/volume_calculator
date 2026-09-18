# -*- coding: utf-8 -*-
"""
Тесты для CLI-режима volume_calc.py.
"""
import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestRunCli:
    """Тесты для run_cli()."""

    def test_valid_file_produces_report(self, tmp_path):
        """CLI-расчёт создаёт файл отчёта."""
        from volume_calc import run_cli

        content = (
            "1    439000.000  2281300.000  198.000\n"
            "2    439010.000  2281300.000  198.000\n"
            "3    439010.000  2281310.000  198.000\n"
            "4    439000.000  2281310.000  198.000\n"
            "5    439005.000  2281305.000  201.000\n"
        )
        filepath = tmp_path / "survey.txt"
        filepath.write_text(content, encoding="utf-8")

        run_cli(str(filepath))

        report_path = tmp_path / "survey_volume_report.txt"
        assert report_path.exists(), "Файл отчёта не создан"
        report_text = report_path.read_text(encoding="utf-8")
        assert "РЕЗУЛЬТАТЫ" in report_text or "Volume" in report_text

    def test_empty_file_prints_error(self, tmp_path, capsys):
        """Пустой файл → сообщение об ошибке, без крэша."""
        from volume_calc import run_cli

        filepath = tmp_path / "empty.txt"
        filepath.write_text("", encoding="utf-8")

        run_cli(str(filepath))
        captured = capsys.readouterr()
        assert "Ошибка" in captured.out or "не найден" in captured.out

    def test_file_with_only_comments(self, tmp_path, capsys):
        """Файл только с комментариями → сообщение об ошибке."""
        from volume_calc import run_cli

        content = "# comment 1\n# comment 2\n// another comment\n"
        filepath = tmp_path / "comments.txt"
        filepath.write_text(content, encoding="utf-8")

        run_cli(str(filepath))
        captured = capsys.readouterr()
        assert "Ошибка" in captured.out or "не найден" in captured.out
