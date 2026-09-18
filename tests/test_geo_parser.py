# -*- coding: utf-8 -*-
"""
Тесты для парсера геодезических координат geo_parser.py.
"""
import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geo_parser import parse_line, load_points_from_file, GeoPoint, _get_integer_digits_count


# ═══════════════════════════════════════════════════════════════════
# _get_integer_digits_count
# ═══════════════════════════════════════════════════════════════════

class TestGetIntegerDigitsCount:
    """Тесты для _get_integer_digits_count."""

    def test_six_digits(self):
        assert _get_integer_digits_count("439007.646") == 6

    def test_seven_digits(self):
        assert _get_integer_digits_count("2281305.861") == 7

    def test_three_digits(self):
        assert _get_integer_digits_count("198.123") == 3

    def test_one_digit(self):
        assert _get_integer_digits_count("5.0") == 1

    def test_negative_number(self):
        assert _get_integer_digits_count("-198.5") == 3

    def test_positive_sign(self):
        assert _get_integer_digits_count("+439007.6") == 6

    def test_comma_decimal(self):
        """Запятая как десятичный разделитель."""
        assert _get_integer_digits_count("439007,646") == 6

    def test_integer_no_decimal(self):
        assert _get_integer_digits_count("12345") == 5

    def test_zero(self):
        assert _get_integer_digits_count("0") == 1

    def test_zero_point_zero(self):
        assert _get_integer_digits_count("0.0") == 1


# ═══════════════════════════════════════════════════════════════════
# parse_line — автоматическое распознавание
# ═══════════════════════════════════════════════════════════════════

class TestParseLine:
    """Тесты для parse_line с автоматическим распознаванием."""

    def test_standard_line_space_separated(self):
        """Стандартная строка: ID X(6) Y(7) H(<=3)."""
        line = "1    439007.646  2281305.861  198.123"
        pt = parse_line(line)
        assert pt is not None
        assert pt.id == "1"
        assert np.isclose(pt.x, 439007.646)
        assert np.isclose(pt.y, 2281305.861)
        assert np.isclose(pt.h, 198.123)

    def test_tab_separated(self):
        """Разделение табуляцией."""
        line = "P1\t439007.646\t2281305.861\t198.123"
        pt = parse_line(line)
        assert pt is not None
        assert pt.id == "P1"
        assert np.isclose(pt.x, 439007.646)

    def test_comma_separated(self):
        """Разделение запятой (CSV)."""
        line = "100,439007.646,2281305.861,198.5"
        pt = parse_line(line)
        assert pt is not None
        assert np.isclose(pt.x, 439007.646)
        assert np.isclose(pt.h, 198.5)

    def test_semicolon_separated(self):
        """Разделение точкой с запятой."""
        line = "P1;439007.646;2281305.861;198.5"
        pt = parse_line(line)
        assert pt is not None
        assert np.isclose(pt.y, 2281305.861)

    def test_comment_hash(self):
        """Строка с комментарием # → None."""
        assert parse_line("# это комментарий") is None

    def test_comment_double_slash(self):
        """Строка с комментарием // → None."""
        assert parse_line("// comment line") is None

    def test_empty_line(self):
        """Пустая строка → None."""
        assert parse_line("") is None
        assert parse_line("   ") is None

    def test_too_few_numbers(self):
        """Менее 3 чисел → None."""
        assert parse_line("P1  439007.646") is None

    def test_negative_height(self):
        """Отрицательная высота."""
        line = "1    439007.646  2281305.861  -5.300"
        pt = parse_line(line)
        assert pt is not None
        assert np.isclose(pt.h, -5.3)

    def test_bom_stripped(self):
        """BOM-символ удаляется."""
        line = "\ufeff1    439007.646  2281305.861  198.0"
        pt = parse_line(line)
        assert pt is not None
        assert pt.id == "1"

    def test_composite_id(self):
        """Составной ID из двух текстовых частей."""
        line = "PT C1    439007.646  2281305.861  198.0"
        pt = parse_line(line)
        assert pt is not None
        # ID может быть "PT_C1" или "PT"
        assert "PT" in pt.id

    def test_comma_decimal_separator(self):
        """Запятая как десятичный разделитель (региональный формат)."""
        # Тут запятая используется и как разделитель полей, и как десятичный разделитель
        # Парсер должен справиться через автоматическое определение
        line = "1\t439007,646\t2281305,861\t198,5"
        pt = parse_line(line)
        assert pt is not None
        assert np.isclose(pt.x, 439007.646, atol=0.001)


# ═══════════════════════════════════════════════════════════════════
# parse_line — явный маппинг колонок
# ═══════════════════════════════════════════════════════════════════

class TestParseLineColumnMapping:
    """Тесты parse_line с column_mapping."""

    def test_explicit_mapping(self):
        """Явный маппинг колонок: id=0, x=1, y=2, z=3."""
        line = "P1  100.0  200.0  50.0"
        mapping = {"id": 0, "x": 1, "y": 2, "z": 3}
        pt = parse_line(line, column_mapping=mapping)
        assert pt is not None
        assert pt.id == "P1"
        assert np.isclose(pt.x, 100.0)
        assert np.isclose(pt.y, 200.0)
        assert np.isclose(pt.h, 50.0)

    def test_reordered_mapping(self):
        """Маппинг с нестандартным порядком: y=1, x=2."""
        line = "P1  200.0  100.0  50.0"
        mapping = {"id": 0, "y": 1, "x": 2, "z": 3}
        pt = parse_line(line, column_mapping=mapping)
        assert pt is not None
        assert np.isclose(pt.x, 100.0)
        assert np.isclose(pt.y, 200.0)


# ═══════════════════════════════════════════════════════════════════
# load_points_from_file
# ═══════════════════════════════════════════════════════════════════

class TestLoadPointsFromFile:
    """Тесты для load_points_from_file."""

    def test_basic_file(self, tmp_path):
        """Файл с 3 точками."""
        content = (
            "1    439000.000  2281300.000  198.000\n"
            "2    439010.000  2281300.000  198.500\n"
            "3    439005.000  2281305.000  201.000\n"
        )
        filepath = tmp_path / "test.txt"
        filepath.write_text(content, encoding="utf-8")

        points = load_points_from_file(str(filepath))
        assert len(points) == 3
        assert all(isinstance(p, GeoPoint) for p in points)

    def test_with_comments_and_blanks(self, tmp_path):
        """Файл с комментариями и пустыми строками — они пропускаются."""
        content = (
            "# Заголовок\n"
            "\n"
            "1    439000.000  2281300.000  198.000\n"
            "// комментарий\n"
            "2    439010.000  2281300.000  198.500\n"
            "\n"
        )
        filepath = tmp_path / "test.txt"
        filepath.write_text(content, encoding="utf-8")

        points = load_points_from_file(str(filepath))
        assert len(points) == 2

    def test_utf8_bom(self, tmp_path):
        """Файл с BOM (utf-8-sig) корректно читается."""
        content = "\ufeff1    439000.000  2281300.000  198.000\n"
        filepath = tmp_path / "test.txt"
        filepath.write_bytes(content.encode("utf-8-sig"))

        points = load_points_from_file(str(filepath))
        assert len(points) == 1

    def test_empty_file(self, tmp_path):
        """Пустой файл → пустой список."""
        filepath = tmp_path / "empty.txt"
        filepath.write_text("", encoding="utf-8")

        points = load_points_from_file(str(filepath))
        assert len(points) == 0

    def test_with_column_mapping(self, tmp_path):
        """Файл с явным маппингом колонок."""
        content = (
            "P1  100.0  200.0  50.0\n"
            "P2  110.0  210.0  55.0\n"
        )
        filepath = tmp_path / "test.txt"
        filepath.write_text(content, encoding="utf-8")

        mapping = {"id": 0, "x": 1, "y": 2, "z": 3}
        points = load_points_from_file(str(filepath), column_mapping=mapping)
        assert len(points) == 2
        assert np.isclose(points[0].x, 100.0)


# Нужен numpy для np.isclose
import numpy as np
