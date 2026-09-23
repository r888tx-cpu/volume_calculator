# -*- coding: utf-8 -*-
import os
import tempfile
import pytest
import numpy as np
import ezdxf

from dxf_importer import inspect_dxf_layers, extract_dxf_geometry, is_dxf_available
from volume_engine import VolumeCalculator


def _create_sample_dxf(filepath: str):
    """Создаёт тестовый чертеж DXF со слоями верха, низа и границы."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Слои
    doc.layers.add("ВЕРХ", color=1)
    doc.layers.add("НИЗ", color=5)
    doc.layers.add("КОНТУР", color=3)
    doc.layers.add("ТЕКСТЫ", color=7)

    # 1. Точки верха на слое "ВЕРХ" (Z = 105.0)
    for x, y in [(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0), (20.0, 30.0)]:
        msp.add_point((x, y, 105.0), dxfattribs={"layer": "ВЕРХ"})

    # 2. Блоки низа на слое "НИЗ" (Z = 100.0) с атрибутами NUM и H
    # Создаем определение блока
    blk = doc.blocks.new(name="PNT_BLOCK")
    blk.add_attdef(tag="NUM", text="0")
    blk.add_attdef(tag="H", text="0.0")

    bot_coords = [(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0), (20.0, 30.0)]
    for i, (x, y) in enumerate(bot_coords):
        block_ref = msp.add_blockref("PNT_BLOCK", (x, y, 100.0), dxfattribs={"layer": "НИЗ"})
        block_ref.add_attrib("NUM", f"B{i+1}")
        block_ref.add_attrib("H", "100.0")

    # 3. Замкнутая полилиния границы (20x20 м, площадь 400 м²)
    boundary_pts = [(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)]
    poly = msp.add_lwpolyline(boundary_pts, close=True, dxfattribs={"layer": "КОНТУР"})

    # 4. Текстовые отметки
    msp.add_text("105.50", dxfattribs={"layer": "ТЕКСТЫ", "insert": (15.0, 25.0, 0.0)})

    doc.saveas(filepath)


def test_dxf_available():
    assert is_dxf_available() is True


def test_inspect_dxf_layers():
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tf:
        dxf_path = tf.name

    try:
        _create_sample_dxf(dxf_path)

        ok, info = inspect_dxf_layers(dxf_path)
        assert ok is True
        assert isinstance(info, dict)

        layers = info["layers"]
        assert "ВЕРХ" in layers
        assert layers["ВЕРХ"]["points_count"] == 5

        assert "НИЗ" in layers
        assert layers["НИЗ"]["inserts_count"] == 5

        assert "КОНТУР" in layers
        assert layers["КОНТУР"]["closed_polylines_count"] == 1

        assert "ТЕКСТЫ" in layers
        assert layers["ТЕКСТЫ"]["texts_count"] == 1

        # Проверка предложений слоев
        assert info["suggested_top_layer"] == "ВЕРХ"
        assert info["suggested_bottom_layer"] == "НИЗ"
        assert info["suggested_boundary_layer"] == "КОНТУР"

        # Проверка обнаруженной границы
        closed = info["closed_polylines"]
        assert len(closed) == 1
        assert abs(closed[0]["area"] - 400.0) < 1e-4

    finally:
        if os.path.exists(dxf_path):
            os.remove(dxf_path)


def test_extract_dxf_geometry_two_surfaces():
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tf:
        dxf_path = tf.name

    try:
        _create_sample_dxf(dxf_path)

        ok, pts, b_arr, is_two_s, msg = extract_dxf_geometry(
            filepath=dxf_path,
            top_layers=["ВЕРХ"],
            bottom_layers=["НИЗ"],
            boundary_layer="КОНТУР",
            coord_swap=True,  # X_cad (10..30) -> Y_geo, Y_cad (20..40) -> X_geo
            include_points=True,
            include_inserts=True,
        )

        assert ok is True
        assert len(pts) == 10
        assert is_two_s is True
        assert b_arr is not None
        assert len(b_arr) == 4

        # Проверяем разделение по поверхностям
        top_pts = [p for p in pts if p.surface_type == "top"]
        bot_pts = [p for p in pts if p.surface_type == "bottom"]
        assert len(top_pts) == 5
        assert len(bot_pts) == 5

        # Проверяем высоты
        assert all(abs(p.h - 105.0) < 1e-3 for p in top_pts)
        assert all(abs(p.h - 100.0) < 1e-3 for p in bot_pts)

        # Проверяем геодезическую смену осей (ГОСТ):
        # CAD X=10, Y=20 -> Geo X=20 (Север), Y=10 (Восток)
        assert any(abs(p.x - 20.0) < 1e-3 and abs(p.y - 10.0) < 1e-3 for p in top_pts)

        # Проверяем расчет объема
        top_arr = np.array([[p.x, p.y, p.h] for p in top_pts])
        bot_arr = np.array([[p.x, p.y, p.h] for p in bot_pts])
        calc = VolumeCalculator(top_arr, bot_arr, boundary_points=b_arr, grid_resolution=1.0, work_type="grading")
        res = calc.calculate()

        assert "error" not in res
        # S = 400 м², H = 5.0 м, V = 2000 м³
        assert abs(res["area_2d"] - 400.0) < 2.0
        assert abs(res["v_fill"] - 2000.0) < 25.0

    finally:
        if os.path.exists(dxf_path):
            os.remove(dxf_path)


def test_extract_dxf_geometry_single_surface_with_texts():
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tf:
        dxf_path = tf.name

    try:
        _create_sample_dxf(dxf_path)

        ok, pts, b_arr, is_two_s, msg = extract_dxf_geometry(
            filepath=dxf_path,
            top_layers=None,
            bottom_layers=None,
            boundary_layer=None,
            coord_swap=False,
            include_points=True,
            include_inserts=False,
            include_texts=True,
        )

        assert ok is True
        assert is_two_s is False
        assert b_arr is None
        # 5 points + 1 text
        assert len(pts) == 6

        # Текстовая отметка должна быть прочитана как 105.50
        text_pt = [p for p in pts if p.id.startswith("T")]
        assert len(text_pt) == 1
        assert abs(text_pt[0].h - 105.50) < 1e-3
        # Без смены осей: X_cad=15, Y_cad=25 -> X_geo=15, Y_geo=25
        assert abs(text_pt[0].x - 15.0) < 1e-3
        assert abs(text_pt[0].y - 25.0) < 1e-3

    finally:
        if os.path.exists(dxf_path):
            os.remove(dxf_path)


def test_dxf_import_invalid_file():
    ok, info = inspect_dxf_layers("non_existent_file_12345.dxf")
    assert ok is False
    assert "не найден" in info.lower()

    ok, pts, _, _, msg = extract_dxf_geometry("non_existent_file_12345.dxf")
    assert ok is False
    assert len(pts) == 0
