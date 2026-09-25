# -*- coding: utf-8 -*-
import os
import tempfile
import pytest
import numpy as np
import ezdxf

from geo_parser import GeoPoint
from volume_engine import VolumeCalculator
from dxf_exporter import export_cartogram_dxf, choose_auto_grid_step, DEFAULT_DXF_LAYERS


def test_choose_auto_grid_step():
    assert choose_auto_grid_step(10.0, 20.0) == 2.0
    assert choose_auto_grid_step(50.0, 40.0) == 5.0
    assert choose_auto_grid_step(120.0, 80.0) == 10.0
    assert choose_auto_grid_step(250.0, 200.0) == 20.0
    assert choose_auto_grid_step(500.0, 500.0) == 50.0


def test_dxf_export_basic():
    # 20x20m flat slab with 1.5m thickness
    top_pts = np.array([
        [0.0, 0.0, 101.5],
        [20.0, 0.0, 101.5],
        [20.0, 20.0, 101.5],
        [0.0, 20.0, 101.5],
        [10.0, 10.0, 101.5]
    ])
    bot_pts = np.array([
        [0.0, 0.0, 100.0],
        [20.0, 0.0, 100.0],
        [20.0, 20.0, 100.0],
        [0.0, 20.0, 100.0],
        [10.0, 10.0, 100.0]
    ])
    boundary = np.array([
        [0.0, 0.0, 100.0],
        [20.0, 0.0, 100.0],
        [20.0, 20.0, 100.0],
        [0.0, 20.0, 100.0]
    ])

    calc = VolumeCalculator(top_pts, bot_pts, boundary_points=boundary, grid_resolution=0.5, work_type="grading")
    res = calc.calculate()
    assert "error" not in res

    geo_points = [
        GeoPoint(id="T1", x=0.0, y=0.0, h=101.5, surface_type="top"),
        GeoPoint(id="T2", x=20.0, y=20.0, h=101.5, surface_type="top"),
        GeoPoint(id="B1", x=0.0, y=0.0, h=100.0, surface_type="bottom"),
        GeoPoint(id="B2", x=20.0, y=20.0, h=100.0, surface_type="bottom"),
    ]

    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tf:
        out_path = tf.name

    try:
        ok, msg = export_cartogram_dxf(
            calc_results=res,
            points=geo_points,
            boundary_indices=[0, 1, 2, 3],
            output_path=out_path,
            grid_step=5.0,
            text_height=0.4,
            include_grid=True,
            include_node_elevations=True,
            include_cell_volumes=True,
            include_zero_line=True,
            include_boundary=True,
            include_survey_points=True,
            include_balance_table=True,
        )
        assert ok is True
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 1000

        # Читаем через ezdxf и проверяем структуру
        doc = ezdxf.readfile(out_path)
        assert doc.dxfversion == "AC1024"  # R2010

        # Проверяем слои
        layer_names = set(layer.dxf.name for layer in doc.layers)
        for expected in ["0_ГРАНИЦА_РАБОТ", "0_СЕТКА_КАРТОГРАММЫ", "ОТМЕТКИ_КРАСНЫЕ", "ОТМЕТКИ_ЧЕРНЫЕ", "РЕЗУЛЬТАТЫ_РАСЧЕТА"]:
            assert expected in layer_names

        # Проверяем элементы в пространстве модели
        msp = doc.modelspace()
        texts = list(msp.query("TEXT"))
        assert len(texts) > 0

        # Проверяем наличие кириллических надписей
        text_contents = [t.dxf.text for t in texts]
        assert any("РЕЗУЛЬТАТЫ РАСЧЕТА ОБЪЕМА" in t for t in text_contents)
        assert any("м³" in t for t in text_contents)
        assert any("м²" in t for t in text_contents)

        # Проверяем полилинию границы
        polylines = list(msp.query("LWPOLYLINE"))
        assert len(polylines) >= 1

    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def test_dxf_export_zero_line_detection():
    # Наклонные поверхности с линией нулевых работ (пересечение)
    top_pts = np.array([
        [0.0, 0.0, 102.0],
        [20.0, 0.0, 98.0],
        [20.0, 20.0, 98.0],
        [0.0, 20.0, 102.0],
    ])
    bot_pts = np.array([
        [0.0, 0.0, 100.0],
        [20.0, 0.0, 100.0],
        [20.0, 20.0, 100.0],
        [0.0, 20.0, 100.0],
    ])
    boundary = np.array([
        [0.0, 0.0, 100.0],
        [20.0, 0.0, 100.0],
        [20.0, 20.0, 100.0],
        [0.0, 20.0, 100.0]
    ])

    calc = VolumeCalculator(top_pts, bot_pts, boundary_points=boundary, grid_resolution=0.5, work_type="grading")
    res = calc.calculate()
    assert res["v_fill"] > 0
    assert res["v_cut"] > 0

    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tf:
        out_path = tf.name

    try:
        ok, msg = export_cartogram_dxf(
            calc_results=res,
            points=[],
            boundary_indices=[0, 1, 2, 3],
            output_path=out_path,
            grid_step=5.0,
            include_zero_line=True,
        )
        assert ok is True
        doc = ezdxf.readfile(out_path)
        msp = doc.modelspace()
        zero_lines = [line for line in msp.query("LINE") if line.dxf.layer == "0_НУЛЕВАЯ_ЛИНИЯ"]
        assert len(zero_lines) > 0
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def test_dxf_export_graceful_failures():
    ok, msg = export_cartogram_dxf({}, [], [], "test.dxf")
    assert ok is False
    assert "Отсутствуют результаты" in msg


def test_dxf_export_all_tabs():
    from dxf_exporter import (
        export_surface_3d_dxf,
        export_tin_dxf,
        export_contours_dxf,
        export_diff_dxf,
        export_plan_2d_dxf,
    )

    top_pts = np.array([
        [0.0, 0.0, 105.0],
        [20.0, 0.0, 105.0],
        [20.0, 20.0, 105.0],
        [0.0, 20.0, 105.0],
        [10.0, 10.0, 106.0]
    ])
    bot_pts = np.array([
        [0.0, 0.0, 100.0],
        [20.0, 0.0, 100.0],
        [20.0, 20.0, 100.0],
        [0.0, 20.0, 100.0],
        [10.0, 10.0, 100.0]
    ])
    boundary = np.array([
        [0.0, 0.0, 100.0],
        [20.0, 0.0, 100.0],
        [20.0, 20.0, 100.0],
        [0.0, 20.0, 100.0]
    ])

    calc = VolumeCalculator(top_pts, bot_pts, boundary_points=boundary, grid_resolution=0.5, work_type="cut")
    res = calc.calculate()
    geo_pts = [
        GeoPoint(id="P1", x=0.0, y=0.0, h=100.0, surface_type="bottom"),
        GeoPoint(id="P2", x=20.0, y=0.0, h=100.0, surface_type="bottom"),
        GeoPoint(id="P3", x=20.0, y=20.0, h=100.0, surface_type="bottom"),
        GeoPoint(id="P4", x=0.0, y=20.0, h=100.0, surface_type="bottom"),
        GeoPoint(id="P5", x=10.0, y=10.0, h=100.0, surface_type="bottom"),
    ]

    with tempfile.TemporaryDirectory() as td:
        # 1. 2D Plan
        p_plan = os.path.join(td, "plan.dxf")
        ok, msg = export_plan_2d_dxf(res, geo_pts, [0, 1, 2, 3], p_plan)
        assert ok is True
        doc = ezdxf.readfile(p_plan)
        assert len(list(doc.modelspace().query("POINT"))) > 0

        # 2. 3D Model
        p_3d = os.path.join(td, "3d.dxf")
        ok, msg = export_surface_3d_dxf(res, geo_pts, [0, 1, 2, 3], p_3d)
        assert ok is True
        doc = ezdxf.readfile(p_3d)
        assert len(list(doc.modelspace().query("3DFACE"))) > 0

        # 3. TIN
        p_tin = os.path.join(td, "tin.dxf")
        ok, msg = export_tin_dxf(res, geo_pts, [0, 1, 2, 3], None, p_tin)
        assert ok is True
        doc = ezdxf.readfile(p_tin)
        assert len(list(doc.modelspace().query("3DFACE"))) > 0
        assert len(list(doc.modelspace().query("LINE"))) > 0

        # 4. Contours
        p_cnt = os.path.join(td, "contours.dxf")
        ok, msg = export_contours_dxf(res, geo_pts, [0, 1, 2, 3], p_cnt, contour_step=0.5)
        assert ok is True
        doc = ezdxf.readfile(p_cnt)
        assert len(list(doc.modelspace().query("LWPOLYLINE"))) > 0

        # 5. Diff
        p_diff = os.path.join(td, "diff.dxf")
        ok, msg = export_diff_dxf(res, geo_pts, [0, 1, 2, 3], p_diff)
        assert ok is True
        doc = ezdxf.readfile(p_diff)
        assert len(list(doc.modelspace().query("TEXT"))) > 0

