# -*- coding: utf-8 -*-
"""
Тесты для Этапа 5: Архитектурные сервисы (SpatialIndexService, ProjectStorageService).
"""
import pytest
import numpy as np
from geo_parser import GeoPoint
from spatial_index import SpatialIndexService


def test_spatial_index_service_basic_queries():
    points = [
        GeoPoint(id="1", x=10.0, y=20.0, h=100.0),
        GeoPoint(id="2", x=30.0, y=40.0, h=102.0),
        GeoPoint(id="3", x=50.0, y=60.0, h=104.0),
    ]
    service = SpatialIndexService(points)

    # 1. Точный поиск ближайшей точки
    idx, dist = service.find_nearest_point(20.0, 10.0)
    assert idx == 0
    assert dist == 0.0

    idx, dist = service.find_nearest_point(41.0, 31.0)
    assert idx == 1
    assert abs(dist - np.hypot(1.0, 1.0)) < 1e-6

    # 2. Поиск при пустом списке
    empty_service = SpatialIndexService([])
    idx, dist = empty_service.find_nearest_point(0, 0)
    assert idx == -1
    assert dist == float("inf")


def test_spatial_index_service_set_points_and_invalidate():
    points = [GeoPoint(id="1", x=0.0, y=0.0, h=10.0)]
    service = SpatialIndexService(points)

    tree1 = service.get_kdtree()
    assert tree1 is not None

    # Добавляем новые точки через set_points
    new_points = [
        GeoPoint(id="A", x=100.0, y=100.0, h=10.0),
        GeoPoint(id="B", x=200.0, y=200.0, h=10.0),
    ]
    service.set_points(new_points)
    assert service._kdtree is None  # кеш сброшен

    idx, dist = service.find_nearest_point(100.0, 100.0)
    assert idx == 0
    assert dist == 0.0


def test_spatial_index_boundary_edge():
    points = [
        GeoPoint(id="1", x=0.0, y=0.0, h=10.0),
        GeoPoint(id="2", x=0.0, y=10.0, h=10.0),
        GeoPoint(id="3", x=10.0, y=10.0, h=10.0),
        GeoPoint(id="4", x=10.0, y=0.0, h=10.0),
        GeoPoint(id="5", x=-2.0, y=5.0, h=10.0),  # ближе всего к первому ребру (0 -> 1)
    ]
    boundary = [0, 1, 2, 3]
    service = SpatialIndexService(points)

    best_k = service.find_nearest_boundary_edge(4, boundary)
    assert best_k == 0  # ребро между 0 и 1


def test_click_tolerance_calculation():
    tol = SpatialIndexService.get_click_tolerance((0, 100), (0, 200))
    assert abs(tol - 200 * 0.035) < 1e-6


def test_project_storage_save_load_backup(tmp_path):
    from project_storage import ProjectStorageService
    
    proj_file = str(tmp_path / "test.volproj")
    data = {
        "points": [{"id": "1", "x": 10.0, "y": 20.0, "h": 30.0}],
        "boundary_indices": [0],
        "numpy_val": np.int64(42),
    }

    # 1. Сохранение
    assert ProjectStorageService.save_project_file(proj_file, data) is True

    # 2. Загрузка
    loaded = ProjectStorageService.load_project_file(proj_file)
    assert loaded is not None
    assert loaded["numpy_val"] == 42
    assert len(loaded["points"]) == 1

    # 3. Повторное сохранение создает .bak
    data["numpy_val"] = 100
    assert ProjectStorageService.save_project_file(proj_file, data) is True
    bak_file = proj_file + ".bak"
    import os
    assert os.path.exists(bak_file)

    # 4. Проверка восстановления из .bak при повреждении основного файла
    with open(proj_file, "w", encoding="utf-8") as f:
        f.write("corrupted json content")

    recovered = ProjectStorageService.load_project_file(proj_file)
    assert recovered is not None
    assert recovered["numpy_val"] == 42  # восстановилось старое значение из .bak


def test_project_storage_folder_utils():
    from project_storage import ProjectStorageService
    
    # Очистка имени от даты
    assert ProjectStorageService.strip_date_from_folder_name("15.09.2026_Карьер") == "Карьер"
    assert ProjectStorageService.strip_date_from_folder_name("Карьер_15.09.2026") == "Карьер"
    assert ProjectStorageService.strip_date_from_folder_name("Карьер") == "Карьер"

