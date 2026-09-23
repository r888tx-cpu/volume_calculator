# -*- coding: utf-8 -*-
"""
Модуль для импорта геодезических данных и контуров из файлов AutoCAD DXF (.dxf).

Поддерживает:
- Чтение 3D точек (POINT);
- Чтение геодезических блоков (INSERT / COGO Points) с разбором атрибутов (номер, отметка);
- Чтение замкнутых полилиний (LWPOLYLINE / POLYLINE) в качестве контура границы работ;
- Чтение структурных 3D-линий и полилиний в качестве точек поверхности;
- Чтение высотных отметок из текста (TEXT / MTEXT);
- Разделение на верхнюю и нижнюю поверхности по слоям;
- Геодезическое преобразование координат по ГОСТ (X_cad = Восток, Y_cad = Север).
"""

import os
import re
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np

try:
    import ezdxf
    from ezdxf.entities import Point, Insert, LWPolyline, Polyline, Text, MText, Face3d
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False

from geo_parser import GeoPoint


def is_dxf_available() -> bool:
    """Проверяет, доступна ли библиотека ezdxf."""
    return EZDXF_AVAILABLE


def inspect_dxf_layers(filepath: str) -> Tuple[bool, Union[Dict[str, Any], str]]:
    """
    Быстро сканирует структуру файла DXF и возвращает сводку по слоям и объектам.

    Возвращает (success: bool, result: Dict | error_message: str)
    """
    if not EZDXF_AVAILABLE:
        return False, "Библиотека ezdxf не установлена. Установите её командой 'pip install ezdxf'."

    if not os.path.exists(filepath):
        return False, f"Файл не найден: {filepath}"

    try:
        doc = ezdxf.readfile(filepath)
        msp = doc.modelspace()
    except Exception as e:
        return False, f"Ошибка открытия DXF файла:\n{e}"

    layers_info: Dict[str, Dict[str, Any]] = {}

    # Регистрируем все существующие в документе слои
    for layer in doc.layers:
        name = layer.dxf.name
        layers_info[name] = {
            "name": name,
            "color": getattr(layer.dxf, "color", 7),
            "points_count": 0,
            "inserts_count": 0,
            "polylines_count": 0,
            "closed_polylines_count": 0,
            "texts_count": 0,
            "faces_count": 0,
            "total_objects": 0,
        }

    closed_polylines: List[Dict[str, Any]] = []

    # Подсчитываем примитивы в пространстве модели
    for entity in msp:
        layer_name = entity.dxf.layer
        if layer_name not in layers_info:
            layers_info[layer_name] = {
                "name": layer_name,
                "color": 7,
                "points_count": 0,
                "inserts_count": 0,
                "polylines_count": 0,
                "closed_polylines_count": 0,
                "texts_count": 0,
                "faces_count": 0,
                "total_objects": 0,
            }

        linfo = layers_info[layer_name]
        dxftype = entity.dxftype()

        if dxftype == "POINT":
            linfo["points_count"] += 1
            linfo["total_objects"] += 1
        elif dxftype == "INSERT":
            linfo["inserts_count"] += 1
            linfo["total_objects"] += 1
        elif dxftype in ("LWPOLYLINE", "POLYLINE"):
            linfo["polylines_count"] += 1
            linfo["total_objects"] += 1
            is_closed = False
            pts_2d = []
            try:
                if dxftype == "LWPOLYLINE":
                    is_closed = bool(entity.is_closed)
                    pts_2d = [(p[0], p[1]) for p in entity.get_points(format="xy")]
                else:
                    is_closed = bool(getattr(entity, "is_closed", False) or entity.dxf.flags & 1)
                    pts_2d = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
            except Exception:
                pass

            if is_closed and len(pts_2d) >= 3:
                linfo["closed_polylines_count"] += 1
                # Вычисление площади многоугольника (формула Гаусса)
                area = 0.5 * abs(sum(pts_2d[i][0] * (pts_2d[(i + 1) % len(pts_2d)][1] - pts_2d[i - 1][1])
                                     for i in range(len(pts_2d))))
                closed_polylines.append({
                    "layer": layer_name,
                    "points_count": len(pts_2d),
                    "area": area,
                    "points": pts_2d,
                })
        elif dxftype in ("TEXT", "MTEXT"):
            linfo["texts_count"] += 1
            linfo["total_objects"] += 1
        elif dxftype == "3DFACE":
            linfo["faces_count"] += 1
            linfo["total_objects"] += 1

    # Фильтруем пустые слои без объектов (но сохраняем если есть хотя бы 1 объект)
    active_layers = {k: v for k, v in layers_info.items() if v["total_objects"] > 0}
    if not active_layers:
        active_layers = layers_info

    # Сортируем замкнутые полилинии по площади (наибольшая первой)
    closed_polylines.sort(key=lambda item: item["area"], reverse=True)

    # Интеллектуальный подбор слоев по ключевым словам
    suggested_top = None
    suggested_bottom = None
    suggested_boundary = None

    top_keywords = ["верх", "top", "проект", "plan", "design", "красн", "red"]
    bot_keywords = ["низ", "bot", "дно", "земля", "рельеф", "факт", "ground", "черн", "black"]
    bound_keywords = ["границ", "контур", "отвод", "bound", "limit", "border"]

    for lname in active_layers:
        ln_low = lname.lower()
        if not suggested_top and any(k in ln_low for k in top_keywords):
            suggested_top = lname
        if not suggested_bottom and any(k in ln_low for k in bot_keywords):
            suggested_bottom = lname
        if not suggested_boundary and any(k in ln_low for k in bound_keywords):
            suggested_boundary = lname

    # Если слой границы не найден по имени, но есть замкнутые полилинии
    if not suggested_boundary and closed_polylines:
        suggested_boundary = closed_polylines[0]["layer"]

    return True, {
        "layers": active_layers,
        "closed_polylines": closed_polylines,
        "suggested_top_layer": suggested_top,
        "suggested_bottom_layer": suggested_bottom,
        "suggested_boundary_layer": suggested_boundary,
    }


def extract_dxf_geometry(
    filepath: str,
    top_layers: Optional[List[str]] = None,
    bottom_layers: Optional[List[str]] = None,
    boundary_layer: Optional[str] = None,
    coord_swap: bool = True,
    include_points: bool = True,
    include_inserts: bool = True,
    include_polylines_as_points: bool = False,
    include_texts: bool = False,
    specific_boundary_points: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[bool, List[GeoPoint], Optional[np.ndarray], bool, str]:
    """
    Извлекает геодезические точки и контур границы из файла DXF.

    Параметры:
    - filepath: путь к DXF-файлу.
    - top_layers: список слоев для верхней поверхности. Если None/пусто и bottom_layers пуст — берутся все активные слои.
    - bottom_layers: список слоев для нижней поверхности.
    - boundary_layer: имя слоя, содержащего замкнутую полилинию границы.
    - coord_swap: если True, X_cad -> Y_geo (Восток), Y_cad -> X_geo (Север) — стандарт ГОСТ.
    - include_points: извлекать примитивы POINT.
    - include_inserts: извлекать блоки INSERT (COGO-точки).
    - include_polylines_as_points: считывать вершины незамкнутых полилиний как точки съёмки.
    - include_texts: считывать числовой текст как отметки.
    - specific_boundary_points: заранее выбранный контур [(x_cad, y_cad), ...].

    Возвращает:
    (success: bool, points: List[GeoPoint], boundary_points: Optional[np.ndarray], is_two_surfaces: bool, msg: str)
    """
    if not EZDXF_AVAILABLE:
        return False, [], None, False, "Библиотека ezdxf не установлена."

    if not os.path.exists(filepath):
        return False, [], None, False, f"Файл не найден: {filepath}"

    try:
        doc = ezdxf.readfile(filepath)
        msp = doc.modelspace()
    except Exception as e:
        return False, [], None, False, f"Ошибка чтения DXF файла:\n{e}"

    top_layers_set = set(top_layers) if top_layers else set()
    bottom_layers_set = set(bottom_layers) if bottom_layers else set()
    is_two_surfaces = bool(top_layers_set and bottom_layers_set)

    def to_geo(x_cad: float, y_cad: float) -> Tuple[float, float]:
        """Преобразует CAD координаты в геодезические (Север X, Восток Y)."""
        if coord_swap:
            return float(y_cad), float(x_cad)
        return float(x_cad), float(y_cad)

    extracted_points: List[GeoPoint] = []
    pt_counter = 1

    # Регулярное выражение для поиска чисел в тексте отметок (например, "145.28", "+2.50", "-0.35")
    re_elevation = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")

    def parse_block_attributes(insert_entity) -> Tuple[str, Optional[float]]:
        """Извлекает ID точки и высоту из атрибутов блока."""
        p_id = ""
        p_h = None
        if hasattr(insert_entity, "attribs"):
            for attrib in insert_entity.attribs:
                tag = attrib.dxf.tag.upper().strip()
                val = attrib.dxf.text.strip()
                if not val:
                    continue
                if not p_id and tag in ("NUM", "NUMBER", "ID", "НОМЕР", "ТОЧКА", "NAME", "PNT", "POINT", "PT"):
                    p_id = val
                elif p_h is None and tag in ("ELEV", "ELEVATION", "H", "Z", "ОТМЕТКА", "ВЫСОТА", "OTM", "ВЫС"):
                    try:
                        p_h = float(val.replace(",", "."))
                    except ValueError:
                        pass
        return p_id, p_h

    # Обход пространства модели
    for entity in msp:
        layer = entity.dxf.layer
        dxftype = entity.dxftype()

        # Определяем тип поверхности для слоя
        surf_type = "auto"
        if is_two_surfaces:
            if layer in top_layers_set:
                surf_type = "top"
            elif layer in bottom_layers_set:
                surf_type = "bottom"
            else:
                # Если слой не выбран ни в верх, ни в низ — пропускаем его точки
                continue
        elif top_layers_set:
            if layer not in top_layers_set:
                continue

        # 1. Точки POINT
        if include_points and dxftype == "POINT":
            loc = entity.dxf.location
            x_geo, y_geo = to_geo(loc.x, loc.y)
            extracted_points.append(GeoPoint(
                id=f"P{pt_counter}",
                x=x_geo,
                y=y_geo,
                h=float(loc.z),
                surface_type=surf_type
            ))
            pt_counter += 1

        # 2. Блоки INSERT
        elif include_inserts and dxftype == "INSERT":
            ins_pt = entity.dxf.insert
            attr_id, attr_h = parse_block_attributes(entity)
            p_id = attr_id if attr_id else f"B{pt_counter}"
            h_val = attr_h if attr_h is not None else float(ins_pt.z)
            x_geo, y_geo = to_geo(ins_pt.x, ins_pt.y)
            extracted_points.append(GeoPoint(
                id=p_id,
                x=x_geo,
                y=y_geo,
                h=h_val,
                surface_type=surf_type
            ))
            pt_counter += 1

        # 3. Вершины полилиний как структурные точки
        elif include_polylines_as_points and dxftype in ("LWPOLYLINE", "POLYLINE"):
            # Не считываем контур границы как съёмочные точки, если этот слой задан как граница
            if boundary_layer and layer == boundary_layer:
                continue
            try:
                if dxftype == "LWPOLYLINE":
                    poly_elev = float(getattr(entity.dxf, "elevation", 0.0))
                    for p in entity.get_points(format="xyb"):
                        x_geo, y_geo = to_geo(p[0], p[1])
                        extracted_points.append(GeoPoint(
                            id=f"L{pt_counter}",
                            x=x_geo,
                            y=y_geo,
                            h=poly_elev,
                            surface_type=surf_type
                        ))
                        pt_counter += 1
                else:
                    for v in entity.vertices:
                        loc = v.dxf.location
                        x_geo, y_geo = to_geo(loc.x, loc.y)
                        extracted_points.append(GeoPoint(
                            id=f"L{pt_counter}",
                            x=x_geo,
                            y=y_geo,
                            h=float(loc.z),
                            surface_type=surf_type
                        ))
                        pt_counter += 1
            except Exception:
                pass

        # 4. Текстовые отметки
        elif include_texts and dxftype in ("TEXT", "MTEXT"):
            try:
                txt = entity.plain_text() if hasattr(entity, "plain_text") else str(entity.dxf.text)
                txt = txt.strip()
                if re_elevation.match(txt):
                    h_val = float(txt.replace(",", "."))
                    ins_pos = getattr(entity.dxf, "insert", getattr(entity.dxf, "align_point", None))
                    if ins_pos:
                        x_geo, y_geo = to_geo(ins_pos.x, ins_pos.y)
                        extracted_points.append(GeoPoint(
                            id=f"T{pt_counter}",
                            x=x_geo,
                            y=y_geo,
                            h=h_val,
                            surface_type=surf_type
                        ))
                        pt_counter += 1
            except Exception:
                pass

    if not extracted_points:
        return False, [], None, is_two_surfaces, "В выбранных слоях DXF-файла не найдено геодезических точек или блоков."

    # Извлечение контура границы
    boundary_arr: Optional[np.ndarray] = None

    if specific_boundary_points and len(specific_boundary_points) >= 3:
        b_geo = [to_geo(p[0], p[1]) for p in specific_boundary_points]
        boundary_arr = np.array(b_geo, dtype=np.float64)
    elif boundary_layer:
        # Ищем замкнутую полилинию на слое boundary_layer с наибольшей площадью
        best_poly = None
        best_area = -1.0
        for entity in msp:
            if entity.dxf.layer != boundary_layer:
                continue
            dxftype = entity.dxftype()
            if dxftype in ("LWPOLYLINE", "POLYLINE"):
                is_closed = False
                pts_2d = []
                try:
                    if dxftype == "LWPOLYLINE":
                        is_closed = bool(entity.is_closed)
                        pts_2d = [(p[0], p[1]) for p in entity.get_points(format="xy")]
                    else:
                        is_closed = bool(getattr(entity, "is_closed", False) or entity.dxf.flags & 1)
                        pts_2d = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
                except Exception:
                    pass

                if is_closed and len(pts_2d) >= 3:
                    area = 0.5 * abs(sum(pts_2d[i][0] * (pts_2d[(i + 1) % len(pts_2d)][1] - pts_2d[i - 1][1])
                                         for i in range(len(pts_2d))))
                    if area > best_area:
                        best_area = area
                        best_poly = pts_2d

        if best_poly:
            b_geo = [to_geo(p[0], p[1]) for p in best_poly]
            boundary_arr = np.array(b_geo, dtype=np.float64)

    msg = f"Успешно импортировано {len(extracted_points)} точек из DXF."
    if boundary_arr is not None:
        msg += f" Загружен контур границы ({len(boundary_arr)} вершин)."

    return True, extracted_points, boundary_arr, is_two_surfaces, msg
