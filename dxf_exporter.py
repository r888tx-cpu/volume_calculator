# -*- coding: utf-8 -*-
"""
Модуль экспорта геодезической картограммы земляных масс в формат DXF (AutoCAD).
Соответствует требованиям ГОСТ 21.508-2020 (СПДС. Правила выполнения рабочей документации
генеральных планов предприятий, сооружений и жилищно-гражданских объектов).

Поддерживает:
- Автоматическую координатную сетку с настраиваемым шагом
- Разметку узлов по ГОСТ (красная проектная отметка, черная фактическая отметка, рабочая отметка со знаком)
- Подписи объемов насыпи (+), выемки (-) и площадей ячеек
- Отрисовку линии нулевых работ (нулевой баланс)
- Отрисовку проектной границы работ (контура сшивания)
- Отображение исходных пикетов (точек съемки)
- Сводную ведомость земляных масс (таблицу баланса)
- Строгое разнесение по именованным слоям с весами линий и цветами
"""

import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import ezdxf
    from ezdxf.enums import TextEntityAlignment
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False


# Стандартные слои DXF с цветами ACI и весами линий (в сотых долях мм)
DEFAULT_DXF_LAYERS = {
    "0_ГРАНИЦА_РАБОТ": {"color": 1, "lineweight": 50},           # 1 = Red, 0.50 мм
    "0_СЕТКА_КАРТОГРАММЫ": {"color": 8, "lineweight": 18},       # 8 = Dark Gray, 0.18 мм
    "0_НУЛЕВАЯ_ЛИНИЯ": {"color": 4, "lineweight": 35, "linetype": "DASHED"}, # 4 = Cyan, 0.35 мм
    "ОТМЕТКИ_КРАСНЫЕ": {"color": 1, "lineweight": 20},           # 1 = Red (проект / верх)
    "ОТМЕТКИ_ЧЕРНЫЕ": {"color": 7, "lineweight": 20},            # 7 = White/Black (земля / низ)
    "ОТМЕТКИ_РАБОЧИЕ_НАСЫПЬ": {"color": 3, "lineweight": 20},    # 3 = Green (+ рабочая)
    "ОТМЕТКИ_РАБОЧИЕ_ВЫЕМКА": {"color": 1, "lineweight": 20},    # 1 = Red (- рабочая)
    "ОБЪЕМЫ_ЯЧЕЕК": {"color": 30, "lineweight": 20},             # 30 = Orange (объем)
    "ПЛОЩАДИ_ЯЧЕЕК": {"color": 8, "lineweight": 15},             # 8 = Gray (площадь)
    "ТОЧКИ_СЪЕМКИ_ВЕРХ": {"color": 1, "lineweight": 15},         # 1 = Red
    "ТОЧКИ_СЪЕМКИ_НИЗ": {"color": 5, "lineweight": 15},          # 5 = Blue
    "ТАБЛИЦА_БАЛАНСА": {"color": 7, "lineweight": 25},           # 7 = White/Black
}


def choose_auto_grid_step(span_x: float, span_y: float) -> float:
    """Подбирает оптимальный инженерный шаг сетки картограммы по габаритам площадки."""
    span = max(span_x, span_y)
    if span <= 25.0:
        return 2.0
    elif span <= 60.0:
        return 5.0
    elif span <= 150.0:
        return 10.0
    elif span <= 300.0:
        return 20.0
    else:
        return 50.0


def export_cartogram_dxf(
    calc_results: Dict[str, Any],
    points: List[Any],
    boundary_indices: List[int],
    output_path: str,
    grid_step: float = 0.0,
    text_height: float = 0.0,
    coord_swap: bool = True,
    include_grid: bool = True,
    include_node_elevations: bool = True,
    include_cell_volumes: bool = True,
    include_zero_line: bool = True,
    include_boundary: bool = True,
    include_survey_points: bool = True,
    include_balance_table: bool = True,
) -> Tuple[bool, str]:
    """
    Экспортирует картограмму земляных масс в формат AutoCAD DXF (R2010).

    calc_results: словарь результатов расчета VolumeCalculator
    points: список объектов GeoPoint
    boundary_indices: список индексов вершин границы
    output_path: путь к сохраняемому файлу .dxf
    grid_step: шаг сетки квадратов (м). Если <= 0, выбирается автоматически
    text_height: высота текста (м). Если <= 0, подбирается под шаг сетки
    coord_swap: если True, X_cad = Y_geo (Восток), Y_cad = X_geo (Север) — стандарт для геодезии в AutoCAD

    Возвращает (success: bool, message: str)
    """
    if not EZDXF_AVAILABLE:
        return False, "Библиотека ezdxf не установлена. Установите её командой 'pip install ezdxf'."

    if not calc_results:
        return False, "Отсутствуют результаты расчета объема."

    boundary = calc_results.get("boundary")
    if boundary is None or len(boundary) < 3:
        if points and boundary_indices and len(boundary_indices) >= 3:
            boundary = np.array([[points[i].x, points[i].y] for i in boundary_indices])
        else:
            return False, "Не задан контур границы работ для картограммы."

    poly_2d = boundary[:, :2]
    x_min, x_max = float(np.min(poly_2d[:, 0])), float(np.max(poly_2d[:, 0]))
    y_min, y_max = float(np.min(poly_2d[:, 1])), float(np.max(poly_2d[:, 1]))
    span_x = x_max - x_min
    span_y = y_max - y_min

    step = float(grid_step) if grid_step > 0.0 else choose_auto_grid_step(span_x, span_y)
    th = float(text_height) if text_height > 0.0 else max(0.20, step * 0.065)

    def to_cad(x_geo: float, y_geo: float) -> Tuple[float, float]:
        """Преобразует геодезические координаты (Север X, Восток Y) в координаты чертежа CAD."""
        if coord_swap:
            return float(y_geo), float(x_geo)
        return float(x_geo), float(y_geo)

    try:
        doc = ezdxf.new("R2010", setup=True)
        msp = doc.modelspace()

        # Регистрация типов линий
        try:
            doc.linetypes.add("DASHED", pattern=[th * 2.0, th * 1.0, -th * 1.0], description="Штриховая нулевая линия")
        except Exception:
            pass

        # Создание слоёв
        for lname, lattr in DEFAULT_DXF_LAYERS.items():
            if lname not in doc.layers:
                layer = doc.layers.add(lname, color=lattr["color"])
                if "lineweight" in lattr:
                    layer.dxf.lineweight = lattr["lineweight"]
                if "linetype" in lattr:
                    layer.dxf.linetype = lattr["linetype"]

        import matplotlib.path as mpl_path
        bound_path = mpl_path.Path(poly_2d)

        # ── 1. Граница работ (контур) ──────────────────────────────────────────
        if include_boundary:
            cad_bound_pts = [to_cad(pt[0], pt[1]) for pt in poly_2d]
            msp.add_lwpolyline(
                cad_bound_pts,
                close=True,
                dxfattribs={"layer": "0_ГРАНИЦА_РАБОТ", "lineweight": 50}
            )

        # ── 2. Интерполяция высот для узлов ────────────────────────────────────
        from scipy.interpolate import RegularGridInterpolator
        gx = calc_results.get("grid_x")
        gy = calc_results.get("grid_y")
        zt_grid = calc_results.get("z_top_grid")
        zb_grid = calc_results.get("z_bot_grid")

        interp_top, interp_bot = None, None
        if gx is not None and gy is not None and zt_grid is not None and zb_grid is not None:
            try:
                xs = np.unique(gx)
                ys = np.unique(gy)
                rgi_top = RegularGridInterpolator((ys, xs), zt_grid, bounds_error=False, fill_value=np.nan)
                rgi_bot = RegularGridInterpolator((ys, xs), zb_grid, bounds_error=False, fill_value=np.nan)
                interp_top = lambda pt: float(rgi_top([pt[1], pt[0]])[0])
                interp_bot = lambda pt: float(rgi_bot([pt[1], pt[0]])[0])
            except Exception:
                pass

        # Fallback интерполяторы из исходных точек
        top_pts_arr = calc_results.get("top_surface_pts")
        if top_pts_arr is None or len(top_pts_arr) == 0:
            top_pts_arr = np.array([[p.x, p.y, p.h] for p in points if getattr(p, "surface_type", None) == "top"])

        bot_pts_arr = calc_results.get("bottom_surface_pts")
        if bot_pts_arr is None or len(bot_pts_arr) == 0:
            bot_pts_arr = np.array([[p.x, p.y, p.h] for p in points if getattr(p, "surface_type", None) == "bottom"])

        from scipy.interpolate import NearestNDInterpolator
        fallback_top = NearestNDInterpolator(top_pts_arr[:, :2], top_pts_arr[:, 2]) if (top_pts_arr is not None and len(top_pts_arr) > 0) else None
        fallback_bot = NearestNDInterpolator(bot_pts_arr[:, :2], bot_pts_arr[:, 2]) if (bot_pts_arr is not None and len(bot_pts_arr) > 0) else None

        # ── 3. Построение координатной сетки квадратов ──────────────────────────
        grid_x_min = math.floor(x_min / step) * step
        grid_x_max = math.ceil(x_max / step) * step
        grid_y_min = math.floor(y_min / step) * step
        grid_y_max = math.ceil(y_max / step) * step

        x_coords = np.arange(grid_x_min, grid_x_max + step * 0.5, step)
        y_coords = np.arange(grid_y_min, grid_y_max + step * 0.5, step)

        node_elevations = {}  # (i, j) -> (z_top, z_bot, dh)

        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                # Проверяем, находится ли узел внутри или вблизи границы (с запасом 1.2 шага сетки)
                dist_check = bound_path.contains_point((x, y), radius=step * 0.75)
                if not dist_check:
                    continue

                z_t, z_b = np.nan, np.nan
                if interp_top is not None:
                    try:
                        val_t = interp_top((x, y))
                        if not np.isnan(val_t):
                            z_t = float(val_t)
                    except Exception:
                        pass
                if np.isnan(z_t) and fallback_top is not None:
                    try:
                        z_t = float(fallback_top(x, y))
                    except Exception:
                        pass

                if interp_bot is not None:
                    try:
                        val_b = interp_bot((x, y))
                        if not np.isnan(val_b):
                            z_b = float(val_b)
                    except Exception:
                        pass
                if np.isnan(z_b) and fallback_bot is not None:
                    try:
                        z_b = float(fallback_bot(x, y))
                    except Exception:
                        pass

                if not np.isnan(z_t) and not np.isnan(z_b):
                    dh = z_t - z_b
                    node_elevations[(i, j)] = (z_t, z_b, dh)

        # ── 4. Отрисовка квадратов и линий сетки ────────────────────────────────
        if include_grid:
            # Отрезки сетки между узлами, входящими в зону работ
            for i, x in enumerate(x_coords):
                for j, y in enumerate(y_coords):
                    # Горизонтальное ребро (j -> j+1)
                    if (i, j) in node_elevations and (i, j + 1) in node_elevations:
                        p1 = to_cad(x, y)
                        p2 = to_cad(x, y_coords[j + 1])
                        msp.add_line(p1, p2, dxfattribs={"layer": "0_СЕТКА_КАРТОГРАММЫ", "lineweight": 18})

                    # Вертикальное ребро (i -> i+1)
                    if (i, j) in node_elevations and (i + 1, j) in node_elevations:
                        p1 = to_cad(x, y)
                        p2 = to_cad(x_coords[i + 1], y)
                        msp.add_line(p1, p2, dxfattribs={"layer": "0_СЕТКА_КАРТОГРАММЫ", "lineweight": 18})

        # ── 5. Подписи отметок в узлах по ГОСТ 21.508-2020 ─────────────────────
        if include_node_elevations:
            cross_sz = th * 0.35
            for (i, j), (z_t, z_b, dh) in node_elevations.items():
                x = x_coords[i]
                y = y_coords[j]
                xc, yc = to_cad(x, y)

                # Маленький центрирующий крестик узла
                msp.add_line((xc - cross_sz, yc), (xc + cross_sz, yc),
                             dxfattribs={"layer": "0_СЕТКА_КАРТОГРАММЫ", "lineweight": 15})
                msp.add_line((xc, yc - cross_sz), (xc, yc + cross_sz),
                             dxfattribs={"layer": "0_СЕТКА_КАРТОГРАММЫ", "lineweight": 15})

                # Рабочая отметка (слева вверху от узла)
                sign_str = "+" if dh > 0.005 else ("-" if dh < -0.005 else "")
                dh_str = f"{sign_str}{abs(dh):.2f}"
                layer_dh = "ОТМЕТКИ_РАБОЧИЕ_НАСЫПЬ" if dh >= 0 else "ОТМЕТКИ_РАБОЧИЕ_ВЫЕМКА"
                t_dh = msp.add_text(dh_str, dxfattribs={"layer": layer_dh, "height": th})
                t_dh.set_placement((xc - th * 0.25, yc + th * 0.35), align=TextEntityAlignment.RIGHT)

                # Красная проектная отметка (справа вверху от узла)
                t_red = msp.add_text(f"{z_t:.2f}", dxfattribs={"layer": "ОТМЕТКИ_КРАСНЫЕ", "height": th})
                t_red.set_placement((xc + th * 0.25, yc + th * 0.35), align=TextEntityAlignment.LEFT)

                # Черная фактическая отметка (справа внизу от узла)
                t_blk = msp.add_text(f"{z_b:.2f}", dxfattribs={"layer": "ОТМЕТКИ_ЧЕРНЫЕ", "height": th})
                t_blk.set_placement((xc + th * 0.25, yc - th * 1.15), align=TextEntityAlignment.LEFT)

        # ── 6. Линия нулевых работ (нулевой баланс) ────────────────────────────
        if include_zero_line:
            eps_zero = 0.005
            # 1. Ребра сетки, где оба конца имеют близкую к нулю рабочую отметку
            drawn_edges = set()
            for (i, j), (zt, zb, dh) in node_elevations.items():
                if abs(dh) <= eps_zero:
                    for ni, nj in [(i + 1, j), (i, j + 1)]:
                        if (ni, nj) in node_elevations and abs(node_elevations[(ni, nj)][2]) <= eps_zero:
                            edge_key = tuple(sorted([(i, j), (ni, nj)]))
                            if edge_key not in drawn_edges:
                                drawn_edges.add(edge_key)
                                p1 = to_cad(x_coords[i], y_coords[j])
                                p2 = to_cad(x_coords[ni], y_coords[nj])
                                msp.add_line(p1, p2, dxfattribs={"layer": "0_НУЛЕВАЯ_ЛИНИЯ", "linetype": "DASHED", "lineweight": 35})

            # 2. Пересечения рёбер ячеек сетки
            for i in range(len(x_coords) - 1):
                for j in range(len(y_coords) - 1):
                    corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
                    edges = [
                        (corners[0], corners[1]),
                        (corners[1], corners[2]),
                        (corners[2], corners[3]),
                        (corners[3], corners[0])
                    ]
                    zero_pts = []
                    for n1, n2 in edges:
                        if n1 in node_elevations and n2 in node_elevations:
                            dh1 = node_elevations[n1][2]
                            dh2 = node_elevations[n2][2]
                            if (dh1 > eps_zero and dh2 < -eps_zero) or (dh1 < -eps_zero and dh2 > eps_zero):
                                frac = abs(dh1) / (abs(dh1) + abs(dh2))
                                x_z = x_coords[n1[0]] + frac * (x_coords[n2[0]] - x_coords[n1[0]])
                                y_z = y_coords[n1[1]] + frac * (y_coords[n2[1]] - y_coords[n1[1]])
                                if bound_path.contains_point((x_z, y_z), radius=0.1):
                                    pt_cad = to_cad(x_z, y_z)
                                    if not any(math.hypot(pt_cad[0] - zp[0], pt_cad[1] - zp[1]) < 1e-3 for zp in zero_pts):
                                        zero_pts.append(pt_cad)

                    if len(zero_pts) == 2:
                        msp.add_line(zero_pts[0], zero_pts[1],
                                     dxfattribs={"layer": "0_НУЛЕВАЯ_ЛИНИЯ", "linetype": "DASHED", "lineweight": 35})

        # ── 7. Объемы и площади в центрах ячеек ────────────────────────────────
        if include_cell_volumes:
            cell_th = th * 0.90
            # Если есть мелкая расчетная сетка, используем её для точной интеграции ячеек
            fine_gx = calc_results.get("grid_x")
            fine_gy = calc_results.get("grid_y")
            fine_dh = calc_results.get("dh_grid")
            fine_mask = calc_results.get("inside_mask")
            fine_res = calc_results.get("grid_resolution", 0.5)
            cell_unit_area = float(fine_res * fine_res)

            for i in range(len(x_coords) - 1):
                x1, x2 = x_coords[i], x_coords[i + 1]
                for j in range(len(y_coords) - 1):
                    y1, y2 = y_coords[j], y_coords[j + 1]

                    # Проверяем, попадает ли центр квадрата или его узлы внутрь границы
                    xc_mid = (x1 + x2) * 0.5
                    yc_mid = (y1 + y2) * 0.5
                    if not bound_path.contains_point((xc_mid, yc_mid), radius=step * 0.6):
                        continue

                    v_f_cell = 0.0
                    v_c_cell = 0.0
                    s_cell = 0.0

                    if fine_gx is not None and fine_gy is not None and fine_dh is not None and fine_mask is not None:
                        # Отбираем точки мелкой сетки, попадающие в данный квадрат
                        box_m = (fine_gx >= x1) & (fine_gx < x2) & (fine_gy >= y1) & (fine_gy < y2) & fine_mask
                        dh_box = fine_dh[box_m]
                        if len(dh_box) > 0:
                            s_cell = float(len(dh_box) * cell_unit_area)
                            v_f_cell = float(np.sum(dh_box[dh_box > 0]) * cell_unit_area)
                            v_c_cell = float(np.abs(np.sum(dh_box[dh_box < 0])) * cell_unit_area)
                    else:
                        # Fallback по 4 узлам
                        active_dhs = [node_elevations[n][2] for n in [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)] if n in node_elevations]
                        if active_dhs:
                            mean_dh = float(np.mean(active_dhs))
                            s_cell = float(step * step)
                            if mean_dh >= 0:
                                v_f_cell = mean_dh * s_cell
                            else:
                                v_c_cell = abs(mean_dh) * s_cell

                    if s_cell <= 0.05 and (v_f_cell + v_c_cell) <= 0.01:
                        continue

                    cad_xc, cad_yc = to_cad(xc_mid, yc_mid)

                    # Формируем подпись объёма и площади
                    lines_to_draw = []
                    if v_f_cell > 0.01:
                        lines_to_draw.append((f"+{v_f_cell:.1f} м³", "ОБЪЕМЫ_ЯЧЕЕК"))
                    if v_c_cell > 0.01:
                        lines_to_draw.append((f"-{v_c_cell:.1f} м³", "ОБЪЕМЫ_ЯЧЕЕК"))
                    if s_cell > 0.05:
                        lines_to_draw.append((f"S={s_cell:.1f} м²", "ПЛОЩАДИ_ЯЧЕЕК"))

                    y_offset = (len(lines_to_draw) - 1) * 0.6 * cell_th
                    for text_val, layer_name in lines_to_draw:
                        t_cell = msp.add_text(text_val, dxfattribs={"layer": layer_name, "height": cell_th})
                        t_cell.set_placement((cad_xc, cad_yc + y_offset), align=TextEntityAlignment.MIDDLE_CENTER)
                        y_offset -= 1.3 * cell_th

        # ── 8. Исходные точки съёмки ──────────────────────────────────────────
        if include_survey_points and points:
            pt_th = th * 0.65
            pt_rad = th * 0.20
            for pt in points:
                xc, yc = to_cad(pt.x, pt.y)
                layer = "ТОЧКИ_СЪЕМКИ_ВЕРХ" if pt.surface_type == "top" else "ТОЧКИ_СЪЕМКИ_НИЗ"

                # Маркер точки (окружность)
                msp.add_circle((xc, yc), radius=pt_rad, dxfattribs={"layer": layer, "lineweight": 15})

                # Номер точки и высотная отметка
                pt_id_str = str(pt.id) if pt.id else ""
                h_str = f"{pt.h:.2f}"
                t_lbl = msp.add_text(f"{pt_id_str} ({h_str})", dxfattribs={"layer": layer, "height": pt_th})
                t_lbl.set_placement((xc + pt_rad * 1.5, yc + pt_rad * 0.5), align=TextEntityAlignment.LEFT)

        # ── 9. Сводная ведомость земляных масс (таблица баланса) ───────────────
        if include_balance_table:
            tb_th = th * 1.10
            # Размещаем таблицу справа от картограммы с отступом
            all_cad_x = [to_cad(pt[0], pt[1])[0] for pt in poly_2d]
            all_cad_y = [to_cad(pt[0], pt[1])[1] for pt in poly_2d]
            tb_x0 = max(all_cad_x) + step * 0.8
            tb_y0 = max(all_cad_y)

            table_rows = [
                ("ВЕДОМОСТЬ ОБЪЕМОВ ЗЕМЛЯНЫХ МАСС", ""),
                ("Нормативный документ:", "ГОСТ 21.508-2020"),
                ("Площадь в плане (2D):", f"{calc_results.get('area_2d', 0.0):.1f} м²"),
                ("Объем насыпи (+):", f"{calc_results.get('v_fill', 0.0):.2f} м³"),
                ("Объем выемки (-):", f"{calc_results.get('v_cut', 0.0):.2f} м³"),
                ("Баланс земляных масс (нетто):", f"{calc_results.get('v_net', 0.0):.2f} м³"),
                ("Средняя толщина слоя:", f"{calc_results.get('avg_thickness', 0.0):.2f} м"),
                ("Максимальная толщина:", f"{calc_results.get('max_thickness', 0.0):.2f} м"),
                ("Минимальная толщина:", f"{calc_results.get('min_thickness', 0.0):.2f} м"),
                ("Шаг сетки картограммы:", f"{step:.1f} м"),
                ("Программа расчета:", "GeoVolumePro"),
            ]

            row_h = tb_th * 1.8
            col1_w = tb_th * 18.0
            col2_w = tb_th * 12.0
            tb_w = col1_w + col2_w

            # Отрисовка рамки таблицы
            cur_y = tb_y0
            for r_idx, (col1, col2) in enumerate(table_rows):
                # Внешний контур строки
                msp.add_lwpolyline(
                    [(tb_x0, cur_y), (tb_x0 + tb_w, cur_y),
                     (tb_x0 + tb_w, cur_y - row_h), (tb_x0, cur_y - row_h)],
                    close=True,
                    dxfattribs={"layer": "ТАБЛИЦА_БАЛАНСА", "lineweight": 25}
                )

                if r_idx == 0:
                    # Заголовок по центру
                    t = msp.add_text(col1, dxfattribs={"layer": "ТАБЛИЦА_БАЛАНСА", "height": tb_th * 1.05})
                    t.set_placement((tb_x0 + tb_w * 0.5, cur_y - row_h * 0.5), align=TextEntityAlignment.MIDDLE_CENTER)
                else:
                    # Разделитель колонок
                    msp.add_line((tb_x0 + col1_w, cur_y), (tb_x0 + col1_w, cur_y - row_h),
                                 dxfattribs={"layer": "ТАБЛИЦА_БАЛАНСА", "lineweight": 18})
                    # Текст колонки 1 (название)
                    t1 = msp.add_text(f" {col1}", dxfattribs={"layer": "ТАБЛИЦА_БАЛАНСА", "height": tb_th})
                    t1.set_placement((tb_x0 + tb_th * 0.4, cur_y - row_h * 0.5), align=TextEntityAlignment.MIDDLE_LEFT)
                    # Текст колонки 2 (значение)
                    t2 = msp.add_text(f"{col2} ", dxfattribs={"layer": "ТАБЛИЦА_БАЛАНСА", "height": tb_th})
                    t2.set_placement((tb_x0 + tb_w - tb_th * 0.4, cur_y - row_h * 0.5), align=TextEntityAlignment.MIDDLE_RIGHT)

                cur_y -= row_h

        # Сохранение файла
        doc.saveas(output_path)
        return True, f"Картограмма успешно экспортирована в DXF:\n{output_path}"

    except Exception as e:
        return False, f"Ошибка при формировании файла DXF: {e}"
