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
- Текстовый блок результатов расчёта (объемы, площади, мощность слоя)
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
    "РЕЗУЛЬТАТЫ_РАСЧЕТА": {"color": 7, "lineweight": 25},       # 7 = White/Black
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


def _add_results_text_block(
    msp,
    calc_results: Dict[str, Any],
    anchor_x: float,
    anchor_y: float,
    text_height: float = 0.35,
):
    """
    Добавляет простой текстовый блок с результатами расчёта в DXF.
    Текст в столбик, без рамок и линий, без указания программы и версии.
    """
    layer = "РЕЗУЛЬТАТЫ_РАСЧЕТА"
    th = text_height
    line_spacing = th * 2.2

    lines = [
        "РЕЗУЛЬТАТЫ РАСЧЕТА ОБЪЕМА",
        "",
        f"Объем насыпи (Fill):    {calc_results.get('v_fill', 0.0):.3f} м³",
        f"Объем выемки (Cut):     {calc_results.get('v_cut', 0.0):.3f} м³",
        f"ИТОГОВЫЙ ОБЪЕМ (Net):   {calc_results.get('v_net', 0.0):.3f} м³",
        "",
        f"Площадь контура (2D):   {calc_results.get('area_2d', 0.0):.3f} м²",
        f"Площадь верха (3D):     {calc_results.get('top_area_3d', 0.0):.3f} м²",
        f"Площадь низа (3D):      {calc_results.get('bot_area_3d', 0.0):.3f} м²",
        "",
        f"Средняя мощность слоя:  {calc_results.get('avg_thickness', 0.0):.3f} м",
        f"Макс. мощность слоя:    {calc_results.get('max_thickness', 0.0):.3f} м",
        f"Мин. мощность слоя:     {calc_results.get('min_thickness', 0.0):.3f} м",
    ]

    cur_y = anchor_y
    for line_text in lines:
        if line_text == "":
            cur_y -= line_spacing * 0.5
            continue
        t = msp.add_text(line_text, dxfattribs={"layer": layer, "height": th})
        t.set_placement((anchor_x, cur_y), align=TextEntityAlignment.LEFT)
        cur_y -= line_spacing


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

                # Отметки в узлах по ГОСТ (выстраиваются в 3 аккуратные строки справа от узла):
                # 1. Красная проектная отметка (Верх)
                t_red = msp.add_text(f"{z_t:.2f}", dxfattribs={"layer": "ОТМЕТКИ_КРАСНЫЕ", "height": th})
                t_red.set_placement((xc + th * 0.25, yc + th * 0.40), align=TextEntityAlignment.LEFT)

                # 2. Черная фактическая отметка (Низ) - на 1 надпись ниже верха
                t_blk = msp.add_text(f"{z_b:.2f}", dxfattribs={"layer": "ОТМЕТКИ_ЧЕРНЫЕ", "height": th})
                t_blk.set_placement((xc + th * 0.25, yc - th * 0.80), align=TextEntityAlignment.LEFT)

                # 3. Рабочая отметка / разница высот (dh) - на 2 надписи ниже верха (под отметкой низа)
                sign_str = "+" if dh > 0.005 else ("-" if dh < -0.005 else "")
                dh_str = f"{sign_str}{abs(dh):.2f}"
                layer_dh = "ОТМЕТКИ_РАБОЧИЕ_НАСЫПЬ" if dh >= 0 else "ОТМЕТКИ_РАБОЧИЕ_ВЫЕМКА"
                t_dh = msp.add_text(dh_str, dxfattribs={"layer": layer_dh, "height": th})
                t_dh.set_placement((xc + th * 0.25, yc - th * 2.00), align=TextEntityAlignment.LEFT)

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

        # ── 9. Текстовый блок результатов расчёта ───────────────
        if include_balance_table:
            all_cad_x = [to_cad(pt[0], pt[1])[0] for pt in poly_2d]
            all_cad_y = [to_cad(pt[0], pt[1])[1] for pt in poly_2d]
            tb_x0 = max(all_cad_x) + step * 0.8
            tb_y0 = max(all_cad_y)
            _add_results_text_block(msp, calc_results, tb_x0, tb_y0, text_height=th * 0.65)

        # Сохранение файла
        doc.saveas(output_path)
        return True, f"Картограмма успешно экспортирована в DXF:\n{output_path}"

    except Exception as e:
        return False, f"Ошибка при формировании файла DXF: {e}"


def export_surface_3d_dxf(
    calc_results: Dict[str, Any],
    points: List[Any],
    boundary_indices: List[int],
    output_path: str,
    coord_swap: bool = True
) -> Tuple[bool, str]:
    """
    Экспортирует 3D модель поверхности в формате AutoCAD DXF (3DFACE).
    Для выемки (cut) экспортируется нижняя поверхность (дно).
    Для насыпи (fill) экспортируется верхняя поверхность (насыпь).
    Для планировки (grading) экспортируются обе поверхности на раздельных слоях.
    """
    if not EZDXF_AVAILABLE:
        return False, "Библиотека ezdxf не установлена."

    try:
        doc = ezdxf.new("R2010")
        doc.header["$DWGCODEPAGE"] = "ANSI_1251"
        msp = doc.modelspace()

        def to_cad(x_geo: float, y_geo: float) -> Tuple[float, float]:
            if coord_swap:
                return float(y_geo), float(x_geo)
            return float(x_geo), float(y_geo)

        # Контур границы
        boundary = calc_results.get("boundary") if calc_results else None
        if (boundary is None or len(boundary) < 3) and points and boundary_indices and len(boundary_indices) >= 3:
            boundary = np.array([[points[i].x, points[i].y, points[i].h] for i in boundary_indices])

        doc.layers.add("0_ГРАНИЦА_РАБОТ", color=1, lineweight=50)
        if boundary is not None and len(boundary) >= 3:
            cad_bound = [to_cad(pt[0], pt[1]) for pt in boundary[:, :2]]
            msp.add_lwpolyline(cad_bound, close=True, dxfattribs={"layer": "0_ГРАНИЦА_РАБОТ", "lineweight": 50})

        # Определение типа земляных работ
        work_type = calc_results.get("work_type", "auto") if calc_results else "auto"
        v_cut = calc_results.get("v_cut", 0.0) if calc_results else 0.0
        v_fill = calc_results.get("v_fill", 0.0) if calc_results else 0.0
        is_cut = (work_type == "cut" or (work_type in ("auto", "grading") and v_cut > v_fill))

        # Для выемок - нижняя поверхность, для насыпей - верхняя
        surf_name = "ДНО_ВЫЕМКА" if is_cut else "ВЕРХ_НАСЫПЬ"
        surf_color = 1 if is_cut else 3
        layer_surf = f"3D_ПОВЕРХНОСТЬ_{surf_name}"
        doc.layers.add(layer_surf, color=surf_color)

        # Получаем точки целевой поверхности
        if is_cut:
            surf_pts = calc_results.get("bottom_surface_pts")
            if surf_pts is None or len(surf_pts) == 0:
                surf_pts = np.array([[p.x, p.y, p.h] for p in points if getattr(p, "surface_type", None) == "bottom"])
        else:
            surf_pts = calc_results.get("top_surface_pts")
            if surf_pts is None or len(surf_pts) == 0:
                surf_pts = np.array([[p.x, p.y, p.h] for p in points if getattr(p, "surface_type", None) == "top"])

        if surf_pts is None or len(surf_pts) == 0:
            surf_pts = np.array([[p.x, p.y, p.h] for p in points])

        # Триангуляция и построение 3DFACE
        if len(surf_pts) >= 3:
            from scipy.spatial import Delaunay
            tri = Delaunay(surf_pts[:, :2])
            for simplex in tri.simplices:
                p1, p2, p3 = surf_pts[simplex[0]], surf_pts[simplex[1]], surf_pts[simplex[2]]
                c1 = to_cad(p1[0], p1[1])
                c2 = to_cad(p2[0], p2[1])
                c3 = to_cad(p3[0], p3[1])
                msp.add_3dface(
                    [(c1[0], c1[1], float(p1[2])),
                     (c2[0], c2[1], float(p2[2])),
                     (c3[0], c3[1], float(p3[2])),
                     (c3[0], c3[1], float(p3[2]))],
                    dxfattribs={"layer": layer_surf}
                )

        # Точки съёмки
        doc.layers.add("ТОЧКИ_СЪЕМКИ", color=7)
        for p in points:
            xc, yc = to_cad(p.x, p.y)
            msp.add_point((xc, yc, float(p.h)), dxfattribs={"layer": "ТОЧКИ_СЪЕМКИ"})
            t = msp.add_text(f"{p.id} ({p.h:.2f})", dxfattribs={"layer": "ТОЧКИ_СЪЕМКИ", "height": 0.5})
            t.set_placement((xc + 0.3, yc + 0.3, float(p.h)), align=TextEntityAlignment.LEFT)

        # Текстовый блок результатов расчёта
        if boundary is not None and len(boundary) >= 3 and calc_results:
            all_cad_x = [to_cad(pt[0], pt[1])[0] for pt in boundary[:, :2]]
            all_cad_y = [to_cad(pt[0], pt[1])[1] for pt in boundary[:, :2]]
            tb_x0 = max(all_cad_x) + (max(all_cad_x) - min(all_cad_x)) * 0.1 + 2.0
            tb_y0 = max(all_cad_y)
            doc.layers.add("РЕЗУЛЬТАТЫ_РАСЧЕТА", color=7, lineweight=25)
            _add_results_text_block(msp, calc_results, tb_x0, tb_y0, text_height=0.35)

        doc.saveas(output_path)
        return True, f"3D модель ({surf_name}) успешно экспортирована в DXF:\n{output_path}"
    except Exception as e:
        return False, f"Ошибка при экспорте 3D модели в DXF: {e}"


def export_tin_dxf(
    calc_results: Dict[str, Any],
    points: List[Any],
    boundary_indices: List[int],
    tin_simplices: Optional[Any],
    output_path: str,
    coord_swap: bool = True
) -> Tuple[bool, str]:
    """
    Экспортирует TIN-триангуляцию в формат AutoCAD DXF:
    3D грани (3DFACE), каркасные ребра (LINE) и отметки вершин.
    Для выемки строится нижняя поверхность, для насыпи - верхняя.
    """
    if not EZDXF_AVAILABLE:
        return False, "Библиотека ezdxf не установлена."

    try:
        doc = ezdxf.new("R2010")
        doc.header["$DWGCODEPAGE"] = "ANSI_1251"
        msp = doc.modelspace()

        def to_cad(x_geo: float, y_geo: float) -> Tuple[float, float]:
            if coord_swap:
                return float(y_geo), float(x_geo)
            return float(x_geo), float(y_geo)

        work_type = calc_results.get("work_type", "auto") if calc_results else "auto"
        v_cut = calc_results.get("v_cut", 0.0) if calc_results else 0.0
        v_fill = calc_results.get("v_fill", 0.0) if calc_results else 0.0
        is_cut = (work_type == "cut" or (work_type in ("auto", "grading") and v_cut > v_fill))
        surf_label = "Дно (выемка)" if is_cut else "Верх (насыпь)"
        color_face = 1 if is_cut else 3

        doc.layers.add("0_ГРАНИЦА_РАБОТ", color=1, lineweight=50)
        doc.layers.add("TIN_ГРАНИ_3D", color=color_face)
        doc.layers.add("TIN_РЕБРА", color=8, lineweight=18)
        doc.layers.add("TIN_ВЕРШИНЫ", color=4, lineweight=15)

        # Контур границы
        boundary = calc_results.get("boundary") if calc_results else None
        if (boundary is None or len(boundary) < 3) and points and boundary_indices and len(boundary_indices) >= 3:
            boundary = np.array([[points[i].x, points[i].y, points[i].h] for i in boundary_indices])

        if boundary is not None and len(boundary) >= 3:
            cad_bound = [to_cad(pt[0], pt[1]) for pt in boundary[:, :2]]
            msp.add_lwpolyline(cad_bound, close=True, dxfattribs={"layer": "0_ГРАНИЦА_РАБОТ", "lineweight": 50})

        # Получаем треугольники TIN с учетом целевой поверхности
        if is_cut:
            surf_pts = calc_results.get("bottom_surface_pts")
            if surf_pts is None or len(surf_pts) == 0:
                surf_pts = np.array([[p.x, p.y, p.h] for p in points if getattr(p, "surface_type", None) == "bottom"])
        else:
            surf_pts = calc_results.get("top_surface_pts")
            if surf_pts is None or len(surf_pts) == 0:
                surf_pts = np.array([[p.x, p.y, p.h] for p in points if getattr(p, "surface_type", None) == "top"])

        if surf_pts is not None and len(surf_pts) >= 3:
            pts_3d = surf_pts
            from scipy.spatial import Delaunay
            tri = Delaunay(pts_3d[:, :2])
            simplices = tri.simplices
        else:
            simplices = tin_simplices if tin_simplices is not None else calc_results.get("active_simplices")
            pts_3d = calc_results.get("pts_3d")
            if pts_3d is None or len(pts_3d) == 0:
                pts_3d = np.array([[p.x, p.y, p.h] for p in points])

            if (simplices is None or len(simplices) == 0) and len(pts_3d) >= 3:
                from scipy.spatial import Delaunay
                tri = Delaunay(pts_3d[:, :2])
                simplices = tri.simplices

        if simplices is not None and len(simplices) > 0 and len(pts_3d) >= 3:
            drawn_edges = set()
            for s in simplices:
                i1, i2, i3 = int(s[0]), int(s[1]), int(s[2])
                if i1 >= len(pts_3d) or i2 >= len(pts_3d) or i3 >= len(pts_3d):
                    continue
                p1, p2, p3 = pts_3d[i1], pts_3d[i2], pts_3d[i3]
                c1 = to_cad(p1[0], p1[1])
                c2 = to_cad(p2[0], p2[1])
                c3 = to_cad(p3[0], p3[1])
                v1 = (c1[0], c1[1], float(p1[2]))
                v2 = (c2[0], c2[1], float(p2[2]))
                v3 = (c3[0], c3[1], float(p3[2]))

                # 3DFACE
                msp.add_3dface([v1, v2, v3, v3], dxfattribs={"layer": "TIN_ГРАНИ_3D"})

                # Каркасные 3D ребра
                for edge in [tuple(sorted([i1, i2])), tuple(sorted([i2, i3])), tuple(sorted([i3, i1]))]:
                    if edge not in drawn_edges:
                        drawn_edges.add(edge)
                        ea = pts_3d[edge[0]]
                        eb = pts_3d[edge[1]]
                        eca = to_cad(ea[0], ea[1])
                        ecb = to_cad(eb[0], eb[1])
                        msp.add_line((eca[0], eca[1], float(ea[2])),
                                     (ecb[0], ecb[1], float(eb[2])),
                                     dxfattribs={"layer": "TIN_РЕБРА"})

        # Вершины TIN
        for idx, p in enumerate(points):
            xc, yc = to_cad(p.x, p.y)
            msp.add_point((xc, yc, float(p.h)), dxfattribs={"layer": "TIN_ВЕРШИНЫ"})
            t = msp.add_text(f"#{idx+1} {p.id} ({p.h:.2f})", dxfattribs={"layer": "TIN_ВЕРШИНЫ", "height": 0.45})
            t.set_placement((xc + 0.25, yc + 0.25, float(p.h)), align=TextEntityAlignment.LEFT)

        # Текстовый блок результатов расчёта
        if boundary is not None and len(boundary) >= 3 and calc_results:
            all_cad_x = [to_cad(pt[0], pt[1])[0] for pt in boundary[:, :2]]
            all_cad_y = [to_cad(pt[0], pt[1])[1] for pt in boundary[:, :2]]
            tb_x0 = max(all_cad_x) + (max(all_cad_x) - min(all_cad_x)) * 0.1 + 2.0
            tb_y0 = max(all_cad_y)
            doc.layers.add("РЕЗУЛЬТАТЫ_РАСЧЕТА", color=7, lineweight=25)
            _add_results_text_block(msp, calc_results, tb_x0, tb_y0, text_height=0.35)

        doc.saveas(output_path)
        return True, f"TIN-триангуляция ({surf_label}) успешно экспортирована в DXF:\n{output_path}"
    except Exception as e:
        return False, f"Ошибка при экспорте TIN в DXF: {e}"


def export_contours_dxf(
    calc_results: Dict[str, Any],
    points: List[Any],
    boundary_indices: List[int],
    output_path: str,
    contour_step: Optional[float] = None,
    coord_swap: bool = True
) -> Tuple[bool, str]:
    """
    Экспортирует топографические горизонтали (изолинии) в формат AutoCAD DXF.
    Для выемки строятся горизонтали нижней поверхности, для насыпи - верхней.
    """
    if not EZDXF_AVAILABLE:
        return False, "Библиотека ezdxf не установлена."

    try:
        doc = ezdxf.new("R2010")
        doc.header["$DWGCODEPAGE"] = "ANSI_1251"
        msp = doc.modelspace()

        def to_cad(x_geo: float, y_geo: float) -> Tuple[float, float]:
            if coord_swap:
                return float(y_geo), float(x_geo)
            return float(x_geo), float(y_geo)

        work_type = calc_results.get("work_type", "auto") if calc_results else "auto"
        v_cut = calc_results.get("v_cut", 0.0) if calc_results else 0.0
        v_fill = calc_results.get("v_fill", 0.0) if calc_results else 0.0
        is_cut = (work_type == "cut" or (work_type in ("auto", "grading") and v_cut > v_fill))
        surf_label = "нижняя поверхность (выемка)" if is_cut else "верхняя поверхность (насыпь)"

        doc.layers.add("0_ГРАНИЦА_РАБОТ", color=1, lineweight=50)
        doc.layers.add("ГОРИЗОНТАЛИ_ОСНОВНЫЕ", color=8, lineweight=25)
        doc.layers.add("ГОРИЗОНТАЛИ_УТОЛЩЕННЫЕ", color=30, lineweight=50)
        doc.layers.add("ОТМЕТКИ_ГОРИЗОНТАЛЕЙ", color=7, lineweight=15)

        # Контур границы
        boundary = calc_results.get("boundary") if calc_results else None
        if (boundary is None or len(boundary) < 3) and points and boundary_indices and len(boundary_indices) >= 3:
            boundary = np.array([[points[i].x, points[i].y, points[i].h] for i in boundary_indices])

        if boundary is not None and len(boundary) >= 3:
            cad_bound = [to_cad(pt[0], pt[1]) for pt in boundary[:, :2]]
            msp.add_lwpolyline(cad_bound, close=True, dxfattribs={"layer": "0_ГРАНИЦА_РАБОТ", "lineweight": 50})

        # Целевая поверхность: нижняя для выемки, верхняя для насыпи
        gx = calc_results.get("grid_x")
        gy = calc_results.get("grid_y")
        z_grid = calc_results.get("z_bot_grid") if is_cut else calc_results.get("z_top_grid")

        # Fallback интерполяция при отсутствии сетки
        if gx is None or gy is None or z_grid is None:
            if is_cut:
                surf_pts = calc_results.get("bottom_surface_pts")
                if surf_pts is None or len(surf_pts) == 0:
                    surf_pts = np.array([[p.x, p.y, p.h] for p in points if getattr(p, "surface_type", None) == "bottom"])
            else:
                surf_pts = calc_results.get("top_surface_pts")
                if surf_pts is None or len(surf_pts) == 0:
                    surf_pts = np.array([[p.x, p.y, p.h] for p in points if getattr(p, "surface_type", None) == "top"])

            if surf_pts is None or len(surf_pts) < 3:
                surf_pts = np.array([[p.x, p.y, p.h] for p in points])

            if len(surf_pts) >= 3:
                from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
                x_min, x_max = float(np.min(surf_pts[:, 0])), float(np.max(surf_pts[:, 0]))
                y_min, y_max = float(np.min(surf_pts[:, 1])), float(np.max(surf_pts[:, 1]))
                gx = np.linspace(x_min, x_max, 100)
                gy = np.linspace(y_min, y_max, 100)
                gx_m, gy_m = np.meshgrid(gx, gy)
                lin = LinearNDInterpolator(surf_pts[:, :2], surf_pts[:, 2])
                near = NearestNDInterpolator(surf_pts[:, :2], surf_pts[:, 2])
                z_grid = lin(gx_m, gy_m)
                mask_nan = np.isnan(z_grid)
                if np.any(mask_nan):
                    z_grid[mask_nan] = near(gx_m[mask_nan], gy_m[mask_nan])

        if z_grid is not None and gx is not None and gy is not None:
            z_valid = z_grid[~np.isnan(z_grid)]
            if len(z_valid) > 0:
                z_min = float(np.min(z_valid))
                z_max = float(np.max(z_valid))
                delta_z = max(z_max - z_min, 0.01)

                c_step = contour_step
                if c_step is None or c_step <= 0:
                    if delta_z <= 0.6:
                        c_step = 0.05
                    elif delta_z <= 1.5:
                        c_step = 0.1
                    elif delta_z <= 4.0:
                        c_step = 0.25
                    elif delta_z <= 10.0:
                        c_step = 0.5
                    elif delta_z <= 25.0:
                        c_step = 1.0
                    else:
                        c_step = 2.0

                start_level = math.floor(z_min / c_step) * c_step
                levels = np.arange(start_level, z_max + c_step * 0.5, c_step)

                # Генерация горизонталей через Matplotlib
                from matplotlib.figure import Figure
                fig = Figure()
                ax = fig.add_subplot(111)
                if gx.ndim == 1:
                    X_m, Y_m = np.meshgrid(gx, gy)
                else:
                    X_m, Y_m = gx, gy

                cs = ax.contour(X_m, Y_m, z_grid, levels=levels)

                for lvl_idx, level in enumerate(cs.levels):
                    is_index = (abs(round(level / (c_step * 5)) * (c_step * 5) - level) < 1e-4)
                    layer_name = "ГОРИЗОНТАЛИ_УТОЛЩЕННЫЕ" if is_index else "ГОРИЗОНТАЛИ_ОСНОВНЫЕ"
                    lw = 50 if is_index else 25

                    segs = cs.allsegs[lvl_idx] if hasattr(cs, "allsegs") else []
                    for seg in segs:
                        if len(seg) < 2:
                            continue
                        cad_pts = [to_cad(pt[0], pt[1]) for pt in seg]
                        msp.add_lwpolyline(cad_pts, dxfattribs={"layer": layer_name, "elevation": float(level), "lineweight": lw})

                        # Подпись отметки горизонтали около середины
                        mid_idx = len(cad_pts) // 2
                        mx, my = cad_pts[mid_idx]
                        t = msp.add_text(f"{level:.2f}", dxfattribs={"layer": "ОТМЕТКИ_ГОРИЗОНТАЛЕЙ", "height": 0.40})
                        t.set_placement((mx + 0.15, my + 0.15, float(level)), align=TextEntityAlignment.LEFT)

        # Текстовый блок результатов расчёта
        if boundary is not None and len(boundary) >= 3 and calc_results:
            all_cad_x = [to_cad(pt[0], pt[1])[0] for pt in boundary[:, :2]]
            all_cad_y = [to_cad(pt[0], pt[1])[1] for pt in boundary[:, :2]]
            tb_x0 = max(all_cad_x) + (max(all_cad_x) - min(all_cad_x)) * 0.1 + 2.0
            tb_y0 = max(all_cad_y)
            doc.layers.add("РЕЗУЛЬТАТЫ_РАСЧЕТА", color=7, lineweight=25)
            _add_results_text_block(msp, calc_results, tb_x0, tb_y0, text_height=0.35)

        doc.saveas(output_path)
        return True, f"Горизонтали ({surf_label}) успешно экспортированы в DXF:\n{output_path}"
    except Exception as e:
        return False, f"Ошибка при экспорте горизонталей в DXF: {e}"


def export_diff_dxf(
    calc_results: Dict[str, Any],
    points: List[Any],
    boundary_indices: List[int],
    output_path: str,
    coord_swap: bool = True
) -> Tuple[bool, str]:
    """
    Экспортирует карту перепада высот (разницы поверхностей) в DXF.
    Включает границу работ, нулевую линию баланса, сетку перепада и отметки.
    """
    if not EZDXF_AVAILABLE:
        return False, "Библиотека ezdxf не установлена."

    try:
        doc = ezdxf.new("R2010")
        doc.header["$DWGCODEPAGE"] = "ANSI_1251"
        msp = doc.modelspace()

        def to_cad(x_geo: float, y_geo: float) -> Tuple[float, float]:
            if coord_swap:
                return float(y_geo), float(x_geo)
            return float(x_geo), float(y_geo)

        doc.layers.add("0_ГРАНИЦА_РАБОТ", color=1, lineweight=50)
        doc.layers.add("0_НУЛЕВАЯ_ЛИНИЯ", color=4, lineweight=35, linetype="DASHED")
        doc.layers.add("ПЕРЕПАД_ВЫСОТ_НАСЫПЬ", color=3)
        doc.layers.add("ПЕРЕПАД_ВЫСОТ_ВЫЕМКА", color=1)
        doc.layers.add("ОТМЕТКИ_РАБОЧИЕ", color=30, lineweight=15)

        # Контур границы
        boundary = calc_results.get("boundary") if calc_results else None
        if (boundary is None or len(boundary) < 3) and points and boundary_indices and len(boundary_indices) >= 3:
            boundary = np.array([[points[i].x, points[i].y, points[i].h] for i in boundary_indices])

        if boundary is not None and len(boundary) >= 3:
            cad_bound = [to_cad(pt[0], pt[1]) for pt in boundary[:, :2]]
            msp.add_lwpolyline(cad_bound, close=True, dxfattribs={"layer": "0_ГРАНИЦА_РАБОТ", "lineweight": 50})

        # Сетка разности
        gx = calc_results.get("grid_x")
        gy = calc_results.get("grid_y")
        zt = calc_results.get("z_top_grid")
        zb = calc_results.get("z_bot_grid")

        if gx is not None and gy is not None and zt is not None and zb is not None:
            dh_grid = zt - zb
            if gx.ndim == 1:
                X_m, Y_m = np.meshgrid(gx, gy)
            else:
                X_m, Y_m = gx, gy

            # Нулевая линия баланса
            from matplotlib.figure import Figure
            fig = Figure()
            ax = fig.add_subplot(111)
            cs = ax.contour(X_m, Y_m, dh_grid, levels=[0.0])
            for segs in cs.allsegs:
                for seg in segs:
                    if len(seg) >= 2:
                        cad_pts = [to_cad(pt[0], pt[1]) for pt in seg]
                        msp.add_lwpolyline(cad_pts, dxfattribs={"layer": "0_НУЛЕВАЯ_ЛИНИЯ", "linetype": "DASHED", "lineweight": 35})

            # Подписи разницы высот в узлах
            step_stride = max(1, len(gx) // 30)
            for r in range(0, dh_grid.shape[0], step_stride):
                for c in range(0, dh_grid.shape[1], step_stride):
                    dh = float(dh_grid[r, c])
                    if not np.isnan(dh):
                        px = float(X_m[r, c])
                        py = float(Y_m[r, c])
                        xc, yc = to_cad(px, py)
                        sign_s = "+" if dh > 0.005 else ("-" if dh < -0.005 else "")
                        layer = "ПЕРЕПАД_ВЫСОТ_НАСЫПЬ" if dh >= 0 else "ПЕРЕПАД_ВЫСОТ_ВЫЕМКА"
                        t = msp.add_text(f"{sign_s}{abs(dh):.2f}", dxfattribs={"layer": layer, "height": 0.4})
                        t.set_placement((xc + 0.1, yc + 0.1, dh), align=TextEntityAlignment.LEFT)

        # Текстовый блок результатов расчёта
        if boundary is not None and len(boundary) >= 3 and calc_results:
            all_cad_x = [to_cad(pt[0], pt[1])[0] for pt in boundary[:, :2]]
            all_cad_y = [to_cad(pt[0], pt[1])[1] for pt in boundary[:, :2]]
            tb_x0 = max(all_cad_x) + (max(all_cad_x) - min(all_cad_x)) * 0.1 + 2.0
            tb_y0 = max(all_cad_y)
            doc.layers.add("РЕЗУЛЬТАТЫ_РАСЧЕТА", color=7, lineweight=25)
            _add_results_text_block(msp, calc_results, tb_x0, tb_y0, text_height=0.35)

        doc.saveas(output_path)
        return True, f"Карта перепада высот успешно экспортирована в DXF:\n{output_path}"
    except Exception as e:
        return False, f"Ошибка при экспорте перепада высот в DXF: {e}"


def export_plan_2d_dxf(
    calc_results: Dict[str, Any],
    points: List[Any],
    boundary_indices: List[int],
    output_path: str,
    coord_swap: bool = True
) -> Tuple[bool, str]:
    """
    Экспортирует 2D план съёмки в формат AutoCAD DXF.
    Включает: контур границы, точки съёмки, отметки в узлах сетки
    (красная проектная / черная фактическая / рабочая разница) и ведомость объёмов.
    """
    if not EZDXF_AVAILABLE:
        return False, "Библиотека ezdxf не установлена."

    try:
        doc = ezdxf.new("R2010")
        doc.header["$DWGCODEPAGE"] = "ANSI_1251"
        msp = doc.modelspace()

        def to_cad(x_geo: float, y_geo: float) -> Tuple[float, float]:
            if coord_swap:
                return float(y_geo), float(x_geo)
            return float(x_geo), float(y_geo)

        work_type = calc_results.get("work_type", "auto") if calc_results else "auto"
        v_cut = calc_results.get("v_cut", 0.0) if calc_results else 0.0
        v_fill = calc_results.get("v_fill", 0.0) if calc_results else 0.0
        is_cut = (work_type == "cut" or (work_type in ("auto", "grading") and v_cut > v_fill))
        surf_label = "2D Plan (cut/bottom)" if is_cut else "2D Plan (fill/top)"

        # Слои
        for lname, lattr in DEFAULT_DXF_LAYERS.items():
            if lname not in doc.layers:
                layer = doc.layers.add(lname, color=lattr["color"])
                if "lineweight" in lattr:
                    layer.dxf.lineweight = lattr["lineweight"]
        doc.layers.add("ТОЧКИ_СЪЕМКИ", color=3 if not is_cut else 1, lineweight=20)
        doc.layers.add("ПОДПИСИ_ТОЧЕК", color=7, lineweight=15)

        # Контур границы
        boundary = calc_results.get("boundary") if calc_results else None
        if (boundary is None or len(boundary) < 3) and points and boundary_indices and len(boundary_indices) >= 3:
            boundary = np.array([[points[i].x, points[i].y, points[i].h] for i in boundary_indices])

        poly_2d = None
        if boundary is not None and len(boundary) >= 3:
            poly_2d = boundary[:, :2]
            cad_bound = [to_cad(pt[0], pt[1]) for pt in poly_2d]
            msp.add_lwpolyline(cad_bound, close=True, dxfattribs={"layer": "0_ГРАНИЦА_РАБОТ", "lineweight": 50})

        # Точки съёмки
        target_surf = "bottom" if is_cut else "top"
        target_pts = [p for p in points if getattr(p, "surface_type", None) in (target_surf, None)]
        if not target_pts:
            target_pts = points
        for idx, p in enumerate(target_pts):
            xc, yc = to_cad(p.x, p.y)
            msp.add_point((xc, yc, float(p.h)), dxfattribs={"layer": "ТОЧКИ_СЪЕМКИ"})
            pid_str = f"#{idx+1}" if not getattr(p, "id", None) else str(p.id)
            t = msp.add_text(f"{pid_str} ({p.h:.2f})", dxfattribs={"layer": "ПОДПИСИ_ТОЧЕК", "height": 0.45})
            t.set_placement((xc + 0.25, yc + 0.25, float(p.h)), align=TextEntityAlignment.LEFT)

        # Отметки в узлах координатной сетки
        if poly_2d is not None and calc_results:
            x_min, x_max = float(np.min(poly_2d[:, 0])), float(np.max(poly_2d[:, 0]))
            y_min, y_max = float(np.min(poly_2d[:, 1])), float(np.max(poly_2d[:, 1]))
            span_x, span_y = x_max - x_min, y_max - y_min
            step = choose_auto_grid_step(span_x, span_y)
            th = max(0.20, step * 0.065)

            import matplotlib.path as mpl_path
            bound_path = mpl_path.Path(poly_2d)

            from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator
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

            top_pts_arr = calc_results.get("top_surface_pts")
            if top_pts_arr is None or len(top_pts_arr) == 0:
                top_pts_arr = np.array([[p.x, p.y, p.h] for p in points if getattr(p, "surface_type", None) == "top"])
            bot_pts_arr = calc_results.get("bottom_surface_pts")
            if bot_pts_arr is None or len(bot_pts_arr) == 0:
                bot_pts_arr = np.array([[p.x, p.y, p.h] for p in points if getattr(p, "surface_type", None) == "bottom"])
            fallback_top = NearestNDInterpolator(top_pts_arr[:, :2], top_pts_arr[:, 2]) \
                if top_pts_arr is not None and len(top_pts_arr) > 0 else None
            fallback_bot = NearestNDInterpolator(bot_pts_arr[:, :2], bot_pts_arr[:, 2]) \
                if bot_pts_arr is not None and len(bot_pts_arr) > 0 else None

            x_coords = np.arange(math.floor(x_min / step) * step,
                                  math.ceil(x_max / step) * step + step * 0.5, step)
            y_coords = np.arange(math.floor(y_min / step) * step,
                                  math.ceil(y_max / step) * step + step * 0.5, step)
            cross_sz = th * 0.35

            for xn in x_coords:
                for yn in y_coords:
                    if not bound_path.contains_point((xn, yn), radius=step * 0.75):
                        continue
                    z_t, z_b = np.nan, np.nan
                    if interp_top:
                        try:
                            v = interp_top((xn, yn))
                            if not np.isnan(v):
                                z_t = v
                        except Exception:
                            pass
                    if np.isnan(z_t) and fallback_top:
                        try:
                            z_t = float(fallback_top(xn, yn))
                        except Exception:
                            pass
                    if interp_bot:
                        try:
                            v = interp_bot((xn, yn))
                            if not np.isnan(v):
                                z_b = v
                        except Exception:
                            pass
                    if np.isnan(z_b) and fallback_bot:
                        try:
                            z_b = float(fallback_bot(xn, yn))
                        except Exception:
                            pass
                    if np.isnan(z_t) or np.isnan(z_b):
                        continue
                    dh = z_t - z_b
                    xc, yc = to_cad(xn, yn)
                    msp.add_line((xc - cross_sz, yc), (xc + cross_sz, yc),
                                 dxfattribs={"layer": "0_СЕТКА_КАРТОГРАММЫ"})
                    msp.add_line((xc, yc - cross_sz), (xc, yc + cross_sz),
                                 dxfattribs={"layer": "0_СЕТКА_КАРТОГРАММЫ"})
                    t_red = msp.add_text(f"{z_t:.2f}", dxfattribs={"layer": "ОТМЕТКИ_КРАСНЫЕ", "height": th})
                    t_red.set_placement((xc + th * 0.25, yc + th * 0.40), align=TextEntityAlignment.LEFT)
                    t_blk = msp.add_text(f"{z_b:.2f}", dxfattribs={"layer": "ОТМЕТКИ_ЧЕРНЫЕ", "height": th})
                    t_blk.set_placement((xc + th * 0.25, yc - th * 0.80), align=TextEntityAlignment.LEFT)
                    sign_str = "+" if dh > 0.005 else ("-" if dh < -0.005 else "")
                    dh_layer = "ОТМЕТКИ_РАБОЧИЕ_НАСЫПЬ" if dh >= 0 else "ОТМЕТКИ_РАБОЧИЕ_ВЫЕМКА"
                    t_dh = msp.add_text(f"{sign_str}{abs(dh):.2f}",
                                        dxfattribs={"layer": dh_layer, "height": th})
                    t_dh.set_placement((xc + th * 0.25, yc - th * 2.00), align=TextEntityAlignment.LEFT)

        # Текстовый блок результатов расчёта
        if calc_results and poly_2d is not None:
            all_cad_x = [to_cad(pt[0], pt[1])[0] for pt in poly_2d]
            all_cad_y = [to_cad(pt[0], pt[1])[1] for pt in poly_2d]
            step_tbl = choose_auto_grid_step(
                float(np.max(poly_2d[:, 0]) - np.min(poly_2d[:, 0])),
                float(np.max(poly_2d[:, 1]) - np.min(poly_2d[:, 1]))
            )
            tb_th = max(0.20, step_tbl * 0.065) * 0.65
            tb_x0 = max(all_cad_x) + step_tbl * 0.8
            tb_y0 = max(all_cad_y)
            _add_results_text_block(msp, calc_results, tb_x0, tb_y0, text_height=tb_th)

        doc.saveas(output_path)
        return True, f"2D план ({surf_label}) успешно экспортирован в DXF:\n{output_path}"
    except Exception as e:
        return False, f"Ошибка при экспорте 2D плана в DXF: {e}"
