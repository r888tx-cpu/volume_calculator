# -*- coding: utf-8 -*-
"""
Модуль для парсинга геодезических координат.

Правила автоматического распознавания:
1) Координата X (Север) — ровно 6 знаков до десятичного разделителя (например 439007.646).
2) Координата Y (Восток) — ровно 7 знаков до десятичного разделителя (например 2281305.861).
3) Высота Z / H — 3 или меньше знаков до разделителя, идет сразу после пары координат (может быть отрицательной).
4) Поддержка пользовательского переопределения колонок (column_mapping).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

@dataclass
class GeoPoint:
    id: str
    x: float  # Север X (6 знаков до точки)
    y: float  # Восток Y (7 знаков до точки)
    h: float  # Высота Z/H (<= 3 знаков до точки)
    raw_line: str = ""
    surface_type: str = "auto"  # "top", "bottom", "boundary", "auto"

def _get_integer_digits_count(val_str: str) -> int:
    """Возвращает количество цифр в целой части числа (до точки или запятой).
    Обрабатывает научную нотацию (e.g. 1.25e+06 → 7 цифр)."""
    s = val_str.strip().lstrip('+-')
    # Научная нотация → вычисляем реальное кол-во знаков
    if 'e' in s.lower():
        try:
            num = abs(float(s.replace(',', '.')))
            if num == 0:
                return 1
            s = f"{num:.0f}"
            return len(s)
        except ValueError:
            pass
    # Обычное число: находим разделитель целой/дробной части
    for sep in ('.', ','):
        pos = s.find(sep)
        if pos != -1:
            return sum(c.isdigit() for c in s[:pos])
    return sum(c.isdigit() for c in s)

def parse_line(line: str, column_mapping: Optional[Dict[str, int]] = None) -> Optional[GeoPoint]:
    line = line.strip().lstrip('\ufeff')
    if not line or line.startswith('#') or line.startswith('//'):
        return None

    if ',' in line or ';' in line or '\t' in line:
        normalized = line.replace('\t', ',').replace(';', ',')
        parts = [p.strip().lstrip('\ufeff') for p in normalized.split(',') if p.strip()]
    else:
        parts = [p.strip().lstrip('\ufeff') for p in line.split() if p.strip()]

    if not parts:
        return None

    # Если задан явный маппинг колонок от пользователя:
    if column_mapping:
        try:
            id_idx = column_mapping.get("id", 0)
            x_idx = column_mapping.get("x", 1)
            y_idx = column_mapping.get("y", 2)
            z_idx = column_mapping.get("z", 3)

            pt_id = parts[id_idx] if 0 <= id_idx < len(parts) else str(1)
            x_val = float(parts[x_idx].replace(',', '.'))
            y_val = float(parts[y_idx].replace(',', '.'))
            z_val = float(parts[z_idx].replace(',', '.'))
            return GeoPoint(id=str(pt_id), x=x_val, y=y_val, h=z_val, raw_line=line)
        except (IndexError, ValueError, KeyError):
            return None

    # Автоматическое распознавание по правилам знаков до разделителя
    numbers_with_indices = []
    for i, p in enumerate(parts):
        if ':' in p:
            continue
        clean_p = p.replace(',', '.')
        try:
            val = float(clean_p)
            digits_before = _get_integer_digits_count(p)
            numbers_with_indices.append((i, val, digits_before, p))
        except ValueError:
            continue

    if len(numbers_with_indices) < 3:
        return None

    x_val = None
    y_val = None
    h_val = None
    x_idx = -1
    y_idx = -1

    # Один проход: ищем X (6 цифр), Y (7 цифр) по правилам разрядности
    for idx, num, digits, _ in numbers_with_indices:
        if digits == 6 and x_val is None:
            x_val, x_idx = num, idx
        elif digits == 7 and y_val is None:
            y_val, y_idx = num, idx

    # Ищем Z/H: <=3 цифр, после координат или любой оставшийся
    if x_val is not None and y_val is not None:
        max_coord_idx = max(x_idx, y_idx)
        for idx, num, digits, _ in numbers_with_indices:
            if idx > max_coord_idx and digits <= 3:
                h_val = num
                break
        if h_val is None:
            for idx, num, digits, _ in numbers_with_indices:
                if idx != x_idx and idx != y_idx and digits <= 3:
                    h_val = num
                    break

    # Резервный поиск по диапазону значений, если знаки нестандартные
    if x_val is None or y_val is None or h_val is None:
        for idx, num, digits, _ in numbers_with_indices:
            if x_val is None and idx != y_idx and (100000 <= abs(num) < 1000000 or digits == 6):
                x_val, x_idx = num, idx
            elif y_val is None and idx != x_idx and (abs(num) >= 1000000 or digits == 7):
                y_val, y_idx = num, idx
        if x_val is not None and y_val is not None and h_val is None:
            for idx, num, digits, _ in numbers_with_indices:
                if idx != x_idx and idx != y_idx:
                    h_val = num
                    break

    if x_val is None or y_val is None or h_val is None:
        return None

    pt_id = parts[0]
    if len(parts) > 1 and 0 < x_idx and 0 < y_idx:
        min_c_idx = min(x_idx, y_idx)
        if min_c_idx >= 2 and not parts[1].replace('.', '').replace('-', '').isdigit() and ':' not in parts[1]:
            pt_id = f"{parts[0]}_{parts[1]}"

    return GeoPoint(id=str(pt_id), x=float(x_val), y=float(y_val), h=float(h_val), raw_line=line)

def load_points_from_file(filepath: str, column_mapping: Optional[Dict[str, int]] = None) -> List[GeoPoint]:
    points = []
    with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
        for line in f:
            pt = parse_line(line, column_mapping=column_mapping)
            if pt:
                points.append(pt)
    return points

