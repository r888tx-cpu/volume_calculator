# -*- coding: utf-8 -*-
"""
Главный скрипт для запуска программы расчета объема.
Использование:
1) Двойной клик или `python volume_calc.py` -> открывает интерактивный графический интерфейс (GUI)
2) `python volume_calc.py [имя_файла.txt]` -> выполняет расчет в командной строке и сохраняет отчет
"""

import sys
import os

class _SafeStream:
    def __init__(self):
        self._buf = []
    def write(self, s):
        if s:
            self._buf.append(s)
            if len(self._buf) > 1000:
                self._buf = self._buf[-500:]
    def flush(self):
        pass

if getattr(sys, "stdout", None) is None:
    sys.stdout = _SafeStream()
if getattr(sys, "stderr", None) is None:
    sys.stderr = _SafeStream()

# Настройка кодировки консоли для Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app_utils import get_app_dir

def run_cli(filepath: str, grid_res: float = 0.2):
    import numpy as np
    from geo_parser import load_points_from_file
    from volume_engine import VolumeCalculator

    print(f"=== Чтение файла координат: {filepath} ===")
    points = load_points_from_file(filepath)
    if not points:
        print("Ошибка: Точки с подходящими координатами не найдены.")
        return

    print(f"Загружено точек: {len(points)}")
    for i, p in enumerate(points[:5]):
        print(f"  {i+1}) ID: {p.id:8s} | X (Север): {p.x:12.3f} | Y (Восток): {p.y:12.3f} | H: {p.h:8.3f}")
    if len(points) > 5:
        print(f"  ... и еще {len(points) - 5} точек.")

    # Разделение точек по surface_type или по медиане высоты (эвристика)
    typed_top = [p for p in points if p.surface_type == "top"]
    typed_bot = [p for p in points if p.surface_type == "bottom"]
    typed_bound = [p for p in points if p.surface_type == "boundary"]

    if typed_top or typed_bot:
        # Есть явная разметка — используем её
        top_list = typed_top if typed_top else []
        bot_list = typed_bot if typed_bot else []
        auto_pts = [p for p in points if p.surface_type == "auto"]
        if auto_pts and not top_list:
            top_list = auto_pts
        elif auto_pts and not bot_list:
            bot_list = auto_pts
        top_pts = np.array([[p.x, p.y, p.h] for p in top_list]) if top_list else np.empty((0, 3))
        bot_pts = np.array([[p.x, p.y, p.h] for p in bot_list]) if bot_list else np.empty((0, 3))
    else:
        # Эвристика: разделение по медиане высоты
        print("  ⚠ Внимание: поверхности не размечены (surface_type). "
              "Разделение по медиане высоты — результат может быть некорректным.")
        h_vals = np.array([p.h for p in points])
        med_h = np.median(h_vals)
        top_pts = np.array([[p.x, p.y, p.h] for p in points if p.h >= med_h])
        bot_pts = np.array([[p.x, p.y, p.h] for p in points if p.h < med_h])

    print("\n=== Расчет объема между поверхностями (сшивание по внешнему контуру) ===")
    calc = VolumeCalculator(top_points=top_pts, bottom_points=bot_pts, grid_resolution=grid_res)
    res = calc.calculate()

    if "error" in res:
        print(f"Ошибка расчета: {res['error']}")
        return

    report = (
        f"\n"
        f"=====================================================\n"
        f"             РЕЗУЛЬТАТЫ РАСЧЕТА ОБЪЕМА               \n"
        f"=====================================================\n"
        f"• Объем насыпи (Fill Volume):     {res['v_fill']:12.2f} м³\n"
        f"• Объем выемки (Cut Volume):      {res['v_cut']:12.2f} м³\n"
        f"-----------------------------------------------------\n"
        f"• ИТОГОВЫЙ ОБЪЕМ (Net Volume):    {res['v_net']:12.2f} м³\n\n"
        f"• Площадь контура в плане (2D):   {res['area_2d']:12.2f} м²\n"
        f"• Площадь верха (3D Surface):     {res['top_area_3d']:12.2f} м²\n"
        f"• Площадь низа (3D Surface):      {res['bot_area_3d']:12.2f} м²\n\n"
        f"• Средняя мощность слоя (пласта): {res['avg_thickness']:12.3f} м\n"
        f"• Максимальная мощность:          {res['max_thickness']:12.3f} м\n"
        f"• Минимальная мощность:           {res['min_thickness']:12.3f} м\n"
        f"• Число расчетных узлов сетки:    {res['num_grid_points']:12d}\n"
        f"• Шаг регулярной сетки:           {res['grid_resolution']:12.2f} м\n"
        f"=====================================================\n"
    )
    print(report)

    # Сохранение отчета
    report_file = os.path.splitext(filepath)[0] + "_volume_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Отчет успешно сохранен в файл: {report_file}")

def main():
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        run_cli(sys.argv[1])
        return

    try:
        from interactive_app import VolumeApp
        app = VolumeApp()
        app.mainloop()
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        cur_dir = get_app_dir()
        try:
            with open(os.path.join(cur_dir, "crash_log.txt"), "w", encoding="utf-8") as f:
                f.write(f"GUI startup error:\n{tb}\n")
        except Exception:
            pass
        try:
            import tkinter.messagebox as mb
            mb.showerror("Ошибка запуска", f"Не удалось запустить приложение:\n\n{e}\n\nПодробности записаны в crash_log.txt")
        except Exception:
            pass
        print("GUI не удалось запустить. Используйте CLI-режим:")
        print(f"  python {os.path.basename(__file__)} <файл_координат.txt>")
        sys.exit(1)

if __name__ == "__main__":
    main()
