# -*- coding: utf-8 -*-
"""
Интерактивное графическое приложение (GUI) на базе Tkinter и Matplotlib
для визуализации точек, интерактивного выбора контура сшивания и расчета объемов.

Доработки:
- Правая кнопка мыши отменяет последние действия (undo).
- Tooltip при наведении курсора на точку (№, H).
- Зум колесом мыши вокруг курсора.
- Блокировка выбора точек при активном инструменте зума/пана.
- Зум не сбрасывается при клике на точку.
- Автосохранение проекта (выделенные точки, контур) в .volproj файл.
- Выбор сохранённых проектов из списка в левой панели.
"""

import os
import sys
import glob
import json
import re
import shutil
import math
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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

import customtkinter as ctk
ctk.set_appearance_mode("system")    # "dark" / "light" / "system" — следует настройке Windows
ctk.set_default_color_theme("blue")  # Базовая тема
ctk.deactivate_automatic_dpi_awareness()
ctk.CTkToplevel._deactivate_windows_window_header_manipulation = True

from customtkinter.windows.widgets.scaling.scaling_tracker import ScalingTracker

# Защита от бага CustomTkinter: ScalingTracker устанавливает window.attributes("-alpha", 0.15),
# а затем падает при вызове block_update_dimensions_event на стандартных окнах tk.Tk / TkinterDnD,
# из-за чего главное окно зависает в полупрозрачном состоянии (15% непрозрачности) и уменьшается в размере.
@classmethod
def _safe_check_dpi_scaling(cls):
    try:
        new_scaling_detected = False
        for window in list(cls.window_widgets_dict.keys()):
            if window.winfo_exists() and not window.state() == "iconic":
                current_dpi_scaling_value = cls.get_window_dpi_scaling(window)
                if current_dpi_scaling_value != cls.window_dpi_scaling_dict.get(window):
                    cls.window_dpi_scaling_dict[window] = current_dpi_scaling_value
                    if hasattr(window, "block_update_dimensions_event"):
                        window.block_update_dimensions_event()
                    try:
                        cls.update_scaling_callbacks_for_window(window)
                    except Exception:
                        pass
                    if hasattr(window, "unblock_update_dimensions_event"):
                        window.unblock_update_dimensions_event()
                    new_scaling_detected = True
        for app in list(cls.window_widgets_dict.keys()):
            try:
                if new_scaling_detected:
                    app.after(cls.loop_pause_after_new_scaling, cls.check_dpi_scaling)
                else:
                    app.after(cls.update_loop_interval, cls.check_dpi_scaling)
                return
            except Exception:
                continue
        cls.update_loop_running = False
    except Exception:
        cls.update_loop_running = False

ScalingTracker.check_dpi_scaling = _safe_check_dpi_scaling


from ui_dialogs import (
    set_window_dark_titlebar,
    ProjectNavigationToolbar,
    PlainOffsetFormatter,
    PointEditDialog,
    CoordinateRemapDialog,
    QuickAddPointDialog,
    FullScreen3DViewer,
    calc_3d_stride,
    TINTableDialog,
    ToolTip,
    add_tooltip,
)

# Кастомная палитра: благородный темно-серый / графит вместо ярко-синего
ctk.ThemeManager.theme["CTkButton"]["fg_color"] = ["#4a5056", "#343a40"]
ctk.ThemeManager.theme["CTkButton"]["hover_color"] = ["#5a6268", "#434a52"]
ctk.ThemeManager.theme["CTkButton"]["border_color"] = ["#6c757d", "#495057"]
ctk.ThemeManager.theme["CTkButton"]["text_color"] = ["#ffffff", "#f8f9fa"]

ctk.ThemeManager.theme["CTkSegmentedButton"]["fg_color"] = ["#ced4da", "#1e2124"]
ctk.ThemeManager.theme["CTkSegmentedButton"]["selected_color"] = ["#ffffff", "#3e444c"]
ctk.ThemeManager.theme["CTkSegmentedButton"]["selected_hover_color"] = ["#f8f9fa", "#4b535d"]
ctk.ThemeManager.theme["CTkSegmentedButton"]["unselected_color"] = ["#dee2e6", "#282c30"]
ctk.ThemeManager.theme["CTkSegmentedButton"]["unselected_hover_color"] = ["#ced4da", "#33383e"]
ctk.ThemeManager.theme["CTkSegmentedButton"]["text_color"] = ["#212529", "#f8f9fa"]

ctk.ThemeManager.theme["CTkCheckBox"]["fg_color"] = ["#4a5056", "#3e444c"]
ctk.ThemeManager.theme["CTkCheckBox"]["hover_color"] = ["#5a6268", "#4b535d"]

ctk.ThemeManager.theme["CTkComboBox"]["button_color"] = ["#4a5056", "#343a40"]
ctk.ThemeManager.theme["CTkComboBox"]["button_hover_color"] = ["#5a6268", "#434a52"]
ctk.ThemeManager.theme["CTkComboBox"]["border_color"] = ["#adb5bd", "#495057"]

ctk.ThemeManager.theme["CTkEntry"]["border_color"] = ["#adb5bd", "#495057"]
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath
from matplotlib.colors import LightSource, LinearSegmentedColormap
from scipy.spatial import ConvexHull, Delaunay

# Дневные палитры картограммы масс (Мягкий дневной рельеф)
# От чистого светлого основания (0.92) к мягкому серому тону (0.56 для насыпи, 0.52 для выемки)
CMAP_SWISS_FILL = LinearSegmentedColormap.from_list(
    "SwissFill",
    [(0.92, 0.92, 0.92), (0.56, 0.56, 0.56)]
)
CMAP_SWISS_CUT = LinearSegmentedColormap.from_list(
    "SwissCut",
    [(0.92, 0.92, 0.92), (0.52, 0.52, 0.52)]
)

def _smooth_grid_np(grid: np.ndarray, passes: int = 2) -> np.ndarray:
    """Быстрое сглаживание 2D сетки скользящим окном без использования scipy.ndimage"""
    res = grid.copy()
    for _ in range(passes):
        padded = np.pad(res, ((1, 1), (1, 1)), mode="edge")
        res = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
            padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0
    return res


def _gaussian_blur_2d_np(arr: np.ndarray, sigma: float = 1.8) -> np.ndarray:
    """Изотропное гауссово сглаживание 2D массива на чистом NumPy (без scipy.ndimage/signal)."""
    rad = int(np.ceil(2.5 * sigma))
    if rad < 1:
        return arr.copy()
    x = np.arange(-rad, rad + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    pad = ((rad, rad), (rad, rad))
    padded = np.pad(arr, pad, mode="edge")
    temp = np.zeros_like(padded)
    for i, w in enumerate(k):
        temp[:, rad:-rad] += w * padded[:, i:padded.shape[1] - (len(k) - 1 - i)]
    out = np.zeros_like(arr)
    for i, w in enumerate(k):
        out += w * temp[i:temp.shape[0] - (len(k) - 1 - i), rad:-rad]
    return out


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Кубическая интерполяция Эрмита (smoothstep) с нулевыми производными на границах."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _compute_swiss_multidirectional_hillshade(
    Z_screen: np.ndarray, step_northing: float, step_easting: float,
    is_cut: bool = False, dh_screen: np.ndarray | None = None
) -> np.ndarray:
    """
    Вычисляет естественную швейцарскую многонаправленную светотеневую отмывку (Swiss Multi-Directional Shading)
    по стандарту David Mark (1992) / GDAL / ESRI, объединенную с равномерным дневным
    гипсометрическим градиентом мощности/глубины (Вариант: Мягкий дневной).
    
    Оси:
    - Z_screen строки (axis 0) — Север (Northing), шаг step_northing
    - Z_screen столбцы (axis 1) — Восток (Easting), шаг step_easting
    
    Для насыпей и выемок:
      - Базовый дневной гипсометрический тон обеспечивает равномерный и отчетливый шаг серого цвета
        между всеми горизонталями мощности/глубины.
      - Многонаправленная светотень с азимута 315° СЗ придает естественный 3D-объем без резких и глухих теней.
      - Дно выемок остается светлым мягким графитом (~0.52), устраняя эффект глубокой ночной черноты.
    """
    step_n = max(1e-4, abs(step_northing))
    step_e = max(1e-4, abs(step_easting))

    # Градиенты по Хорну (Horn's 3x3 filter) с подавлением шума
    zp = np.pad(Z_screen, 1, mode="edge")
    p = ((zp[2:, 2:] + 2.0 * zp[1:-1, 2:] + zp[:-2, 2:]) -
         (zp[2:, :-2] + 2.0 * zp[1:-1, :-2] + zp[:-2, :-2])) / (8.0 * step_e)
    q = ((zp[2:, 2:] + 2.0 * zp[2:, 1:-1] + zp[2:, :-2]) -
         (zp[:-2, 2:] + 2.0 * zp[:-2, 1:-1] + zp[:-2, :-2])) / (8.0 * step_n)

    slope = np.arctan(np.sqrt(p**2 + q**2))
    # Направление вниз по склону (-p, -q): -p по Востоку, -q по Северу
    aspect = np.mod(np.arctan2(-p, -q), 2.0 * np.pi)

    angles = [315.0, 270.0, 360.0, 225.0]
    weights = [0.50, 0.20, 0.20, 0.10]
    zenith = np.radians(45.0)

    sh = np.zeros_like(Z_screen)
    for a_deg, w in zip(angles, weights):
        az_rad = np.radians(a_deg)
        s = np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(az_rad - aspect)
        sh += w * s

    # Центрированная дельта светотени относительно нейтральной плоскости (45° zenith)
    delta_sh = sh - np.cos(zenith)

    # Нормализованная мощность/глубина dh_norm для равномерного градиента уровней
    if dh_screen is not None:
        dh_valid = dh_screen[~np.isnan(dh_screen)]
        dh_max = float(np.nanmax(dh_valid)) if len(dh_valid) > 0 else 1.0
        dh_norm = np.clip(np.nan_to_num(dh_screen, nan=0.0) / (dh_max if dh_max > 0 else 1.0), 0.0, 1.0)
    else:
        dh_norm = np.zeros_like(Z_screen)

    if not is_cut:
        # Насыпь: дневная база (0.92 у подошвы -> 0.56 у гребня) + мягкая светотень 315°
        base_tone = 0.92 - 0.36 * dh_norm
        return np.clip(base_tone + 0.55 * delta_sh, 0.34, 0.98)
    else:
        # Выемка: дневная база (0.92 у бровки -> 0.52 на дне ямы) + мягкая светотень 315°
        base_tone = 0.92 - 0.40 * dh_norm
        return np.clip(base_tone + 0.48 * delta_sh, 0.34, 0.96)

# Поддержка Drag & Drop — tkinterdnd2 (если установлен)
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False

from geo_parser import GeoPoint, load_points_from_file, parse_line
from app_utils import get_app_dir, get_projects_dir, __version__, APP_NAME, APP_TITLE
from spatial_index import SpatialIndexService, polygon_self_intersects
from project_storage import ProjectStorageService, NumpyJSONEncoder

# Вспомогательные диалоги (PointEditDialog, CoordinateRemapDialog, QuickAddPointDialog, FullScreen3DViewer)
# вынесены в модуль ui_dialogs.py


_AppBase = TkinterDnD.Tk if _DND_AVAILABLE else tk.Tk

class VolumeApp(_AppBase):
    TAB_2D: str = "2D Схема в плане"
    TAB_3D: str = "3D Поверхности"
    TAB_DIFF: str = "Картограмма масс"
    TAB_TIN: str = "TIN Триангуляция"
    TAB_CONTOURS: str = "Горизонтали"
    TAB_TABLE: str = "Таблица точек"

    LOD_LABEL_THRESHOLD: int = 50  # Порог авто-показа подписей: только при <= 50 точек в кадре
    _point_labels_mode: str = "on"
    _labels_update_timer = None
    btn_labels_mode = None
    _split_height_cache = None
    _work_type_cache = None
    _point_surfaces_cache = None
    _boundary_inside_cache = None
    _boundary_indices_set = None
    _boundary_interpolator_cache = None
    _boundary_path_cache = None
    _local_surface_threshold = None
    _pan_annotations_hidden: bool = False
    _contour_drag_source_idx = None
    _box_select_start = None
    _current_labeled_points: list = []

    def __init__(self):
        super().__init__()
        self._current_labeled_points = []
        self.title(f"{APP_TITLE} — Расчет объема земляных масс по координатам (TIN / Сетка)")
        self.geometry("1280x820")
        self.minsize(1000, 650)
        try:
            self.attributes("-alpha", 1.0)
        except Exception:
            pass

        # Данные
        self.points: list[GeoPoint] = []
        self.top_indices = set()
        self.bottom_indices = set()
        self.boundary_indices = []
        self.calc_results = None
        self.current_mode = tk.StringVar(value="select_boundary")
        self._current_file_path = None
        self._current_project_dir = None
        self._column_mapping = None

        # Выделение точек рамкой (Box Selection)
        self._selected_points: set[int] = set()
        self._box_select_start = None
        self._box_select_moved = False
        self._box_select_rect_artist = None
        self._shift_pressed = False
        self._ctrl_pressed = False

        # Кеш локальной интерполяции контура и классификации поверхностей
        self._boundary_interpolator_cache = None
        self._boundary_path_cache = None
        self._local_surface_threshold = None
        self._split_height_cache = None
        self._work_type_cache = None
        self._point_surfaces_cache = None
        self._boundary_inside_cache = None
        self._boundary_indices_set = None

        # Перемещение схемы мышью (Pan)
        self._pan_start = None
        self._pan_dragged = False

        # Интерактивное редактирование контура (перетаскивание вершины по ПКМ и встраивание по double-click ЛКМ)
        self._contour_drag_source_idx = None
        self._contour_drag_pos = None
        self._contour_drag_moved = False
        self._contour_drag_target_idx = None
        self._contour_drag_start_coord = None
        self._contour_drag_artists = []
        self._contour_drag_line1 = None
        self._contour_drag_line2 = None
        self._contour_drag_guide = None
        self._contour_drag_ring = None
        self._blit_bg = None
        self._points_kdtree = None
        self._points_coords_len = 0
        self._pending_single_click_timer = None

        # Стек для отмены действий
        self._undo_stack = []

        # TIN-триангуляция (вкладка геодезических треугольников)
        self._tin_excluded: set = set()    # индексы исключённых треугольников
        self._tin_simplices = None         # np.ndarray simplices из Delaunay
        self._tin_pts_2d = None            # np.ndarray XY всех точек для TIN
        self._tin_selected_idx = None      # индекс выбранного треугольника
        self._tin_pan_start = None         # стартовая позиция панорамирования TIN
        self._tin_pan_dragged = False
        self._tin_view_initialized = False
        self._tin_patches = []             # список MplPolygon патчей для быстрой подсветки
        self._tin_updating_selection = False # флаг блокировки рекурсивных событий Treeview
        self._tin_custom_simplices = None  # пользовательская триангуляция (после переброски рёбер)
        self._diff_colorbar = None         # сохранённый colorbar картограммы мощности
        self._suppress_tab_switch = False  # подавление перехода на картограмму при авто-расчёте
        self._tin_table_visible = True     # видимость таблицы треугольников на вкладке TIN
        self._is_tab_fullscreen = False    # полноэкранный режим текущей вкладки (скрыта левая панель)
        self._saved_sash_pos = 420
        self._saved_win_state = "normal"
        self._toolbar_volume_labels = []   # центральные метки отображения объемов в тулбарах
        self._toolbars = []                # список панелей инструментов холстов
        self._fs_surface_panels = {}       # плавающие панели переключателей поверхностей для холстов
        self._pan_draw_timers = {}         # активные таймеры плавной перерисовки панорамирования

        # Флаги ленивого рендеринга неактивных вкладок (Dirty Flags)
        self._tin_dirty = False
        self._table_dirty = False
        self._3d_dirty = False
        self._diff_dirty = False
        self._2d_dirty = False
        self._contours_dirty = False
        self._raw_contours_cache = None
        self._contours_view_initialized = False
        self._contours_pan_start = None
        self._contours_pan_dragged = False
        self._auto_save_timer = None
        self._boundary_calc_timer = None

        # Быстрые постоянные графические элементы 2D схемы
        self._contour_line_artist = None
        self._contour_fill_artist = None
        self._contour_order_artists = []
        self._selected_scatter_artist = None
        self._scatter_top_artist = None
        self._scatter_bot_artist = None
        self._scatter_bound_artist = None
        self._is_separate_surfaces = False
        self._2d_visible_mask_cache = None

        # Видимость поверхностей (чекбоксы слева)
        self._show_top    = tk.BooleanVar(value=True)
        self._show_bottom = tk.BooleanVar(value=True)
        self._point_labels_mode = "on"    # "on" (показывать), "off" (скрыть)
        self._labels_update_timer = None
        self.btn_labels_mode = None

        self.canvas_2d = None
        self.canvas_3d = None
        self.canvas_diff = None
        self.canvas_tin = None
        self.canvas_contours = None
        self.fig_2d = None
        self.fig_3d = None
        self.fig_diff = None
        self.fig_tin = None
        self.fig_contours = None
        self.ax_2d = None
        self.ax_3d = None
        self.ax_diff = None
        self.ax_tin = None
        self.ax_contours = None
        self._toolbar_contours = None
        self.cbo_contour_step = None
        self._show_contour_labels = None
        self.lbl_contours_stats = None
        self.tree = None
        self.tree_tin = None
        self.lbl_tin_stats = None
        self.lbl_table_stats = None
        self._tin_min_angle = tk.IntVar(value=7)
        self._tin_max_iter = tk.IntVar(value=4)

        self._init_ui()
        self._migrate_legacy_projects()
        self._scan_saved_projects()

        # Иконка приложения — устанавливается ПОСЛЕ построения UI,
        # иначе iconbitmap() на Windows сбрасывает геометрию PanedWindow.
        try:
            candidates = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_icon.ico"),
                os.path.join(os.path.dirname(sys.executable), "_internal", "app_icon.ico"),
                os.path.join(os.path.dirname(sys.executable), "app_icon.ico"),
            ]
            if hasattr(sys, "_MEIPASS"):
                candidates.insert(0, os.path.join(sys._MEIPASS, "app_icon.ico"))
            for p in candidates:
                if os.path.exists(p):
                    self.iconbitmap(p)
                    break
        except Exception:
            pass

        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Отслеживание клавиш-модификаторов Shift и Ctrl для рамки выделения
        self.bind("<KeyPress-Shift_L>", lambda e: self._set_modifier("shift", True), add="+")
        self.bind("<KeyRelease-Shift_L>", lambda e: self._set_modifier("shift", False), add="+")
        self.bind("<KeyPress-Shift_R>", lambda e: self._set_modifier("shift", True), add="+")
        self.bind("<KeyRelease-Shift_R>", lambda e: self._set_modifier("shift", False), add="+")
        self.bind("<KeyPress-Control_L>", lambda e: self._set_modifier("ctrl", True), add="+")
        self.bind("<KeyRelease-Control_L>", lambda e: self._set_modifier("ctrl", False), add="+")
        self.bind("<KeyPress-Control_R>", lambda e: self._set_modifier("ctrl", True), add="+")
        self.bind("<KeyRelease-Control_R>", lambda e: self._set_modifier("ctrl", False), add="+")
        self.bind("<Delete>", lambda e: self._on_delete_key(), add="+")

        # Drag & Drop — регистрируем обработчик перетаскивания файлов на окно
        if _DND_AVAILABLE:
            try:
                self.drop_target_register(DND_FILES)
                self.dnd_bind("<<Drop>>", self._on_file_drop)
            except Exception:
                pass

    def _set_modifier(self, mod: str, is_down: bool):
        if mod == "shift":
            self._shift_pressed = is_down
        elif mod == "ctrl":
            self._ctrl_pressed = is_down

    def _is_shift_down(self, event=None) -> bool:
        if getattr(self, "_shift_pressed", False):
            return True
        if event is not None:
            if getattr(event, "key", None) == "shift":
                return True
            gui_ev = getattr(event, "guiEvent", None)
            if gui_ev is not None and (getattr(gui_ev, "state", 0) & 0x0001):
                return True
        return False

    def _is_ctrl_down(self, event=None) -> bool:
        if getattr(self, "_ctrl_pressed", False):
            return True
        if event is not None:
            if getattr(event, "key", None) in ("control", "ctrl"):
                return True
            gui_ev = getattr(event, "guiEvent", None)
            if gui_ev is not None and (getattr(gui_ev, "state", 0) & 0x0004):
                return True
        return False

    def _init_ui(self):
        # Левая панель с фиксированной шириной 360 px (рассчитана под строку отчета + 8 знаков,
        # полностью исключает дрожание, рывки и дёрганье окон, освобождая место для подписей осей графиков)
        self.left_frame = ctk.CTkFrame(self, width=360, corner_radius=0)
        self.left_frame.pack_propagate(False)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 2), pady=5)

        self.right_frame = ctk.CTkFrame(self, corner_radius=0)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 5), pady=5)

        self._build_left_panel()
        self._build_right_panel()
        self._apply_ttk_dark_style()
        self._lift_canvas_overlays()

    def _build_left_panel(self):
        # Панель переключения темы в самом верху (компактно)
        header_bar = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        header_bar.pack(fill=tk.X, padx=8, pady=(6, 2))

        self.btn_theme_toggle_left = ctk.CTkButton(
            header_bar,
            text="🌓 Тема",
            height=26,
            font=ctk.CTkFont(size=11),
            command=self._toggle_app_theme
        )
        self.btn_theme_toggle_left.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        add_tooltip(self.btn_theme_toggle_left, "Переключить тему оформления (Тёмная / Светлая)")

        self.btn_readme = ctk.CTkButton(
            header_bar,
            text="📖 Справка",
            height=26,
            font=ctk.CTkFont(size=11),
            command=self._open_readme
        )
        self.btn_readme.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        add_tooltip(self.btn_readme, "Открыть подробное иллюстрированное руководство пользователя (HTML)")

        # Внутренний scrollable контейнер для левой панели (гарантирует что всё влезет!)
        inner = ctk.CTkScrollableFrame(self.left_frame, fg_color="transparent")
        inner.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 4))

        # Защита от горизонтального сдвига содержимого при клике или фокусе
        def _prevent_hscroll(event=None):
            try:
                if inner._parent_canvas.xview()[0] != 0.0:
                    inner._parent_canvas.xview_moveto(0.0)
            except Exception:
                pass

        inner._parent_canvas.bind("<Configure>", _prevent_hscroll, add="+")
        inner.bind("<FocusIn>", _prevent_hscroll, add="+")
        inner._parent_canvas.bind("<FocusIn>", _prevent_hscroll, add="+")

        def _group(parent, title):
            """Создаёт группу (аналог LabelFrame) как CTkFrame с заголовком"""
            f = ctk.CTkFrame(parent)
            f.pack(fill=tk.X, pady=(0, 3))
            ctk.CTkLabel(f, text=title,
                         font=ctk.CTkFont(size=11, weight="bold"),
                         anchor="w", height=16).pack(fill=tk.X, padx=8, pady=(8, 8))
            body = ctk.CTkFrame(f, fg_color="transparent")
            body.pack(fill=tk.X, padx=6, pady=(0, 3))
            return body

        # === 1. Проекты и файлы координат ===
        grp_load = _group(inner, " 1. Проекты и файлы координат ")

        self.cbo_files = None

        # Кнопки импорта файла и раздельных съемок
        btn_box1 = ctk.CTkFrame(grp_load, fg_color="transparent")
        btn_box1.pack(fill=tk.X, pady=(0, 2))
        b_imp = ctk.CTkButton(btn_box1, text="Импорт файла...", width=0, height=24,
                              command=self._open_file_dialog)
        b_imp.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        add_tooltip(b_imp, "Импортировать новый файл координат (TXT, CSV, DAT, XYZ, PTS, DXF) в проект")

        b_sep = ctk.CTkButton(btn_box1, text="Верх / Низ...", width=0, height=24,
                              command=self._open_separate_files_dialog)
        b_sep.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        add_tooltip(b_sep, "Импортировать две раздельные съемки для верхней и нижней поверхностей")

        b_ref = ctk.CTkButton(btn_box1, text="⟳", width=30, height=24,
                              command=self._scan_saved_projects)
        b_ref.pack(side=tk.RIGHT)
        add_tooltip(b_ref, "Обновить список проектов и файлов на диске")

        # Кнопка настройки осей и колонок
        btn_box_remap = ctk.CTkFrame(grp_load, fg_color="transparent")
        btn_box_remap.pack(fill=tk.X, pady=(0, 2))
        b_remap = ctk.CTkButton(btn_box_remap, text="🔀 Настройка осей и колонок (X, Y, Z)...", width=0, height=24,
                                command=self._open_remap_dialog)
        b_remap.pack(fill=tk.X)
        add_tooltip(b_remap, "Настроить соответствие колонок файла: X (Север), Y (Восток), Z (Высота), Разделитель")

        # Выбор сохраненного проекта (через полстроки по высоте)
        app_folder_name = os.path.basename(os.path.normpath(get_app_dir())) or "GeoVolumePro"
        ctk.CTkLabel(grp_load, text=f"Выбор проекта ({app_folder_name}/Projects):", anchor="w", height=16).pack(fill=tk.X, pady=(8, 0))
        self.cbo_projects = ctk.CTkComboBox(grp_load, values=[], height=26,
                                             command=self._on_cbo_project_selected)
        self.cbo_projects.pack(fill=tk.X, pady=(1, 2))
        add_tooltip(self.cbo_projects, f"Выбор сохраненного проекта из папки {app_folder_name}/Projects")

        # Кнопки Открыть папку и Удалить проект
        btn_box2 = ctk.CTkFrame(grp_load, fg_color="transparent")
        btn_box2.pack(fill=tk.X, pady=(0, 2))
        b_folder = ctk.CTkButton(btn_box2, text="📁 Открыть папку", width=0, height=24,
                                 command=self._open_project_folder)
        b_folder.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        add_tooltip(b_folder, "Открыть папку текущего проекта в Проводнике Windows")

        b_del = ctk.CTkButton(btn_box2, text="Удалить проект", width=0, height=24,
                              fg_color="#8B2020", hover_color="#A02828",
                              command=self._delete_selected_project)
        b_del.pack(side=tk.RIGHT, padx=(2, 0))
        add_tooltip(b_del, "Удалить выбранный проект и связанные данные")

        self.lbl_file_info = ctk.CTkLabel(grp_load, text="Проект не загружен",
                                           text_color=("#52525b", "#a1a1aa"),
                                           font=ctk.CTkFont(size=10), anchor="w", height=16)
        self.lbl_file_info.pack(fill=tk.X, pady=(1, 0))

        # === 2. Режим и контур сшивания ===
        grp_contour = _group(inner, " 2. Режим и контур сшивания ")

        ctk.CTkLabel(grp_contour, text="Режим работы (ЛКМ на схеме):", anchor="w", height=16).pack(fill=tk.X)
        _mode_values = [
            "1. Интерактивный контур (ЛКМ)",
            "2. Авто-контур (Выпуклая оболочка)",
            "3. Назначение: Верхняя поверхность",
            "4. Назначение: Нижняя поверхность",
            "5. Добавить точку (ЛКМ на схеме)",
            "6. Рамка выделения (Shift+ЛКМ)",
        ]
        self.cbo_mode = ctk.CTkComboBox(grp_contour, values=_mode_values, height=26,
                                         state="readonly",
                                         command=self._on_cbo_mode_selected_cmd)
        self.cbo_mode.set(_mode_values[0])
        self.cbo_mode.pack(fill=tk.X, pady=(1, 2))
        add_tooltip(self.cbo_mode, "Выбор активного режима работы курсора на 2D схеме")

        btn_contour_box = ctk.CTkFrame(grp_contour, fg_color="transparent")
        btn_contour_box.pack(fill=tk.X, pady=(0, 1))
        b_undo = ctk.CTkButton(btn_contour_box, text="↩ Отмена", width=0, height=24,
                               command=self._undo_last_action)
        b_undo.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        add_tooltip(b_undo, "Отменить последнее действие (контур, назначение поверхности)")

        b_reset = ctk.CTkButton(btn_contour_box, text="Сброс", width=0, height=24,
                                fg_color="gray40", hover_color="gray30",
                                command=self._reset_boundary)
        b_reset.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        add_tooltip(b_reset, "Сбросить текущий контур сшивания и вернуть автоматический режим")

        b_split = ctk.CTkButton(btn_contour_box, text="Авто-Z", width=0, height=24,
                                command=self._auto_split_by_height)
        b_split.pack(side=tk.RIGHT, expand=True, fill=tk.X)
        add_tooltip(b_split, "Автоматически разделить точки на верхнюю и нижнюю поверхности по перепаду высот")

        self.lbl_contour_hint = ctk.CTkLabel(
            grp_contour,
            text="💡 Shift+ЛКМ — рамка | 2×ЛКМ — в ребро | ПКМ — тянуть",
            text_color=("#52525b", "#a1a1aa"),
            font=ctk.CTkFont(size=9),
            anchor="w",
            height=16
        )
        self.lbl_contour_hint.pack(fill=tk.X, pady=(1, 0))

        # === 3. Параметры и расчет ===
        grp_calc = _group(inner, " 3. Параметры и расчет ")

        res_box = ctk.CTkFrame(grp_calc, fg_color="transparent")
        res_box.pack(fill=tk.X, pady=(0, 1))
        ctk.CTkLabel(res_box, text="Сетка (м):", height=16).pack(side=tk.LEFT)
        self.ent_res = ctk.CTkEntry(res_box, width=54, height=24)
        self.ent_res.insert(0, "0.2")
        self.ent_res.pack(side=tk.LEFT, padx=(4, 6))
        add_tooltip(self.ent_res, "Шаг регулярной расчетной сетки интерполяции в метрах (по умолчанию 0.2 м)")

        self.btn_run_calc = ctk.CTkButton(res_box, text="Расчет", width=0, height=24,
                                           command=self.calculate_volume)
        self.btn_run_calc.pack(side=tk.RIGHT, expand=True, fill=tk.X)
        add_tooltip(self.btn_run_calc, "Запустить триангуляцию TIN, сеточный расчет и вычисление объемов")

        # === 4. Результаты расчета ===
        self.grp_results = ctk.CTkFrame(inner)
        self.grp_results.pack(fill=tk.BOTH, expand=True, pady=(0, 3))
        ctk.CTkLabel(self.grp_results, text=" 4. Результаты расчета ",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     anchor="w", height=16).pack(fill=tk.X, padx=8, pady=(8, 8))

        # tk.Text с полосой прокрутки CTkScrollbar
        txt_frame = ctk.CTkFrame(self.grp_results, fg_color="transparent")
        txt_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 2))

        txt_sb = ctk.CTkScrollbar(txt_frame, width=12)
        txt_sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.txt_results = tk.Text(txt_frame, height=16, width=42,
                                   font=("Consolas", 9), wrap=tk.WORD,
                                   relief=tk.FLAT, borderwidth=0,
                                   yscrollcommand=txt_sb.set)
        txt_sb.configure(command=self.txt_results.yview)
        self.txt_results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.txt_results.insert(tk.END, "Загрузите координаты и нажмите 'Рассчитать объем'.")
        self.txt_results.config(state=tk.DISABLED)
        self._update_text_widget_colors()

        # Строка статуса автосохранения отчёта и схем
        self.lbl_report_status = ctk.CTkLabel(self.grp_results, text="",
                                               text_color=("#52525b", "#a1a1aa"),
                                               font=ctk.CTkFont(size=10),
                                               anchor="w", wraplength=340, height=16)
        self.lbl_report_status.pack(anchor=tk.W, padx=6, pady=(0, 2))



    # ==================== СТИЛИЗАЦИЯ TTK / TEXT ПОД ТЕМУ CTK ====================

    def _apply_windows_dark_titlebar(self, dark: bool = True):
        """Устанавливает тёмно-серый заголовок и рамку окна на Windows 10/11 через DWM API"""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            self.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id()) or self.winfo_id()

            # 1. Immersive dark mode (Windows 10 build 19041+ / Windows 11)
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19
            val = ctypes.c_int(1 if dark else 0)
            res = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                ctypes.byref(val), ctypes.sizeof(val)
            )
            if res != 0:
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1,
                    ctypes.byref(val), ctypes.sizeof(val)
                )

            # 2. Windows 11 custom caption color: DWMWA_CAPTION_COLOR = 35, DWMWA_TEXT_COLOR = 36
            if dark:
                cap_color = ctypes.c_int(0x00292521)  # #212529 in 0x00BBGGRR
                txt_color = ctypes.c_int(0x00FFFFFF)
            else:
                cap_color = ctypes.c_int(0x00FAF9F8)  # #F8F9FA
                txt_color = ctypes.c_int(0x00292521)

            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 35, ctypes.byref(cap_color), ctypes.sizeof(cap_color)
            )
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 36, ctypes.byref(txt_color), ctypes.sizeof(txt_color)
            )
        except Exception:
            pass

    def _apply_ttk_dark_style(self):
        """Применяет стиль к ttk.Treeview и другим ttk-элементам в соответствии с темой CTk"""
        is_dark = ctk.get_appearance_mode() == "Dark"

        # Применяем тёмно-серый заголовок и рамку окна на Windows
        self._apply_windows_dark_titlebar(is_dark)

        style = ttk.Style()
        if "clam" in style.theme_names():
            try:
                style.theme_use("clam")
            except Exception:
                pass

        if is_dark:
            bg       = "#212529"
            bg2      = "#2b2b2b"
            fg       = "#f0f2f5"
            sel_bg   = "#3d444d"
            head_bg  = "#1e2124"
            head_fg  = "#ffffff"
        else:
            bg       = "#f8f9fa"
            bg2      = "#ffffff"
            fg       = "#212529"
            sel_bg   = "#ced4da"
            head_bg  = "#e9ecef"
            head_fg  = "#212529"

        # Внешний фон главного окна и разделителя
        try:
            self.configure(bg=bg)
        except Exception:
            pass

        # Treeview
        style.configure("Treeview",
                        background=bg2, foreground=fg,
                        fieldbackground=bg2, rowheight=24,
                        borderwidth=0)
        style.configure("Treeview.Heading",
                        background=head_bg, foreground=head_fg,
                        relief="flat", font=("Segoe UI", 9, "bold"))
        style.map("Treeview",
                  background=[("selected", sel_bg)],
                  foreground=[("selected", "#ffffff")])
        style.map("Treeview.Heading",
                  background=[("active", sel_bg)])

        # PanedWindow sash
        style.configure("TPanedwindow", background=bg)
        # Scrollbar
        style.configure("Vertical.TScrollbar", background=bg)
        style.configure("Horizontal.TScrollbar", background=bg)

        # Обновление тегов списка треугольников TIN под тему
        if getattr(self, "tree_tin", None) is not None:
            if is_dark:
                self.tree_tin.tag_configure("active", foreground="#ffffff", background="")
                self.tree_tin.tag_configure("excluded", foreground="#ff6b6b", background="#3d2020")
            else:
                self.tree_tin.tag_configure("active", foreground="#212529", background="")
                self.tree_tin.tag_configure("excluded", foreground="#c0392b", background="#fdf2f2")

        if getattr(self, "lbl_tin_stats", None) is not None:
            self.lbl_tin_stats.configure(text_color="#e0e0e0" if is_dark else "#1a5276")

        if getattr(self, "txt_results", None) is not None:
            self._update_text_widget_colors()

        tb_bg = "#212529" if is_dark else "#f8f9fa"
        tb_fg = "#ffffff" if is_dark else "#212529"
        btn_bg = "#2b3035" if is_dark else "#ffffff"
        btn_fg = "#f8f9fa" if is_dark else "#212529"
        btn_active = "#3d444b" if is_dark else "#e9ecef"
        btn_highlight = "#495057" if is_dark else "#ced4da"

        for tb in getattr(self, "_toolbars", []):
            try:
                tb.config(bg=tb_bg)
                if hasattr(tb, "_message_label"):
                    tb._message_label.config(bg=tb_bg, fg=tb_fg)
                for btn_attr in ("_btn_save_png", "_btn_dxf", "_btn_report"):
                    b = getattr(tb, btn_attr, None)
                    if b is not None:
                        b.config(
                            bg=btn_bg,
                            fg=btn_fg,
                            activebackground=btn_active,
                            activeforeground=btn_fg,
                            highlightbackground=btn_highlight,
                            highlightcolor=btn_highlight
                        )
            except Exception:
                pass

        self._update_toolbar_volume_labels()
        self._apply_all_canvases_theme(is_dark)

    def _update_text_widget_colors(self):
        """Синхронизирует цвета tk.Text с текущей темой CTk"""
        is_dark = ctk.get_appearance_mode() == "Dark"
        if is_dark:
            self.txt_results.config(bg="#212529", fg="#f0f2f5",
                                    insertbackground="#f0f2f5",
                                    selectbackground="#3d444d")
        else:
            self.txt_results.config(bg="#f8f9fa", fg="#212529",
                                    insertbackground="#212529",
                                    selectbackground="#adb5bd")

    def _get_canvas_text_color(self) -> str:
        """Возвращает контрастный цвет текста для подписей точек по фону ax_2d."""
        try:
            if hasattr(self, "ax_2d") and self.ax_2d is not None:
                fc = self.ax_2d.get_facecolor()
                lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
                return "#f0f2f5" if lum < 0.5 else "#111111"
        except Exception:
            pass
        is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False
        return "#f0f2f5" if is_dark else "#111111"

    def _apply_axes_theme(self, ax, fig=None, is_dark=None):
        """Применяет цветовую тему (светлую или тёмную) к фигуре и осям Matplotlib."""
        if ax is None:
            return
        if is_dark is None:
            is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False

        fig_bg = "#212529" if is_dark else "#ffffff"
        ax_bg  = "#1e2124" if is_dark else "#ffffff"
        fg_col = "#e0e0e0" if is_dark else "#212529"
        grid_col = "#343a40" if is_dark else "#d0d0d0"
        spine_col = "#495057" if is_dark else "#888888"

        if fig is not None:
            try:
                fig.patch.set_facecolor(fig_bg)
            except Exception:
                pass
        try:
            ax.set_facecolor(ax_bg)
        except Exception:
            pass

        try:
            ax.tick_params(colors=fg_col, which="both", labelsize=9)
            spines = getattr(ax, "spines", {})
            if isinstance(spines, dict):
                for spine in spines.values():
                    spine.set_color(spine_col)
            if hasattr(ax, "xaxis"):
                if hasattr(ax.xaxis, "label"):
                    ax.xaxis.label.set_color(fg_col)
                    try:
                        ax.xaxis.label.set_size(9)
                    except Exception:
                        pass
                if hasattr(ax.xaxis, "offsetText"):
                    ax.xaxis.offsetText.set_color(fg_col)
                    try:
                        ax.xaxis.offsetText.set_size(9)
                    except Exception:
                        pass
            if hasattr(ax, "yaxis"):
                if hasattr(ax.yaxis, "label"):
                    ax.yaxis.label.set_color(fg_col)
                    try:
                        ax.yaxis.label.set_size(9)
                    except Exception:
                        pass
                if hasattr(ax.yaxis, "offsetText"):
                    ax.yaxis.offsetText.set_color(fg_col)
                    try:
                        ax.yaxis.offsetText.set_size(9)
                    except Exception:
                        pass
            for title_attr in ("title", "_left_title", "_right_title"):
                t = getattr(ax, title_attr, None)
                if t is not None:
                    try:
                        t.set_color(fg_col)
                    except Exception:
                        pass
            ax.grid(True, linestyle="--", alpha=0.5, color=grid_col)
        except Exception:
            pass

        # Для 3D осей
        if hasattr(ax, "xaxis") and hasattr(ax.xaxis, "set_pane_color"):
            try:
                pane_color = (0.13, 0.15, 0.17, 1.0) if is_dark else (0.95, 0.95, 0.95, 1.0)
                ax.xaxis.set_pane_color(pane_color)
                ax.yaxis.set_pane_color(pane_color)
                ax.zaxis.set_pane_color(pane_color)
                if hasattr(ax, "zaxis") and hasattr(ax.zaxis, "label"):
                    ax.zaxis.label.set_color(fg_col)
            except Exception:
                pass

        # Легенда
        try:
            leg = ax.get_legend()
            if leg is not None:
                leg.get_frame().set_facecolor(ax_bg)
                leg.get_frame().set_edgecolor(spine_col)
                for t in leg.get_texts():
                    t.set_color(fg_col)
        except Exception:
            pass

        # Обновляем фон углов кнопки «Сброс контура»
        btn_reset = self.__dict__.get("btn_reset_contour")
        if btn_reset is not None:
            try:
                btn_reset.configure(bg_color=ax_bg)
            except Exception:
                pass

    def _apply_all_canvases_theme(self, is_dark=None):
        """Актуализирует темы всех графиков (2D, 3D, Diff, TIN) и нативных подписей."""
        if is_dark is None:
            is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False
        fig_bg = "#212529" if is_dark else "#ffffff"

        # 1. 2D Схема в плане
        if getattr(self, "ax_2d", None) is not None:
            self._apply_axes_theme(self.ax_2d, getattr(self, "fig_2d", None), is_dark)
            if getattr(self, "canvas_2d", None) is not None:
                try:
                    if hasattr(self.canvas_2d, "get_tk_widget"):
                        self.canvas_2d.get_tk_widget().configure(bg=fig_bg)
                except Exception:
                    pass
                self.canvas_2d.draw_idle()

            if getattr(self, "points", None):
                try:
                    self._redraw_2d(reset_view=False)
                except Exception:
                    pass

            if self._has_tk_canvas_widget():
                try:
                    tk_canvas = self.canvas_2d.get_tk_widget()
                    tk_canvas.itemconfigure("point_label", fill=self._get_canvas_text_color())
                except Exception:
                    pass

        # 2. 3D Поверхности
        if hasattr(self, "ax_3d") and self.ax_3d is not None:
            self._apply_axes_theme(self.ax_3d, getattr(self, "fig_3d", None), is_dark)
            if hasattr(self, "canvas_3d") and self.canvas_3d is not None:
                try:
                    if hasattr(self.canvas_3d, "get_tk_widget"):
                        self.canvas_3d.get_tk_widget().configure(bg=fig_bg)
                except Exception:
                    pass
                self.canvas_3d.draw_idle()

        # 3. Картограмма масс
        if hasattr(self, "ax_diff") and self.ax_diff is not None:
            self._apply_axes_theme(self.ax_diff, getattr(self, "fig_diff", None), is_dark)
            if hasattr(self, "canvas_diff") and self.canvas_diff is not None:
                try:
                    if hasattr(self.canvas_diff, "get_tk_widget"):
                        self.canvas_diff.get_tk_widget().configure(bg=fig_bg)
                except Exception:
                    pass
                self.canvas_diff.draw_idle()

        # 4. TIN Триангуляция
        if hasattr(self, "ax_tin") and self.ax_tin is not None:
            self._apply_axes_theme(self.ax_tin, getattr(self, "fig_tin", None), is_dark)
            if hasattr(self, "canvas_tin") and self.canvas_tin is not None:
                try:
                    if hasattr(self.canvas_tin, "get_tk_widget"):
                        self.canvas_tin.get_tk_widget().configure(bg=fig_bg)
                except Exception:
                    pass
                self.canvas_tin.draw_idle()

        # 5. Горизонтали
        if hasattr(self, "ax_contours") and self.ax_contours is not None:
            self._apply_axes_theme(self.ax_contours, getattr(self, "fig_contours", None), is_dark)
            if hasattr(self, "canvas_contours") and self.canvas_contours is not None:
                try:
                    if hasattr(self.canvas_contours, "get_tk_widget"):
                        self.canvas_contours.get_tk_widget().configure(bg=fig_bg)
                except Exception:
                    pass
                self.canvas_contours.draw_idle()

        # 6. Плавающие панели поверхностей и панель селекции
        for p in getattr(self, "_fs_surface_panels", {}).values():
            try:
                p.configure(bg_color=fig_bg)
            except Exception:
                pass
        if getattr(self, "frame_selection_bar", None) is not None:
            try:
                self.frame_selection_bar.configure(bg_color=fig_bg)
            except Exception:
                pass

    def _toggle_app_theme(self):
        """Переключает тёмную/светлую тему приложения"""
        current = ctk.get_appearance_mode()
        new_mode = "Light" if current == "Dark" else "Dark"
        ctk.set_appearance_mode(new_mode)
        btn_left = self.__dict__.get("btn_theme_toggle_left")
        if btn_left:
            next_hint = "Светлая" if new_mode == "Dark" else "Тёмная"
            btn_left.configure(text=f"🌓 Включить: {next_hint} тема")
        self.after(50, self._apply_ttk_dark_style)

    def _build_right_panel(self):
        self.TAB_2D = "2D Схема в плане"
        self.TAB_3D = "3D Поверхности"
        self.TAB_DIFF = "Картограмма масс"
        self.TAB_TIN = "TIN Триангуляция"
        self.TAB_CONTOURS = "Горизонтали"
        self.TAB_TABLE = "Таблица точек"

        self.tabview = ctk.CTkTabview(
            self.right_frame,
            command=self._on_tabview_change,
            anchor="nw"
        )
        self.tabview.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        try:
            self.tabview._segmented_button.configure(
                font=ctk.CTkFont(size=11, weight="bold"),
                selected_color=("#ffffff", "#3e444c"),
                selected_hover_color=("#f8f9fa", "#4b535d"),
                unselected_color=("#dee2e6", "#282c30"),
                unselected_hover_color=("#ced4da", "#33383e"),
                text_color=("#212529", "#f8f9fa"),
            )
        except Exception:
            pass

        self.tab_2d = self.tabview.add(self.TAB_2D)
        self.tab_3d = self.tabview.add(self.TAB_3D)
        self.tab_diff = self.tabview.add(self.TAB_DIFF)
        self.tab_tin = self.tabview.add(self.TAB_TIN)
        self.tab_contours = self.tabview.add(self.TAB_CONTOURS)
        self.tab_table = self.tabview.add(self.TAB_TABLE)

        self._build_2d_tab()
        self._build_3d_tab()
        self._build_diff_tab()
        self._build_tin_tab()
        self._build_contours_tab()
        self._build_table_tab()

        # Настройка двойного щелчка по названию вкладки для разворачивания на весь экран
        self._setup_tab_double_click()

        # Заменяем tabview.set на безопасный метод без задержек after(100), вызывающих скрытие вкладок
        self.tabview.set = self._select_tab
        self._select_tab(self.TAB_2D)
        self._redraw_2d(reset_view=True)

    def _open_readme(self):
        """Открывает файл документации readme.html в системном браузере по умолчанию"""
        import webbrowser
        candidates = []
        if getattr(sys, "frozen", False):
            exe_dir = os.path.dirname(sys.executable)
            candidates.append(os.path.join(exe_dir, "readme.html"))
            if hasattr(sys, "_MEIPASS"):
                candidates.append(os.path.join(sys._MEIPASS, "readme.html"))
        app_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(app_dir, "readme.html"))
        candidates.append(os.path.join(os.getcwd(), "readme.html"))

        target = None
        for p in candidates:
            if os.path.isfile(p):
                target = p
                break

        if target:
            try:
                webbrowser.open(f"file:///{os.path.abspath(target).replace(os.sep, '/')}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось открыть справку:\n{e}")
        else:
            messagebox.showwarning("Предупреждение", "Файл документации readme.html не найден.")

    def _setup_tab_double_click(self):
        """Настраивает обработку двойного щелчка по названию вкладки для разворачивания на весь экран"""
        try:
            buttons_dict = getattr(self.tabview._segmented_button, "_buttons_dict", {})
            for name, btn in buttons_dict.items():
                btn.bind("<Double-Button-1>", lambda e, n=name: self._on_tab_title_double_click(n, e), add="+")

            try:
                self.tabview._segmented_button._canvas.bind(
                    "<Double-Button-1>", lambda e: self._toggle_tab_fullscreen(), add="+"
                )
            except Exception:
                pass

            # Клавиша Esc также возвращает к обычному экрану, если вкладка развернута на весь экран
            self.bind("<Escape>", self._on_escape_key, add="+")
        except Exception:
            pass

    def _on_tab_title_double_click(self, tab_name: str, event=None):
        """Обработка двойного щелчка по заголовку вкладки: переключение на неё и разворачивание/сворачивание"""
        if tab_name and hasattr(self, "tabview") and self.tabview.get() != tab_name:
            self.tabview.set(tab_name)
        self._toggle_tab_fullscreen()

    def _toggle_tab_fullscreen(self, event=None):
        """Разворачивание активной вкладки на весь экран / возврат к обычному экрану программы по двойному щелчку"""
        is_fs = getattr(self, "_is_tab_fullscreen", False)
        if not is_fs:
            self._saved_win_state = self.state()
            self._is_tab_fullscreen = True

            try:
                self.left_frame.pack_forget()
            except Exception:
                pass

            if self.state() != "zoomed":
                try:
                    self.state("zoomed")
                except Exception:
                    pass
        else:
            self._is_tab_fullscreen = False

            prev_state = getattr(self, "_saved_win_state", "normal")
            if prev_state != "zoomed" and self.state() == "zoomed":
                try:
                    self.state(prev_state)
                except Exception:
                    pass

            try:
                self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 2), pady=5, before=self.right_frame)
            except Exception:
                pass

        self._update_toolbar_volume_labels()
        self._lift_canvas_overlays()
        self.after_idle(self._on_tab_fullscreen_changed)

    def _update_toolbar_volume_labels(self):
        """Обновляет центральную строку с объемами насыпи и выемки в строке инструментов графиков"""
        is_fs = getattr(self, "_is_tab_fullscreen", False)
        labels = getattr(self, "_toolbar_volume_labels", [])
        if not labels:
            return

        if getattr(self, "calc_results", None):
            r = self.calc_results
            v_fill = r.get("v_fill", 0.0)
            v_cut = r.get("v_cut", 0.0)
            v_net = r.get("v_net", 0.0)
            if is_fs:
                text = (
                    f"Насыпь (Fill): {v_fill:.3f} м³   |   "
                    f"Выемка (Cut): {v_cut:.3f} м³   |   "
                    f"ИТОГ (Net): {v_net:.3f} м³"
                )
            else:
                text = f"Fill: {v_fill:.3f} м³  |  Cut: {v_cut:.3f} м³  |  Net: {v_net:.3f} м³"
        else:
            text = ""

        valid_labels = []
        is_dark = ctk.get_appearance_mode() == "Dark"
        lbl_fg = "#ffffff" if is_dark else "#212529"
        for lbl in labels:
            try:
                if lbl.winfo_exists():
                    parent_bg = lbl.master.cget("bg")
                    lbl.config(text=text, fg=lbl_fg, bg=parent_bg)
                    valid_labels.append(lbl)
            except Exception:
                pass
        self._toolbar_volume_labels = valid_labels

    def _apply_fig_layout(self, fig, pad: float = 0.5):
        """Применяет компактный tight_layout, максимизируя полезную площадь схем"""
        if not fig:
            return
        try:
            fig.tight_layout(pad=pad, rect=[0.005, 0.035, 0.995, 0.975])
        except Exception:
            try:
                fig.tight_layout(pad=pad)
            except Exception:
                pass

    def _get_labels_btn_text(self) -> str:
        mode = getattr(self, "_point_labels_mode", "on")
        if mode == "on":
            return "🏷️ Подписи: Вкл"
        else:
            return "🏷️ Подписи: Выкл"

    def _toggle_labels_mode(self):
        cur = getattr(self, "_point_labels_mode", "on")
        self._point_labels_mode = "off" if cur == "on" else "on"

        if hasattr(self, "btn_labels_mode") and self.btn_labels_mode is not None:
            try:
                self.btn_labels_mode.configure(text=self._get_labels_btn_text())
            except Exception:
                pass

        self._redraw_2d(reset_view=False)

    def _create_canvas_surface_panel(self, canvas_widget, show_labels_btn=False):
        """Создает компактную плавающую панель с переключателями в правом верхнем углу холста"""
        panel = ctk.CTkFrame(
            canvas_widget,
            fg_color=("#f8f9fa", "#252930"),
            bg_color=("#ffffff", "#212529"),
            border_width=1,
            border_color=("#ced4da", "#495057"),
            corner_radius=0,
        )
        cb_top = ctk.CTkCheckBox(
            panel,
            text="▲ Верхняя",
            variable=self._show_top,
            command=self._on_surface_vis_toggle,
            width=16,
            height=16,
            checkbox_width=16,
            checkbox_height=16,
            bg_color=("#f8f9fa", "#252930"),
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        cb_top.pack(side=tk.LEFT, padx=(8, 6), pady=3)
        add_tooltip(cb_top, "Показать или скрыть точки и рельеф верхней поверхности")

        cb_bot = ctk.CTkCheckBox(
            panel,
            text="▼ Нижняя",
            variable=self._show_bottom,
            command=self._on_surface_vis_toggle,
            width=16,
            height=16,
            checkbox_width=16,
            checkbox_height=16,
            bg_color=("#f8f9fa", "#252930"),
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        cb_bot.pack(side=tk.LEFT, padx=(4, 6), pady=3)
        add_tooltip(cb_bot, "Показать или скрыть точки и рельеф нижней поверхности")

        if show_labels_btn:
            self.btn_labels_mode = ctk.CTkButton(
                panel,
                text=self._get_labels_btn_text(),
                width=0,
                height=22,
                corner_radius=0,
                bg_color=("#f8f9fa", "#252930"),
                fg_color=("#e9ecef", "#343a40"),
                hover_color=("#dee2e6", "#495057"),
                text_color=("#212529", "#f8f9fa"),
                font=ctk.CTkFont(size=11, weight="bold"),
                command=self._toggle_labels_mode,
            )
            self.btn_labels_mode.pack(side=tk.LEFT, padx=(2, 6), pady=3)
            add_tooltip(self.btn_labels_mode, "Режим подписей номеров и отметок точек (Все / Отключены)")

        return panel

    def _update_canvas_surface_switches(self):
        """Показывает переключатели поверхностей в правом верхнем углу холста на активной вкладке (как в обычном, так и в полноэкранном режиме)"""
        current_tab = self.tabview.get() if hasattr(self, "tabview") else None

        for tab_name, panel in getattr(self, "_fs_surface_panels", {}).items():
            try:
                if tab_name == current_tab:
                    panel.place(relx=1.0, y=6, anchor="ne", x=-8)
                    panel.lift()
                else:
                    panel.place_forget()
            except Exception:
                pass

    _update_fullscreen_surface_switches = _update_canvas_surface_switches

    def _lift_canvas_overlays(self):
        """Поднимает плавающие элементы управления поверх холста"""
        for tb in getattr(self, "_toolbars", []):
            try:
                tb.lift()
            except Exception:
                pass
        self._update_canvas_surface_switches()
        self._position_reset_contour_button()
        if hasattr(self, "_update_selection_bar"):
            self._update_selection_bar()
        if hasattr(self, "btn_readme"):
            try:
                self.btn_readme.lift()
            except Exception:
                pass
        if hasattr(self, "btn_theme_toggle"):
            try:
                self.btn_theme_toggle.lift()
            except Exception:
                pass

    def _on_tab_fullscreen_changed(self):
        """Адаптирует и перерисовывает графику текущей вкладки после изменения режима экрана"""
        try:
            if not hasattr(self, "tabview"):
                return
            current_tab = self.tabview.get()
            if current_tab == self.TAB_2D:
                if hasattr(self, "fig_2d"):
                    self._apply_fig_layout(self.fig_2d, pad=0.5)
                if hasattr(self, "canvas_2d"):
                    self.canvas_2d.draw_idle()
            elif current_tab == self.TAB_DIFF:
                if hasattr(self, "fig_diff"):
                    self._apply_fig_layout(self.fig_diff, pad=0.5)
                if hasattr(self, "canvas_diff"):
                    self.canvas_diff.draw_idle()
            elif current_tab == self.TAB_TIN:
                if hasattr(self, "fig_tin"):
                    self._apply_fig_layout(self.fig_tin, pad=0.5)
                if hasattr(self, "_fit_tin_view"):
                    self._fit_tin_view()
            elif current_tab == getattr(self, "TAB_CONTOURS", "Горизонтали"):
                if hasattr(self, "fig_contours"):
                    self._apply_fig_layout(self.fig_contours, pad=0.5)
                if hasattr(self, "_fit_contours_view"):
                    self._fit_contours_view()
            elif current_tab == self.TAB_3D:
                if hasattr(self, "fig_3d"):
                    self._apply_fig_layout(self.fig_3d, pad=0.5)
                if hasattr(self, "canvas_3d"):
                    self.canvas_3d.draw_idle()
            self._lift_canvas_overlays()
        except Exception:
            pass

    def _on_escape_key(self, event=None):
        """Нажатие Esc — возврат к обычному экрану программы, если вкладка развернута"""
        if getattr(self, "_is_tab_fullscreen", False):
            self._toggle_tab_fullscreen()

    def _on_delete_key(self, event=None):
        """Нажатие Delete — удаление выделенных точек (если фокус не в поле ввода)"""
        try:
            focused = self.focus_get()
            if focused and isinstance(focused, (tk.Entry, ttk.Entry, tk.Text)):
                return
            if focused and hasattr(focused, "_entry") and isinstance(focused._entry, tk.Entry):
                return
            if getattr(self, "_selected_points", None):
                self._delete_point(-1)
        except Exception:
            pass

    def _select_tab(self, tab_name: str):
        """Безопасное синхронное переключение вкладки без задержек after() и коллизий CTkTabview"""
        if not hasattr(self, "tabview") or tab_name not in self.tabview._tab_dict:
            return

        for name, frame in self.tabview._tab_dict.items():
            if name != tab_name:
                frame.grid_forget()
        self.tabview._current_name = tab_name
        self.tabview._segmented_button.set(tab_name)
        self.tabview._set_grid_current_tab()
        self._on_tabview_change()
        self._lift_canvas_overlays()


    def _format_coord_display(self, y_east, x_north):
        """Формат координат курсора на холсте (Север X, Восток Y)."""
        if y_east is None or x_north is None:
            return ""
        return f"X (Север): {x_north:.3f},  Y (Восток): {y_east:.3f}"

    def _format_coord_diff(self, y_east, x_north):
        """Быстрое отображение координат и фактической высоты/мощности слоя под курсором на картограмме (O(1))."""
        if y_east is None or x_north is None:
            return ""
        coord_str = f"X (Север): {x_north:.3f},  Y (Восток): {y_east:.3f}"
        r = getattr(self, "calc_results", None)
        if r is not None:
            gx = r.get("grid_x")
            gy = r.get("grid_y")
            if gx is not None and gy is not None and gx.size > 0 and gy.size > 0:
                min_x, max_x = float(gx.min()), float(gx.max())
                min_y, max_y = float(gy.min()), float(gy.max())
                if min_x <= x_north <= max_x and min_y <= y_east <= max_y:
                    dx1 = abs(gx[0, 1] - gx[0, 0]) if gx.shape[1] > 1 else 0.0
                    dx0 = abs(gx[1, 0] - gx[0, 0]) if gx.shape[0] > 1 else 0.0
                    step_x = dx1 if dx1 > 1e-6 else (dx0 if dx0 > 1e-6 else max((max_x - min_x) / max(gx.shape[1] - 1, 1), 0.1))

                    dy0 = abs(gy[1, 0] - gy[0, 0]) if gy.shape[0] > 1 else 0.0
                    dy1 = abs(gy[0, 1] - gy[0, 0]) if gy.shape[1] > 1 else 0.0
                    step_y = dy0 if dy0 > 1e-6 else (dy1 if dy1 > 1e-6 else max((max_y - min_y) / max(gy.shape[0] - 1, 1), 0.1))

                    r_idx = int(round((y_east - min_y) / max(step_y, 1e-6)))
                    c_idx = int(round((x_north - min_x) / max(step_x, 1e-6)))
                    show_top = self._show_top.get() if hasattr(self, "_show_top") else True
                    show_bottom = self._show_bottom.get() if hasattr(self, "_show_bottom") else True
                    is_cut = self._is_excavation() if hasattr(self, "_is_excavation") else False

                    if show_top and show_bottom:
                        grid = r.get("dh_grid")
                        prefix = "ΔH:"
                        unit = "м"
                        fmt_sign = "+" if not is_cut else "-"
                    elif show_top:
                        grid = r.get("z_top_grid")
                        prefix = "Z_верх:"
                        unit = "м"
                        fmt_sign = ""
                    elif show_bottom:
                        grid = r.get("z_bot_grid")
                        prefix = "Z_низ:"
                        unit = "м"
                        fmt_sign = ""
                    else:
                        grid = None

                    if grid is not None and 0 <= r_idx < grid.shape[0] and 0 <= c_idx < grid.shape[1]:
                        val = float(grid[r_idx, c_idx])
                        if not np.isnan(val):
                            if fmt_sign == "+":
                                s_val = f"+{val:.2f}" if val > 0 else f"{val:.2f}"
                            elif fmt_sign == "-":
                                s_val = f"-{abs(val):.2f}"
                            else:
                                s_val = f"{val:.2f}"
                            return f"[{prefix} {s_val} {unit}]   {coord_str}"
        return coord_str

    def _format_coord_contours(self, y_east, x_north):
        """Быстрое отображение координат и отметок высот Z_верх/Z_низ или Z под курсором на плане горизонталей (O(1))."""
        if y_east is None or x_north is None:
            return ""
        coord_str = f"X (Север): {x_north:.3f},  Y (Восток): {y_east:.3f}"
        data = self._get_contours_data() if hasattr(self, "_get_contours_data") else None
        if data is not None:
            gx = data.get("grid_x")
            gy = data.get("grid_y")
            if gx is not None and gy is not None and gx.size > 0 and gy.size > 0:
                min_x, max_x = float(gx.min()), float(gx.max())
                min_y, max_y = float(gy.min()), float(gy.max())
                if min_x <= x_north <= max_x and min_y <= y_east <= max_y:
                    dx1 = abs(gx[0, 1] - gx[0, 0]) if gx.shape[1] > 1 else 0.0
                    dx0 = abs(gx[1, 0] - gx[0, 0]) if gx.shape[0] > 1 else 0.0
                    step_x = dx1 if dx1 > 1e-6 else (dx0 if dx0 > 1e-6 else max((max_x - min_x) / max(gx.shape[1] - 1, 1), 0.1))

                    dy0 = abs(gy[1, 0] - gy[0, 0]) if gy.shape[0] > 1 else 0.0
                    dy1 = abs(gy[0, 1] - gy[0, 0]) if gy.shape[1] > 1 else 0.0
                    step_y = dy0 if dy0 > 1e-6 else (dy1 if dy1 > 1e-6 else max((max_y - min_y) / max(gy.shape[0] - 1, 1), 0.1))

                    r_idx = int(round((y_east - min_y) / max(step_y, 1e-6)))
                    c_idx = int(round((x_north - min_x) / max(step_x, 1e-6)))
                    z_top_grid = data.get("z_top_grid")
                    z_bot_grid = data.get("z_bot_grid")
                    is_single = data.get("is_single_survey", False)
                    show_top = self._show_top.get() if hasattr(self, "_show_top") else True
                    show_bottom = self._show_bottom.get() if hasattr(self, "_show_bottom") else True

                    h_parts = []
                    if is_single:
                        if z_top_grid is not None and 0 <= r_idx < z_top_grid.shape[0] and 0 <= c_idx < z_top_grid.shape[1]:
                            zt = float(z_top_grid[r_idx, c_idx])
                            if not np.isnan(zt):
                                h_parts.append(f"Z: {zt:.2f} м")
                    else:
                        if show_top and z_top_grid is not None and 0 <= r_idx < z_top_grid.shape[0] and 0 <= c_idx < z_top_grid.shape[1]:
                            zt = float(z_top_grid[r_idx, c_idx])
                            if not np.isnan(zt):
                                h_parts.append(f"Z_верх: {zt:.2f} м")
                        if show_bottom and z_bot_grid is not None and 0 <= r_idx < z_bot_grid.shape[0] and 0 <= c_idx < z_bot_grid.shape[1]:
                            zb = float(z_bot_grid[r_idx, c_idx])
                            if not np.isnan(zb):
                                h_parts.append(f"Z_низ: {zb:.2f} м")
                    if h_parts:
                        return f"[{' | '.join(h_parts)}]   {coord_str}"
        return coord_str

    def _build_2d_tab(self):
        is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False
        title_color = "#e0e0e0" if is_dark else "#212529"
        fig_bg = "#212529" if is_dark else "#ffffff"

        self.fig_2d = Figure(figsize=(6, 5), dpi=100)
        self.ax_2d = self.fig_2d.add_subplot(111)
        self.ax_2d.set_xlabel("Восток Y (м)", color=title_color)
        self.ax_2d.set_ylabel("Север X (м)", color=title_color)
        self.ax_2d.grid(True, linestyle="--", alpha=0.5)
        self.ax_2d.format_coord = self._format_coord_display
        self.ax_2d.xaxis.set_major_formatter(PlainOffsetFormatter(useOffset=True))
        self.ax_2d.yaxis.set_major_formatter(PlainOffsetFormatter(useOffset=True))
        self._apply_axes_theme(self.ax_2d, self.fig_2d, is_dark)

        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, master=self.tab_2d)
        cw_2d = self.canvas_2d.get_tk_widget()
        cw_2d.configure(bg=fig_bg)
        cw_2d.pack(fill=tk.BOTH, expand=True)
        self._fs_surface_panels[self.TAB_2D] = self._create_canvas_surface_panel(cw_2d, show_labels_btn=True)

        # Кнопка «Сброс контура» под надписью «Контур сшивания»
        self.btn_reset_contour = ctk.CTkButton(
            cw_2d,
            text="Сброс контура",
            command=self._reset_boundary,
            fg_color="#c0392b",
            hover_color="#962d22",
            text_color="#ffffff",
            bg_color=("#ffffff", "#1e2124"),
            height=24,
            corner_radius=0,
            border_width=1,
            border_color=("#a93226", "#78281f"),
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        cw_2d.bind("<Configure>", lambda e: self._position_reset_contour_button(), add="+")
        add_tooltip(self.btn_reset_contour, "Сбросить текущий контур сшивания и вернуть автоматический режим")

        # Плавающая панель пакетного управления выделенными точками
        self.frame_selection_bar = ctk.CTkFrame(
            cw_2d,
            fg_color=("#ffffff", "#212529"),
            bg_color=("#ffffff", "#212529"),
            corner_radius=0,
            border_width=1,
            border_color=("#bdc3c7", "#495057")
        )
        self.lbl_sel_count = ctk.CTkLabel(
            self.frame_selection_bar,
            text="Выделено: 0 т.",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("#2c3e50", "#ecf0f1")
        )
        self.lbl_sel_count.pack(side=tk.LEFT, padx=(8, 6), pady=3)

        self.btn_sel_bottom = ctk.CTkButton(
            self.frame_selection_bar,
            text="▼ В нижнюю",
            width=0,
            height=24,
            fg_color="#d35400",
            hover_color="#a04000",
            text_color="#ffffff",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self._batch_assign_surface("bottom")
        )
        self.btn_sel_bottom.pack(side=tk.LEFT, padx=(0, 4), pady=3)
        add_tooltip(self.btn_sel_bottom, "Перенести все выделенные точки на нижнюю поверхность")

        self.btn_sel_top = ctk.CTkButton(
            self.frame_selection_bar,
            text="▲ В верхнюю",
            width=0,
            height=24,
            fg_color="#1565c0",
            hover_color="#0d3b7a",
            text_color="#ffffff",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self._batch_assign_surface("top")
        )
        self.btn_sel_top.pack(side=tk.LEFT, padx=(0, 4), pady=3)
        add_tooltip(self.btn_sel_top, "Перенести все выделенные точки на верхнюю поверхность")

        self.btn_sel_contour = ctk.CTkButton(
            self.frame_selection_bar,
            text="⬡ В контур",
            width=0,
            height=24,
            fg_color="#27ae60",
            hover_color="#1e8449",
            text_color="#ffffff",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._build_boundary_from_selected_points
        )
        self.btn_sel_contour.pack(side=tk.LEFT, padx=(0, 4), pady=3)
        add_tooltip(self.btn_sel_contour, "Построить контур сшивания вокруг выделенных точек (Convex Hull)")

        self.btn_sel_delete = ctk.CTkButton(
            self.frame_selection_bar,
            text="❌ Удалить",
            width=0,
            height=24,
            fg_color="#c0392b",
            hover_color="#962d22",
            text_color="#ffffff",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self._delete_point(-1)
        )
        self.btn_sel_delete.pack(side=tk.LEFT, padx=(0, 4), pady=3)
        add_tooltip(self.btn_sel_delete, "Удалить все выделенные точки из проекта")

        self.btn_sel_clear = ctk.CTkButton(
            self.frame_selection_bar,
            text="✕ Отмена",
            width=0,
            height=24,
            fg_color="gray45",
            hover_color="gray35",
            text_color="#ffffff",
            font=ctk.CTkFont(size=11),
            command=self._clear_selected_points
        )
        self.btn_sel_clear.pack(side=tk.LEFT, padx=(0, 8), pady=3)
        add_tooltip(self.btn_sel_clear, "Снять выделение со всех точек")

        # Отслеживание изменения границ осей для мгновенного скрытия/показа подписей точек
        self.ax_2d.callbacks.connect("xlim_changed", lambda ax: self._update_2d_annotations_visibility())
        self.ax_2d.callbacks.connect("ylim_changed", lambda ax: self._update_2d_annotations_visibility())

        self._toolbar_2d = ProjectNavigationToolbar(self.canvas_2d, cw_2d, pack_toolbar=False)
        self._toolbar_2d.place(relx=0.0, rely=1.0, anchor="sw", relwidth=1.0, height=26)
        self._setup_custom_toolbar_buttons(self._toolbar_2d, self.fig_2d, self.TAB_2D)
        self._toolbars.append(self._toolbar_2d)

        # Обработка мыши для панорамирования, клика по точкам и зума
        self.canvas_2d.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas_2d.mpl_connect("button_release_event", self._on_canvas_release)
        self.canvas_2d.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self.canvas_2d.mpl_connect("scroll_event", self._on_canvas_scroll)
        self.canvas_2d.mpl_connect("resize_event", lambda e: self._update_tk_canvas_labels_positions())

    def _position_reset_contour_button(self):
        """Позиционирует кнопку 'Сброс контура' строго под надписью 'Контур сшивания'
        с выравниванием по её правому краю, либо скрывает кнопку, если вкладка не активна или контур пуст."""
        if not hasattr(self, "btn_reset_contour") or not hasattr(self, "canvas_2d"):
            return
        if getattr(self, "tabview", None) and self.tabview.get() != self.TAB_2D:
            self.btn_reset_contour.place_forget()
            return

        # Если контур не сформирован, скрываем кнопку
        if len(getattr(self, "boundary_indices", [])) < 2:
            self.btn_reset_contour.place_forget()
            return

        cw_2d = self.canvas_2d.get_tk_widget()
        leg = self.ax_2d.get_legend()
        if leg is not None and leg.get_visible():
            try:
                renderer = self.canvas_2d.get_renderer()
                bbox = leg.get_window_extent(renderer)
                fig_h = self.fig_2d.bbox.height or cw_2d.winfo_height()
                x = int(bbox.x1)
                y = int(fig_h - bbox.y0) + 4
                self.btn_reset_contour.place(x=x, y=y, anchor="ne")
                self.btn_reset_contour.lift()
                return
            except Exception:
                pass

        # Резервное позиционирование в верхнем правом углу под панелью поверхностей
        self.btn_reset_contour.place(relx=1.0, x=-26, y=60, anchor="ne")
        self.btn_reset_contour.lift()

    def _has_tk_canvas_widget(self) -> bool:
        """Проверяет, доступен ли нативный Tkinter Canvas виджет для сверхбыстрой отрисовки текста."""
        try:
            if not hasattr(self, "canvas_2d") or self.canvas_2d is None:
                return False
            if not hasattr(self.canvas_2d, "get_tk_widget"):
                return False
            widget = self.canvas_2d.get_tk_widget()
            return isinstance(widget, tk.Canvas)
        except Exception:
            return False

    def _clear_tk_canvas_labels(self):
        """Удаляет нативные текстовые метки точек с Tkinter Canvas."""
        if not self._has_tk_canvas_widget():
            return
        try:
            tk_canvas = self.canvas_2d.get_tk_widget()
            tk_canvas.delete("point_label")
        except Exception:
            pass

    def _compute_label_screen_pos(self, dx: float, dy: float, bbox, fig_h: float):
        """Вычисляет экранные координаты и anchor для подписи точки, гарантируя невылезание за ax.bbox."""
        x_min, x_max = bbox.x0, bbox.x1
        y_disp_min, y_disp_max = bbox.y0, bbox.y1

        # Если точка физически вне видимой области графика -> скрыть
        if dx < x_min or dx > x_max or dy < y_disp_min or dy > y_disp_max:
            return 0.0, 0.0, "sw", False

        # Защита от вылезания за верхнюю границу графика
        near_top = (dy > y_disp_max - 28.0)
        if near_top:
            v_anchor = "n"
            tk_y = fig_h - dy + 6.0
        else:
            v_anchor = "s"
            tk_y = fig_h - dy - 4.0

        # Защита от вылезания за правую границу графика
        near_right = (dx > x_max - 54.0)
        if near_right:
            h_anchor = "e"
            tk_x = dx - 5.0
        else:
            h_anchor = "w"
            tk_x = dx + 5.0

        anchor = v_anchor + h_anchor
        return tk_x, tk_y, anchor, True

    def _render_tk_canvas_labels(self, pts_to_label: list):
        """Создает нативные текстовые метки точек на Tkinter Canvas (10-12 мс для 500 точек).
        Точки за пределами ax.bbox помечаются state='hidden'. Текст поднимается поверх подложки Matplotlib.
        """
        self._current_labeled_points = list(pts_to_label) if pts_to_label else []
        if not self._has_tk_canvas_widget():
            return
        try:
            tk_canvas = self.canvas_2d.get_tk_widget()
            tk_canvas.delete("point_label")
            if not pts_to_label or not hasattr(self, "ax_2d") or self.ax_2d is None:
                return

            bbox = self.ax_2d.bbox
            fig_h = self.fig_2d.bbox.height or tk_canvas.winfo_height()

            coords = np.array([[p.y, p.x] for p in pts_to_label], dtype=float)
            disp_coords = self.ax_2d.transData.transform(coords)

            text_color = self._get_canvas_text_color()

            for i, p in enumerate(pts_to_label):
                dx, dy = disp_coords[i]
                tk_x, tk_y, anchor, visible = self._compute_label_screen_pos(dx, dy, bbox, fig_h)
                state = "normal" if visible else "hidden"
                tk_canvas.create_text(
                    tk_x, tk_y,
                    text=f"{p.id}\nH:{p.h:.3f}",
                    anchor=anchor,
                    font=("Segoe UI", 8),
                    fill=text_color,
                    state=state,
                    tags=("point_label", f"pt_{i}")
                )
            tk_canvas.tag_raise("point_label")
        except Exception:
            pass

    def _update_tk_canvas_labels_positions(self):
        """Мгновенно обновляет экранные координаты существующих текстовых меток точек (10 мс для 500 точек).
        Вызывается при панорамировании (drag), зуме (scroll) и изменении размера окна (resize).
        """
        if not self._has_tk_canvas_widget() or not getattr(self, "_current_labeled_points", None):
            return
        if not hasattr(self, "ax_2d") or self.ax_2d is None:
            return

        try:
            tk_canvas = self.canvas_2d.get_tk_widget()
            bbox = self.ax_2d.bbox
            fig_h = self.fig_2d.bbox.height or tk_canvas.winfo_height()

            pts = self._current_labeled_points
            coords = np.array([[p.y, p.x] for p in pts], dtype=float)
            disp_coords = self.ax_2d.transData.transform(coords)

            for i in range(len(pts)):
                dx, dy = disp_coords[i]
                tag = f"pt_{i}"
                tk_x, tk_y, anchor, visible = self._compute_label_screen_pos(dx, dy, bbox, fig_h)
                if visible:
                    tk_canvas.coords(tag, tk_x, tk_y)
                    tk_canvas.itemconfigure(tag, anchor=anchor, state="normal")
                else:
                    tk_canvas.itemconfigure(tag, state="hidden")
            tk_canvas.tag_raise("point_label")
        except Exception:
            pass

    def _update_2d_annotations_visibility(self):
        """Скрывает подписи точек, если сама точка пересекла любую границу холста,
        и планирует адаптивную синхронизацию подписей (LOD) при зуме."""
        if self._has_tk_canvas_widget():
            self._update_tk_canvas_labels_positions()
            self._schedule_viewport_annotations_update(delay_ms=120)
            return

        if not hasattr(self, "ax_2d") or not hasattr(self, "_point_annotations"):
            return
        if not self._point_annotations:
            self._schedule_viewport_annotations_update()
            return
        try:
            xlim = self.ax_2d.get_xlim()
            ylim = self.ax_2d.get_ylim()
            y_min, y_max = min(xlim), max(xlim)
            x_min, x_max = min(ylim), max(ylim)

            pan_hidden = getattr(self, "_pan_annotations_hidden", False)
            # Быстрая проверка границ в пространстве данных без поштучного вызова transData
            for item in self._point_annotations:
                ann = item[0]
                pt_y, pt_x = item[1]
                kind = item[2]
                if pt_y < y_min or pt_y > y_max or pt_x < x_min or pt_x > x_max:
                    ann.set_visible(False)
                else:
                    if kind == "bound" or not pan_hidden:
                        ann.set_visible(True)
                    else:
                        ann.set_visible(False)
        except Exception:
            pass

        self._schedule_viewport_annotations_update(delay_ms=120)

    def _schedule_viewport_annotations_update(self, delay_ms: int = 120):
        """Планирует отложенное обновление подписей (debouncing) при зуме/панорамировании."""
        try:
            if getattr(self, "_labels_update_timer", None) is not None:
                self.after_cancel(self._labels_update_timer)
        except Exception:
            pass
        try:
            self._labels_update_timer = self.after(delay_ms, self._sync_viewport_annotations)
        except Exception:
            self._labels_update_timer = None

    def _sync_viewport_annotations(self):
        """Синхронизирует текстовые подписи точек с текущей видимой областью холста."""
        self._labels_update_timer = None
        if not hasattr(self, "ax_2d") or not hasattr(self, "points") or not self.points:
            return
        if getattr(self, "canvas_2d", None) is None:
            return
        if getattr(self, "tabview", None) and self.tabview.get() != self.TAB_2D:
            return

        mode = getattr(self, "_point_labels_mode", "on")
        if mode != "on":
            self._clear_point_text_annotations()
            self.canvas_2d.draw_idle()
            return

        try:
            xlim = self.ax_2d.get_xlim()
            ylim = self.ax_2d.get_ylim()
            y_min, y_max = min(xlim), max(xlim)
            x_min, x_max = min(ylim), max(ylim)
        except Exception:
            return

        # Быстрый поиск видимых точек через NumPy SpatialIndex
        spatial = self._get_spatial_service()
        vis_indices = spatial.get_points_in_bbox(y_min, y_max, x_min, x_max)

        show_top = self._show_top.get() if hasattr(self, "_show_top") else True
        show_bottom = self._show_bottom.get() if hasattr(self, "_show_bottom") else True
        if not show_top and not show_bottom:
            self._clear_point_text_annotations()
            self.canvas_2d.draw_idle()
            return

        pts_to_label = []
        for idx in vis_indices:
            idx = int(idx)
            if not self._is_point_visible_on_2d(idx):
                continue
            pts_to_label.append(self.points[idx])

        if not pts_to_label:
            self._clear_point_text_annotations()
            self.canvas_2d.draw_idle()
            return

        if self._has_tk_canvas_widget():
            curr_pts = getattr(self, "_current_labeled_points", [])
            curr_ids = [getattr(p, "id", None) for p in curr_pts]
            new_ids = [getattr(p, "id", None) for p in pts_to_label]
            if curr_ids == new_ids and curr_ids:
                self._update_tk_canvas_labels_positions()
            else:
                self._render_tk_canvas_labels(pts_to_label)
            self.canvas_2d.draw_idle()
            return

        # Fallback для Matplotlib backend (тесты и headless)
        curr_point_pts = {item[1] for item in self._point_annotations if item[2] == "point"}
        new_point_pts = {(p.y, p.x) for p in pts_to_label}
        if curr_point_pts == new_point_pts:
            # Если точки те же, но были скрыты во время панорамирования — восстанавливаем видимость
            self._pan_annotations_hidden = False
            for item in self._point_annotations:
                if item[2] == "point":
                    item[0].set_visible(True)
            self.canvas_2d.draw_idle()
            return

        # Набор изменился — пересоздаем подписи точек (их <= 70 благодаря децимации, создаются моментально)
        self._pan_annotations_hidden = False
        self._clear_point_text_annotations()
        text_col = self._get_canvas_text_color()
        for p in pts_to_label:
            ann = self.ax_2d.annotate(
                f"{p.id}\nH:{p.h:.3f}", (p.y, p.x),
                textcoords="offset points", xytext=(5, 5),
                fontsize=9.0, alpha=0.9, zorder=5, clip_on=True,
                color=text_col
            )
            self._point_annotations.append((ann, (p.y, p.x), "point"))

        self.canvas_2d.draw_idle()

    def _clear_point_text_annotations(self):
        """Удаляет с холста только аннотации точек (сохраняя номера контура)."""
        self._clear_tk_canvas_labels()
        self._current_labeled_points = []
        kept = []
        for ann, coords, kind in self._point_annotations:
            if kind == "point":
                try:
                    ann.remove()
                except Exception:
                    pass
            else:
                kept.append((ann, coords, kind))
        self._point_annotations = kept

    def _build_3d_tab(self):
        # Верхняя панель инструментов 3D (компактная, 30 px)
        top_bar_3d = ctk.CTkFrame(self.tab_3d, fg_color="transparent", height=30)
        top_bar_3d.pack(fill=tk.X, side=tk.TOP, padx=4, pady=(2, 2))

        b_fs = ctk.CTkButton(top_bar_3d, text="⛶ На весь экран (Чистый 3D)", width=0, height=24, font=ctk.CTkFont(size=11),
                             command=self._open_fullscreen_3d)
        b_fs.pack(side=tk.LEFT, padx=(0, 6))
        add_tooltip(b_fs, "Развернуть 3D модель рельефа в отдельное полноэкранное окно без панелей (также двойной клик)")

        b_reset = ctk.CTkButton(top_bar_3d, text="🔄 Исходный ракурс", width=0, height=24, font=ctk.CTkFont(size=11),
                                command=self._reset_3d_view)
        b_reset.pack(side=tk.LEFT, padx=(0, 4))
        add_tooltip(b_reset, "Сбросить угол обзора, масштаб и поворот 3D модели к начальному состоянию")

        b_top = ctk.CTkButton(top_bar_3d, text="🔝 Вид сверху (План)", width=0, height=24, font=ctk.CTkFont(size=11),
                              command=self._top_3d_view)
        b_top.pack(side=tk.LEFT, padx=(0, 4))
        add_tooltip(b_top, "Установить вид строго сверху (в плане) для ортогонального обзора")

        cb_bot_3d = ctk.CTkCheckBox(
            top_bar_3d,
            text="▼ Нижняя",
            variable=self._show_bottom,
            command=self._on_surface_vis_toggle,
            width=16,
            height=16,
            checkbox_width=16,
            checkbox_height=16,
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        cb_bot_3d.pack(side=tk.RIGHT, padx=(4, 6))
        add_tooltip(cb_bot_3d, "Показать или скрыть точки и рельеф нижней поверхности")

        cb_top_3d = ctk.CTkCheckBox(
            top_bar_3d,
            text="▲ Верхняя",
            variable=self._show_top,
            command=self._on_surface_vis_toggle,
            width=16,
            height=16,
            checkbox_width=16,
            checkbox_height=16,
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        cb_top_3d.pack(side=tk.RIGHT, padx=(4, 4))
        add_tooltip(cb_top_3d, "Показать или скрыть точки и рельеф верхней поверхности")

        from mpl_toolkits.mplot3d import Axes3D
        is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False
        title_color = "#e0e0e0" if is_dark else "#212529"
        fig_bg = "#212529" if is_dark else "#ffffff"

        self.fig_3d = Figure(figsize=(6, 5), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(111, projection="3d")
        self.ax_3d.set_xlabel("Y (Восток)", color=title_color)
        self.ax_3d.set_ylabel("X (Север)", color=title_color)
        self.ax_3d.set_zlabel("H (Высота)", color=title_color)
        self._apply_axes_theme(self.ax_3d, self.fig_3d, is_dark)

        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.tab_3d)
        cw_3d = self.canvas_3d.get_tk_widget()
        cw_3d.configure(bg=fig_bg)
        cw_3d.pack(fill=tk.BOTH, expand=True)
        cw_3d.bind("<Double-Button-1>", lambda e: self._open_fullscreen_3d())

        self._toolbar_3d = ProjectNavigationToolbar(self.canvas_3d, cw_3d, pack_toolbar=False)
        self._toolbar_3d.place(relx=0.0, rely=1.0, anchor="sw", relwidth=1.0, height=26)
        self._setup_custom_toolbar_buttons(self._toolbar_3d, self.fig_3d, self.TAB_3D)
        self._toolbars.append(self._toolbar_3d)

    def _build_diff_tab(self):
        is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False
        title_color = "#e0e0e0" if is_dark else "#212529"
        fig_bg = "#212529" if is_dark else "#ffffff"

        self.fig_diff = Figure(figsize=(6, 5), dpi=100)
        self.ax_diff = self.fig_diff.add_subplot(111)
        self.ax_diff.set_xlabel("Восток Y (м)", color=title_color)
        self.ax_diff.set_ylabel("Север X (м)", color=title_color)
        self.ax_diff.grid(True, linestyle="--", alpha=0.5)
        self.ax_diff.format_coord = self._format_coord_diff
        self.ax_diff.xaxis.set_major_formatter(PlainOffsetFormatter(useOffset=True))
        self.ax_diff.yaxis.set_major_formatter(PlainOffsetFormatter(useOffset=True))
        self._apply_axes_theme(self.ax_diff, self.fig_diff, is_dark)

        self.canvas_diff = FigureCanvasTkAgg(self.fig_diff, master=self.tab_diff)
        cw_diff = self.canvas_diff.get_tk_widget()
        cw_diff.configure(bg=fig_bg)
        cw_diff.pack(fill=tk.BOTH, expand=True)
        self._fs_surface_panels[self.TAB_DIFF] = self._create_canvas_surface_panel(cw_diff)

        self._toolbar_diff = ProjectNavigationToolbar(self.canvas_diff, cw_diff, pack_toolbar=False)
        self._toolbar_diff.place(relx=0.0, rely=1.0, anchor="sw", relwidth=1.0, height=26)
        self._setup_custom_toolbar_buttons(self._toolbar_diff, self.fig_diff, self.TAB_DIFF)
        self._toolbars.append(self._toolbar_diff)

        # Зум и панорамирование как на вкладке «Схема в плане»
        self._diff_pan_start = None
        self._diff_pan_dragged = False
        self.canvas_diff.mpl_connect("button_press_event", self._on_diff_canvas_press)
        self.canvas_diff.mpl_connect("button_release_event", self._on_diff_canvas_release)
        self.canvas_diff.mpl_connect("motion_notify_event", self._on_diff_canvas_motion)
        self.canvas_diff.mpl_connect("scroll_event", self._on_diff_canvas_scroll)

    def _throttled_canvas_draw(self, canvas, delay_ms: int = 25):
        """Плавная отрисовка холста при интерактивном перетаскивании/панорамировании (FPS gate).
        Предотвращает накопление очереди перерисовок при частых событиях движения мыши."""
        if canvas is None:
            return
        if not hasattr(self, "_pan_draw_timers") or not isinstance(self._pan_draw_timers, dict):
            self._pan_draw_timers = {}
        if canvas in self._pan_draw_timers:
            return

        def _do_draw():
            self._pan_draw_timers.pop(canvas, None)
            try:
                canvas.draw()
            except Exception:
                pass

        try:
            if hasattr(self, "after") and callable(self.after):
                self._pan_draw_timers[canvas] = self.after(delay_ms, _do_draw)
            else:
                canvas.draw_idle()
        except Exception:
            canvas.draw_idle()

    def _flush_canvas_draw(self, canvas):
        """Немедленно сбрасывает таймеры панорамирования и запрашивает актуальную отрисовку холста."""
        if canvas is None:
            return
        if hasattr(self, "_pan_draw_timers") and isinstance(self._pan_draw_timers, dict):
            timer = self._pan_draw_timers.pop(canvas, None)
            if timer is not None:
                try:
                    if hasattr(self, "after_cancel") and callable(self.after_cancel):
                        self.after_cancel(timer)
                except Exception:
                    pass
        try:
            canvas.draw_idle()
        except Exception:
            pass

    def _on_diff_canvas_press(self, event):
        """Нажатие мыши на картограмме — запоминаем стартовое положение для панорамирования"""
        try:
            if hasattr(self, "_toolbar_diff") and self._toolbar_diff.mode != "":
                return
        except Exception:
            pass
        if event.x is None or event.y is None:
            return
        self._diff_pan_start = (
            event.x, event.y,
            event.xdata, event.ydata,
            self.ax_diff.get_xlim(),
            self.ax_diff.get_ylim(),
            event.button
        )
        self._diff_pan_dragged = False

    def _on_diff_canvas_motion(self, event):
        """Панорамирование картограммы при перетаскивании мышью"""
        if self._diff_pan_start is None or event.x is None or event.y is None:
            return
        start_px_x, start_px_y, _, _, orig_xlim, orig_ylim, btn = self._diff_pan_start
        dx_px = event.x - start_px_x
        dy_px = event.y - start_px_y
        drag_threshold = 3 if btn in (2, 3) else 8
        if abs(dx_px) > drag_threshold or abs(dy_px) > drag_threshold:
            self._diff_pan_dragged = True
        if self._diff_pan_dragged:
            bbox = self.ax_diff.bbox
            if bbox.width > 0 and bbox.height > 0:
                dx_data = dx_px * (orig_xlim[1] - orig_xlim[0]) / bbox.width
                dy_data = dy_px * (orig_ylim[1] - orig_ylim[0]) / bbox.height
                self.ax_diff.set_xlim(orig_xlim[0] - dx_data, orig_xlim[1] - dx_data)
                self.ax_diff.set_ylim(orig_ylim[0] - dy_data, orig_ylim[1] - dy_data)
                self._throttled_canvas_draw(self.canvas_diff, delay_ms=25)

    def _on_diff_canvas_release(self, event):
        """Отпускание мыши на картограмме"""
        was_dragged = self._diff_pan_dragged
        self._diff_pan_start = None
        self._diff_pan_dragged = False
        if was_dragged:
            self._flush_canvas_draw(self.canvas_diff)

    def _on_diff_canvas_scroll(self, event):
        """Зум колесом мыши вокруг курсора на картограмме"""
        if event.xdata is None or event.ydata is None:
            return
        base_scale = 1.25
        scale_factor = 1.0 / base_scale if event.button == "up" else base_scale
        cur_xlim = self.ax_diff.get_xlim()
        cur_ylim = self.ax_diff.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        new_width  = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        self.ax_diff.set_xlim([xdata - new_width  * (1 - relx), xdata + new_width  * relx])
        self.ax_diff.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self._throttled_canvas_draw(self.canvas_diff, delay_ms=25)

    def _build_table_tab(self):
        # Панель инструментов редактирования точек над таблицей
        tbl_toolbar = ctk.CTkFrame(self.tab_table, fg_color="transparent", height=40)
        tbl_toolbar.pack(fill=tk.X, side=tk.TOP, padx=5, pady=4)

        b_add = ctk.CTkButton(tbl_toolbar, text="➕ Добавить точку", width=0, height=24,
                              font=ctk.CTkFont(size=11), command=self._add_point_dialog)
        b_add.pack(side=tk.LEFT, padx=(0, 4))
        add_tooltip(b_add, "Вручную ввести координаты (Север X, Восток Y, Высота H) новой точки")

        b_edit = ctk.CTkButton(tbl_toolbar, text="✏️ Редактировать", width=0, height=24,
                               font=ctk.CTkFont(size=11), command=self._edit_selected_point_dialog)
        b_edit.pack(side=tk.LEFT, padx=(0, 4))
        add_tooltip(b_edit, "Редактировать координаты или имя выбранной точки (также двойной клик)")

        b_del = ctk.CTkButton(tbl_toolbar, text="🗑️ Удалить точку", width=0, height=24,
                              fg_color="#8B2020", hover_color="#A02828",
                              font=ctk.CTkFont(size=11), command=self._delete_selected_point)
        b_del.pack(side=tk.LEFT, padx=(0, 4))
        add_tooltip(b_del, "Удалить выбранную точку из проекта")

        b_toggle = ctk.CTkButton(tbl_toolbar, text="🔄 Сменить тип", width=0, height=24,
                                 font=ctk.CTkFont(size=11), command=self._toggle_selected_point_type)
        b_toggle.pack(side=tk.LEFT, padx=(0, 4))
        add_tooltip(b_toggle, "Переключить принадлежность точки (Верхняя / Нижняя / Контур / Авто)")

        b_remap = ctk.CTkButton(tbl_toolbar, text="🔀 Колонки...", width=0, height=24,
                                font=ctk.CTkFont(size=11), command=self._open_remap_dialog)
        b_remap.pack(side=tk.LEFT, padx=(0, 6))
        add_tooltip(b_remap, "Настроить сопоставление столбцов таблицы с осями X, Y, Z")

        self.lbl_table_stats = ctk.CTkLabel(tbl_toolbar, text="Всего: 0 точек",
                                             font=ctk.CTkFont(size=10, weight="bold"))
        self.lbl_table_stats.pack(side=tk.RIGHT, padx=5)

        # Контейнер для Treeview и Scrollbar
        tbl_container = ttk.Frame(self.tab_table)
        tbl_container.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        cols = ("idx", "id", "x", "y", "h", "surface")
        self.tree = ttk.Treeview(tbl_container, columns=cols, show="headings", selectmode="browse")
        self.tree.heading("idx", text="№")
        self.tree.heading("id", text="Имя / ID")
        self.tree.heading("x", text="Север X (м)")
        self.tree.heading("y", text="Восток Y (м)")
        self.tree.heading("h", text="Высота H (м)")
        self.tree.heading("surface", text="Тип поверхности")

        self.tree.column("idx", width=45, anchor=tk.CENTER)
        self.tree.column("id", width=100, anchor=tk.CENTER)

        self.tree.column("x", width=140, anchor=tk.E)
        self.tree.column("y", width=140, anchor=tk.E)
        self.tree.column("h", width=110, anchor=tk.E)
        self.tree.column("surface", width=140, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(tbl_container, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<Double-1>", lambda e: self._edit_selected_point_dialog())

    def _build_tin_tab(self):
        """Строит вкладку геодезических треугольников (TIN) с компактной панелью инструментов"""
        # Единая компактная верхняя панель инструментов TIN (30 px)
        top_bar = ctk.CTkFrame(self.tab_tin, fg_color="transparent", height=30)
        top_bar.pack(fill=tk.X, side=tk.TOP, padx=4, pady=(2, 2))

        # Статус — справа
        self.lbl_tin_stats = ctk.CTkLabel(top_bar, text="Треугольников: —",
                                           font=ctk.CTkFont(size=10, weight="bold"),
                                           text_color=("#1a5276", "#e0e0e0"))
        self.lbl_tin_stats.pack(side=tk.RIGHT, padx=(6, 4))

        b_fit = ctk.CTkButton(top_bar, text="🔍 В фокус", width=0, height=24, font=ctk.CTkFont(size=11),
                              command=self._fit_tin_view)
        b_fit.pack(side=tk.LEFT, padx=(0, 3))
        add_tooltip(b_fit, "Вписать всю триангуляцию в границы экрана")

        b_ex = ctk.CTkButton(top_bar, text="✂ Исключить", width=0, height=24, font=ctk.CTkFont(size=11),
                             command=self._tin_toggle_selected)
        b_ex.pack(side=tk.LEFT, padx=(0, 3))
        add_tooltip(b_ex, "Исключить/вернуть выбранный треугольник")

        b_res = ctk.CTkButton(top_bar, text="↩ Сброс", width=0, height=24, font=ctk.CTkFont(size=11),
                              command=self._tin_reset)
        b_res.pack(side=tk.LEFT, padx=(0, 3))
        add_tooltip(b_res, "Сбросить исключения треугольников")

        b_table = ctk.CTkButton(top_bar, text="📊 Таблица", width=0, height=24, font=ctk.CTkFont(size=11),
                                command=self._open_tin_table_dialog)
        b_table.pack(side=tk.LEFT, padx=(0, 3))
        add_tooltip(b_table, "Открыть полноэкранную интерактивную таблицу треугольников с сортировкой и фильтрацией")

        self.btn_toggle_tin_table = None

        b_del = ctk.CTkButton(top_bar, text="🔄 Делоне", width=0, height=24, font=ctk.CTkFont(size=11),
                              command=self._tin_reset_to_delaunay)
        b_del.pack(side=tk.LEFT, padx=(0, 4))
        add_tooltip(b_del, "Сбросить сетку к стандартной триангуляции Делоне")

        b_opt = ctk.CTkButton(top_bar, text="⚡ Оптимизация", width=0, height=24, font=ctk.CTkFont(size=11),
                              command=self._auto_optimize_tin_edges)
        b_opt.pack(side=tk.LEFT, padx=(0, 4))
        add_tooltip(b_opt, "Автоматическая оптимизация рёбер триангуляции по минимальному углу")

        ctk.CTkLabel(top_bar, text="Угол:", font=ctk.CTkFont(size=10)).pack(side=tk.LEFT)
        self._tin_min_angle = tk.IntVar(value=7)
        ttk.Spinbox(top_bar, from_=1, to=45, increment=1,
                    textvariable=self._tin_min_angle, width=5,
                    font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(2, 2))
        ctk.CTkLabel(top_bar, text="°", font=ctk.CTkFont(size=10)).pack(side=tk.LEFT, padx=(0, 4))

        ctk.CTkLabel(top_bar, text="Итер:", font=ctk.CTkFont(size=10)).pack(side=tk.LEFT)
        self._tin_max_iter = tk.IntVar(value=4)
        ttk.Spinbox(top_bar, from_=1, to=50, increment=1,
                    textvariable=self._tin_max_iter, width=5,
                    font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(2, 2))

        is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False
        title_color = "#e0e0e0" if is_dark else "#212529"
        fig_bg = "#212529" if is_dark else "#ffffff"

        self.fig_tin = Figure(figsize=(6, 4), dpi=100)
        self.ax_tin = self.fig_tin.add_subplot(111)
        self.ax_tin.set_xlabel("Восток Y (м)", color=title_color)
        self.ax_tin.set_ylabel("Север X (м)", color=title_color)
        self.ax_tin.grid(True, linestyle="--", alpha=0.4)
        self.ax_tin.format_coord = self._format_coord_display
        self.ax_tin.xaxis.set_major_formatter(PlainOffsetFormatter(useOffset=True))
        self.ax_tin.yaxis.set_major_formatter(PlainOffsetFormatter(useOffset=True))
        self._apply_axes_theme(self.ax_tin, self.fig_tin, is_dark)

        self.canvas_tin = FigureCanvasTkAgg(self.fig_tin, master=self.tab_tin)
        cw_tin = self.canvas_tin.get_tk_widget()
        cw_tin.configure(bg=fig_bg)
        cw_tin.pack(fill=tk.BOTH, expand=True)
        self._fs_surface_panels[self.TAB_TIN] = self._create_canvas_surface_panel(cw_tin)

        self._toolbar_tin = ProjectNavigationToolbar(self.canvas_tin, cw_tin, pack_toolbar=False)
        self._toolbar_tin.place(relx=0.0, rely=1.0, anchor="sw", relwidth=1.0, height=26)
        self._setup_custom_toolbar_buttons(self._toolbar_tin, self.fig_tin, self.TAB_TIN)
        self._toolbars.append(self._toolbar_tin)

        # Панорамирование мышью (ЛКМ перетаскивание), зум колесом и клик по треугольнику
        self.canvas_tin.mpl_connect("button_press_event", self._on_tin_canvas_press)
        self.canvas_tin.mpl_connect("button_release_event", self._on_tin_canvas_release)
        self.canvas_tin.mpl_connect("motion_notify_event", self._on_tin_canvas_motion)
        self.canvas_tin.mpl_connect("scroll_event", self._on_tin_canvas_scroll)

        # Скрытые объекты для совместимости с внешними вызовами и методами
        self.paned_tin = None
        self._tin_bottom_frame = None
        self.tree_tin = None

    def _build_contours_tab(self):
        """Строит вкладку топографических горизонталей (изогипс) с компактной панелью инструментов"""
        top_bar = ctk.CTkFrame(self.tab_contours, fg_color="transparent", height=30)
        top_bar.pack(fill=tk.X, side=tk.TOP, padx=4, pady=(2, 2))

        # Статус / диапазон высот — справа
        self.lbl_contours_stats = ctk.CTkLabel(
            top_bar, text="Горизонтали: —",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=("#1a5276", "#e0e0e0")
        )
        self.lbl_contours_stats.pack(side=tk.RIGHT, padx=(6, 4))

        b_fit = ctk.CTkButton(
            top_bar, text="🔍 В фокус", width=0, height=24, font=ctk.CTkFont(size=11),
            command=self._fit_contours_view
        )
        b_fit.pack(side=tk.LEFT, padx=(0, 4))
        add_tooltip(b_fit, "Вписать план с горизонталями в границы экрана")

        lbl_step = ctk.CTkLabel(top_bar, text="Шаг (м):", font=ctk.CTkFont(size=10))
        lbl_step.pack(side=tk.LEFT, padx=(4, 2))

        self.cbo_contour_step = ttk.Combobox(
            top_bar,
            values=["Авто", "0.05", "0.1", "0.25", "0.5", "1.0", "2.0", "5.0"],
            state="readonly",
            width=7,
            font=("Segoe UI", 9)
        )
        self.cbo_contour_step.set("Авто")
        self.cbo_contour_step.bind("<<ComboboxSelected>>", lambda e: self._on_contour_settings_changed())
        self.cbo_contour_step.pack(side=tk.LEFT, padx=(0, 6))
        add_tooltip(self.cbo_contour_step, "Шаг сечения рельефа горизонталями (в метрах)")

        self._show_contour_labels = tk.BooleanVar(value=True)
        cb_labels = ctk.CTkCheckBox(
            top_bar,
            text="Отметки",
            variable=self._show_contour_labels,
            command=self._on_contour_settings_changed,
            width=16,
            height=16,
            checkbox_width=16,
            checkbox_height=16,
            font=ctk.CTkFont(size=11)
        )
        cb_labels.pack(side=tk.LEFT, padx=(0, 6))
        add_tooltip(cb_labels, "Отображать числовые отметки высот на горизонталях")

        is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False
        title_color = "#e0e0e0" if is_dark else "#212529"
        fig_bg = "#212529" if is_dark else "#ffffff"

        self.fig_contours = Figure(figsize=(6, 4), dpi=100)
        self.ax_contours = self.fig_contours.add_subplot(111)
        self.ax_contours.set_xlabel("Восток Y (м)", color=title_color)
        self.ax_contours.set_ylabel("Север X (м)", color=title_color)
        self.ax_contours.grid(True, linestyle="--", alpha=0.4)
        self.ax_contours.format_coord = self._format_coord_contours
        self.ax_contours.xaxis.set_major_formatter(PlainOffsetFormatter(useOffset=True))
        self.ax_contours.yaxis.set_major_formatter(PlainOffsetFormatter(useOffset=True))
        self._apply_axes_theme(self.ax_contours, self.fig_contours, is_dark)

        self.canvas_contours = FigureCanvasTkAgg(self.fig_contours, master=self.tab_contours)
        cw_contours = self.canvas_contours.get_tk_widget()
        cw_contours.configure(bg=fig_bg)
        cw_contours.pack(fill=tk.BOTH, expand=True)
        self._fs_surface_panels[self.TAB_CONTOURS] = self._create_canvas_surface_panel(cw_contours)

        self._toolbar_contours = ProjectNavigationToolbar(self.canvas_contours, cw_contours, pack_toolbar=False)
        self._toolbar_contours.place(relx=0.0, rely=1.0, anchor="sw", relwidth=1.0, height=26)
        self._setup_custom_toolbar_buttons(self._toolbar_contours, self.fig_contours, self.TAB_CONTOURS)
        self._toolbars.append(self._toolbar_contours)

        self.canvas_contours.mpl_connect("button_press_event", self._on_contours_canvas_press)
        self.canvas_contours.mpl_connect("button_release_event", self._on_contours_canvas_release)
        self.canvas_contours.mpl_connect("motion_notify_event", self._on_contours_canvas_motion)
        self.canvas_contours.mpl_connect("scroll_event", self._on_contours_canvas_scroll)

    def _on_contour_settings_changed(self, *args):
        """Вызывается при изменении настроек шага или отметок горизонталей"""
        self._redraw_contours()

    def _fit_contours_view(self):
        """Устанавливает границы ax_contours точно в фокус отображаемых данных с отступом 5%"""
        if not hasattr(self, "ax_contours") or self.ax_contours is None:
            return

        pts_y = []
        pts_x = []

        if len(self.boundary_indices) >= 3:
            boundary_set = set(self.boundary_indices)
            bound_pts = np.array([[self.points[i].x, self.points[i].y] for i in self.boundary_indices])
            bound_path = MplPath(bound_pts)

            for i, p in enumerate(self.points):
                if i in boundary_set or bound_path.contains_point((p.x, p.y), radius=1e-5):
                    pts_y.append(p.y)
                    pts_x.append(p.x)

        if not pts_y or not pts_x:
            pts_y = [p.y for p in self.points]
            pts_x = [p.x for p in self.points]

        if not pts_y or not pts_x:
            if self.calc_results:
                boundary = self.calc_results.get("boundary")
                if boundary is not None and len(boundary) > 0:
                    pts_y = boundary[:, 1].tolist()
                    pts_x = boundary[:, 0].tolist()

        if not pts_y or not pts_x:
            return

        min_y, max_y = min(pts_y), max(pts_y)
        min_x, max_x = min(pts_x), max(pts_x)

        span_y = max(max_y - min_y, 1.0)
        span_x = max(max_x - min_x, 1.0)

        pad_y = span_y * 0.05
        pad_x = span_x * 0.05

        self.ax_contours.set_xlim(min_y - pad_y, max_y + pad_y)
        self.ax_contours.set_ylim(min_x - pad_x, max_x + pad_x)
        self.ax_contours.set_aspect("equal", adjustable="datalim")
        if hasattr(self, "canvas_contours") and self.canvas_contours is not None:
            self.canvas_contours.draw_idle()

    def _get_contours_data(self):
        """Возвращает данные для построения горизонталей:
        1. Если выполнен расчет объема (self.calc_results) и есть контур (len(self.boundary_indices) >= 3),
           возвращает расчетную сетку и полигон сшивания.
        2. Иначе строит сетку напрямую по имеющимся точкам съемки (self.points), независимо от наличия контура.
        """
        if not self.points or len(self.points) < 3:
            return None

        # 1. Если есть готовый расчет объема внутри контура
        if getattr(self, "calc_results", None) is not None and len(getattr(self, "boundary_indices", [])) >= 3:
            r = self.calc_results
            gx = r.get("grid_x")
            gy = r.get("grid_y")
            z_top = r.get("z_top_grid")
            z_bot = r.get("z_bot_grid")
            if gx is not None and gy is not None:
                return {
                    "grid_x": gx,
                    "grid_y": gy,
                    "z_top_grid": z_top,
                    "z_bot_grid": z_bot,
                    "boundary": r.get("boundary"),
                    "is_single_survey": False,
                }

        # 2. Если контур не задан или расчет объема еще не выполнялся —
        # строим интерполированную сетку напрямую по имеющимся точкам съемки
        cache = getattr(self, "_raw_contours_cache", None)
        pts_len = len(self.points)
        bnd_len = len(getattr(self, "boundary_indices", []))
        if cache is not None and cache.get("pts_len") == pts_len and cache.get("bnd_len") == bnd_len:
            return cache.get("data")

        try:
            from scipy.interpolate import LinearNDInterpolator

            pts_x = np.array([p.x for p in self.points], dtype=np.float64)
            pts_y = np.array([p.y for p in self.points], dtype=np.float64)
            pts_h = np.array([p.h for p in self.points], dtype=np.float64)

            min_x, max_x = float(np.min(pts_x)), float(np.max(pts_x))
            min_y, max_y = float(np.min(pts_y)), float(np.max(pts_y))
            span_max = max(max_x - min_x, max_y - min_y)
            if span_max < 1e-4:
                return None

            res = max(span_max / 350.0, 0.2)
            gx = np.arange(min_x, max_x + res, res)
            gy = np.arange(min_y, max_y + res, res)
            grid_x, grid_y = np.meshgrid(gx, gy)

            is_two = self._is_two_surfaces()
            top_indices = [i for i, p in enumerate(self.points) if p.surface_type == "top"]
            bot_indices = [i for i, p in enumerate(self.points) if p.surface_type == "bottom"]
            has_two_explicit = (len(top_indices) >= 3 and len(bot_indices) >= 3)

            boundary = None
            if len(getattr(self, "boundary_indices", [])) >= 3:
                boundary = np.array([[self.points[i].x, self.points[i].y] for i in self.boundary_indices if 0 <= i < len(self.points)])

            if is_two or has_two_explicit:
                interp_top = LinearNDInterpolator(np.column_stack((pts_x[top_indices], pts_y[top_indices])), pts_h[top_indices])
                interp_bot = LinearNDInterpolator(np.column_stack((pts_x[bot_indices], pts_y[bot_indices])), pts_h[bot_indices])
                z_top = interp_top(grid_x, grid_y)
                z_bot = interp_bot(grid_x, grid_y)
                is_single = False
            else:
                interp_all = LinearNDInterpolator(np.column_stack((pts_x, pts_y)), pts_h)
                z_all = interp_all(grid_x, grid_y)
                z_top = z_all
                z_bot = None
                is_single = True

            data = {
                "grid_x": grid_x,
                "grid_y": grid_y,
                "z_top_grid": z_top,
                "z_bot_grid": z_bot,
                "boundary": boundary,
                "is_single_survey": is_single,
            }
            self._raw_contours_cache = {
                "pts_len": pts_len,
                "bnd_len": bnd_len,
                "data": data,
            }
            return data
        except Exception:
            return None

    def _redraw_contours(self):
        """Отрисовывает топографические горизонтали рельефа с векторной обрезкой по контуру (если задан) или по всем точкам съемки"""
        if not hasattr(self, "ax_contours") or self.ax_contours is None:
            return

        is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False
        title_color = "#e0e0e0" if is_dark else "#212529"

        cur_xlim = self.ax_contours.get_xlim() if getattr(self, "_contours_view_initialized", False) else None
        cur_ylim = self.ax_contours.get_ylim() if getattr(self, "_contours_view_initialized", False) else None

        self.ax_contours.clear()
        self._apply_axes_theme(self.ax_contours, getattr(self, "fig_contours", None), is_dark)
        self.ax_contours.set_xlabel("Восток Y (м)", color=title_color)
        self.ax_contours.set_ylabel("Север X (м)", color=title_color)
        self.ax_contours.grid(True, linestyle="--", alpha=0.35)
        self.ax_contours.format_coord = self._format_coord_contours
        self.ax_contours.xaxis.set_major_formatter(PlainOffsetFormatter(useOffset=True))
        self.ax_contours.yaxis.set_major_formatter(PlainOffsetFormatter(useOffset=True))

        if not self.points or len(self.points) < 3:
            self.ax_contours.text(
                0.5, 0.5,
                "Для построения горизонталей загрузите файл с точками съемки\n(требуется минимум 3 точки)",
                transform=self.ax_contours.transAxes,
                ha="center", va="center",
                fontsize=11, color=title_color, alpha=0.7
            )
            if hasattr(self, "lbl_contours_stats") and self.lbl_contours_stats is not None:
                if hasattr(self.lbl_contours_stats, "configure"):
                    self.lbl_contours_stats.configure(text="Горизонтали: точки не загружены")
                elif hasattr(self.lbl_contours_stats, "setText"):
                    self.lbl_contours_stats.setText("Горизонтали: точки не загружены")
            if hasattr(self, "canvas_contours") and self.canvas_contours is not None:
                self.canvas_contours.draw_idle()
            return

        data = self._get_contours_data()
        if data is None:
            self.ax_contours.text(
                0.5, 0.5,
                "Недостаточно данных для построения горизонталей\n(точки коллинеарны или имеют одинаковые координаты)",
                transform=self.ax_contours.transAxes,
                ha="center", va="center",
                fontsize=11, color=title_color, alpha=0.7
            )
            if hasattr(self, "lbl_contours_stats") and self.lbl_contours_stats is not None:
                if hasattr(self.lbl_contours_stats, "configure"):
                    self.lbl_contours_stats.configure(text="Горизонтали: ошибка построения")
                elif hasattr(self.lbl_contours_stats, "setText"):
                    self.lbl_contours_stats.setText("Горизонтали: ошибка построения")
            if hasattr(self, "canvas_contours") and self.canvas_contours is not None:
                self.canvas_contours.draw_idle()
            return

        gx = data.get("grid_x")
        gy = data.get("grid_y")
        z_top = data.get("z_top_grid")
        z_bot = data.get("z_bot_grid")
        boundary = data.get("boundary")
        is_single = data.get("is_single_survey", False)

        if gx is None or gy is None:
            return

        show_top = self._show_top.get() if hasattr(self, "_show_top") else True
        show_bottom = self._show_bottom.get() if hasattr(self, "_show_bottom") else True

        if not show_top and not show_bottom:
            self.ax_contours.text(
                0.5, 0.5,
                "Отображение поверхностей отключено.\nВключите чекбокс '▲ Верхняя' или '▼ Нижняя' в правом верхнем углу.",
                transform=self.ax_contours.transAxes,
                ha="center", va="center",
                fontsize=11, color=title_color, alpha=0.7
            )
            if hasattr(self, "lbl_contours_stats") and self.lbl_contours_stats is not None:
                if hasattr(self.lbl_contours_stats, "configure"):
                    self.lbl_contours_stats.configure(text="Поверхности скрыты")
                elif hasattr(self.lbl_contours_stats, "setText"):
                    self.lbl_contours_stats.setText("Поверхности скрыты")
            if hasattr(self, "canvas_contours") and self.canvas_contours is not None:
                self.canvas_contours.draw_idle()
            return

        # Полигон отсечения по контуру границы съёмки (если контур задан)
        poly_clip = None
        bound_closed_yx = None
        if boundary is not None and len(boundary) >= 3:
            boundary_yx = np.column_stack((boundary[:, 1], boundary[:, 0]))
            bound_closed_yx = np.vstack([boundary_yx, boundary_yx[0]])
            poly_clip = MplPolygon(boundary_yx, closed=True, facecolor="none", edgecolor="none", transform=self.ax_contours.transData)
            self.ax_contours.add_patch(poly_clip)

        # Сбор диапазонов высот
        all_z_valid = []
        if is_single:
            survey_valid = z_top[~np.isnan(z_top)] if z_top is not None else np.array([])
            if len(survey_valid) > 0:
                all_z_valid.append(survey_valid)
        else:
            top_valid = z_top[~np.isnan(z_top)] if (show_top and z_top is not None) else np.array([])
            bot_valid = z_bot[~np.isnan(z_bot)] if (show_bottom and z_bot is not None) else np.array([])
            if len(top_valid) > 0:
                all_z_valid.append(top_valid)
            if len(bot_valid) > 0:
                all_z_valid.append(bot_valid)

        if not all_z_valid:
            return

        merged_z = np.concatenate(all_z_valid)
        z_min_total = float(np.min(merged_z))
        z_max_total = float(np.max(merged_z))
        delta_z = max(z_max_total - z_min_total, 0.01)

        # Определение шага горизонталей
        step_str = self.cbo_contour_step.get() if (hasattr(self, "cbo_contour_step") and self.cbo_contour_step is not None) else "Авто"
        if step_str == "Авто" or not step_str:
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
            elif delta_z <= 60.0:
                c_step = 2.0
            else:
                c_step = 5.0
        else:
            try:
                c_step = float(step_str.replace(",", "."))
                if c_step <= 0:
                    c_step = 0.5
            except Exception:
                c_step = 0.5

        # Формирование инфо-строки
        stat_parts = [f"Шаг h = {c_step:g} м"]
        if is_single:
            stat_parts.append(f"Рельеф съемки: {z_min_total:.2f}..{z_max_total:.2f} м ({len(self.points)} точек)")
        else:
            if show_top and len(top_valid) > 0:
                stat_parts.append(f"Верх: {float(np.min(top_valid)):.2f}..{float(np.max(top_valid)):.2f} м")
            if show_bottom and len(bot_valid) > 0:
                stat_parts.append(f"Низ: {float(np.min(bot_valid)):.2f}..{float(np.max(bot_valid)):.2f} м")
        if hasattr(self, "lbl_contours_stats") and self.lbl_contours_stats is not None:
            if hasattr(self.lbl_contours_stats, "configure"):
                self.lbl_contours_stats.configure(text=" | ".join(stat_parts))
            elif hasattr(self.lbl_contours_stats, "setText"):
                self.lbl_contours_stats.setText(" | ".join(stat_parts))

        show_labels = self._show_contour_labels.get() if hasattr(self, "_show_contour_labels") and self._show_contour_labels is not None else True
        fmt_digits = 2 if c_step < 0.1 else (1 if c_step < 1.0 or any(abs(round(v, 1) - v) > 1e-4 for v in [z_min_total, z_max_total]) else 1)
        fmt_str = f"%.{fmt_digits}f"

        legend_lines = []
        legend_labels = []

        # Легкие маркеры точек съемки на заднем плане
        pts_y = [p.y for p in self.points]
        pts_x = [p.x for p in self.points]
        pt_dot_col = "#94a3b8" if is_dark else "#64748b"
        self.ax_contours.scatter(
            pts_y, pts_x, s=3.5, color=pt_dot_col,
            alpha=0.35 if len(self.points) < 5000 else 0.18,
            zorder=1, label="_nolegend_"
        )

        def _draw_surface_contours(grid_z, is_top_surface, custom_label=None):
            if grid_z is None:
                return
            z_clean = grid_z[~np.isnan(grid_z)]
            if len(z_clean) == 0:
                return
            z_min_s = float(np.min(z_clean))
            z_max_s = float(np.max(z_clean))
            if z_max_s - z_min_s < 1e-4:
                return

            first_level = np.floor(z_min_s / c_step) * c_step
            last_level = np.ceil(z_max_s / c_step) * c_step
            levels = np.arange(first_level, last_level + c_step * 0.5, c_step)
            if len(levels) == 0:
                return

            index_mult = 5
            index_levels = [lvl for lvl in levels if abs(round(round(lvl / c_step) % index_mult)) < 1e-4]
            inter_levels = [lvl for lvl in levels if abs(round(round(lvl / c_step) % index_mult)) >= 1e-4]

            if is_top_surface:
                col_idx = "#f59e0b" if is_dark else "#b45309"
                col_sub = "#d97706" if is_dark else "#d97706"
                ls_style = "-"
                alpha_idx = 0.95
                alpha_sub = 0.65
                surf_label = custom_label or "Верхняя поверхность"
            else:
                col_idx = "#38bdf8" if is_dark else "#0369a1"
                col_sub = "#0ea5e9" if is_dark else "#0284c7"
                ls_style = "--" if (show_top and not is_single) else "-"
                alpha_idx = 0.95
                alpha_sub = 0.60
                surf_label = custom_label or "Нижняя поверхность"

            cs_sub = None
            if len(inter_levels) > 0:
                try:
                    cs_sub = self.ax_contours.contour(
                        gy, gx, grid_z, levels=inter_levels,
                        colors=col_sub, linewidths=0.75, linestyles=ls_style,
                        alpha=alpha_sub, zorder=2
                    )
                    if poly_clip is not None:
                        for coll in getattr(cs_sub, "collections", []):
                            coll.set_clip_path(poly_clip)
                except Exception:
                    pass

            if len(index_levels) > 0:
                try:
                    cs_idx = self.ax_contours.contour(
                        gy, gx, grid_z, levels=index_levels,
                        colors=col_idx, linewidths=1.45, linestyles=ls_style,
                        alpha=alpha_idx, zorder=3
                    )
                    if poly_clip is not None:
                        for coll in getattr(cs_idx, "collections", []):
                            coll.set_clip_path(poly_clip)
                    if show_labels:
                        self.ax_contours.clabel(
                            cs_idx, inline=True, fontsize=8.5, fmt=fmt_str,
                            inline_spacing=8, use_clabeltext=True
                        )
                except Exception:
                    pass
            elif show_labels and cs_sub is not None:
                try:
                    self.ax_contours.clabel(
                        cs_sub, inline=True, fontsize=8.0, fmt=fmt_str,
                        inline_spacing=8, use_clabeltext=True
                    )
                except Exception:
                    pass

            legend_lines.append(matplotlib.lines.Line2D([0], [0], color=col_idx, lw=1.6, linestyle=ls_style))
            legend_labels.append(surf_label)

        if is_single:
            use_top_palette = show_top or not show_bottom
            _draw_surface_contours(z_top, is_top_surface=use_top_palette, custom_label="Рельеф съемки")
        else:
            if show_bottom:
                _draw_surface_contours(z_bot, is_top_surface=False)
            if show_top:
                _draw_surface_contours(z_top, is_top_surface=True)

        if bound_closed_yx is not None:
            bound_col = "#10b981" if is_dark else "#059669"
            self.ax_contours.plot(
                bound_closed_yx[:, 0], bound_closed_yx[:, 1],
                color=bound_col, linewidth=1.6, linestyle="-",
                zorder=4, label="Граница съемки"
            )
            legend_lines.append(matplotlib.lines.Line2D([0], [0], color=bound_col, lw=1.6))
            legend_labels.append("Граница съемки")

        if legend_lines:
            leg_bg = "#252930" if is_dark else "#ffffff"
            leg = self.ax_contours.legend(
                legend_lines, legend_labels,
                loc="upper left", framealpha=0.88,
                facecolor=leg_bg, edgecolor="#495057" if is_dark else "#ced4da",
                fontsize=9
            )
            for text in leg.get_texts():
                text.set_color(title_color)

        if cur_xlim is not None and cur_ylim is not None and not np.isnan(cur_xlim[0]):
            self.ax_contours.set_xlim(cur_xlim)
            self.ax_contours.set_ylim(cur_ylim)
            self.ax_contours.set_aspect("equal", adjustable="datalim")
        else:
            self._fit_contours_view()
            self._contours_view_initialized = True

        if hasattr(self, "canvas_contours") and self.canvas_contours is not None:
            self.canvas_contours.draw_idle()

    def _on_contours_canvas_press(self, event):
        """Нажатие кнопки мыши на холсте горизонталей"""
        try:
            if hasattr(self, "_toolbar_contours") and self._toolbar_contours.mode != "":
                return
        except Exception:
            pass

        if event.x is None or event.y is None or event.inaxes != self.ax_contours:
            return

        self._contours_pan_start = (
            event.x,
            event.y,
            event.xdata,
            event.ydata,
            self.ax_contours.get_xlim(),
            self.ax_contours.get_ylim(),
            event.button
        )
        self._contours_pan_dragged = False

    def _on_contours_canvas_motion(self, event):
        """Перемещение мыши — плавное панорамирование плана горизонталей"""
        if getattr(self, "_contours_pan_start", None) is None or event.x is None or event.y is None:
            return

        start_px_x, start_px_y, _, _, orig_xlim, orig_ylim, btn = self._contours_pan_start
        dx_px = event.x - start_px_x
        dy_px = event.y - start_px_y

        drag_threshold = 3 if btn in (2, 3) else 6
        if abs(dx_px) > drag_threshold or abs(dy_px) > drag_threshold:
            self._contours_pan_dragged = True

        if self._contours_pan_dragged:
            bbox = self.ax_contours.bbox
            if bbox.width > 0 and bbox.height > 0:
                dx_data = dx_px * (orig_xlim[1] - orig_xlim[0]) / bbox.width
                dy_data = dy_px * (orig_ylim[1] - orig_ylim[0]) / bbox.height
                self.ax_contours.set_xlim(orig_xlim[0] - dx_data, orig_xlim[1] - dx_data)
                self.ax_contours.set_ylim(orig_ylim[0] - dy_data, orig_ylim[1] - dy_data)
                self._throttled_canvas_draw(self.canvas_contours, delay_ms=25)

    def _on_contours_canvas_release(self, event):
        """Отпускание кнопки мыши на холсте горизонталей"""
        if getattr(self, "_contours_pan_start", None) is None:
            return

        was_dragged = self._contours_pan_dragged
        self._contours_pan_start = None
        self._contours_pan_dragged = False

        if was_dragged:
            self._flush_canvas_draw(self.canvas_contours)

    def _on_contours_canvas_scroll(self, event):
        """Зумирование колесом мыши вокруг курсора на плане горизонталей"""
        if event.xdata is None or event.ydata is None or event.inaxes != self.ax_contours:
            return
        base_scale = 1.25
        scale_factor = 1.0 / base_scale if event.button == "up" else base_scale

        cur_xlim = self.ax_contours.get_xlim()
        cur_ylim = self.ax_contours.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax_contours.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax_contours.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self.ax_contours.set_aspect("equal", adjustable="datalim")
        self._throttled_canvas_draw(self.canvas_contours, delay_ms=25)



    # ==================== УПРАВЛЕНИЕ ПРОЕКТАМИ В ПАПКЕ PROJECTS ====================

    def _migrate_legacy_projects(self):
        """Автоматически перемещает файлы проектов и координат в папку Projects/ДД.ММ.ГГГГ_Имя/"""
        cur_dir = get_app_dir()
        proj_root = get_projects_dir()
        os.makedirs(proj_root, exist_ok=True)

        ignored_files = {"requirements.txt", "readme.md", "run_volume_calc.bat", "volumecalculator.spec"}

        # Быстрая проверка: если в корне нет файлов для миграции, не выполняем тяжелое сканирование
        has_root_files = bool(glob.glob(os.path.join(cur_dir, "*.volproj*")))
        if not has_root_files:
            for ext in ["*.txt", "*.csv", "*.dat", "*.xyz", "*.pts"]:
                for tf in glob.glob(os.path.join(cur_dir, ext)):
                    fn = os.path.basename(tf).lower()
                    if fn not in ignored_files and not fn.endswith("_volume_report.txt"):
                        has_root_files = True
                        break
                if has_root_files:
                    break
        if not has_root_files:
            return

        # 1. Нормализуем имена существующих папок в Projects → формат «ДД.ММ.ГГГГ_Имя»
        try:
            for item in list(os.listdir(proj_root)):
                item_path = os.path.join(proj_root, item)
                if not os.path.isdir(item_path):
                    continue

                is_prefix_dated = bool(re.match(r'^\d{2}[._]\d{2}[._]\d{4}_', item))
                if is_prefix_dated:
                    continue  # уже в правильном формате

                # Получаем дату из суффикса или из времени папки
                suffix_match = re.search(r'_(\d{2}[._]\d{2}[._]\d{4})(_\d+)?$', item)
                if suffix_match:
                    folder_date = suffix_match.group(1).replace('_', '.').replace('.', '.')
                    # Нормализуем дату к формату ДД.ММ.ГГГГ
                    folder_date = re.sub(r'[_]', '.', folder_date)
                    clean_base = self._strip_date_from_folder_name(item)
                else:
                    # Папка без даты — берём дату модификации
                    try:
                        mtime = os.path.getmtime(item_path)
                        folder_date = datetime.fromtimestamp(mtime).strftime("%d.%m.%Y")
                    except Exception:
                        folder_date = datetime.now().strftime("%d.%m.%Y")
                    clean_base = re.sub(r'[<>:"/\\|?*]', '_', item).strip('. ')

                new_name = f"{folder_date}_{clean_base}"
                new_path = os.path.join(proj_root, new_name)
                counter = 1
                while os.path.exists(new_path) and new_path != item_path:
                    new_name = f"{folder_date}_{clean_base}_{counter}"
                    new_path = os.path.join(proj_root, new_name)
                    counter += 1

                if new_path != item_path:
                    try:
                        os.rename(item_path, new_path)
                    except Exception:
                        pass
        except Exception:
            pass

        # 2. Миграция файлов проектов из корня
        for vf in glob.glob(os.path.join(cur_dir, "*.volproj*")):
            base_name = os.path.basename(vf).split(".")[0]
            proj_name, target_dir = self._create_project_folder(base_name)
            dest = os.path.join(target_dir, os.path.basename(vf))
            try:
                if os.path.exists(vf) and not os.path.exists(dest):
                    shutil.move(vf, dest)
            except Exception:
                pass

        # 3. Миграция текстовых файлов координат из корня
        for ext in ["*.txt", "*.csv", "*.dat", "*.xyz", "*.pts"]:
            for tf in glob.glob(os.path.join(cur_dir, ext)):
                fn = os.path.basename(tf)
                if fn.lower() in ignored_files:
                    continue
                if fn.endswith("_volume_report.txt"):
                    base_name = fn.replace("_volume_report.txt", "")
                else:
                    base_name = os.path.splitext(fn)[0]
                proj_name, target_dir = self._create_project_folder(base_name)
                dest = os.path.join(target_dir, fn)
                try:
                    if os.path.exists(tf) and not os.path.exists(dest):
                        shutil.move(tf, dest)
                except Exception:
                    pass


    def _scan_saved_projects(self):
        """Сканирует папку Projects и отображает проекты по названиям папок"""
        proj_root = get_projects_dir()
        self._project_folders = {}  # folder_name -> folder_path

        folder_names = []
        if os.path.exists(proj_root):
            for item in sorted(os.listdir(proj_root)):
                full_p = os.path.join(proj_root, item)
                if os.path.isdir(full_p):
                    self._project_folders[item] = full_p
                    folder_names.append(item)

        self.cbo_projects.configure(values=folder_names)

        # Если есть проекты и ничего не выбрано — выбираем проект и автоматически загружаем его
        if folder_names:
            cur_sel = self.cbo_projects.get()
            if not cur_sel or cur_sel not in folder_names:
                last_proj = self._get_last_active_project_name()
                if last_proj and last_proj in folder_names:
                    target_proj = last_proj
                else:
                    try:
                        target_proj = max(folder_names, key=lambda fn: os.path.getmtime(self._project_folders[fn]))
                    except Exception:
                        target_proj = folder_names[0]

                self.cbo_projects.set(target_proj)
                if target_proj in self._project_folders:
                    self._update_files_combobox_for_project(self._project_folders[target_proj])
                self.after(50, lambda p=target_proj: self._on_cbo_project_selected(p))
            else:
                self._update_files_combobox_for_project(self._project_folders[cur_sel])
        else:
            self.cbo_projects.set("")
            if getattr(self, "cbo_files", None) is not None:
                self.cbo_files.configure(values=[])
                self.cbo_files.set("")

    def _update_files_combobox_for_project(self, folder_path: str):
        """Обновляет список файлов координат для выбранной папки проекта"""
        if getattr(self, "cbo_files", None) is None:
            return
        coord_files = []
        for ext in ["*.txt", "*.csv", "*.dat", "*.xyz", "*.pts", "*.TXT", "*.CSV", "*.DAT"]:
            for f in glob.glob(os.path.join(folder_path, ext)):
                fn = os.path.basename(f)
                if not fn.endswith("_volume_report.txt") and not fn.endswith("_report.txt"):
                    coord_files.append(fn)

        coord_files = sorted(list(set(coord_files)))
        self.cbo_files.configure(values=coord_files)
        if coord_files:
            self.cbo_files.set(coord_files[0])
        else:
            self.cbo_files.set("")

    def _on_cbo_project_selected(self, value):
        """При выборе папки проекта в списке автоматически загружает его"""
        proj_name = value if value else self.cbo_projects.get()
        if proj_name and proj_name in self._project_folders:
            folder_path = self._project_folders[proj_name]
            self._update_files_combobox_for_project(folder_path)
            self.load_project_from_folder(folder_path)

    def _open_project_folder(self):
        """Открывает папку текущего проекта в Проводнике Windows"""
        proj_name = self.cbo_projects.get() if hasattr(self.cbo_projects, "get") else ""
        folder_path = None
        if proj_name and proj_name in getattr(self, "_project_folders", {}):
            folder_path = self._project_folders[proj_name]
        elif getattr(self, "_current_project_dir", None) and os.path.isdir(self._current_project_dir):
            folder_path = self._current_project_dir
        else:
            from app_utils import get_projects_dir
            folder_path = get_projects_dir()

        if folder_path and os.path.exists(folder_path):
            try:
                os.startfile(folder_path)
            except Exception:
                import subprocess
                subprocess.Popen(["explorer", os.path.normpath(folder_path)])
        else:
            messagebox.showinfo("Информация", "Папка проекта не найдена.")

    _load_selected_project_button = _open_project_folder

    def _on_cbo_file_selected(self, event):
        """При выборе конкретного файла координат внутри проекта (ttk legacy)"""
        fn = self.cbo_files.get()
        if fn and self._current_project_dir:
            fpath = os.path.join(self._current_project_dir, fn)
            if os.path.exists(fpath):
                self.load_file(fpath)

    def _on_cbo_file_selected_cmd(self, value):
        """При выборе файла координат (CTkComboBox command callback)"""
        fn = value if value else self.cbo_files.get()
        if fn and self._current_project_dir:
            fpath = os.path.join(self._current_project_dir, fn)
            if os.path.exists(fpath):
                self.load_file(fpath)

    def _on_cbo_mode_selected(self, event):
        """Обрабатывает выбор режима из выпадающего списка (ttk legacy)"""
        self._on_cbo_mode_selected_cmd(self.cbo_mode.get())

    def _on_cbo_mode_selected_cmd(self, value):
        """Обрабатывает выбор режима из выпадающего списка (CTkComboBox command callback)"""
        mode_text_map = {
            "1. Интерактивный контур (ЛКМ)": "select_boundary",
            "2. Авто-контур (Выпуклая оболочка)": "auto_hull",
            "3. Назначение: Верхняя поверхность": "assign_top",
            "4. Назначение: Нижняя поверхность": "assign_bottom",
            "5. Добавить точку (ЛКМ на схеме)": "add_point",
            "6. Рамка выделения (Shift+ЛКМ)": "box_select",
        }
        mode = mode_text_map.get(value, "select_boundary")
        self.current_mode.set(mode)
        self._on_mode_change()

    @staticmethod
    def _strip_date_from_folder_name(folder_name: str) -> str:
        """Возвращает базовое имя папки без даты-префикса ДД.ММ.ГГГГ_ или суффикса _ДД.ММ.ГГГГ"""
        return ProjectStorageService.strip_date_from_folder_name(folder_name)

    def _create_project_folder(self, base_name: str) -> tuple[str, str]:
        """
        Создает подпапку проекта в папке Projects.
        Формат: ДД.ММ.ГГГГ_ИмяПроекта (дата в начале имени).
        Возвращает (имя_папки, полный_путь_к_папке).
        """
        return ProjectStorageService.create_project_folder(get_projects_dir(), base_name)


    def _on_file_drop(self, event):
        """Обработчик перетаскивания файла координат на окно программы (Drag & Drop).
        Поведение идентично импорту через диалог: создаётся папка проекта, файл копируется."""
        raw = event.data.strip()
        # tkinterdnd2 возвращает пути в фигурных скобках если есть пробелы, или через пробелы
        # Парсим первый файл из списка
        if raw.startswith("{"):
            # Формат: {path1} {path2} ...
            end = raw.find("}")
            f = raw[1:end].strip() if end != -1 else raw.strip("{}")
        else:
            # Несколько файлов через пробел — берём первый
            f = raw.split()[0]

        if not f or not os.path.isfile(f):
            messagebox.showwarning("Drag & Drop", f"Не удалось прочитать файл:\n{f}")
            return

        # Стандартная схема импорта — та же что в _open_file_dialog
        base_name = os.path.splitext(os.path.basename(f))[0]
        proj_name, target_dir = self._create_project_folder(base_name)

        dest_file = os.path.join(target_dir, os.path.basename(f))
        if os.path.abspath(f) != os.path.abspath(dest_file):
            try:
                shutil.copy2(f, dest_file)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось скопировать файл в проект:\n{e}")
                return

        self._scan_saved_projects()
        self.cbo_projects.set(proj_name)
        self._current_project_dir = target_dir
        self._update_files_combobox_for_project(target_dir)
        self.load_file(dest_file)

    def _open_file_dialog(self):
        """Импорт внешнего файла координат или чертежа DXF и создание новой папки проекта в Projects"""
        f = filedialog.askopenfilename(
            initialdir=get_projects_dir(),
            title="Выберите файл с координатами или чертеж DXF для импорта",
            filetypes=[
                ("Все поддерживаемые форматы", "*.txt *.csv *.dat *.xyz *.pts *.dxf"),
                ("Чертежи AutoCAD DXF", "*.dxf"),
                ("Текстовые файлы координат", "*.txt *.csv *.dat *.xyz *.pts"),
                ("Все файлы", "*.*")
            ]
        )
        if not f:
            return

        base_name = os.path.splitext(os.path.basename(f))[0]
        proj_name, target_dir = self._create_project_folder(base_name)

        dest_file = os.path.join(target_dir, os.path.basename(f))
        if os.path.abspath(f) != os.path.abspath(dest_file):
            try:
                shutil.copy2(f, dest_file)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось скопировать файл в проект:\n{e}")
                return

        self._scan_saved_projects()
        self.cbo_projects.set(proj_name)
        self._current_project_dir = target_dir
        self._update_files_combobox_for_project(target_dir)
        self.load_file(dest_file)

    def _open_dxf_import_dialog(self):
        """Диалог выбора и импорта чертежа AutoCAD DXF с автоматическим определением слоев"""
        f = filedialog.askopenfilename(
            initialdir=get_projects_dir(),
            title="Выберите чертеж AutoCAD DXF для импорта",
            filetypes=[("Чертежи AutoCAD DXF", "*.dxf"), ("Все файлы", "*.*")]
        )
        if not f:
            return

        base_name = os.path.splitext(os.path.basename(f))[0]
        proj_name, target_dir = self._create_project_folder(base_name)

        dest_file = os.path.join(target_dir, os.path.basename(f))
        if os.path.abspath(f) != os.path.abspath(dest_file):
            try:
                shutil.copy2(f, dest_file)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось скопировать DXF в проект:\n{e}")
                return

        self._scan_saved_projects()
        self.cbo_projects.set(proj_name)
        self._current_project_dir = target_dir
        self._update_files_combobox_for_project(target_dir)
        self.load_file(dest_file)

    def _open_separate_files_dialog(self):
        f_top = filedialog.askopenfilename(
            initialdir=get_projects_dir(),
            title="1/2 Выберите файл ВЕРХНЕЙ поверхности",
            filetypes=[("Файлы съёмки (TXT, CSV, DXF)", "*.txt *.csv *.dat *.xyz *.pts *.dxf"), ("Все файлы", "*.*")]
        )
        if not f_top:
            return
        f_bot = filedialog.askopenfilename(
            initialdir=get_projects_dir(),
            title="2/2 Выберите файл НИЖНЕЙ поверхности",
            filetypes=[("Файлы съёмки (TXT, CSV, DXF)", "*.txt *.csv *.dat *.xyz *.pts *.dxf"), ("Все файлы", "*.*")]
        )
        if not f_bot:
            return

        if f_top.lower().endswith(".dxf"):
            from dxf_importer import extract_dxf_geometry
            ok, pts_top, _, _, _ = extract_dxf_geometry(f_top)
            if not ok or not pts_top:
                messagebox.showerror("Ошибка импорта DXF", f"Не удалось прочитать точки из DXF верха: {f_top}")
                return
        else:
            pts_top = load_points_from_file(f_top)
            if not pts_top and self._file_has_data_lines(f_top):
                try:
                    self.attributes("-alpha", 1.0)
                except Exception:
                    pass
                dlg_top = CoordinateRemapDialog(self, f_top, initial_mapping=None)
                try:
                    self.attributes("-alpha", 1.0)
                except Exception:
                    pass
                if dlg_top.result_mapping is not None:
                    pts_top = load_points_from_file(f_top, column_mapping=dlg_top.result_mapping)

        if f_bot.lower().endswith(".dxf"):
            from dxf_importer import extract_dxf_geometry
            ok, pts_bot, _, _, _ = extract_dxf_geometry(f_bot)
            if not ok or not pts_bot:
                messagebox.showerror("Ошибка импорта DXF", f"Не удалось прочитать точки из DXF низа: {f_bot}")
                return
        else:
            pts_bot = load_points_from_file(f_bot)
            if not pts_bot and self._file_has_data_lines(f_bot):
                try:
                    self.attributes("-alpha", 1.0)
                except Exception:
                    pass
                dlg_bot = CoordinateRemapDialog(self, f_bot, initial_mapping=None)
                try:
                    self.attributes("-alpha", 1.0)
                except Exception:
                    pass
                if dlg_bot.result_mapping is not None:
                    pts_bot = load_points_from_file(f_bot, column_mapping=dlg_bot.result_mapping)

        if not pts_top or not pts_bot:
            messagebox.showerror(
                "Ошибка импорта",
                f"Не удалось распознать точки в одном из файлов:\n"
                f"• Верхняя поверхность: {len(pts_top) if pts_top else 0} точек\n"
                f"• Нижняя поверхность: {len(pts_bot) if pts_bot else 0} точек\n\n"
                f"Убедитесь, что оба файла содержат корректные строки координат или чертеж DXF."
            )
            return

        self.calc_results = None
        self._is_separate_surfaces = True
        for p in pts_top:
            p.surface_type = "top"
        for p in pts_bot:
            p.surface_type = "bottom"

        base_name = f"{os.path.splitext(os.path.basename(f_top))[0]}_комплекс"
        proj_name, target_dir = self._create_project_folder(base_name)

        dest_top = os.path.join(target_dir, os.path.basename(f_top))
        dest_bot = os.path.join(target_dir, os.path.basename(f_bot))
        try:
            if os.path.abspath(f_top) != os.path.abspath(dest_top):
                shutil.copy2(f_top, dest_top)
            if os.path.abspath(f_bot) != os.path.abspath(dest_bot):
                shutil.copy2(f_bot, dest_bot)
        except Exception:
            pass

        self.points = pts_top + pts_bot
        self._current_project_dir = target_dir
        self._current_file_path = dest_top
        self.boundary_indices = []
        self._undo_stack.clear()
        self._auto_classify_initial()

        self.lbl_file_info.configure(text=f"{proj_name} (Верх: {len(pts_top)}, Низ: {len(pts_bot)})")
        self._update_all_views(reset_view=True)
        self.save_project_state(self._current_file_path)
        self._scan_saved_projects()
        self.cbo_projects.set(proj_name)
        if len(self.boundary_indices) >= 3:
            self._schedule_boundary_calc(300)
        else:
            try:
                self._select_tab(self.TAB_2D)
            except Exception:
                pass


    # ==================== СОХРАНЕНИЕ / ЗАГРУЗКА ПРОЕКТА ====================

    def _schedule_auto_save(self, delay_ms: int = 1500):
        """Отложенное автосохранение проекта с дебаунсингом для исключения лагов UI при кликах"""
        if getattr(self, "_auto_save_timer", None) is not None:
            try:
                self.after_cancel(self._auto_save_timer)
            except Exception:
                pass
            self._auto_save_timer = None
        self._auto_save_timer = self.after(delay_ms, self._execute_delayed_auto_save)

    def _execute_delayed_auto_save(self):
        self._auto_save_timer = None
        if getattr(self, "_current_file_path", None):
            self.save_project_state(self._current_file_path)

    def _schedule_boundary_calc(self, delay_ms: int = 400):
        """Отложенный авто-расчет объема при редактировании контура (дебаунсинг серии кликов)."""
        if getattr(self, "_boundary_calc_timer", None) is not None:
            try:
                self.after_cancel(self._boundary_calc_timer)
            except Exception:
                pass
            self._boundary_calc_timer = None
        self._boundary_calc_timer = self.after(delay_ms, self._execute_delayed_boundary_calc)

    def _execute_delayed_boundary_calc(self):
        self._boundary_calc_timer = None
        if len(getattr(self, "boundary_indices", [])) >= 3:
            self._calc_boundary_silent()
            current_tab = self.tabview.get() if hasattr(self, "tabview") else None
            if current_tab == self.TAB_2D:
                self._redraw_2d(reset_view=False)

    def _get_project_file_path(self, filepath: str) -> str:
        if self._current_project_dir:
            folder_name = os.path.basename(self._current_project_dir)
            # Имя .volproj файла — без даты, чтобы файлы не засорялись датами
            file_base = self._strip_date_from_folder_name(folder_name) or folder_name
            return os.path.join(self._current_project_dir, f"{file_base}.volproj")
        base, _ = os.path.splitext(filepath)
        return base + ".volproj"

    def save_project_state(self, filepath: str = None) -> bool:
        if not self.points:
            return False

        if not filepath and self._current_file_path:
            filepath = self._current_file_path
        if not filepath:
            return False

        proj_path = self._get_project_file_path(filepath)
        temp_path = proj_path + ".tmp"
        bak_path = proj_path + ".bak"

        data = {
            "source_file": os.path.basename(filepath),
            "column_mapping": self._column_mapping,
            "points": [
                {
                    "id": str(p.id),
                    "x": float(p.x),
                    "y": float(p.y),
                    "h": float(p.h),
                    "surface_type": str(p.surface_type)
                }
                for p in self.points
            ],
            "boundary_indices": [int(idx) for idx in self.boundary_indices],
            "tin_excluded": sorted(int(i) for i in self._tin_excluded),
            "tin_simplices": [list(map(int, s)) for s in self._tin_simplices] if self._tin_simplices is not None else [],
            "has_custom_tin": (self._tin_custom_simplices is not None),
            "is_separate_surfaces": self._is_two_surfaces(),
        }


        return ProjectStorageService.save_project_file(proj_path, data)

    def _reset_all_caches_for_new_project(self):
        """Полный сброс всех кешей, триангуляций и состояний при переключении или загрузке нового проекта"""
        self.calc_results = None
        self._selected_points.clear()
        self._undo_stack.clear()
        self._invalidate_boundary_cache()
        self._boundary_interpolator_cache = None
        self._boundary_path_cache = None
        self._boundary_inside_cache = None
        self._split_height_cache = None
        self._work_type_cache = None
        self._point_surfaces_cache = None
        self._local_surface_threshold = None
        self._spatial_service = None
        self._points_kdtree = None
        self._points_coords_len = 0
        self._tin_excluded = set()
        self._tin_custom_simplices = None
        self._tin_simplices = None
        self._tin_pts_2d = None
        self._tin_selected_idx = None
        self._tin_patches = []
        self._tin_view_initialized = False
        self._tin_dirty = True
        self._table_dirty = True
        self._3d_dirty = True
        self._diff_dirty = True
        self._2d_dirty = True
        self._contours_dirty = True
        self._raw_contours_cache = None
        self._contours_view_initialized = False

    def _get_app_config_path(self):
        app_data = os.environ.get("LOCALAPPDATA") or get_app_dir()
        cfg_dir = os.path.join(app_data, "GeoVolumePro")
        try:
            os.makedirs(cfg_dir, exist_ok=True)
        except Exception:
            cfg_dir = get_app_dir()
        return os.path.join(cfg_dir, "app_config.json")

    def _get_last_active_project_name(self) -> str:
        try:
            cfg_p = self._get_app_config_path()
            if os.path.exists(cfg_p):
                with open(cfg_p, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    return cfg.get("last_project", "")
        except Exception:
            pass
        return ""

    def _save_last_active_project_name(self, proj_name: str):
        if not proj_name or "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
            return
        try:
            cfg_p = self._get_app_config_path()
            cfg = {}
            if os.path.exists(cfg_p):
                try:
                    with open(cfg_p, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                except Exception:
                    cfg = {}
            cfg["last_project"] = proj_name
            with open(cfg_p, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def load_project_from_folder(self, folder_path: str):
        """Загружает проект из указанной папки Projects/<Name>/"""
        self._current_project_dir = folder_path
        proj_name = os.path.basename(folder_path)

        # 1. Ищем .volproj файл
        volproj_files = glob.glob(os.path.join(folder_path, "*.volproj"))
        if volproj_files:
            self.load_project_from_file(volproj_files[0])
            return

        # 2. Ищем файл координат
        coord_files = ProjectStorageService.list_coordinate_files(folder_path)
        if coord_files:
            self.load_file(os.path.join(folder_path, coord_files[0]))
            return

        messagebox.showinfo("Информация", f"В папке проекта '{proj_name}' не найдены файлы данных.")

    def load_project_from_file(self, proj_path: str):
        """Загружает проект по пути к .volproj файлу"""
        data = ProjectStorageService.load_project_file(proj_path)

        if data is None:
            answer = messagebox.askyesno(
                "Повреждённый файл проекта",
                f"Файл проекта повреждён:\n{os.path.basename(proj_path)}\n\n"
                f"Удалить повреждённый файл и загрузить точки заново из исходного файла координат?"
            )
            if answer:
                try:
                    if os.path.exists(proj_path):
                        os.remove(proj_path)
                    bak = proj_path + ".bak"
                    if os.path.exists(bak):
                        os.remove(bak)
                    self._scan_saved_projects()
                except Exception as del_e:
                    messagebox.showerror("Ошибка", f"Не удалось удалить файл:\n{del_e}")
            return

        try:
            source_file = data.get("source_file", "")
            self._column_mapping = data.get("column_mapping", None)
            folder_path = os.path.dirname(proj_path)
            self._current_project_dir = folder_path

            source_path = os.path.join(folder_path, source_file) if source_file else proj_path
            if not os.path.exists(source_path):
                for ext in [".txt", ".csv", ".dat", ".TXT", ".CSV", ".DAT"]:
                    cands = glob.glob(os.path.join(folder_path, f"*{ext}"))
                    if cands:
                        source_path = cands[0]
                        break

            pts = []
            for p in data.get("points", []):
                pts.append(
                    GeoPoint(
                        id=str(p.get("id", "")),
                        x=float(p["x"]),
                        y=float(p["y"]),
                        h=float(p["h"]),
                        surface_type=str(p.get("surface_type", "auto"))
                    )
                )

            self._reset_all_caches_for_new_project()
            self.points = pts
            self._is_separate_surfaces = bool(data.get("is_separate_surfaces", False))
            raw_bounds = data.get("boundary_indices", [])
            self.boundary_indices = [int(i) for i in raw_bounds if 0 <= int(i) < len(self.points)]
            self._tin_excluded = set(int(i) for i in data.get("tin_excluded", []))

            n_pts = len(self.points)
            raw_simplices = data.get("tin_simplices", [])
            has_custom = data.get("has_custom_tin", False)
            valid_simplices = []
            if raw_simplices:
                for s in raw_simplices:
                    if len(s) == 3 and all(0 <= int(v) < n_pts for v in s):
                        valid_simplices.append([int(v) for v in s])

            if has_custom and valid_simplices and len(valid_simplices) > 0:
                self._tin_custom_simplices = [tuple(s) for s in valid_simplices]
                self._tin_simplices = np.array(valid_simplices, dtype=int)
                self._tin_dirty = False
            else:
                self._tin_custom_simplices = None
                self._tin_simplices = None
                self._tin_dirty = True

            self._current_file_path = source_path
            self._save_last_active_project_name(os.path.basename(folder_path))

            proj_display = os.path.basename(folder_path)
            self.lbl_file_info.configure(
                text=f"{proj_display} ({len(self.points)} точек) [сохранён]"
            )
            self._update_all_views(reset_view=True)

            # Авто-расчёт и переход на схему в плане при загрузке/смене проекта
            if len(self.boundary_indices) >= 3:
                self.after(200, self._auto_calc_and_show_plan)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить данные проекта:\n{str(e)}")

    def _delete_selected_project(self):
        """Удаляет выбранную папку проекта из Projects"""
        display_name = self.cbo_projects.get()
        if not display_name or display_name not in self._project_folders:
            messagebox.showinfo("Информация", "Выберите проект для удаления.")
            return

        folder_path = self._project_folders[display_name]

        if messagebox.askyesno("Подтверждение", f"Удалить папку проекта '{display_name}' и все файлы внутри?"):
            try:
                shutil.rmtree(folder_path, ignore_errors=True)
                self.points = []
                self.boundary_indices = []
                self.calc_results = None
                self._current_file_path = None
                self._current_project_dir = None
                self._column_mapping = None
                self._scan_saved_projects()
                self._update_all_views(reset_view=True)
                self.lbl_file_info.configure(text="Проект удален")
                messagebox.showinfo("Успешно", f"Проект '{display_name}' удален.")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось удалить проект:\n{str(e)}")

    def _on_closing(self):
        try:
            if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
                try:
                    self.destroy()
                except Exception:
                    pass
                return
        except Exception:
            pass

        try:
            if getattr(self, "_auto_save_timer", None) is not None:
                try:
                    self.after_cancel(self._auto_save_timer)
                except Exception:
                    pass
                self._auto_save_timer = None
        except Exception:
            pass

        try:
            if getattr(self, "points", None) and getattr(self, "_current_file_path", None):
                self.save_project_state(self._current_file_path)
        except Exception:
            pass

        try:
            self.quit()
        except Exception:
            pass

        try:
            self.destroy()
        except Exception:
            pass

        try:
            os._exit(0)
        except Exception:
            sys.exit(0)

    # ==================== ЗАГРУЗКА И ПЕРЕОПРЕДЕЛЕНИЕ ФАЙЛОВ ====================

    @staticmethod
    def _file_has_data_lines(filepath: str) -> bool:
        """Проверяет, есть ли в файле хотя бы одна непустая строка с данными (не комментарий)."""
        if not filepath or not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith('#') and not s.startswith('//'):
                        return True
        except Exception:
            pass
        return False

    def load_file(self, filepath: str, column_mapping: dict = None):
        self._current_file_path = filepath
        self._current_project_dir = os.path.dirname(filepath)

        if filepath.lower().endswith(".dxf"):
            self._import_dxf_file(filepath)
            return

        if column_mapping is not None:
            self._column_mapping = column_mapping

        # Если маппинг не был передан принудительно, проверяем .volproj
        if column_mapping is None:
            proj_path = self._get_project_file_path(filepath)
            if os.path.exists(proj_path):
                self.load_project_from_file(proj_path)
                return
            self._column_mapping = None

        pts = load_points_from_file(filepath, column_mapping=self._column_mapping)
        if not pts:
            if not self._file_has_data_lines(filepath):
                messagebox.showwarning(
                    "Предупреждение",
                    f"Файл {os.path.basename(filepath)} пуст или не содержит данных."
                )
                return

            if column_mapping is None:
                # Координаты не соответствуют правилу 6/7 — сразу открываем окно переопределения
                try:
                    self.attributes("-alpha", 1.0)
                except Exception:
                    pass
                dlg = CoordinateRemapDialog(self, filepath, initial_mapping=None)
                try:
                    self.attributes("-alpha", 1.0)
                except Exception:
                    pass

                if dlg.result_mapping is not None:
                    self._column_mapping = dlg.result_mapping
                    self.load_file(filepath, column_mapping=self._column_mapping)
                return
            else:
                messagebox.showwarning(
                    "Предупреждение",
                    f"В файле {os.path.basename(filepath)} не удалось распознать координаты с выбранными колонками."
                )
                return
        self._reset_all_caches_for_new_project()
        self.points = pts
        self._is_separate_surfaces = False
        proj_display = os.path.basename(self._current_project_dir) if self._current_project_dir else os.path.basename(filepath)
        self.lbl_file_info.configure(text=f"{proj_display} ({len(pts)} точек)")
        self.boundary_indices = []
        self._auto_classify_initial()
        self._update_all_views(reset_view=True)
        self.save_project_state(filepath)

        # Автоматический расчёт объёма и переход на схему в плане после загрузки
        if len(self.boundary_indices) >= 3:
            self.after(200, self._auto_calc_and_show_plan)
        else:
            try:
                self._select_tab(self.TAB_2D)
            except Exception:
                pass

    def _import_dxf_file(self, filepath: str):
        """Импортирует съёмку и контур из файла чертежа AutoCAD DXF"""
        self._current_file_path = filepath
        self._current_project_dir = os.path.dirname(filepath)

        from ui_dialogs import DxfImportDialog
        dlg = DxfImportDialog(self, filepath)
        self.wait_window(dlg)

        if not getattr(dlg, "result_points", None):
            return

        self._reset_all_caches_for_new_project()
        pts = list(dlg.result_points)
        self.boundary_indices = []

        if dlg.result_boundary is not None and len(dlg.result_boundary) >= 3:
            b_indices = []
            mean_h = float(np.mean([p.h for p in pts])) if pts else 0.0
            for k, (bx, by) in enumerate(dlg.result_boundary):
                found_idx = -1
                for i, p in enumerate(pts):
                    if math.hypot(p.x - bx, p.y - by) < 0.05:
                        found_idx = i
                        break
                if found_idx != -1:
                    b_indices.append(found_idx)
                else:
                    new_pt = GeoPoint(id=f"BND{k+1}", x=float(bx), y=float(by), h=mean_h, surface_type="boundary")
                    pts.append(new_pt)
                    b_indices.append(len(pts) - 1)
            self.boundary_indices = b_indices

        self.points = pts
        self._is_separate_surfaces = dlg.is_two_surfaces

        proj_display = os.path.basename(self._current_project_dir) if self._current_project_dir else os.path.basename(filepath)
        n_top = sum(1 for p in pts if p.surface_type == "top")
        n_bot = sum(1 for p in pts if p.surface_type == "bottom")
        if self._is_separate_surfaces and n_top > 0 and n_bot > 0:
            self.lbl_file_info.configure(text=f"{proj_display} (DXF Верх: {n_top}, Низ: {n_bot})")
        else:
            self.lbl_file_info.configure(text=f"{proj_display} ({len(pts)} точек DXF)")

        if not self.boundary_indices:
            self._auto_classify_initial()

        self._update_all_views(reset_view=True)
        self.save_project_state(filepath)

        if len(self.boundary_indices) >= 3:
            self.after(200, self._auto_calc_and_show_plan)
        else:
            try:
                self._select_tab(self.TAB_2D)
            except Exception:
                pass

    def _open_remap_dialog(self):
        """Открывает диалог ручного переопределения колонок координат"""
        if not self._current_file_path or not os.path.exists(self._current_file_path):
            messagebox.showinfo("Информация", "Сначала откройте или выберите файл координат.")
            return

        try:
            self.attributes("-alpha", 1.0)
        except Exception:
            pass

        dlg = CoordinateRemapDialog(self, self._current_file_path, initial_mapping=self._column_mapping)

        try:
            self.attributes("-alpha", 1.0)
        except Exception:
            pass

        if dlg.result_mapping is not None:
            self._column_mapping = dlg.result_mapping
            self.load_file(self._current_file_path, column_mapping=self._column_mapping)
            messagebox.showinfo("Успешно", "Координаты успешно переопределены и пересчитаны!")

    # ==================== ЛОГИКА КОНТУРА И ПОВЕРХНОСТЕЙ ====================

    def _auto_classify_initial(self, force_hull: bool = False):
        if not self.points:
            return
        xy = np.array([[p.x, p.y] for p in self.points])
        if len(xy) >= 3 and (len(xy) <= 7 or force_hull):
            try:
                hull = ConvexHull(xy)
                self.boundary_indices = [int(v) for v in hull.vertices]
            except Exception:
                self.boundary_indices = list(range(len(self.points)))
        else:
            self.boundary_indices = []

        if getattr(self, "_is_separate_surfaces", False):
            # Внимание: если режим двух поверхностей уже выставлен (например, при раздельном импорте двух файлов),
            # сохраняем исходное разделение точек на top и bottom!
            pass
        elif self._detect_two_surfaces_heuristic():
            self._is_separate_surfaces = True
            h_vals = [p.h for p in self.points]
            split_candidate = (min(h_vals) + max(h_vals)) / 2.0
            for p in self.points:
                p.surface_type = "top" if p.h > split_candidate else "bottom"
        else:
            self._is_separate_surfaces = False

        if getattr(self, "_is_separate_surfaces", False) and (len(self.points) <= 7 or force_hull):
            self._adjust_work_zone_boundary_for_two_surfaces(xy)

        self._invalidate_boundary_cache()

    def _adjust_work_zone_boundary_for_two_surfaces(self, xy: np.ndarray):
        """
        Если в режиме двух поверхностей одна поверхность существенно меньше другой по площади в плане
        (например, дно котлована/выемки на большой окружающей площадке, или локальная насыпь на обширном основании),
        автоматически выставляет расчетную границу по контуру МЕНЬШЕЙ поверхности (зоне фактических работ),
        чтобы избежать фиктивной экстраполяции высот за пределы рабочей зоны.
        """
        try:
            idx_top = [i for i, p in enumerate(self.points) if p.surface_type == "top"]
            idx_bot = [i for i, p in enumerate(self.points) if p.surface_type == "bottom"]
            if len(idx_top) < 3 or len(idx_bot) < 3:
                return

            xy_top = xy[idx_top]
            xy_bot = xy[idx_bot]

            hull_top = ConvexHull(xy_top)
            hull_bot = ConvexHull(xy_bot)

            poly_top = xy_top[hull_top.vertices]
            poly_bot = xy_bot[hull_bot.vertices]

            from volume_engine import polygon_area_2d
            area_top = polygon_area_2d(poly_top)
            area_bot = polygon_area_2d(poly_bot)

            if area_top <= 1e-4 or area_bot <= 1e-4:
                return

            # 1. Дно котлована/выемки (bot) существенно меньше дневной поверхности (top)
            if area_bot < area_top * 0.85:
                path_top = MplPath(poly_top)
                if (np.mean(path_top.contains_points(xy_bot, radius=0.1)) >= 0.75 and
                        np.any(path_top.contains_points(xy_bot, radius=-0.35))):
                    self.boundary_indices = [int(idx_bot[v]) for v in hull_bot.vertices]
                    return

            # 2. Насыпь (top) существенно меньше окружающего основания (bot)
            if area_top < area_bot * 0.85:
                path_bot = MplPath(poly_bot)
                if (np.mean(path_bot.contains_points(xy_top, radius=0.1)) >= 0.75 and
                        np.any(path_bot.contains_points(xy_top, radius=-0.35))):
                    self.boundary_indices = [int(idx_top[v]) for v in hull_top.vertices]
                    return
        except Exception:
            pass

    def _on_mode_change(self):
        mode = self.current_mode.get() if hasattr(self.current_mode, "get") else self.current_mode
        if mode == "auto_hull":
            self._auto_classify_initial(force_hull=True)
        self._update_all_views()

    def _reset_boundary(self):
        self.boundary_indices = []
        self.calc_results = None
        self._undo_stack.clear()
        self._tin_excluded.clear()
        self._tin_custom_simplices = None
        self._invalidate_boundary_cache()
        self._update_all_views()
        self._schedule_auto_save()

    def _auto_split_by_height(self):
        if not self.points:
            return
        self._invalidate_boundary_cache()
        if self._detect_two_surfaces_heuristic() or getattr(self, "_is_separate_surfaces", False):
            self._is_separate_surfaces = True
            h_vals = [p.h for p in self.points]
            split_candidate = (min(h_vals) + max(h_vals)) / 2.0
            for p in self.points:
                p.surface_type = "top" if p.h > split_candidate else "bottom"
        else:
            self._is_separate_surfaces = False
            work_type = self._detect_work_type_from_points()
            inside_mask = self._ensure_boundary_inside_mask() if len(self.boundary_indices) >= 3 else None
            for i, p in enumerate(self.points):
                if i in self.boundary_indices:
                    p.surface_type = "boundary"
                elif inside_mask is not None and not (bool(inside_mask[i]) if i < len(inside_mask) else False):
                    p.surface_type = "top" if work_type == "cut" else "bottom"
                else:
                    p.surface_type = self._get_point_surface(p, work_type=work_type)
        self._tin_excluded.clear()
        self._tin_custom_simplices = None
        self._update_all_views()
        self._schedule_auto_save()

    # ==================== РЕДАКТИРОВАНИЕ И ДОБАВЛЕНИЕ ТОЧЕК ====================

    def _add_point_dialog(self):
        """Диалог добавления новой точки"""
        next_id = f"т{len(self.points) + 1}"
        dlg = PointEditDialog(self, title="➕ Добавление новой точки", default_id=next_id)
        if dlg.result:
            id_val, x_val, y_val, h_val, surf_val = dlg.result
            if surf_val == "auto":
                mean_bound, split_h, work_type = self._get_split_height()
                dummy_pt = GeoPoint(id=id_val, x=x_val, y=y_val, h=h_val, surface_type="auto")
                surf_val = self._get_point_surface(dummy_pt, mean_bound, split_h, work_type)
            new_pt = GeoPoint(id=id_val, x=x_val, y=y_val, h=h_val, surface_type=surf_val)
            self.points.append(new_pt)
            new_idx = len(self.points) - 1

            if surf_val == "boundary":
                self.boundary_indices.append(new_idx)

            self._invalidate_boundary_cache()
            self._tin_excluded.clear()
            self._tin_custom_simplices = None
            self._update_all_views()
            self._schedule_auto_save()
            if len(self.boundary_indices) >= 3:
                self._suppress_tab_switch = True
                self._schedule_boundary_calc(400)
            messagebox.showinfo("Успешно", f"Точка ID '{id_val}' успешно добавлена!")

    def _edit_selected_point_dialog(self):
        """Диалог редактирования выбранной точки"""
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Информация", "Выберите точку в таблице для редактирования.")
            return

        item = sel[0]
        vals = self.tree.item(item, "values")
        idx = int(vals[0]) - 1

        if not (0 <= idx < len(self.points)):
            return

        pt = self.points[idx]
        dlg = PointEditDialog(self, title=f"✏️ Редактирование точки #{idx+1} (ID: {pt.id})", point=pt)
        if dlg.result:
            id_val, x_val, y_val, h_val, surf_val = dlg.result
            pt.id = id_val
            pt.x = x_val
            pt.y = y_val
            pt.h = h_val
            pt.surface_type = surf_val

            if surf_val == "boundary" and idx not in self.boundary_indices:
                self.boundary_indices.append(idx)
            elif surf_val != "boundary" and idx in self.boundary_indices:
                self.boundary_indices.remove(idx)

            self._invalidate_boundary_cache()
            self._update_all_views()
            self._schedule_auto_save()
            if len(self.boundary_indices) >= 3:
                self._suppress_tab_switch = True
                self._schedule_boundary_calc(400)

    def _delete_selected_point(self):
        """Удаление выбранной точки из таблицы"""
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Информация", "Выберите точку в таблице для удаления.")
            return

        item = sel[0]
        vals = self.tree.item(item, "values")
        idx = int(vals[0]) - 1

        if not (0 <= idx < len(self.points)):
            return

        pt = self.points[idx]
        if messagebox.askyesno("Подтверждение", f"Удалить точку #{idx+1} (ID: {pt.id}, H: {pt.h:.3f})?"):
            del self.points[idx]

            # Корректировка индексов контура
            new_bounds = []
            for b in self.boundary_indices:
                if b == idx:
                    continue
                elif b > idx:
                    new_bounds.append(b - 1)
                else:
                    new_bounds.append(b)
            self.boundary_indices = new_bounds
            self._invalidate_boundary_cache()
            self._tin_excluded.clear()
            self._tin_custom_simplices = None

            self._update_all_views()
            self._schedule_auto_save()
            if len(self.boundary_indices) >= 3:
                self._suppress_tab_switch = True
                self._schedule_boundary_calc(400)

    def _toggle_selected_point_type(self):
        """Быстрое циклическое переключение типа поверхности выбранной точки"""
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Информация", "Выберите точку в таблице для смены типа.")
            return

        item = sel[0]
        idx = int(self.tree.item(item, "values")[0]) - 1
        if not (0 <= idx < len(self.points)):
            return

        pt = self.points[idx]
        types = ["auto", "top", "bottom", "boundary"]
        cur_idx = types.index(pt.surface_type) if pt.surface_type in types else 0
        pt.surface_type = types[(cur_idx + 1) % len(types)]

        if pt.surface_type == "boundary" and idx not in self.boundary_indices:
            self.boundary_indices.append(idx)
        elif pt.surface_type != "boundary" and idx in self.boundary_indices:
            self.boundary_indices.remove(idx)

        self._invalidate_boundary_cache()
        self._update_all_views()
        self._schedule_auto_save()
        if len(self.boundary_indices) >= 3:
            self._suppress_tab_switch = True
            self._schedule_boundary_calc(400)

    # ==================== ИНТЕРАКТИВНОСТЬ 2D ХОЛСТА (ПЕРЕМЕЩЕНИЕ И ВЫБОР) ====================

    def _get_spatial_service(self) -> SpatialIndexService:
        svc = self.__dict__.get("_spatial_index")
        pts = getattr(self, "points", [])
        if svc is None:
            svc = SpatialIndexService(pts)
            self._spatial_index = svc
        elif getattr(svc, "_points", None) is not pts:
            svc.set_points(pts)
        return svc

    def _invalidate_points_spatial_index(self):
        """Инвалидирует кеш пространственного индекса cKDTree при изменении списка точек."""
        self._get_spatial_service().invalidate()
        self._2d_visible_mask_cache = None

    def _get_points_kdtree(self):
        """Возвращает сбалансированное KD-дерево точек для O(log N) поиска ближайшей точки."""
        return self._get_spatial_service().get_kdtree()

    def _find_nearest_point(self, click_y: float, click_x: float, visible_only: bool = True) -> tuple[int, float]:
        idx, dist = self._get_spatial_service().find_nearest_point(click_y, click_x)
        if visible_only and idx != -1 and hasattr(self, "_is_point_visible_on_2d") and not self._is_point_visible_on_2d(idx):
            tol = self._get_click_tolerance()
            tree = self._get_spatial_service().get_kdtree()
            if tree is not None:
                try:
                    k_cand = min(30, len(self.points))
                    dists, indices = tree.query([click_y, click_x], k=k_cand, distance_upper_bound=tol)
                    if np.isscalar(dists):
                        dists = [dists]
                        indices = [indices]
                    for d, cand_idx in zip(dists, indices):
                        if np.isinf(d) or cand_idx >= len(self.points):
                            break
                        if self._is_point_visible_on_2d(cand_idx):
                            return int(cand_idx), float(d)
                    return -1, float("inf")
                except Exception:
                    pass
            mask = self._ensure_2d_visible_mask() if hasattr(self, "_ensure_2d_visible_mask") else None
            if mask is not None:
                vis_indices = np.flatnonzero(mask)
                if len(vis_indices) == 0:
                    return -1, float("inf")
                coords = self._get_spatial_service()._ensure_coords()
                vis_coords = coords[vis_indices]
                d2 = (vis_coords[:, 0] - click_y) ** 2 + (vis_coords[:, 1] - click_x) ** 2
                min_loc = int(np.argmin(d2))
                min_d2 = d2[min_loc]
                return int(vis_indices[min_loc]), float(np.sqrt(min_d2))
            # Fallback
            best_idx, best_dist = -1, float("inf")
            for i, p in enumerate(self.points):
                if self._is_point_visible_on_2d(i):
                    d = (p.y - click_y) ** 2 + (p.x - click_x) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_idx = i
            return best_idx, float(np.sqrt(best_dist)) if best_idx != -1 else float("inf")
        return idx, dist

    def _get_click_tolerance(self) -> float:
        try:
            xlim = self.ax_2d.get_xlim()
            ylim = self.ax_2d.get_ylim()
            if isinstance(xlim[0], (int, float)) and isinstance(ylim[0], (int, float)):
                return SpatialIndexService.get_click_tolerance(xlim, ylim)
        except Exception:
            pass
        return 1.0

    def _find_nearest_boundary_edge(self, pt_idx: int) -> int:
        """
        Находит индекс ребра контура k, к которому точка pt_idx ближе всего.
        Ребро k соединяет boundary_indices[k] и boundary_indices[(k + 1) % N].
        Возвращает k (от 0 до N - 1).
        """
        return self._get_spatial_service().find_nearest_boundary_edge(pt_idx, self.boundary_indices)

    def _ensure_box_select_artist(self):
        """Гарантирует существование постоянного анимированного прямоугольника выделения."""
        rect = self.__dict__.get("_box_select_rect_artist", None)
        ax_2d = self.__dict__.get("ax_2d", None)
        if rect is None or getattr(rect, "axes", None) != ax_2d:
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (0, 0), 0, 0,
                fill=True, facecolor="#3498db", alpha=0.25,
                edgecolor="#2980b9", linestyle="--", linewidth=1.5,
                zorder=20, animated=True, visible=False
            )
            if ax_2d is not None:
                ax_2d.add_patch(rect)
            self._box_select_rect_artist = rect
        return rect

    def _ensure_contour_drag_artists(self):
        """Гарантирует существование постоянных анимированных элементов для перетаскивания вершины."""
        ax_2d = self.__dict__.get("ax_2d", None)
        if ax_2d is None:
            return
        l1 = self.__dict__.get("_contour_drag_line1", None)
        if l1 is None or getattr(l1, "axes", None) != ax_2d:
            self._contour_drag_line1, = ax_2d.plot(
                [], [], color="#ff8c00", linestyle="--", linewidth=2.2, zorder=11, animated=True, visible=False
            )
        l2 = self.__dict__.get("_contour_drag_line2", None)
        if l2 is None or getattr(l2, "axes", None) != ax_2d:
            self._contour_drag_line2, = ax_2d.plot(
                [], [], color="#ff8c00", linestyle="--", linewidth=2.2, zorder=11, animated=True, visible=False
            )
        guide = self.__dict__.get("_contour_drag_guide", None)
        if guide is None or getattr(guide, "axes", None) != ax_2d:
            self._contour_drag_guide, = ax_2d.plot(
                [], [], color="#e74c3c", linestyle=":", linewidth=1.5, alpha=0.7, zorder=10, animated=True, visible=False
            )
        ring = self.__dict__.get("_contour_drag_ring", None)
        if ring is None or getattr(ring, "axes", None) != ax_2d:
            self._contour_drag_ring, = ax_2d.plot(
                [], [], marker="o", markersize=12, markerfacecolor="none",
                markeredgecolor="#ff8c00", markeredgewidth=2.5, zorder=12, animated=True, visible=False
            )

    def _hide_contour_drag_artists(self):
        """Скрывает временные анимированные художники перетаскивания вершины."""
        for name in ("_contour_drag_line1", "_contour_drag_line2", "_contour_drag_guide", "_contour_drag_ring"):
            item = self.__dict__.get(name, None)
            if item is not None:
                item.set_visible(False)

    def _capture_blit_bg(self):
        """Захватывает растровый буфер осей для быстрого аппаратного blitting."""
        canvas_2d = self.__dict__.get("canvas_2d", None)
        ax_2d = self.__dict__.get("ax_2d", None)
        if canvas_2d is None or ax_2d is None:
            self._blit_bg = None
            return
        try:
            self._hide_contour_drag_artists()
            rect = self.__dict__.get("_box_select_rect_artist", None)
            if rect is not None:
                rect.set_visible(False)
            if hasattr(canvas_2d, "copy_from_bbox"):
                self._blit_bg = canvas_2d.copy_from_bbox(ax_2d.bbox)
            else:
                self._blit_bg = None
        except Exception:
            self._blit_bg = None

    def _insert_point_into_boundary(self, min_idx: int):
        """Встраивает точку min_idx в ближайшее ребро существующего контура с сохранением порядка обхода"""
        if not self.points or min_idx < 0 or min_idx >= len(self.points):
            return
        if min_idx in self.boundary_indices:
            return
        N = len(self.boundary_indices)
        if N < 2:
            insert_pos = N
        elif N == 2:
            insert_pos = 1
        else:
            best_k = self._find_nearest_boundary_edge(min_idx)
            insert_pos = best_k + 1

        old_type = self.points[min_idx].surface_type
        self.points[min_idx].surface_type = "boundary"
        if N < 2:
            self.boundary_indices.append(min_idx)
        else:
            self.boundary_indices.insert(insert_pos, min_idx)

        self._undo_stack.append(("boundary_insert", insert_pos, min_idx, old_type))
        self._invalidate_boundary_cache()
        self._update_2d_contour_only()
        self._redraw_2d()
        self._table_dirty = True
        self._tin_dirty = True
        self._schedule_auto_save()
        if len(self.boundary_indices) >= 3:
            self._suppress_tab_switch = True
            self._schedule_boundary_calc(400)

    def _on_canvas_press(self, event):
        """Нажатие кнопки мыши на 2D холсте"""
        try:
            state = self.tab_2d.winfo_children()
            for w in state:
                if isinstance(w, NavigationToolbar2Tk):
                    if w.mode != "":
                        return
        except Exception:
            pass

        if event.x is None or event.y is None:
            return

        # 1. Двойной клик ЛКМ: встраивание новой точки в ближайшее ребро контура (только для режимов контура)
        if event.button == 1 and getattr(event, "dblclick", False):
            # Отменяем отложенный одиночный клик
            if getattr(self, "_pending_single_click_timer", None) is not None:
                try:
                    self.after_cancel(self._pending_single_click_timer)
                except Exception:
                    pass
                self._pending_single_click_timer = None

            self._pan_start = None
            self._pan_dragged = False

            cur_mode = self.current_mode.get() if hasattr(self, "current_mode") else ""
            if cur_mode in ("select_boundary", "auto_hull"):
                click_y = event.xdata
                click_x = event.ydata
                if click_y is not None and click_x is not None and self.points:
                    min_idx, min_dist = self._find_nearest_point(click_y, click_x)
                    tol = self._get_click_tolerance()
                    if min_dist <= tol and min_idx != -1 and min_idx not in self.boundary_indices:
                        self._insert_point_into_boundary(min_idx)
            return

        # 2. Нажатие ПКМ (кнопка 3): контекстное меню точки или перетаскивание вершины контура
        if event.button == 3:
            self._hide_contour_drag_artists()
            self._rclick_point_idx = None
            self._rclick_start = None

            click_y = event.xdata
            click_x = event.ydata
            if click_y is not None and click_x is not None and self.points:
                min_idx, min_dist = self._find_nearest_point(click_y, click_x)
                tol = self._get_click_tolerance()
                if min_dist <= tol and min_idx != -1:
                    self._rclick_point_idx = min_idx
                    self._rclick_start = (event.x, event.y)
                    if min_idx in self.boundary_indices:
                        # Нажата ПКМ на вершине контура -> включаем режим drag вершины
                        self._contour_drag_source_idx = min_idx
                        self._contour_drag_pos = self.boundary_indices.index(min_idx)
                        self._contour_drag_start_coord = (event.x, event.y)
                        self._contour_drag_moved = False
                        self._contour_drag_target_idx = None
                        self._capture_blit_bg()
                    return

            # Если нажато мимо точек:
            self._contour_drag_source_idx = None
            self._contour_drag_pos = None
            self._contour_drag_moved = False
            self._contour_drag_target_idx = None
            return

        # 3. Нажатие ЛКМ (кнопка 1): рамка выделения или панорамирование схемы
        if event.button == 1:
            is_box_mode = (self.current_mode.get() == "box_select")
            is_shift = self._is_shift_down(event)

            if is_box_mode or is_shift:
                self._box_select_start = (
                    event.xdata, event.ydata,
                    event.x, event.y
                )
                self._box_select_moved = False
                self._pan_start = None
                self._capture_blit_bg()
                return

            self._pan_start = (
                event.x,
                event.y,
                event.xdata,
                event.ydata,
                self.ax_2d.get_xlim(),
                self.ax_2d.get_ylim(),
                event.button
            )
            self._pan_dragged = False

    def _on_canvas_motion(self, event):
        """Перемещение мыши: перетаскивание вершины контура по ПКМ, рамка выделения или панорамирование схемы по ЛКМ"""
        if event.x is None or event.y is None:
            return

        # 1. Перетаскивание вершины контура по ПКМ
        if getattr(self, "_contour_drag_source_idx", None) is not None:
            self._handle_contour_drag_motion(event)
            return

        # 2. Перемещение рамки выделения
        if getattr(self, "_box_select_start", None) is not None:
            self._handle_box_select_motion(event)
            return

        # 3. Панорамирование схемы по ЛКМ
        if self._pan_start is None:
            return

        start_px_x, start_px_y, _, _, orig_xlim, orig_ylim, btn = self._pan_start
        if btn != 1:
            return

        dx_px = event.x - start_px_x
        dy_px = event.y - start_px_y

        drag_threshold = 8
        if abs(dx_px) > drag_threshold or abs(dy_px) > drag_threshold:
            self._pan_dragged = True

        # Панорамирование при перетаскивании ЛКМ (с динамическим обновлением нативного текста или скрытием)
        if self._pan_dragged:
            bbox = self.ax_2d.bbox
            if bbox.width > 0 and bbox.height > 0:
                dx_data = dx_px * (orig_xlim[1] - orig_xlim[0]) / bbox.width
                dy_data = dy_px * (orig_ylim[1] - orig_ylim[0]) / bbox.height
                self.ax_2d.set_xlim(orig_xlim[0] - dx_data, orig_xlim[1] - dx_data)
                self.ax_2d.set_ylim(orig_ylim[0] - dy_data, orig_ylim[1] - dy_data)

                if self._has_tk_canvas_widget():
                    self._update_tk_canvas_labels_positions()
                else:
                    # При активном перетаскивании скрываем подписи точек, чтобы панорамирование шло на 60 FPS
                    if not getattr(self, "_pan_annotations_hidden", False):
                        for item in getattr(self, "_point_annotations", []):
                            if item[2] == "point":
                                item[0].set_visible(False)
                        self._pan_annotations_hidden = True

                self._throttled_canvas_draw(self.canvas_2d, delay_ms=25)

    def _handle_box_select_motion(self, event):
        """Интерактивное отображение рамки выделения при зажатой ЛКМ с аппаратным blitting"""
        if event.x is None or event.y is None or self._box_select_start is None:
            return
        start_y, start_x, px_x, px_y = self._box_select_start
        dist_px = np.hypot(event.x - px_x, event.y - px_y)
        if dist_px > 4:
            self._box_select_moved = True

        if not self._box_select_moved or event.xdata is None or event.ydata is None:
            return

        cur_y = event.xdata
        cur_x = event.ydata

        y_min = min(start_y, cur_y)
        y_max = max(start_y, cur_y)
        x_min = min(start_x, cur_x)
        x_max = max(start_x, cur_x)

        rect = self._ensure_box_select_artist()
        rect.set_xy((y_min, x_min))
        rect.set_width(y_max - y_min)
        rect.set_height(x_max - x_min)
        rect.set_visible(True)

        if self._blit_bg is not None and hasattr(self.canvas_2d, "restore_region") and hasattr(self.canvas_2d, "blit"):
            try:
                self.canvas_2d.restore_region(self._blit_bg)
                self.ax_2d.draw_artist(rect)
                self.canvas_2d.blit(self.ax_2d.bbox)
                return
            except Exception:
                pass
        self.canvas_2d.draw_idle()

    def _handle_box_select_release(self, event):
        """Завершение выделения прямоугольной рамкой на 2D холсте"""
        start_y, start_x, px_x, px_y = self._box_select_start
        was_moved = getattr(self, "_box_select_moved", False)
        self._box_select_start = None
        self._box_select_moved = False

        if getattr(self, "_box_select_rect_artist", None) is not None:
            self._box_select_rect_artist.set_visible(False)
        self._blit_bg = None

        cur_y = event.xdata
        cur_x = event.ydata
        is_ctrl = self._is_ctrl_down(event)

        if was_moved and cur_y is not None and cur_x is not None:
            y_min = min(start_y, cur_y)
            y_max = max(start_y, cur_y)
            x_min = min(start_x, cur_x)
            x_max = max(start_x, cur_x)

            candidate_indices = self._get_spatial_service().get_points_in_bbox(y_min, y_max, x_min, x_max)
            if hasattr(self, "_is_point_visible_on_2d"):
                selected = {int(i) for i in candidate_indices if self._is_point_visible_on_2d(int(i))}
            else:
                selected = {int(i) for i in candidate_indices}

            if is_ctrl:
                self._selected_points.update(selected)
            else:
                self._selected_points = selected
        else:
            # Одиночный клик в режиме рамки или с Shift
            if cur_y is not None and cur_x is not None and self.points:
                min_idx, min_dist = self._find_nearest_point(cur_y, cur_x)
                tol = self._get_click_tolerance()
                if min_dist <= tol and min_idx != -1:
                    if min_idx in self._selected_points:
                        self._selected_points.remove(min_idx)
                    else:
                        self._selected_points.add(min_idx)
                else:
                    if not is_ctrl:
                        self._selected_points.clear()

        self._update_2d_selection_only()
        self._update_selection_bar()

    def _handle_contour_drag_motion(self, event):
        """Интерактивное отображение перемещения вершины контура при зажатой ПКМ с аппаратным blitting"""
        if event.xdata is None or event.ydata is None:
            return

        # Проверяем порог смещения в пикселях от точки нажатия
        start_px = getattr(self, "_contour_drag_start_coord", None)
        if start_px is not None:
            dist_px = np.hypot(event.x - start_px[0], event.y - start_px[1])
            if dist_px > 4:
                self._contour_drag_moved = True

        if not self._contour_drag_moved:
            return

        cur_y = event.xdata
        cur_x = event.ydata

        # Ищем, не наведён ли курсор на свободную точку для привязки
        cand_idx, cand_dist = self._find_nearest_point(cur_y, cur_x)
        tol = self._get_click_tolerance()
        is_snap = (cand_dist <= tol and cand_idx != -1 and 
                   cand_idx != self._contour_drag_source_idx and 
                   cand_idx not in self.boundary_indices)

        if is_snap:
            target_pt = self.points[cand_idx]
            snap_y, snap_x = target_pt.y, target_pt.x
            self._contour_drag_target_idx = cand_idx
        else:
            snap_y, snap_x = cur_y, cur_x
            self._contour_drag_target_idx = None

        self._ensure_contour_drag_artists()

        # Резиновая линия: соединяем соседние вершины контура с текущей позицией
        N = len(self.boundary_indices)
        pos = self._contour_drag_pos
        has_l1, has_l2 = False, False

        if N >= 3 and pos is not None:
            prev_idx = self.boundary_indices[(pos - 1) % N]
            next_idx = self.boundary_indices[(pos + 1) % N]
            p_prev = self.points[prev_idx]
            p_next = self.points[next_idx]
            self._contour_drag_line1.set_data([p_prev.y, snap_y], [p_prev.x, snap_x])
            self._contour_drag_line1.set_visible(True)
            self._contour_drag_line2.set_data([snap_y, p_next.y], [snap_x, p_next.x])
            self._contour_drag_line2.set_visible(True)
            has_l1, has_l2 = True, True
        elif N == 2 and pos is not None:
            other_idx = self.boundary_indices[1 - pos]
            p_other = self.points[other_idx]
            self._contour_drag_line1.set_data([p_other.y, snap_y], [p_other.x, snap_x])
            self._contour_drag_line1.set_visible(True)
            self._contour_drag_line2.set_visible(False)
            has_l1 = True
        else:
            self._contour_drag_line1.set_visible(False)
            self._contour_drag_line2.set_visible(False)

        # Пунктирная линия-указатель от старой вершины к новой
        has_guide = False
        if self._contour_drag_source_idx is not None and 0 <= self._contour_drag_source_idx < len(self.points):
            src_pt = self.points[self._contour_drag_source_idx]
            self._contour_drag_guide.set_data([src_pt.y, snap_y], [src_pt.x, snap_x])
            self._contour_drag_guide.set_visible(True)
            has_guide = True
        else:
            self._contour_drag_guide.set_visible(False)

        # Подсветка целевой точки ярким кольцом
        if is_snap:
            self._contour_drag_ring.set_data([snap_y], [snap_x])
            self._contour_drag_ring.set_visible(True)
        else:
            self._contour_drag_ring.set_visible(False)

        # Blitting
        if self._blit_bg is not None and hasattr(self.canvas_2d, "restore_region") and hasattr(self.canvas_2d, "blit"):
            try:
                self.canvas_2d.restore_region(self._blit_bg)
                if has_l1:
                    self.ax_2d.draw_artist(self._contour_drag_line1)
                if has_l2:
                    self.ax_2d.draw_artist(self._contour_drag_line2)
                if has_guide:
                    self.ax_2d.draw_artist(self._contour_drag_guide)
                if is_snap:
                    self.ax_2d.draw_artist(self._contour_drag_ring)
                self.canvas_2d.blit(self.ax_2d.bbox)
                return
            except Exception:
                pass

        self.canvas_2d.draw_idle()

    def _on_canvas_release(self, event):
        """Отпускание кнопки мыши на 2D холсте"""
        # Скрываем временные линии перетаскивания
        self._hide_contour_drag_artists()
        self._blit_bg = None

        # 0. Завершение выделения прямоугольной рамкой
        if self.__dict__.get("_box_select_start", None) is not None:
            self._handle_box_select_release(event)
            return

        # 1. Завершение перетаскивания контура по ПКМ
        if self.__dict__.get("_contour_drag_source_idx", None) is not None:
            source_idx = self._contour_drag_source_idx
            pos = self._contour_drag_pos
            moved = self._contour_drag_moved
            target_idx = self._contour_drag_target_idx

            self._contour_drag_source_idx = None
            self._contour_drag_pos = None
            self._contour_drag_moved = False
            self._contour_drag_target_idx = None

            if moved:
                # Если курсор отпущен над свободной точкой
                if target_idx is not None and 0 <= target_idx < len(self.points) and target_idx not in self.boundary_indices:
                    old_idx = source_idx
                    old_type_target = self.points[target_idx].surface_type
                    old_type_old = self.points[old_idx].surface_type
                    self.boundary_indices[pos] = target_idx
                    self.points[target_idx].surface_type = "boundary"
                    self.points[old_idx].surface_type = "auto"
                    self._undo_stack.append(("boundary_replace", pos, old_idx, target_idx, old_type_target, old_type_old))
                    self._invalidate_boundary_cache()
                    self._redraw_2d()
                    self._table_dirty = True
                    self._tin_dirty = True
                    self._schedule_auto_save()
                    if len(self.boundary_indices) >= 3:
                        self._suppress_tab_switch = True
                        self._schedule_boundary_calc(400)
                else:
                    # Перетащили впустую — просто восстанавливаем вид
                    self._update_2d_contour_only()
                self._rclick_point_idx = None
                return
            else:
                # Одиночный клик ПКМ без движения на вершине контура -> показать контекстное меню
                pt_idx = self.__dict__.get("_rclick_point_idx", None)
                self._rclick_point_idx = None
                if pt_idx is not None and 0 <= pt_idx < len(self.points):
                    self._show_point_context_menu(pt_idx, event)
                return

        # Нажатие ПКМ на любой другой точке (не в контуре) или на пустом месте при наличии выделения
        if getattr(event, "button", None) == 3:
            pt_idx = self.__dict__.get("_rclick_point_idx", None)
            self._rclick_point_idx = None
            if pt_idx is not None and 0 <= pt_idx < len(self.points):
                self._show_point_context_menu(pt_idx, event)
            elif getattr(self, "_selected_points", None):
                first_sel = next(iter(self._selected_points))
                self._show_point_context_menu(first_sel, event)
            return

        # 2. Обработка ЛКМ (кнопка 1)
        if self._pan_start is None:
            return

        press_x, press_y, press_xdata, press_ydata, orig_xlim, orig_ylim, btn = self._pan_start
        was_dragged = self._pan_dragged
        self._pan_start = None
        self._pan_dragged = False

        if was_dragged:
            # Было перемещение схемы — восстанавливаем видимость и актуализируем подписи
            self._flush_canvas_draw(self.canvas_2d)
            self._pan_annotations_hidden = False
            self._schedule_viewport_annotations_update(delay_ms=20)
            return

        if btn == 1:
            # Если координаты отпускания потеряны (курсор ушёл к краю осей), восстанавливаем из точки нажатия
            if (getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None) and press_xdata is not None and press_ydata is not None:
                event.xdata = press_xdata
                event.ydata = press_ydata

            cur_mode = self.current_mode.get() if hasattr(self, "current_mode") else ""
            if cur_mode in ("assign_top", "assign_bottom", "add_point"):
                if getattr(self, "_pending_single_click_timer", None) is not None:
                    try:
                        self.after_cancel(self._pending_single_click_timer)
                    except Exception:
                        pass
                    self._pending_single_click_timer = None
                self._handle_point_click(event)
                return

            # Одиночный клик ЛКМ с небольшой задержкой (чтобы отличить от двойного клика)
            if getattr(self, "_pending_single_click_timer", None) is not None:
                try:
                    self.after_cancel(self._pending_single_click_timer)
                except Exception:
                    pass
            self._pending_single_click_timer = self.after(
                200, lambda ev=event: self._execute_delayed_single_click(ev)
            )

    def _execute_delayed_single_click(self, event):
        self._pending_single_click_timer = None
        self._handle_point_click(event)

    def _handle_point_click(self, event):
        """Обработка клика ЛКМ на 2D схеме"""
        click_y = event.xdata   # Y (Восток) — горизонтальная ось
        click_x = event.ydata   # X (Север)  — вертикальная ось

        if click_y is None or click_x is None:
            return

        mode = self.current_mode.get()

        # Режим добавления новой точки — клик в любом месте схемы
        if mode == "add_point":
            next_id = f"т{len(self.points) + 1}"
            dlg = QuickAddPointDialog(self, x_val=click_x, y_val=click_y, default_id=next_id)
            if dlg.result:
                id_val, x_val, y_val, h_val, surf_val = dlg.result
                if surf_val == "auto":
                    mean_bound, split_h, work_type = self._get_split_height()
                    dummy_pt = GeoPoint(id=id_val, x=x_val, y=y_val, h=h_val, surface_type="auto")
                    surf_val = self._get_point_surface(dummy_pt, mean_bound, split_h, work_type)
                new_pt = GeoPoint(id=id_val, x=x_val, y=y_val, h=h_val, surface_type=surf_val)
                self.points.append(new_pt)
                new_idx = len(self.points) - 1
                if surf_val == "boundary":
                    self.boundary_indices.append(new_idx)
                self._tin_excluded.clear()   # триангуляция изменилась
                self._tin_custom_simplices = None
                self._update_all_views()
                self._schedule_auto_save()
                # Автопересчёт если контур готов
                if len(self.boundary_indices) >= 3:
                    self._suppress_tab_switch = True
                    self._schedule_boundary_calc(400)
            return

        # Остальные режимы — клик должен попасть на существующую точку
        if not self.points:
            return

        min_idx, min_dist = self._find_nearest_point(click_y, click_x)
        tol = self._get_click_tolerance()

        if min_dist > tol:
            if getattr(self, "_selected_points", None):
                self._selected_points.clear()
                self._update_selection_bar()
                self._update_2d_selection_only()
            return

        if mode == "box_select":
            if min_idx in self._selected_points:
                self._selected_points.remove(min_idx)
            else:
                self._selected_points.add(min_idx)
            self._update_2d_selection_only()
            self._update_selection_bar()
            return

        elif mode in ("select_boundary", "auto_hull"):
            if min_idx not in self.boundary_indices:
                old_type = self.points[min_idx].surface_type
                self.points[min_idx].surface_type = "boundary"
                self.boundary_indices.append(min_idx)
                self._undo_stack.append(("boundary_add", min_idx, old_type))
            else:
                old_type = self.points[min_idx].surface_type
                self.boundary_indices.remove(min_idx)
                self.points[min_idx].surface_type = "auto"
                self._undo_stack.append(("boundary_remove", min_idx, old_type))
            self._invalidate_boundary_cache()
            self._redraw_2d()
            self._table_dirty = True
            self._tin_dirty = True
            self._schedule_auto_save()
            # Пересчёт объёма при каждом изменении контура с дебаунсингом
            if len(self.boundary_indices) >= 3:
                self._suppress_tab_switch = True
                self._schedule_boundary_calc(400)

        elif mode == "assign_top":
            self._assign_point_surface(min_idx, "top", toggle_same=True)

        elif mode == "assign_bottom":
            self._assign_point_surface(min_idx, "bottom", toggle_same=True)


    def _undo_last_action(self):
        if not self._undo_stack:
            return
        action = self._undo_stack.pop()

        if action[0] == "boundary_add":
            idx = action[1]
            old_type = action[2] if len(action) > 2 else "auto"
            if idx in self.boundary_indices:
                self.boundary_indices.remove(idx)
            if 0 <= idx < len(self.points):
                self.points[idx].surface_type = old_type
            self._invalidate_boundary_cache()
        elif action[0] == "boundary_remove":
            idx = action[1]
            old_type = action[2] if len(action) > 2 else "boundary"
            if idx not in self.boundary_indices:
                self.boundary_indices.append(idx)
            if 0 <= idx < len(self.points):
                self.points[idx].surface_type = old_type
            self._invalidate_boundary_cache()
        elif action[0] == "boundary_insert":
            pos, idx = action[1], action[2]
            old_type = action[3] if len(action) > 3 else "auto"
            if 0 <= pos < len(self.boundary_indices) and self.boundary_indices[pos] == idx:
                self.boundary_indices.pop(pos)
            elif idx in self.boundary_indices:
                self.boundary_indices.remove(idx)
            if 0 <= idx < len(self.points):
                self.points[idx].surface_type = old_type
            self._invalidate_boundary_cache()
        elif action[0] == "boundary_replace":
            pos, old_idx, new_idx = action[1], action[2], action[3]
            old_type_new = action[4] if len(action) > 4 else "auto"
            old_type_old = action[5] if len(action) > 5 else "boundary"
            if 0 <= pos < len(self.boundary_indices) and self.boundary_indices[pos] == new_idx:
                self.boundary_indices[pos] = old_idx
            if 0 <= new_idx < len(self.points):
                self.points[new_idx].surface_type = old_type_new
            if 0 <= old_idx < len(self.points):
                self.points[old_idx].surface_type = old_type_old
            self._invalidate_boundary_cache()
        elif action[0] == "boundary_set":
            old_boundary = action[1]
            old_surfaces = action[2]
            self.boundary_indices = list(old_boundary)
            for i, st in enumerate(old_surfaces):
                if 0 <= i < len(self.points):
                    self.points[i].surface_type = st
            self._invalidate_boundary_cache()
        elif action[0] == "assign":
            idx = action[1]
            old_type = action[2]
            was_bound = action[3] if len(action) > 3 else (old_type == "boundary")
            if 0 <= idx < len(self.points):
                self.points[idx].surface_type = old_type
                if was_bound and idx not in self.boundary_indices:
                    self.boundary_indices.append(idx)
            self._invalidate_boundary_cache()
            self._table_dirty = True
            self._tin_dirty = True
            self._2d_dirty = False
            self._update_all_views()
            self._schedule_auto_save()
            if len(self.boundary_indices) >= 3:
                self._suppress_tab_switch = True
                self._schedule_boundary_calc(400)
        elif action[0] == "batch_assign":
            # ("batch_assign", [(idx, old_type, was_bound), ...])
            for item in action[1]:
                idx, old_type = item[0], item[1]
                was_bound = item[2] if len(item) > 2 else (old_type == "boundary")
                if 0 <= idx < len(self.points):
                    self.points[idx].surface_type = old_type
                    if was_bound and idx not in self.boundary_indices:
                        self.boundary_indices.append(idx)
            self._invalidate_boundary_cache()
            self._table_dirty = True
            self._tin_dirty = True
            self._2d_dirty = False
            self._update_all_views()
            self._schedule_auto_save()
            if len(self.boundary_indices) >= 3:
                self._suppress_tab_switch = True
                self._schedule_boundary_calc(400)
        elif action[0] == "tin_flip":
            _, t1, t2, old_t1, old_t2, _, _ = action
            if self._tin_simplices is not None and t1 < len(self._tin_simplices) and t2 < len(self._tin_simplices):
                simplices_list = [list(s) for s in self._tin_simplices]
                simplices_list[t1] = list(old_t1)
                simplices_list[t2] = list(old_t2)
                self._tin_simplices = np.array(simplices_list)
                self._tin_custom_simplices = [tuple(s) for s in simplices_list]
        elif action[0] == "auto_flip_group":
            # Откат группы авто-флипов в обратном порядке
            flip_list = action[1]
            if self._tin_simplices is not None:
                simplices_list = [list(s) for s in self._tin_simplices]
                for flip in reversed(flip_list):
                    _, t1, t2, old_t1, old_t2, _, _ = flip
                    if t1 < len(simplices_list) and t2 < len(simplices_list):
                        simplices_list[t1] = list(old_t1)
                        simplices_list[t2] = list(old_t2)
                self._tin_simplices = np.array(simplices_list)
                self._tin_custom_simplices = [tuple(s) for s in simplices_list]
            self.lbl_tin_stats.configure(text="↩ Авто-оптимизация отменена")

        tv = self.__dict__.get("tabview", None)
        current_tab = tv.get() if tv is not None else None
        self._redraw_2d()
        tab_table = getattr(self, "TAB_TABLE", "Таблица точек")
        tab_tin = getattr(self, "TAB_TIN", "Триангуляция TIN")
        if current_tab == tab_table:
            self._update_table()
            self._table_dirty = False
        else:
            self._table_dirty = True

        if current_tab == tab_tin:
            self._redraw_tin(reset_view=False)
            self._tin_dirty = False
        else:
            self._tin_dirty = True

        self._schedule_auto_save()
        # Пересчёт объёма после отмены изменений контура или поверхностей
        if action[0] in ("boundary_add", "boundary_remove", "boundary_insert", "boundary_replace", "boundary_set", "assign", "batch_assign") and len(self.boundary_indices) >= 3:
            self._suppress_tab_switch = True
            self._schedule_boundary_calc(400)

    def _update_selection_bar(self):
        """Обновляет видимость и статус плавающей панели выделенных точек на 2D холсте"""
        if not hasattr(self, "frame_selection_bar"):
            return
        n = len(getattr(self, "_selected_points", set()))
        if n == 0:
            self.frame_selection_bar.place_forget()
            return

        self.lbl_sel_count.configure(text=f"Выделено: {n} т.")
        # Размещаем по центру вверху холста 2D
        if getattr(self, "tabview", None) and self.tabview.get() == self.TAB_2D:
            self.frame_selection_bar.place(relx=0.5, y=34, anchor="n")
            self.frame_selection_bar.lift()
        else:
            self.frame_selection_bar.place_forget()

    def _batch_assign_surface(self, target_surf: str):
        """Пакетное назначение выбранным точкам целевой поверхности ('top' или 'bottom')"""
        if not self._selected_points:
            return
        self._assign_point_surface(-1, target_surf, toggle_same=False)

    def _clear_selected_points(self):
        """Сброс выделения точек"""
        self._selected_points.clear()
        self._update_2d_selection_only()
        self._update_selection_bar()

    def _build_boundary_from_selected_points(self):
        """Строит контур сшивания (Convex Hull) вокруг группы выделенных точек"""
        if len(getattr(self, "boundary_indices", [])) >= 3:
            messagebox.showwarning(
                "Контур сшивания",
                "Контур сшивания уже существует.\nЧтобы построить новый контур вокруг выделенных точек, сначала сбросьте текущий контур (кнопка 'Сброс контура')."
            )
            return

        sel = getattr(self, "_selected_points", set())
        if len(sel) < 3:
            messagebox.showwarning(
                "Контур сшивания",
                "Для построения контура сшивания необходимо выделить как минимум 3 точки."
            )
            return

        sel_indices = [idx for idx in sel if 0 <= idx < len(self.points)]
        if len(sel_indices) < 3:
            messagebox.showwarning(
                "Контур сшивания",
                "Для построения контура сшивания необходимо выделить как минимум 3 точки."
            )
            return

        pts_coords = np.array([[self.points[i].x, self.points[i].y] for i in sel_indices])

        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts_coords)
            boundary_idx = [sel_indices[v] for v in hull.vertices]
        except Exception as ex:
            messagebox.showerror(
                "Ошибка построения контура",
                f"Не удалось построить контур вокруг выделенных точек:\n{ex}"
            )
            return

        old_boundary = list(self.boundary_indices)
        old_surfaces = [p.surface_type for p in self.points]
        self._undo_stack.append(("boundary_set", old_boundary, old_surfaces))

        self.boundary_indices = boundary_idx
        for idx in self.boundary_indices:
            self.points[idx].surface_type = "boundary"

        self._tin_excluded.clear()
        self._tin_custom_simplices = None
        self._tin_dirty = True
        self._invalidate_boundary_cache()

        self._selected_points.clear()
        self._update_selection_bar()
        self._update_all_views()
        self._schedule_auto_save()

        if len(self.boundary_indices) >= 3:
            self._suppress_tab_switch = True
            self._schedule_boundary_calc(400)

    def _assign_point_surface(self, target_idx: int, target_surf: str, toggle_same: bool = False):
        """Назначает целевую поверхность для точки (или выделенной группы точек).
        При toggle_same=True повторный клик по одиночной точке с тем же типом поверхности сбрасывает её в 'auto'.
        """
        if not self.points:
            return
        undo_items = []

        if target_idx in self._selected_points and len(self._selected_points) > 1:
            indices = sorted(list(self._selected_points))
        elif 0 <= target_idx < len(self.points):
            indices = [target_idx]
        elif self._selected_points:
            indices = sorted(list(self._selected_points))
        else:
            return

        for idx in indices:
            if 0 <= idx < len(self.points):
                old_type = self.points[idx].surface_type
                actual_target = target_surf
                if toggle_same and len(indices) == 1 and old_type == target_surf:
                    actual_target = "auto"

                if old_type != actual_target:
                    was_in_bound = (idx in self.boundary_indices)
                    undo_items.append((idx, old_type, was_in_bound))
                    self.points[idx].surface_type = actual_target
                    if idx in self.boundary_indices:
                        self.boundary_indices.remove(idx)

        if undo_items:
            self._undo_stack.append(("batch_assign", undo_items))

        self._selected_points.clear()
        self._update_selection_bar()
        self._invalidate_boundary_cache()
        self._tin_excluded.clear()
        self._tin_custom_simplices = None
        self._tin_dirty = True
        self._table_dirty = True
        self._2d_dirty = False
        self._update_all_views()
        self._schedule_auto_save()
        if len(self.boundary_indices) >= 3:
            self._suppress_tab_switch = True
            self._schedule_boundary_calc(400)

    def _delete_point(self, target_idx: int = -1):
        """Удаляет точку (или выделенную группу точек)"""
        if not self.points:
            return

        if target_idx in self._selected_points and len(self._selected_points) > 1:
            indices = sorted(list(self._selected_points), reverse=True)
            msg = f"Удалить выбранные точки ({len(indices)} шт.)?"
        elif 0 <= target_idx < len(self.points):
            indices = [target_idx]
            pt = self.points[target_idx]
            msg = f"Удалить точку #{target_idx + 1} (ID: {pt.id}, H: {pt.h:.2f})?"
        elif self._selected_points:
            indices = sorted(list(self._selected_points), reverse=True)
            msg = f"Удалить выбранные точки ({len(indices)} шт.)?"
        else:
            return

        if not messagebox.askyesno("Подтверждение удаления", msg):
            return

        for idx in indices:
            if 0 <= idx < len(self.points):
                del self.points[idx]
                # Корректировка индексов контура
                new_bounds = []
                for b in self.boundary_indices:
                    if b == idx:
                        continue
                    elif b > idx:
                        new_bounds.append(b - 1)
                    else:
                        new_bounds.append(b)
                self.boundary_indices = new_bounds

        self._selected_points.clear()
        self._update_selection_bar()
        self._invalidate_points_spatial_index()
        self._invalidate_boundary_cache()
        self._tin_excluded.clear()
        self._tin_custom_simplices = None
        self._tin_dirty = True
        self._table_dirty = True
        self._2d_dirty = False
        self._update_all_views()
        self._schedule_auto_save()
        if len(self.boundary_indices) >= 3:
            self._suppress_tab_switch = True
            self._schedule_boundary_calc(400)

    def _show_point_context_menu(self, pt_idx: int, event):
        """Отображает контекстное меню для точки на 2D схеме по ПКМ"""
        if not self.points or not (0 <= pt_idx < len(self.points)):
            return

        pt = self.points[pt_idx]

        # Если точка не была выделена, выделяем её для наглядности
        if pt_idx not in self._selected_points:
            self._selected_points = {pt_idx}
            self._update_2d_selection_only()
            self._update_selection_bar()

        num_sel = len(self._selected_points)
        is_multi = (num_sel > 1 and pt_idx in self._selected_points)

        is_dark = ctk.get_appearance_mode() == "Dark"
        bg = "#2b2b2b" if is_dark else "#f8f9fa"
        fg = "#f0f2f5" if is_dark else "#212529"
        active_bg = "#3a3a3a" if is_dark else "#e9ecef"
        active_fg = "#38bdf8" if is_dark else "#0d6efd"

        menu = tk.Menu(
            self,
            tearoff=0,
            bg=bg,
            fg=fg,
            activebackground=active_bg,
            activeforeground=active_fg,
            font=("Segoe UI", 10)
        )

        surf_map = {
            "top": "Верх",
            "bottom": "Низ",
            "boundary": "Контур",
            "auto": "Авто"
        }
        surf_str = surf_map.get(pt.surface_type, pt.surface_type)
        if is_multi:
            hdr_text = f"Выделено точек: {num_sel}"
        else:
            hdr_text = f"Точка #{pt_idx + 1} (ID: {pt.id}, {surf_str})"

        menu.add_command(label=hdr_text, state=tk.DISABLED)
        menu.add_separator()

        lbl_top = f"Перенести в верхнюю поверхность ({num_sel} т.)" if is_multi else "Перенести в верхнюю поверхность"
        lbl_bot = f"Перенести в нижнюю поверхность ({num_sel} т.)" if is_multi else "Перенести в нижнюю поверхность"
        lbl_auto = f"Сбросить в авто-определение ({num_sel} т.)" if is_multi else "Сбросить в авто-определение"
        lbl_bound = f"⬡ Построить контур вокруг выделенных ({num_sel} т.)" if is_multi else "⬡ Построить контур сшивания"
        lbl_del = f"Удалить ({num_sel} т.)" if is_multi else "Удалить"

        menu.add_command(
            label=lbl_top,
            command=lambda: self._assign_point_surface(pt_idx, "top")
        )
        menu.add_command(
            label=lbl_bot,
            command=lambda: self._assign_point_surface(pt_idx, "bottom")
        )
        menu.add_command(
            label=lbl_auto,
            command=lambda: self._assign_point_surface(pt_idx, "auto")
        )
        menu.add_command(
            label=lbl_bound,
            command=self._build_boundary_from_selected_points
        )
        menu.add_separator()
        menu.add_command(
            label=lbl_del,
            command=lambda: self._delete_point(pt_idx)
        )
        menu.add_separator()
        lbl_cancel = f"Отмена (снять выделение {num_sel} т.)" if is_multi else "Отмена (снять выделение)"
        menu.add_command(
            label=lbl_cancel,
            command=self._clear_selected_points
        )

        # Вычисляем экранные координаты курсора
        gui_ev = getattr(event, "guiEvent", None)
        if gui_ev is not None and hasattr(gui_ev, "x_root") and hasattr(gui_ev, "y_root"):
            root_x = gui_ev.x_root
            root_y = gui_ev.y_root
        else:
            root_x = self.winfo_pointerx()
            root_y = self.winfo_pointery()

        try:
            menu.tk_popup(root_x, root_y)
        finally:
            menu.grab_release()


    def _on_canvas_scroll(self, event):
        if event.xdata is None or event.ydata is None:
            return
        base_scale = 1.25
        scale_factor = 1.0 / base_scale if event.button == "up" else base_scale

        cur_xlim = self.ax_2d.get_xlim()
        cur_ylim = self.ax_2d.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax_2d.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax_2d.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])

        if self._has_tk_canvas_widget():
            self._update_tk_canvas_labels_positions()
        else:
            # При активном зумировании скрываем подписи точек, чтобы зум был моментальным (60 FPS)
            for item in getattr(self, "_point_annotations", []):
                if item[2] == "point":
                    item[0].set_visible(False)

        self._schedule_viewport_annotations_update(delay_ms=120)
        self._throttled_canvas_draw(self.canvas_2d, delay_ms=25)

    # ==================== РАСЧЕТ И ВИЗУАЛИЗАЦИЯ ====================

    def _auto_calc_and_show_plan(self):
        """Автоматический расчёт объёма с переходом на вкладку '2D схема в плане'."""
        self._suppress_tab_switch = True
        try:
            self.calculate_volume()
        finally:
            self._suppress_tab_switch = False
        # Переключаемся на вкладку схемы в плане только если расчёт прошёл
        if self.calc_results:
            try:
                self._select_tab(self.TAB_2D)
            except Exception:
                pass

    def _calc_boundary_silent(self):
        """Моментальный расчёт объёма при редактировании контура без переключения вкладки."""
        self._suppress_tab_switch = True
        try:
            self.calculate_volume(silent=True)
        finally:
            self._suppress_tab_switch = False

    def calculate_volume(self, use_tin: bool = None, silent: bool = False):
        """Расчёт объёма. По умолчанию использует активную триангуляцию TIN с учётом всех переброшенных рёбер."""
        if not self.points:
            if not silent:
                messagebox.showwarning("Внимание", "Сначала загрузите координаты.")
            return

        try:
            res_val = float(self.ent_res.get().replace(",", "."))
        except ValueError:
            res_val = 0.2

        if len(self.boundary_indices) >= 3:
            bound_pts = np.array([[self.points[i].x, self.points[i].y, self.points[i].h]
                                   for i in self.boundary_indices])
        else:
            bound_pts = None

        is_two_surfs = self._is_two_surfaces()
        mean_bound, split_h, work_type = self._get_split_height()
        is_cut = (work_type == "cut")

        top_list, bot_list = [], []
        inside_mask = self._ensure_boundary_inside_mask() if (not is_two_surfs and len(self.boundary_indices) >= 3) else None
        for i, p in enumerate(self.points):
            if is_two_surfs:
                # В режиме двух независимых поверхностей точки гарантированно и строго идут в свои массивы
                if p.surface_type == "top":
                    top_list.append([p.x, p.y, p.h])
                elif p.surface_type == "bottom":
                    bot_list.append([p.x, p.y, p.h])
                continue

            if i in self.boundary_indices or p.surface_type == "boundary":
                continue

            # В режиме сшивания по контуру точки за пределами контура не участвуют в расчёте объема данного контура
            if inside_mask is not None and not (bool(inside_mask[i]) if i < len(inside_mask) else False):
                continue

            surf = self._get_point_surface(p, mean_bound, split_h, work_type)
            if surf == "top":
                top_list.append([p.x, p.y, p.h])
            elif surf == "bottom":
                bot_list.append([p.x, p.y, p.h])

        top_arr = np.array(top_list) if top_list else np.empty((0, 3))
        bot_arr = np.array(bot_list) if bot_list else np.empty((0, 3))

        work_type_param = "grading" if is_two_surfs else ("cut" if is_cut else "fill")

        try:
            from volume_engine import VolumeCalculator
            calc = VolumeCalculator(
                top_points=top_arr,
                bottom_points=bot_arr,
                boundary_points=bound_pts,
                grid_resolution=res_val,
                work_type=work_type_param,
                stitching=not is_two_surfs
            )

            # Если TIN ещё не был построен или контур изменился (_tin_dirty) — строим его
            if (self._tin_simplices is None or len(self._tin_simplices) == 0 or getattr(self, "_tin_dirty", False)) and len(self.boundary_indices) >= 3:
                self._redraw_tin(reset_view=False)
                self._tin_dirty = False

            # Приоритет: если сформированы треугольники TIN (в т.ч. пользовательские переброски),
            # рассчитываем методом TIN с точным построением картограммы и 3D
            pts_3d = np.array([[p.x, p.y, p.h] for p in self.points])
            if not is_two_surfs and self._tin_simplices is not None and len(self._tin_simplices) > 0 and len(self.boundary_indices) >= 3:
                tin_surface_param = "bottom" if is_cut else "top"
                res = calc.calculate_with_custom_tin(
                    pts_2d=self._tin_pts_2d if self._tin_pts_2d is not None else pts_3d[:, :2],
                    simplices=self._tin_simplices,
                    excluded_simplex_indices=self._tin_excluded,
                    pts_3d=pts_3d,
                    tin_surface=tin_surface_param
                )
            else:
                res = calc.calculate()

            if "error" in res and self._tin_simplices is not None:
                # Если сохраненная или кастомная триангуляция не подошла,
                # автоматически перестраиваем стандартную триангуляцию Делоне заново и повторяем
                self._tin_custom_simplices = None
                self._tin_simplices = None
                self._tin_dirty = True
                self._rebuild_tin()
                if self._tin_simplices is not None and len(self._tin_simplices) > 0:
                    res = calc.calculate_with_custom_tin(
                        pts_2d=self._tin_pts_2d if self._tin_pts_2d is not None else pts_3d[:, :2],
                        simplices=self._tin_simplices,
                        excluded_simplex_indices=self._tin_excluded,
                        pts_3d=pts_3d,
                        tin_surface=tin_surface_param
                    )
                if "error" in res:
                    res = calc.calculate()

            if "error" in res:
                if not silent:
                    messagebox.showerror("Ошибка расчета", res["error"])
                return

            self.calc_results = res
            self._display_results(res)

            # Ленивое обновление 3D поверхности и картограммы
            current_tab = self.tabview.get() if hasattr(self, "tabview") else None
            if not getattr(self, "_suppress_tab_switch", False):
                try:
                    if current_tab == self.TAB_2D:
                        self._select_tab(self.TAB_DIFF)
                        current_tab = self.TAB_DIFF
                except Exception:
                    pass

            if current_tab == self.TAB_3D:
                self._redraw_3d(res)
                self._3d_dirty = False
                self._diff_dirty = True
                self._contours_dirty = True
            elif current_tab == self.TAB_DIFF:
                self._redraw_diff(res)
                self._diff_dirty = False
                self._3d_dirty = True
                self._contours_dirty = True
            elif current_tab == getattr(self, "TAB_CONTOURS", "Горизонтали"):
                self._redraw_contours()
                self._contours_dirty = False
                self._3d_dirty = True
                self._diff_dirty = True
            else:
                self._3d_dirty = True
                self._diff_dirty = True
                self._contours_dirty = True

            # Автосохранение результатов в текстовый файл
            self._auto_save_report(res)

            # Автосохранение состояния проекта с дебаунсингом
            if self._current_file_path:
                self._schedule_auto_save()

        except Exception as e:
            if not silent:
                messagebox.showerror("Ошибка", f"Ошибка расчета объема:\n{str(e)}")

    def _auto_save_report(self, r: dict):
        """Автоматически сохраняет отчёт в текстовый файл папки проекта"""
        if not self._current_project_dir:
            return
        folder_name = os.path.basename(self._current_project_dir)
        base_name = self._strip_date_from_folder_name(folder_name) or folder_name
        report_path = os.path.join(self._current_project_dir, f"{base_name}_report.txt")
        try:
            report_text = self.txt_results.get("1.0", tk.END)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            self.lbl_report_status.configure(
                text=f"✓ Отчёт: {os.path.basename(report_path)}")
        except Exception as e:
            self.lbl_report_status.configure(text=f"⚠ Не удалось сохранить отчёт: {e}")

    def _display_results(self, r: dict):
        self.txt_results.config(state=tk.NORMAL)
        self.txt_results.delete("1.0", tk.END)

        is_tin = r.get("custom_tin", False)
        method_line = (
            f"• Метод:                  {'TIN (прямой)':>16}\n"
            f"• Активных треугольников: {r.get('num_triangles', 0):>10d}\n"
            if is_tin else
            f"• Метод:                  {'Интерполяция на сетке':>16}\n"
            f"• Расчетных узлов сетки:  {r.get('num_grid_points', 0):>10d}\n"
            f"• Шаг сетки:              {r.get('grid_resolution', 0):>10.3f} м\n"
        )

        report = (
            f"=== {APP_TITLE} ===\n"
            f"=== РЕЗУЛЬТАТЫ РАСЧЕТА ОБЪЕМА ===\n"
            f"• Объем насыпи (Fill):    {r['v_fill']:>10.3f} м³\n"
            f"• Объем выемки (Cut):     {r['v_cut']:>10.3f} м³\n"
            f"------------------------------------\n"
            f"• ИТОГОВЫЙ ОБЪЕМ (Net):   {r['v_net']:>10.3f} м³\n\n"
            f"• Площадь контура (2D):   {r['area_2d']:>10.3f} м²\n"
            f"• Площадь верха (3D):     {r['top_area_3d']:>10.3f} м²\n"
            f"• Площадь низа (3D):      {r['bot_area_3d']:>10.3f} м²\n\n"
            f"• Средняя мощность слоя:  {r['avg_thickness']:>10.3f} м\n"
            f"• Макс. мощность слоя:    {r['max_thickness']:>10.3f} м\n"
            f"• Мин. мощность слоя:     {r['min_thickness']:>10.3f} м\n"
            + method_line
        )
        self.txt_results.insert(tk.END, report)
        self.txt_results.config(state=tk.DISABLED)
        self._update_toolbar_volume_labels()

    def _update_all_views(self, reset_view=False):
        if reset_view:
            self._tin_view_initialized = False
            self._contours_view_initialized = False
        current_tab = self.tabview.get() if hasattr(self, "tabview") else None
        self._redraw_2d(reset_view=reset_view)
        if current_tab == self.TAB_TABLE:
            self._update_table()
            self._table_dirty = False
        else:
            self._table_dirty = True

        if current_tab == self.TAB_TIN:
            self._redraw_tin(reset_view=reset_view)
            self._tin_dirty = False
        else:
            self._tin_dirty = True

        if current_tab == getattr(self, "TAB_CONTOURS", "Горизонтали"):
            self._redraw_contours()
            self._contours_dirty = False
        else:
            self._contours_dirty = True

        if current_tab == self.TAB_3D and self.calc_results:
            self._redraw_3d(self.calc_results)
            self._3d_dirty = False
        else:
            self._3d_dirty = True

        if current_tab == self.TAB_DIFF and self.calc_results:
            self._redraw_diff(self.calc_results)
            self._diff_dirty = False
        else:
            self._diff_dirty = True

        self._update_toolbar_volume_labels()


    def _detect_two_surfaces_heuristic(self) -> bool:
        """Эвристическое определение съемки с двумя параллельными/независимыми поверхностями (например, фундаменты, перекрытия)."""
        if not self.points or len(self.points) < 6:
            return False
        try:
            coords = np.array([[p.x, p.y, p.h] for p in self.points], dtype=float)
            h_vals = coords[:, 2]
            h_min, h_max = float(np.min(h_vals)), float(np.max(h_vals))
            if h_max - h_min < 0.04:
                return False
            span = max(float(np.ptp(coords[:, 0])), float(np.ptp(coords[:, 1])))
            if span < 1e-4:
                return False

            from scipy.spatial import cKDTree
            r_tol = min(0.35, max(0.06, span * 0.05))
            tree = cKDTree(coords[:, :2])
            pairs = tree.query_pairs(r=r_tol)
            vert_pairs = [(i, j) for i, j in pairs if abs(coords[i, 2] - coords[j, 2]) >= 0.04]

            # 1. Значительная доля точек состоит из вертикальных пар (например, углы фундамента или плиты от 5 см)
            if len(vert_pairs) >= 3 and (len(vert_pairs) * 2 >= len(self.points) * 0.25):
                return True

            # 2. Истинный бимодальный разрыв по высоте (gap >= 0.20 м без промежуточных точек)
            # и перекрытие в плане (IoU > 0.45 и area_ratio >= 0.50)
            if h_max - h_min < 0.20:
                return False
            split_candidate = (h_min + h_max) / 2.0
            idx_top = [i for i, h in enumerate(h_vals) if h > split_candidate]
            idx_bot = [i for i, h in enumerate(h_vals) if h <= split_candidate]
            if len(idx_top) >= 3 and len(idx_bot) >= 3:
                # Если нет вертикальных пар, но один высотный ярус целиком окружён другим
                # (все внешние вершины ConvexHull принадлежат только одному ярусу, а второй лежит строго внутри),
                # то это тело с наклонными откосами (насыпь/котлован), а не две параллельные поверхности!
                try:
                    hull = ConvexHull(coords[:, :2])
                    hull_indices = set(hull.vertices)
                    top_on_hull = any(i in hull_indices for i in idx_top)
                    bot_on_hull = any(i in hull_indices for i in idx_bot)
                    if not (top_on_hull and bot_on_hull):
                        return False
                except Exception:
                    pass

                gap = float(np.min(coords[idx_top, 2]) - np.max(coords[idx_bot, 2]))
                if gap >= 0.20:
                    bbox_top = (np.min(coords[idx_top, 0]), np.max(coords[idx_top, 0]),
                                np.min(coords[idx_top, 1]), np.max(coords[idx_top, 1]))
                    bbox_bot = (np.min(coords[idx_bot, 0]), np.max(coords[idx_bot, 0]),
                                np.min(coords[idx_bot, 1]), np.max(coords[idx_bot, 1]))
                    dx_overlap = max(0.0, min(bbox_top[1], bbox_bot[1]) - max(bbox_top[0], bbox_bot[0]))
                    dy_overlap = max(0.0, min(bbox_top[3], bbox_bot[3]) - max(bbox_top[2], bbox_bot[2]))
                    area_overlap = dx_overlap * dy_overlap
                    area_top = max(1e-4, (bbox_top[1] - bbox_top[0]) * (bbox_top[3] - bbox_top[2]))
                    area_bot = max(1e-4, (bbox_bot[1] - bbox_bot[0]) * (bbox_bot[3] - bbox_bot[2]))
                    area_union = area_top + area_bot - area_overlap
                    iou = area_overlap / max(1e-4, area_union)
                    area_ratio = min(area_top, area_bot) / max(area_top, area_bot)
                    if iou > 0.45 and area_ratio >= 0.50:
                        return True
        except Exception:
            return False
        return False

    def _is_two_surfaces(self) -> bool:
        """Проверяет, является ли проект расчетом по двум независимым поверхностям (Верх и Низ)."""
        if getattr(self, "__dict__", {}).get("_is_separate_surfaces", False):
            return True
        if not self.points:
            return False
        n_top = sum(1 for p in self.points if p.surface_type == "top")
        n_bot = sum(1 for p in self.points if p.surface_type == "bottom")
        has_bound_pts = any(p.surface_type == "boundary" for p in self.points)
        if n_top >= 3 and n_bot >= 3 and not has_bound_pts:
            return True
        return False

    def _invalidate_boundary_cache(self):
        """Сбрасывает кешированный интерполятор контура и классификацию поверхностей при любых изменениях геометрии"""
        self._boundary_interpolator_cache = None
        self._boundary_path_cache = None
        self._local_surface_threshold = None
        self._split_height_cache = None
        self._work_type_cache = None
        self._point_surfaces_cache = None
        self._boundary_inside_cache = None
        self._boundary_indices_set = None
        self._2d_visible_mask_cache = None

    def _get_boundary_path(self):
        """Возвращает кешированный mpl_path.Path полигона контура сшивания."""
        cached = getattr(self, "_boundary_path_cache", None)
        if cached is not None:
            return cached
        if not self.points or len(self.boundary_indices) < 3:
            self._boundary_path_cache = None
            return None
        valid_indices = [i for i in self.boundary_indices if 0 <= i < len(self.points)]
        if len(valid_indices) < 3:
            self._boundary_path_cache = None
            return None
        try:
            coords = np.array([[self.points[i].x, self.points[i].y] for i in valid_indices], dtype=float)
            path = MplPath(coords)
            self._boundary_path_cache = path
            return path
        except Exception:
            self._boundary_path_cache = None
            return None

    def _ensure_boundary_inside_mask(self) -> np.ndarray:
        """Возвращает булев массив размера len(self.points), где True если точка внутри контура сшивания."""
        cached = getattr(self, "_boundary_inside_cache", None)
        if cached is not None and len(cached) == len(self.points):
            return cached
        if not self.points:
            self._boundary_inside_cache = np.array([], dtype=bool)
            return self._boundary_inside_cache

        bound_path = self._get_boundary_path()
        if bound_path is None:
            mask = np.ones(len(self.points), dtype=bool)
        else:
            coords = np.array([[p.x, p.y] for p in self.points], dtype=float)
            mask = bound_path.contains_points(coords, radius=1e-5)
        self._boundary_inside_cache = mask
        return mask

    def _ensure_point_surfaces(self) -> list[str]:
        """Возвращает кешированный список поверхностей ('top', 'bottom', 'boundary') для всех точек."""
        cached = getattr(self, "_point_surfaces_cache", None)
        if cached is not None and len(cached) == len(self.points):
            return cached
        if not self.points:
            self._point_surfaces_cache = []
            return self._point_surfaces_cache

        mean_bound, split_h, work_type = self._get_split_height()
        bound_set = getattr(self, "_boundary_indices_set", None)
        if bound_set is None:
            bound_set = set(self.boundary_indices)
            self._boundary_indices_set = bound_set

        is_two = self._is_two_surfaces()
        has_contour = len(self.boundary_indices) >= 3

        interp = None
        thresh = 0.15
        inside_mask = None
        if not is_two and has_contour:
            interp = self._get_boundary_interpolator()
            thresh = getattr(self, "_local_surface_threshold", None)
            if thresh is None:
                thresh = self._compute_local_surface_threshold(interp)
                self._local_surface_threshold = thresh
            inside_mask = self._ensure_boundary_inside_mask()

        surfaces = []
        for i, p in enumerate(self.points):
            if is_two:
                if p.surface_type in ("top", "bottom"):
                    surfaces.append(p.surface_type)
                else:
                    surfaces.append("top" if p.h > split_h else "bottom")
            elif p.surface_type == "top":
                surfaces.append("top")
            elif p.surface_type == "bottom":
                surfaces.append("bottom")
            elif i in bound_set or p.surface_type == "boundary":
                surfaces.append("boundary")
            elif not has_contour:
                surfaces.append("top" if p.h > split_h else "bottom")
            else:
                is_inside = bool(inside_mask[i]) if (inside_mask is not None and i < len(inside_mask)) else False
                if is_inside:
                    z_bound_local = interp(p.x, p.y)
                    if work_type == "cut":
                        local_depth = z_bound_local - p.h
                        surfaces.append("bottom" if local_depth > thresh else "top")
                    else:
                        local_height = p.h - z_bound_local
                        surfaces.append("top" if local_height > thresh else "bottom")
                else:
                    # Точка за пределами контура: для насыпи окружающая земля — bottom, для выемки — top
                    surfaces.append("top" if work_type == "cut" else "bottom")

        self._point_surfaces_cache = surfaces
        return surfaces

    def _ensure_2d_visible_mask(self) -> np.ndarray:
        """Возвращает кешированный векторизованный булев массив размера len(self.points) видимости точек на 2D плане."""
        cached = self.__dict__.get("_2d_visible_mask_cache", None)
        if cached is not None and isinstance(cached, np.ndarray) and len(cached) == len(self.points):
            return cached
        if not self.points:
            self._2d_visible_mask_cache = np.array([], dtype=bool)
            return self._2d_visible_mask_cache

        show_top = self._show_top.get() if hasattr(self, "_show_top") else True
        show_bottom = self._show_bottom.get() if hasattr(self, "_show_bottom") else True
        if not show_top and not show_bottom:
            self._2d_visible_mask_cache = np.zeros(len(self.points), dtype=bool)
            return self._2d_visible_mask_cache

        surfaces = self._ensure_point_surfaces()
        n = len(self.points)
        mask = np.zeros(n, dtype=bool)

        is_two = self._is_two_surfaces()
        mean_bound, split_h, work_type = self._get_split_height()
        is_cut = (work_type == "cut")

        bound_path = self._get_boundary_path() if not is_two and len(self.boundary_indices) >= 3 else None
        inside_mask = self._ensure_boundary_inside_mask() if bound_path is not None else None

        for i in range(n):
            pt_surf = surfaces[i] if i < len(surfaces) else "bottom"
            if is_two:
                mask[i] = show_top if pt_surf == "top" else show_bottom
                continue

            if pt_surf == "boundary":
                mask[i] = show_top if is_cut else show_bottom
                continue

            if inside_mask is not None:
                is_inside = bool(inside_mask[i]) if i < len(inside_mask) else True
                if not is_inside:
                    if not show_top:
                        mask[i] = False
                    elif not show_bottom and not is_cut:
                        mask[i] = False
                    else:
                        mask[i] = True
                    continue

            mask[i] = show_top if pt_surf == "top" else show_bottom

        self._2d_visible_mask_cache = mask
        return mask

    def _is_point_visible_on_2d(self, i: int) -> bool:
        """Определяет, должна ли точка отображаться на 2D-схеме с учётом фильтра поверхностей (быстро O(1))."""
        if not self.points or not (0 <= i < len(self.points)):
            return False
        mask = self._ensure_2d_visible_mask()
        if i < len(mask):
            return bool(mask[i])
        return False

    def _get_boundary_interpolator(self):
        """Возвращает функцию z_bound(x, y), интерполирующую отметку бровки контура сшивания.
        Использует 2D триангуляцию Делоне вершин контура с барицентрической интерполяцией,
        а для точек вне выпуклой оболочки вершин контура — IDW от 3 ближайших точек контура.
        """
        cached = getattr(self, "_boundary_interpolator_cache", None)
        if cached is not None:
            return cached

        if not self.points or len(self.boundary_indices) < 3:
            bound_h = [self.points[i].h for i in self.boundary_indices if 0 <= i < len(self.points)]
            mean_val = float(np.mean(bound_h)) if bound_h else 0.0
            def fallback_interp(qx, qy):
                return mean_val
            self._boundary_interpolator_cache = fallback_interp
            return fallback_interp

        valid_indices = [i for i in self.boundary_indices if 0 <= i < len(self.points)]
        bound_xy = np.array([[self.points[i].x, self.points[i].y] for i in valid_indices], dtype=float)
        bound_h = np.array([self.points[i].h for i in valid_indices], dtype=float)

        tri = None
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(bound_xy)
        except Exception:
            tri = None

        def interp_fn(qx: float, qy: float) -> float:
            q = np.array([qx, qy], dtype=float)
            if tri is not None:
                s = tri.find_simplex([q])[0]
                if s >= 0:
                    b = tri.transform[s, :2].dot(q - tri.transform[s, 2])
                    w = np.array([b[0], b[1], 1.0 - b.sum()])
                    v_idx = tri.simplices[s]
                    return float(np.dot(w, bound_h[v_idx]))

            # Fallback: IDW от 3 ближайших вершин контура
            dists = np.hypot(bound_xy[:, 0] - qx, bound_xy[:, 1] - qy)
            min_d = np.min(dists)
            if min_d < 1e-6:
                return float(bound_h[np.argmin(dists)])

            k = min(3, len(bound_xy))
            k_idx = np.argsort(dists)[:k]
            weights = 1.0 / np.maximum(dists[k_idx] ** 2, 1e-12)
            return float(np.sum(weights * bound_h[k_idx]) / np.sum(weights))

        self._boundary_interpolator_cache = interp_fn
        return interp_fn

    def _compute_local_surface_threshold(self, interp=None) -> float:
        """Вычисляет адаптивный порог заглубления/возвышения для классификации рельефа"""
        if interp is None:
            interp = self._get_boundary_interpolator()
        if not self.points or len(self.boundary_indices) < 3:
            return 0.15

        bound_set = getattr(self, "_boundary_indices_set", None)
        if bound_set is None:
            bound_set = set(self.boundary_indices)
            self._boundary_indices_set = bound_set

        inside_mask = self._ensure_boundary_inside_mask()

        diffs = []
        for i, p in enumerate(self.points):
            if i in bound_set or p.surface_type in ("boundary", "top", "bottom"):
                continue
            if not (bool(inside_mask[i]) if i < len(inside_mask) else False):
                continue
            z_loc = interp(p.x, p.y)
            diffs.append(abs(p.h - z_loc))

        # Fallback если внутри контура нет подходящих точек
        if not diffs:
            for i, p in enumerate(self.points):
                if i in bound_set or p.surface_type in ("boundary", "top", "bottom"):
                    continue
                z_loc = interp(p.x, p.y)
                diffs.append(abs(p.h - z_loc))

        if not diffs:
            return 0.15

        max_diff = float(np.max(diffs))
        return float(np.clip(0.25 * max_diff, 0.08, 0.20))

    def _detect_work_type_from_points(self) -> str:
        """Определяет тип земляных работ ('cut' или 'fill') по локальным разностям рельефа относительно контура."""
        cached = getattr(self, "_work_type_cache", None)
        if cached is not None:
            return cached

        if not self.points:
            return "cut"

        if self._is_two_surfaces():
            top_h = [p.h for p in self.points if p.surface_type == "top"]
            bot_h = [p.h for p in self.points if p.surface_type == "bottom"]
            mean_top = float(np.mean(top_h)) if top_h else 0.0
            mean_bot = float(np.mean(bot_h)) if bot_h else 0.0
            res = "fill" if mean_top >= mean_bot else "cut"
            self._work_type_cache = res
            return res

        if len(self.boundary_indices) < 3:
            return "cut"

        bound_set = getattr(self, "_boundary_indices_set", None)
        if bound_set is None:
            bound_set = set(self.boundary_indices)
            self._boundary_indices_set = bound_set

        interp = self._get_boundary_interpolator()
        inside_mask = self._ensure_boundary_inside_mask()

        # При наличии контура тип работ определяется строго по точкам внутри контура сшивания!
        diffs = []
        for i, p in enumerate(self.points):
            if i in bound_set or p.surface_type == "boundary":
                continue
            if not (bool(inside_mask[i]) if i < len(inside_mask) else False):
                continue
            z_loc = interp(p.x, p.y)
            diffs.append(p.h - z_loc)

        # Fallback: если строго внутри контура точек нет, используем все точки не на границе
        if not diffs:
            for i, p in enumerate(self.points):
                if i in bound_set or p.surface_type == "boundary":
                    continue
                z_loc = interp(p.x, p.y)
                diffs.append(p.h - z_loc)

        if not diffs:
            res = "cut"
        else:
            mean_diff = float(np.mean(diffs))
            if mean_diff > 1e-4:
                res = "fill"
            elif mean_diff < -1e-4:
                res = "cut"
            else:
                above = sum(1 for d in diffs if d > 0)
                below = sum(1 for d in diffs if d < 0)
                res = "fill" if above >= below else "cut"

        self._work_type_cache = res
        return res

    def _get_split_height(self) -> tuple[float, float, str]:
        """Вычисляет среднюю высоту контура, разделяющую высоту и тип земляных работ.
        Returns: (mean_bound, split_h, work_type)
        """
        cached = getattr(self, "_split_height_cache", None)
        if cached is not None:
            return cached

        if not self.points:
            return 0.0, 0.0, "cut"

        work_type = self._detect_work_type_from_points()

        if self._is_two_surfaces():
            top_h = [p.h for p in self.points if p.surface_type == "top"]
            bot_h = [p.h for p in self.points if p.surface_type == "bottom"]
            mean_top = float(np.mean(top_h)) if top_h else 0.0
            mean_bot = float(np.mean(bot_h)) if bot_h else 0.0
            split_h = (mean_top + mean_bot) / 2.0
            bound_h = [self.points[i].h for i in self.boundary_indices if 0 <= i < len(self.points)]
            mean_bound = float(np.mean(bound_h)) if bound_h else split_h
            res = (mean_bound, split_h, work_type)
            self._split_height_cache = res
            return res

        if len(self.boundary_indices) < 3:
            res = (0.0, 0.0, work_type)
            self._split_height_cache = res
            return res

        bound_h = [self.points[i].h for i in self.boundary_indices if 0 <= i < len(self.points)]
        mean_bound = float(np.mean(bound_h)) if bound_h else 0.0

        bound_set = getattr(self, "_boundary_indices_set", None)
        if bound_set is None:
            bound_set = set(self.boundary_indices)
            self._boundary_indices_set = bound_set

        inside_mask = self._ensure_boundary_inside_mask()
        inside_pts = [p for i, p in enumerate(self.points) if i not in bound_set and (bool(inside_mask[i]) if i < len(inside_mask) else False)]
        if not inside_pts:
            inside_pts = [p for i, p in enumerate(self.points) if i not in bound_set]

        if not inside_pts:
            res = (mean_bound, mean_bound, work_type)
            self._split_height_cache = res
            return res

        inside_h = [p.h for p in inside_pts]
        mean_inside = float(np.mean(inside_h))
        split_h = (mean_bound + mean_inside) / 2.0
        res = (mean_bound, split_h, work_type)
        self._split_height_cache = res
        return res

    def _get_point_surface(self, p, mean_bound: float = None, split_h: float = None, work_type: str = None) -> str:
        """Возвращает фактическую принадлежность точки к поверхности: 'top', 'bottom' или 'boundary'.
        Приоритет:
        1. Ручное назначение ('top' или 'bottom') — абсолютный безусловный приоритет.
        2. Принадлежность к контуру сшивания ('boundary').
        3. Авто-определение ('auto') по локальной интерполяции отметки контура Z_bound(x, y).
        """
        if p.surface_type == "top":
            return "top"
        if p.surface_type == "bottom":
            return "bottom"
        if p.surface_type == "boundary":
            return "boundary"

        # Проверка вхождения в контур сшивания
        bound_set = getattr(self, "_boundary_indices_set", None)
        if bound_set is None:
            bound_set = set(self.boundary_indices)
            self._boundary_indices_set = bound_set

        if hasattr(p, 'idx') and p.idx in bound_set:
            return "boundary"
        for idx in self.boundary_indices:
            if 0 <= idx < len(self.points) and self.points[idx] is p:
                return "boundary"

        # В режиме двух независимых поверхностей или если контур ещё не построен (< 3 точек)
        if self._is_two_surfaces() or len(self.boundary_indices) < 3:
            if split_h is None or work_type is None:
                mean_bound, split_h, work_type = self._get_split_height()
            return "top" if p.h > split_h else "bottom"

        if work_type is None:
            work_type = self._detect_work_type_from_points()

        bound_path = self._get_boundary_path()
        is_inside = bound_path.contains_point((p.x, p.y), radius=1e-5) if bound_path is not None else True
        if not is_inside:
            return "top" if work_type == "cut" else "bottom"

        interp = self._get_boundary_interpolator()
        z_bound_local = interp(p.x, p.y)

        thresh = getattr(self, "_local_surface_threshold", None)
        if thresh is None:
            thresh = self._compute_local_surface_threshold(interp)
            self._local_surface_threshold = thresh

        if work_type == "cut":
            local_depth = z_bound_local - p.h
            return "bottom" if local_depth > thresh else "top"
        else:
            local_height = p.h - z_bound_local
            return "top" if local_height > thresh else "bottom"

    def _is_excavation(self) -> bool:
        """Определяет, является ли текущий рельеф выемкой (True) или насыпью (False)."""
        if self.points and len(self.boundary_indices) >= 3:
            return self._detect_work_type_from_points() == "cut"

        if self.calc_results:
            r = self.calc_results
            work_type = r.get("work_type")
            if work_type == "cut":
                return True
            if work_type == "fill":
                return False
            v_cut = r.get("v_cut", 0.0)
            v_fill = r.get("v_fill", 0.0)
            if v_cut > 0 and v_cut > v_fill:
                return True
            if v_fill > 0 and v_fill > v_cut:
                return False

        return False

    def _redraw_2d(self, reset_view=False):
        old_xlim = None
        old_ylim = None
        if not reset_view:
            try:
                cur_x = self.ax_2d.get_xlim()
                cur_y = self.ax_2d.get_ylim()
                if cur_x != (0.0, 1.0) or cur_y != (0.0, 1.0):
                    old_xlim = cur_x
                    old_ylim = cur_y
            except Exception:
                pass

        self.ax_2d.clear()
        is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False
        title_color = "#e0e0e0" if is_dark else "#212529"
        self.ax_2d.set_xlabel("Восток Y (м)", color=title_color)
        self.ax_2d.set_ylabel("Север X (м)", color=title_color)
        self._apply_axes_theme(self.ax_2d, self.fig_2d, is_dark)
        self.ax_2d.grid(True, linestyle="--", alpha=0.5)
        self.ax_2d.format_coord = self._format_coord_display
        self.ax_2d.xaxis.set_major_formatter(PlainOffsetFormatter(useOffset=True))
        self.ax_2d.yaxis.set_major_formatter(PlainOffsetFormatter(useOffset=True))

        self._point_annotations = []
        self._clear_tk_canvas_labels()
        self._current_labeled_points = []
        self._contour_line_artist = None
        self._contour_fill_artist = None
        self._contour_order_artists = []
        self._selected_scatter_artist = None
        self._scatter_top_artist = None
        self._scatter_bot_artist = None
        self._scatter_bound_artist = None
        self._blit_bg = None
        self._box_select_rect_artist = None
        self._contour_drag_line1 = None
        self._contour_drag_line2 = None
        self._contour_drag_guide = None
        self._contour_drag_ring = None

        if not self.points:
            self.canvas_2d.draw()
            self._position_reset_contour_button()
            return

        show_top    = self._show_top.get()
        show_bottom = self._show_bottom.get()
        mean_bound, split_h, work_type = self._get_split_height()
        is_cut = (work_type == "cut")
        should_label = False
        pts_to_label = []

        # 1. Линия контура и полигон заливки
        if len(self.boundary_indices) >= 2:
            bx = [self.points[i].x for i in self.boundary_indices if 0 <= i < len(self.points)]
            by = [self.points[i].y for i in self.boundary_indices if 0 <= i < len(self.points)]
            if len(self.boundary_indices) >= 3:
                bx_c = bx + [bx[0]]
                by_c = by + [by[0]]
            else:
                bx_c = bx
                by_c = by
            is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False
            contour_col = "#00e676" if is_dark else "#27ae60"
            fill_col = "#00e676" if is_dark else "#2ecc71"
            line, = self.ax_2d.plot(by_c, bx_c, "-", color=contour_col, linewidth=2.5, label="Контур сшивания", zorder=3)
            self._contour_line_artist = line
            if len(self.boundary_indices) >= 3:
                poly = self.ax_2d.fill(by_c, bx_c, color=fill_col, alpha=0.15)[0]
                self._contour_fill_artist = poly
        else:
            is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False
            contour_col = "#00e676" if is_dark else "#27ae60"
            line, = self.ax_2d.plot([], [], "-", color=contour_col, linewidth=2.5, label="Контур сшивания", zorder=3)
            self._contour_line_artist = line
            self._contour_fill_artist = None

        # 2. Номера порядка обхода контура (#1, #2, ...)
        show_bound_labels = (show_top and show_bottom) or (is_cut and show_top) or (not is_cut and show_bottom)
        if show_bound_labels and len(self.boundary_indices) >= 2:
            is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False
            order_col = "#00e676" if is_dark else "#004d00"
            for order, idx in enumerate(self.boundary_indices):
                if 0 <= idx < len(self.points):
                    pt = self.points[idx]
                    ann = self.ax_2d.annotate(
                        f"#{order+1}", (pt.y, pt.x),
                        textcoords="offset points", xytext=(-12, -12),
                        fontsize=9, fontweight="bold", color=order_col,
                        zorder=5, clip_on=True
                    )
                    self._contour_order_artists.append(ann)
                    self._point_annotations.append((ann, (pt.y, pt.x), "bound"))

        # 3. Векторизованная отрисовка точек
        if show_top or show_bottom:
            is_two_surfs = self._is_two_surfaces()
            is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False
            top_pts, bot_pts, bound_pts = [], [], []

            surfaces = self._ensure_point_surfaces()
            for i, p in enumerate(self.points):
                if not self._is_point_visible_on_2d(i):
                    continue

                pt_surf = surfaces[i] if i < len(surfaces) else "bottom"
                if is_two_surfs:
                    if pt_surf == "top":
                        top_pts.append(p)
                    else:
                        bot_pts.append(p)
                else:
                    if pt_surf == "boundary":
                        bound_pts.append(p)
                    elif pt_surf == "top":
                        top_pts.append(p)
                    else:
                        bot_pts.append(p)

            # Яркие, контрастные цвета для темной темы и классические для светлой
            top_color = "#00e5ff" if is_dark else "#1565c0"
            bot_color = "#ff9100" if is_dark else "#d35400"
            bound_color = "#00e676" if is_dark else "#1565c0"
            edge_top = "#ffffff" if is_dark else "#0d3b7a"
            edge_bot = "#ffffff" if is_dark else "#a04000"
            edge_bnd = "#ffffff" if is_dark else "#0d3b7a"

            if top_pts:
                self._scatter_top_artist = self.ax_2d.scatter(
                    [p.y for p in top_pts], [p.x for p in top_pts],
                    color=top_color, marker="^", s=18 if is_dark else 14,
                    edgecolors=edge_top, linewidth=0.6 if is_dark else 0.5, zorder=4,
                    label="Верхняя" if is_two_surfs else "_nolegend_"
                )
            if bot_pts:
                self._scatter_bot_artist = self.ax_2d.scatter(
                    [p.y for p in bot_pts], [p.x for p in bot_pts],
                    color=bot_color, marker="v", s=18 if is_dark else 14,
                    edgecolors=edge_bot, linewidth=0.6 if is_dark else 0.5, zorder=4,
                    label="Нижняя" if is_two_surfs else "_nolegend_"
                )
            if bound_pts:
                self._scatter_bound_artist = self.ax_2d.scatter(
                    [p.y for p in bound_pts], [p.x for p in bound_pts],
                    color=bound_color, marker="s", s=22 if is_dark else 18,
                    edgecolors=edge_bnd, linewidth=0.6 if is_dark else 0.5, zorder=4,
                    label="_nolegend_"
                )

            # Аннотации точек: режим Вкл/Выкл
            all_pts_to_label = top_pts + bot_pts + bound_pts
            mode = getattr(self, "_point_labels_mode", "on")
            should_label = (mode == "on")
            if should_label:
                if old_xlim is not None and old_ylim is not None:
                    y_min, y_max = min(old_xlim), max(old_xlim)
                    x_min, x_max = min(old_ylim), max(old_ylim)
                    pts_to_label = [p for p in all_pts_to_label if y_min <= p.y <= y_max and x_min <= p.x <= x_max]
                else:
                    pts_to_label = all_pts_to_label
            else:
                pts_to_label = []

            if should_label:
                if not self._has_tk_canvas_widget():
                    text_col = self._get_canvas_text_color()
                    for p in pts_to_label:
                        ann = self.ax_2d.annotate(
                            f"{p.id}\nH:{p.h:.3f}", (p.y, p.x),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=9.0, alpha=0.9, zorder=5, clip_on=True,
                            color=text_col
                        )
                        self._point_annotations.append((ann, (p.y, p.x), "point"))

        # 4. Подсветка выделенных точек рамкой (Box Selection)
        sel_ys, sel_xs = [], []
        if getattr(self, "_selected_points", None):
            sel_ys = [self.points[i].y for i in self._selected_points if 0 <= i < len(self.points)]
            sel_xs = [self.points[i].x for i in self._selected_points if 0 <= i < len(self.points)]

        is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False
        self._selected_scatter_artist = self.ax_2d.scatter(
            sel_ys, sel_xs,
            s=140,
            facecolors="none",
            edgecolors="#ffd700" if is_dark else "#f39c12",
            linewidth=2.5,
            zorder=12,
            label="_nolegend_"
        )

        self.ax_2d.set_aspect("equal", adjustable="datalim")
        leg = self.ax_2d.legend(loc="upper right", fontsize=8)
        if leg is not None:
            is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False
            ax_bg = "#1e2124" if is_dark else "#ffffff"
            spine_col = "#495057" if is_dark else "#cccccc"
            fg_col = "#e0e0e0" if is_dark else "#212529"
            try:
                leg.get_frame().set_facecolor(ax_bg)
                leg.get_frame().set_edgecolor(spine_col)
                for t in leg.get_texts():
                    t.set_color(fg_col)
            except Exception:
                pass

        if old_xlim is not None and old_ylim is not None:
            self.ax_2d.set_xlim(old_xlim)
            self.ax_2d.set_ylim(old_ylim)
        else:
            self.ax_2d.autoscale(True)
            self._apply_fig_layout(self.fig_2d, pad=0.5)

        self._ensure_box_select_artist()
        self._ensure_contour_drag_artists()
        self._update_2d_annotations_visibility()
        self.canvas_2d.draw()
        if self._has_tk_canvas_widget():
            if should_label:
                self._render_tk_canvas_labels(pts_to_label)
            else:
                self._clear_tk_canvas_labels()
                self._current_labeled_points = []
        self._position_reset_contour_button()
        self._update_selection_bar()

    def _update_2d_contour_only(self):
        """Быстрое дифференциальное обновление контура сшивания без ax.clear() и без пересоздания точек."""
        if not hasattr(self, "ax_2d") or self.ax_2d is None:
            return
        if getattr(self, "_contour_line_artist", None) is None:
            self._redraw_2d()
            return

        # 1. Линия контура
        if len(self.boundary_indices) >= 2:
            bx = [self.points[i].x for i in self.boundary_indices if 0 <= i < len(self.points)]
            by = [self.points[i].y for i in self.boundary_indices if 0 <= i < len(self.points)]
            if len(self.boundary_indices) >= 3:
                bx_c = bx + [bx[0]]
                by_c = by + [by[0]]
            else:
                bx_c = bx
                by_c = by
            self._contour_line_artist.set_data(by_c, bx_c)

            # 2. Заливка контура
            if self._contour_fill_artist is not None:
                try:
                    self._contour_fill_artist.remove()
                except Exception:
                    pass
                self._contour_fill_artist = None

            if len(self.boundary_indices) >= 3:
                poly = self.ax_2d.fill(by_c, bx_c, color="#2ecc71", alpha=0.15)[0]
                self._contour_fill_artist = poly
        else:
            self._contour_line_artist.set_data([], [])
            if self._contour_fill_artist is not None:
                try:
                    self._contour_fill_artist.remove()
                except Exception:
                    pass
                self._contour_fill_artist = None

        # 3. Номера обхода контура (#1, #2, ...)
        for ann in getattr(self, "_contour_order_artists", []):
            try:
                ann.remove()
            except Exception:
                pass
        self._contour_order_artists = []

        show_top = self._show_top.get() if hasattr(self, "_show_top") else True
        show_bottom = self._show_bottom.get() if hasattr(self, "_show_bottom") else True
        mean_bound, split_h, work_type = self._get_split_height()
        is_cut = (work_type == "cut")
        show_bound_labels = (show_top and show_bottom) or (is_cut and show_top) or (not is_cut and show_bottom)

        if show_bound_labels and len(self.boundary_indices) >= 2:
            is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False
            order_col = "#00e676" if is_dark else "#004d00"
            for order, idx in enumerate(self.boundary_indices):
                if 0 <= idx < len(self.points):
                    pt = self.points[idx]
                    ann = self.ax_2d.annotate(
                        f"#{order+1}", (pt.y, pt.x),
                        textcoords="offset points", xytext=(-12, -12),
                        fontsize=9, fontweight="bold", color=order_col,
                        zorder=5, clip_on=True
                    )
                    self._contour_order_artists.append(ann)

        self.canvas_2d.draw_idle()
        self._position_reset_contour_button()
        self._update_selection_bar()

    def _update_2d_selection_only(self):
        """Быстрое обновление подсветки выделенных точек без перерисовки всей схемы."""
        if not hasattr(self, "ax_2d") or self.ax_2d is None:
            return
        if getattr(self, "_selected_scatter_artist", None) is None:
            self._redraw_2d()
            return

        if getattr(self, "_selected_points", None):
            coords = np.array([[self.points[i].y, self.points[i].x]
                               for i in self._selected_points if 0 <= i < len(self.points)])
            if len(coords) > 0:
                self._selected_scatter_artist.set_offsets(coords)
            else:
                self._selected_scatter_artist.set_offsets(np.empty((0, 2)))
        else:
            self._selected_scatter_artist.set_offsets(np.empty((0, 2)))

        self.canvas_2d.draw_idle()
        self._update_selection_bar()

    def _on_surface_vis_toggle(self):
        """Вызывается при изменении чекбоксов видимости поверхностей — перерисовывает активную вкладку и помечает остальные dirty."""
        self._2d_visible_mask_cache = None
        current_tab = self.tabview.get() if hasattr(self, "tabview") else None
        if current_tab == self.TAB_2D:
            self._redraw_2d()
            self._3d_dirty = True
            self._diff_dirty = True
            self._tin_dirty = True
            self._contours_dirty = True
        elif current_tab == self.TAB_3D:
            if self.calc_results:
                self._redraw_3d(self.calc_results)
                self._3d_dirty = False
            self._2d_dirty = True
            self._diff_dirty = True
            self._tin_dirty = True
            self._contours_dirty = True
        elif current_tab == self.TAB_DIFF:
            if self.calc_results:
                self._redraw_diff(self.calc_results)
                self._diff_dirty = False
            self._2d_dirty = True
            self._3d_dirty = True
            self._tin_dirty = True
            self._contours_dirty = True
        elif current_tab == self.TAB_TIN:
            self._redraw_tin()
            self._tin_dirty = False
            self._2d_dirty = True
            self._3d_dirty = True
            self._diff_dirty = True
            self._contours_dirty = True
        elif current_tab == getattr(self, "TAB_CONTOURS", "Горизонтали"):
            self._redraw_contours()
            self._contours_dirty = False
            self._2d_dirty = True
            self._3d_dirty = True
            self._diff_dirty = True
            self._tin_dirty = True
        else:
            self._redraw_2d()
            self._3d_dirty = True
            self._diff_dirty = True
            self._tin_dirty = True
            self._contours_dirty = True
        self._lift_canvas_overlays()

    def _redraw_3d(self, r: dict):
        if self.canvas_3d is None:
            return

        show_top    = self._show_top.get()
        show_bottom = self._show_bottom.get()

        self.ax_3d.clear()
        is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False
        title_color = "#e0e0e0" if is_dark else "#212529"
        self.ax_3d.set_xlabel("Y (Восток, м)", color=title_color)
        self.ax_3d.set_ylabel("X (Север, м)", color=title_color)
        self.ax_3d.set_zlabel("H (Высота, м)", color=title_color)
        self._apply_axes_theme(self.ax_3d, self.fig_3d, is_dark)

        # Верхняя (Upper) — физически верхняя поверхность z_top_g.
        # Нижняя (Lower) — физически нижняя поверхность z_bot_g.
        z_top_g = r.get("z_top_grid")
        z_bot_g = r.get("z_bot_grid")
        if z_top_g is None or z_bot_g is None:
            self.canvas_3d.draw()
            return

        is_cut = self._is_excavation()
        grid_upper, grid_lower = z_top_g, z_bot_g

        is_two_surfs = self._is_two_surfaces()
        has_tin = (not is_two_surfs and r.get("custom_tin") and r.get("pts_3d") is not None
                   and r.get("active_simplices") is not None
                   and len(r["active_simplices"]) > 0)

        pts_3d = r.get("pts_3d")
        active_s = r.get("active_simplices")
        r_step, c_step = calc_3d_stride(grid_upper if grid_upper is not None else grid_lower, target_dim=45)

        if show_top and show_bottom:
            if not is_cut:
                # Насыпь: верхняя поверхность — тело насыпи (треугольниками TIN!),
                # нижняя — основание (рельеф)
                if has_tin:
                    self.ax_3d.plot_trisurf(pts_3d[:, 1], pts_3d[:, 0], pts_3d[:, 2],
                                            triangles=active_s, cmap="copper", alpha=0.85,
                                            edgecolor="#4a2e12", linewidth=0.5)
                else:
                    self.ax_3d.plot_surface(r["grid_y"], r["grid_x"], grid_upper,
                                            cmap="copper", alpha=0.75, edgecolor="none", rstride=r_step, cstride=c_step)
                self.ax_3d.plot_surface(r["grid_y"], r["grid_x"], grid_lower,
                                        cmap="Blues_r", alpha=0.50, edgecolor="none", rstride=r_step, cstride=c_step)
            else:
                # Выемка: верхняя поверхность — дневная земля,
                # нижняя — дно котлована (треугольниками TIN!)
                self.ax_3d.plot_surface(r["grid_y"], r["grid_x"], grid_upper,
                                        cmap="copper", alpha=0.50, edgecolor="none", rstride=r_step, cstride=c_step)
                if has_tin:
                    self.ax_3d.plot_trisurf(pts_3d[:, 1], pts_3d[:, 0], pts_3d[:, 2],
                                            triangles=active_s, cmap="Blues_r", alpha=0.85,
                                            edgecolor="#154360", linewidth=0.5)
                else:
                    self.ax_3d.plot_surface(r["grid_y"], r["grid_x"], grid_lower,
                                        cmap="Blues_r", alpha=0.75, edgecolor="none", rstride=r_step, cstride=c_step)
        elif show_top:
            # Только верхняя: в насыпи верхняя поверхность построена по TIN
            if has_tin and not is_cut:
                self.ax_3d.plot_trisurf(pts_3d[:, 1], pts_3d[:, 0], pts_3d[:, 2],
                                        triangles=active_s, cmap="copper", alpha=0.85,
                                        edgecolor="#4a2e12", linewidth=0.5)
            else:
                self.ax_3d.plot_surface(r["grid_y"], r["grid_x"], grid_upper,
                                        cmap="copper", alpha=0.85, edgecolor="none", rstride=r_step, cstride=c_step)
        elif show_bottom:
            # Только нижняя: в выемке нижняя поверхность (котлован) построена по TIN
            if has_tin and is_cut:
                self.ax_3d.plot_trisurf(pts_3d[:, 1], pts_3d[:, 0], pts_3d[:, 2],
                                        triangles=active_s, cmap="Blues_r", alpha=0.85,
                                        edgecolor="#154360", linewidth=0.5)
            else:
                self.ax_3d.plot_surface(r["grid_y"], r["grid_x"], grid_lower,
                                        cmap="Blues_r", alpha=0.85, edgecolor="none", rstride=r_step, cstride=c_step)

        if not is_two_surfs and len(self.boundary_indices) >= 3:
            b_pts = r["boundary"]
            self.ax_3d.plot(list(b_pts[:, 1]) + [b_pts[0, 1]],
                            list(b_pts[:, 0]) + [b_pts[0, 0]],
                            list(b_pts[:, 2]) + [b_pts[0, 2]],
                            "g-", linewidth=3, label="Контур сшивания")

        # Явно задаём все три оси по фактическому диапазону данных с небольшим отступом.
        # Без этого matplotlib 3D может растянуть оси в 5× раз относительно данных.
        grid_y = r.get("grid_y")
        grid_x = r.get("grid_x")
        all_z = np.concatenate([z_top_g[~np.isnan(z_top_g)].ravel(),
                                 z_bot_g[~np.isnan(z_bot_g)].ravel()])
        if grid_y is not None and grid_x is not None and len(all_z) > 0:
            y_min, y_max = float(np.nanmin(grid_y)), float(np.nanmax(grid_y))
            x_min, x_max = float(np.nanmin(grid_x)), float(np.nanmax(grid_x))
            z_min, z_max = float(np.min(all_z)),     float(np.max(all_z))

            def _pad(lo, hi, pct):
                d = (hi - lo) * pct
                return lo - d, hi + d

            self.ax_3d.set_xlim(*_pad(y_min, y_max, 0.05))
            self.ax_3d.set_ylim(*_pad(x_min, x_max, 0.05))
            self.ax_3d.set_zlim(*_pad(z_min, z_max, 0.15))

        self._apply_fig_layout(self.fig_3d, pad=0.5)
        self.canvas_3d.draw()

    def _open_fullscreen_3d(self):
        """Открывает отдельное полноэкранное окно для детального и красивого изучения 3D рельефа"""
        if not self.calc_results:
            if self.points and len(self.boundary_indices) >= 3:
                self.calculate_volume()
            else:
                messagebox.showinfo("Информация", "Сначала загрузите координаты и задайте контур для построения 3D модели.")
                return
        if not self.calc_results:
            return
        FullScreen3DViewer(self, self.calc_results, self.points, self.boundary_indices)

    def _reset_3d_view(self):
        """Сбрасывает ракурс встроенного 3D графика к виду по умолчанию"""
        self.ax_3d.view_init(elev=35, azim=-60)
        self.canvas_3d.draw_idle()

    def _top_3d_view(self):
        """Переключает встроенный 3D график на вид сверху (план)"""
        self.ax_3d.view_init(elev=90, azim=-90)
        self.canvas_3d.draw_idle()

    def _redraw_diff(self, r: dict):
        if self.canvas_diff is None:
            return

        show_top    = self._show_top.get()
        show_bottom = self._show_bottom.get()

        self.fig_diff.clear()
        self.ax_diff = self.fig_diff.add_subplot(111)
        is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False
        title_color = "#e0e0e0" if is_dark else "#212529"
        self.ax_diff.set_xlabel("Восток Y (м)", color=title_color)
        self.ax_diff.set_ylabel("Север X (м)", color=title_color)
        self.ax_diff.grid(True, linestyle="--", alpha=0.5)
        self.ax_diff.format_coord = self._format_coord_diff
        self.ax_diff.xaxis.set_major_formatter(PlainOffsetFormatter(useOffset=True))
        self.ax_diff.yaxis.set_major_formatter(PlainOffsetFormatter(useOffset=True))
        self._apply_axes_theme(self.ax_diff, self.fig_diff, is_dark)

        z_top_g = r.get("z_top_grid")
        z_bot_g = r.get("z_bot_grid")
        if z_top_g is None or z_bot_g is None:
            self.canvas_diff.draw()
            return

        is_cut = self._is_excavation()
        grid_upper_data = z_top_g
        grid_lower_data = z_bot_g

        if is_cut:
            label_upper = "Высота верхней (существующий рельеф) Z (м)"
            label_lower = "Высота нижней (дно котлована) Z (м)"
            title_upper = "Картограмма: Верхняя поверхность (существующий рельеф)"
            title_lower = "Картограмма: Нижняя поверхность (дно котлована)"
            cmap_dh   = CMAP_SWISS_CUT
            title_dh  = "Картограмма мощности выемки ΔH (м)"
        else:
            label_upper = "Высота верхней (гребень / насыпь) Z (м)"
            label_lower = "Высота нижней (существующий рельеф / основание) Z (м)"
            title_upper = "Картограмма: Верхняя поверхность (гребень / насыпь)"
            title_lower = "Картограмма: Нижняя поверхность (существующий рельеф / основание)"
            cmap_dh   = CMAP_SWISS_FILL
            title_dh  = "Картограмма мощности насыпи ΔH (м)"

        if show_top and show_bottom:
            grid_data = r["dh_grid"]
            cmap      = cmap_dh
            cb_label  = "Мощность выемки ΔH (м)" if is_cut else "Мощность насыпи ΔH (м)"
            title     = title_dh
        elif show_top:
            grid_data = grid_upper_data
            cmap      = "copper"
            cb_label  = label_upper
            title     = title_upper
        elif show_bottom:
            grid_data = grid_lower_data
            cmap      = "Blues_r" if is_cut else "Blues"
            cb_label  = label_lower
            title     = title_lower
        else:
            self._apply_axes_theme(self.ax_diff, self.fig_diff, is_dark)
            if len(self.boundary_indices) >= 3 and r.get("boundary") is not None:
                b_pts = r["boundary"]
                self.ax_diff.plot(list(b_pts[:, 1]) + [b_pts[0, 1]],
                                  list(b_pts[:, 0]) + [b_pts[0, 0]],
                                  "r--", linewidth=2, label="Контур сшивания")
                self.ax_diff.legend(loc="upper right")
            self.ax_diff.set_aspect("equal", adjustable="datalim")
            self._apply_fig_layout(self.fig_diff, pad=0.5)
            self.canvas_diff.draw()
            return

        self._apply_axes_theme(self.ax_diff, self.fig_diff, is_dark)

        # Формируем полигон контура для строгой векторной обрезки (исключает любые артефакты снаружи)
        poly_clip = None
        if len(self.boundary_indices) >= 3 and r.get("boundary") is not None:
            b_pts = r["boundary"]
            boundary_yx = np.column_stack([b_pts[:, 1], b_pts[:, 0]])
            poly_clip = MplPolygon(boundary_yx, closed=True, facecolor="none", edgecolor="none", transform=self.ax_diff.transData)
            self.ax_diff.add_patch(poly_clip)

        valid = grid_data[~np.isnan(grid_data)]
        levels = 24 if len(valid) > 0 and float(np.ptp(valid)) > 1e-4 else None

        if show_top and show_bottom:
            # 1. ЕСТЕСТВЕННЫЙ ШВЕЙЦАРСКИЙ СВЕТОТЕНЕВОЙ РЕЛЬЕФ (Swiss Multi-Directional Shading)
            if len(valid) > 0 and float(np.ptp(valid)) > 1e-4:
                try:
                    relief_surf = grid_upper_data if not is_cut else grid_lower_data
                    nan_mask = np.isnan(grid_data)
                    if not np.all(nan_mask):
                        fill_val = float(np.nanmin(relief_surf)) if not is_cut else float(np.nanmax(relief_surf))
                        filled_surf = np.where(nan_mask, fill_val, relief_surf)
                        sm_surf = _gaussian_blur_2d_np(filled_surf, sigma=2.2)
                        gx_m = r["grid_x"]
                        gy_m = r["grid_y"]
                        step_x = abs(gx_m[0, 1] - gx_m[0, 0]) if gx_m.shape[1] > 1 and abs(gx_m[0, 1] - gx_m[0, 0]) > 1e-6 else 1.0
                        step_y = abs(gy_m[1, 0] - gy_m[0, 0]) if gy_m.shape[0] > 1 and abs(gy_m[1, 0] - gy_m[0, 0]) > 1e-6 else 1.0

                        # В массивах расчётной сетки ось 0 соответствует gy (Восток / ось X графика),
                        # а ось 1 соответствует gx (Север / ось Y графика).
                        # Для корректного совмещения с осями imshow (где строки - это ось Y, а столбцы - ось X)
                        # транспонируем матрицу рельефа (.T):
                        sm_screen = sm_surf.T
                        dh_screen = grid_data.T

                        # Швейцарская многонаправленная отмывка с дневным гипсометрическим градиентом
                        hs = _compute_swiss_multidirectional_hillshade(
                            sm_screen, step_x, step_y, is_cut=is_cut, dh_screen=dh_screen
                        )
                        extent = [float(gy_m.min()), float(gy_m.max()), float(gx_m.min()), float(gx_m.max())]
                        im = self.ax_diff.imshow(
                            hs, extent=extent, origin="lower",
                            cmap="gray", vmin=0.20, vmax=1.0,
                            interpolation="bilinear"
                        )
                        if poly_clip is not None:
                            im.set_clip_path(poly_clip)
                except Exception:
                    pass

            # 2. Чёткие контрастные горизонтали равной мощности шагом 0.5 м
            if len(valid) > 0 and float(np.ptp(valid)) > 1e-4:
                try:
                    dh_min = float(np.nanmin(valid))
                    dh_max = float(np.nanmax(valid))
                    rng = dh_max - dh_min
                    if rng < 1.0:
                        c_step = 0.2
                    elif rng < 6.0:
                        c_step = 0.5
                    else:
                        c_step = 1.0
                    c_levels = np.arange(np.ceil(dh_min / c_step) * c_step, dh_max, c_step)
                    if len(c_levels) >= 2:
                        grid_m = np.ma.masked_where(np.isnan(grid_data), grid_data)
                        cs = self.ax_diff.contour(r["grid_y"], r["grid_x"], grid_m, levels=c_levels, colors="#111827", linewidths=0.95, alpha=0.90)
                        if poly_clip is not None:
                            try:
                                cs.set_clip_path(poly_clip)
                            except Exception:
                                if hasattr(cs, "collections"):
                                    for coll in cs.collections:
                                        coll.set_clip_path(poly_clip)
                        fmt = "-%.1f м" if is_cut else "+%.1f м"
                        self.ax_diff.clabel(cs, inline=True, fontsize=8.5, fmt=fmt, inline_spacing=6)
                except Exception:
                    pass

            # 3. Шкала мощности / глубины (согласована с дневной гипсометрической палитрой)
            dh_min = float(np.nanmin(valid)) if len(valid) > 0 else 0.0
            dh_max = float(np.nanmax(valid)) if len(valid) > 0 else 1.0
            norm = matplotlib.colors.Normalize(vmin=dh_min, vmax=dh_max)
            sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_dh)
            cb = self.fig_diff.colorbar(sm, ax=self.ax_diff, label=cb_label)
            if is_cut:
                cb.ax.invert_yaxis()
            self._diff_colorbar = cb
        else:
            # Отображение отдельной поверхности (верх или низ) с адаптивным LOD шагом
            r_step, c_step = calc_3d_stride(grid_data, target_dim=120)
            gy_sub = r["grid_y"][::r_step, ::c_step]
            gx_sub = r["grid_x"][::r_step, ::c_step]
            gd_sub = grid_data[::r_step, ::c_step]
            c = self.ax_diff.contourf(gy_sub, gx_sub, gd_sub, levels=levels, cmap=cmap)
            if poly_clip is not None:
                try:
                    c.set_clip_path(poly_clip)
                except Exception:
                    pass
            self.fig_diff.colorbar(c, ax=self.ax_diff, label=cb_label)

        if len(self.boundary_indices) >= 3 and r.get("boundary") is not None:
            b_pts = r["boundary"]
            self.ax_diff.plot(list(b_pts[:, 1]) + [b_pts[0, 1]],
                              list(b_pts[:, 0]) + [b_pts[0, 0]],
                              "r--", linewidth=2, label="Контур")

        self.ax_diff.set_aspect("equal", adjustable="datalim")
        self.ax_diff.legend(loc="upper right")
        self._apply_fig_layout(self.fig_diff, pad=0.5)
        self.canvas_diff.draw()
        if hasattr(self, "_toolbar_diff"):
            try:
                self._toolbar_diff.update()
            except Exception:
                pass

    def _update_table(self):
        if self.tree is None:
            return

        for item in self.tree.get_children():
            self.tree.delete(item)

        mean_bound, split_h, work_type = self._get_split_height()
        n_top = 0
        n_bot = 0
        n_bound = len(self.boundary_indices)

        is_two_surfs = self._is_two_surfaces()
        surfaces = self._ensure_point_surfaces()
        bound_indices_map = {idx: pos for pos, idx in enumerate(self.boundary_indices)}
        for i, p in enumerate(self.points):
            if not is_two_surfs and i in bound_indices_map:
                surf = f"Контур (#{bound_indices_map[i]+1})"
            else:
                actual_surf = surfaces[i] if i < len(surfaces) else "bottom"
                if p.surface_type == "top":
                    surf = "Верхняя"
                    n_top += 1
                elif p.surface_type == "bottom":
                    surf = "Нижняя"
                    n_bot += 1
                else:
                    surf = f"auto ({'Верх' if actual_surf == 'top' else 'Низ'})"
                    if actual_surf == "top":
                        n_top += 1
                    else:
                        n_bot += 1

            self.tree.insert("", tk.END, values=(i+1, p.id, f"{p.x:.3f}", f"{p.y:.3f}", f"{p.h:.3f}", surf))

        self.lbl_table_stats.configure(
            text=f"Всего: {len(self.points)} т. | Контур: {n_bound} | Верх: {n_top} | Низ: {n_bot}"
        )

    def _get_report_icon_path(self):
        """Возвращает путь к иконке сохранения отчета, генерируя её при необходимости"""
        candidates = []
        if hasattr(sys, "_MEIPASS"):
            candidates.append(os.path.join(sys._MEIPASS, "report_icon.png"))
        candidates.append(os.path.join(os.path.dirname(sys.executable), "_internal", "report_icon.png"))
        candidates.append(os.path.join(os.path.dirname(sys.executable), "report_icon.png"))
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_icon.png"))

        for p in candidates:
            if os.path.exists(p):
                return p

        # Автоматическая генерация иконки, если файл отсутствует
        icon_path = candidates[-1]
        try:
            from PIL import Image, ImageDraw
            im = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
            draw = ImageDraw.Draw(im)
            s = 48.0 / 24.0
            w = int(1.8 * s)
            poly = [(4*s, 2*s), (14*s, 2*s), (19*s, 7*s), (19*s, 21*s), (4*s, 21*s)]
            draw.polygon(poly, outline=(0, 0, 0, 255), width=w)
            draw.line([(14*s, 2*s), (14*s, 7*s), (19*s, 7*s)], fill=(0, 0, 0, 255), width=w)
            draw.line([(7*s, 11*s), (16*s, 11*s)], fill=(0, 0, 0, 255), width=int(1.5*s))
            draw.line([(7*s, 14.5*s), (16*s, 14.5*s)], fill=(0, 0, 0, 255), width=int(1.5*s))
            draw.line([(7*s, 18*s), (12*s, 18*s)], fill=(0, 0, 0, 255), width=int(1.5*s))
            im_24 = im.resize((24, 24), Image.Resampling.LANCZOS)
            im_24.save(icon_path)
            large_path = icon_path.replace(".png", "_large.png")
            im.save(large_path)
        except Exception:
            pass
        return icon_path

    def _save_tab_image(self, fig, tab_name: str):
        """Сохраняет изображение схемы в папку проекта с именем активной вкладки"""
        proj_dir = self._current_project_dir if self._current_project_dir else get_projects_dir()
        try:
            os.makedirs(proj_dir, exist_ok=True)
        except Exception:
            pass
        filename = f"{tab_name}.png"
        out_path = os.path.join(proj_dir, filename)

        temp_annotations = []
        if fig == getattr(self, "fig_2d", None) and getattr(self, "_current_labeled_points", None):
            try:
                text_col = self._get_canvas_text_color()
                for p in self._current_labeled_points:
                    ann = self.ax_2d.annotate(
                        f"{p.id}\nH:{p.h:.3f}", (p.y, p.x),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=9.0, alpha=0.9, zorder=5, clip_on=True,
                        color=text_col
                    )
                    temp_annotations.append(ann)
            except Exception:
                pass

        try:
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            self.lbl_report_status.configure(text=f"✓ Схема сохранена: {filename}")
            messagebox.showinfo("Сохранено", f"Изображение «{tab_name}» успешно сохранено в папку проекта:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить изображение:\n{e}")
        finally:
            for ann in temp_annotations:
                try:
                    ann.remove()
                except Exception:
                    pass

    def _save_report_to_project(self):
        """Сохраняет результаты расчета напрямую в текстовый файл папки проекта"""
        if not self.calc_results:
            messagebox.showwarning("Внимание", "Сначала выполните расчет объема.")
            return
        proj_dir = self._current_project_dir if self._current_project_dir else get_projects_dir()
        try:
            os.makedirs(proj_dir, exist_ok=True)
        except Exception:
            pass
        folder_name = os.path.basename(proj_dir) if self._current_project_dir else "Проект"
        base_name = self._strip_date_from_folder_name(folder_name) or folder_name
        report_path = os.path.join(proj_dir, f"{base_name}_report.txt")
        try:
            report_text = self.txt_results.get("1.0", tk.END)
            with open(report_path, "w", encoding="utf-8") as out:
                out.write(report_text)
            self.lbl_report_status.configure(text=f"✓ Отчёт сохранен: {os.path.basename(report_path)}")
            messagebox.showinfo("Сохранено", f"Отчет успешно сохранен в папку проекта:\n{report_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить отчет:\n{e}")

    def _get_themed_toolbar_icon(self, name: str, is_dark: bool):
        """Возвращает адаптированную по контрасту PhotoImage для кнопок тулбара под светлую/тёмную тему."""
        if not hasattr(self, "_cached_toolbar_icons"):
            self._cached_toolbar_icons = {}
        cache_key = (name, is_dark)
        if cache_key in self._cached_toolbar_icons:
            return self._cached_toolbar_icons[cache_key]

        from PIL import Image, ImageDraw, ImageTk
        import numpy as np

        size = 18
        if name == "save":
            try:
                import matplotlib.cbook as cbook
                p = cbook._get_data_path('images/filesave_large.png')
                if not os.path.exists(p):
                    p = cbook._get_data_path('images/filesave.png')
                im = Image.open(p).convert("RGBA")
                arr = np.array(im)
                mask = arr[..., 3] > 0
                if is_dark:
                    arr[mask, 0:3] = 245
                else:
                    arr[mask, 0] = 33
                    arr[mask, 1] = 37
                    arr[mask, 2] = 41
                im_res = Image.fromarray(arr).resize((size, size), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(im_res, master=self)
                self._cached_toolbar_icons[cache_key] = photo
                return photo
            except Exception:
                return None
        elif name == "report":
            try:
                color = (245, 245, 245, 255) if is_dark else (33, 37, 41, 255)
                im = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
                draw = ImageDraw.Draw(im)
                s = 48.0 / 24.0
                w = int(2.0 * s)
                poly = [(4*s, 2*s), (14*s, 2*s), (19*s, 7*s), (19*s, 22*s), (4*s, 22*s)]
                draw.polygon(poly, outline=color, width=w)
                draw.line([(14*s, 2*s), (14*s, 7*s), (19*s, 7*s)], fill=color, width=w)
                draw.line([(7*s, 11*s), (16*s, 11*s)], fill=color, width=int(1.8*s))
                draw.line([(7*s, 15*s), (16*s, 15*s)], fill=color, width=int(1.8*s))
                draw.line([(7*s, 19*s), (13*s, 19*s)], fill=color, width=int(1.8*s))
                im_res = im.resize((size, size), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(im_res, master=self)
                self._cached_toolbar_icons[cache_key] = photo
                return photo
            except Exception:
                return None
        return None

    def _setup_custom_toolbar_buttons(self, toolbar, fig, tab_name: str):
        """Настраивает аккуратные кнопки у левого края панели холста: PNG, DXF, Отчет"""
        is_dark = ctk.get_appearance_mode() == "Dark"
        tb_bg = "#212529" if is_dark else "#f8f9fa"
        tb_fg = "#ffffff" if is_dark else "#212529"
        btn_bg = "#2b3035" if is_dark else "#ffffff"
        btn_fg = "#f8f9fa" if is_dark else "#212529"
        btn_active = "#3d444b" if is_dark else "#e9ecef"
        btn_highlight = "#495057" if is_dark else "#ced4da"

        try:
            toolbar.config(bg=tb_bg, bd=0, highlightthickness=0)
        except Exception:
            pass

        toolbar.save_figure = lambda *args: self._save_tab_image(fig, tab_name)

        # Скрываем стандартную системную кнопку сохранения Matplotlib
        if "Save" in getattr(toolbar, "_buttons", {}):
            try:
                toolbar._buttons["Save"].pack_forget()
            except Exception:
                pass

        btn_save_png = tk.Button(
            master=toolbar,
            text="💾 Сохранить PNG",
            relief="solid",
            borderwidth=1,
            bg=btn_bg,
            fg=btn_fg,
            activebackground=btn_active,
            activeforeground=btn_fg,
            highlightbackground=btn_highlight,
            highlightcolor=btn_highlight,
            highlightthickness=1,
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=2,
            cursor="hand2",
            command=lambda: self._save_tab_image(fig, tab_name)
        )
        toolbar._btn_save_png = btn_save_png
        btn_save_png.pack(side=tk.LEFT, padx=(6, 2), pady=2)
        add_tooltip(btn_save_png, f"Сохранить графическое изображение текущей вкладки '{tab_name}' в формате PNG")

        btn_dxf = tk.Button(
            master=toolbar,
            text="📐 Экспорт DXF",
            relief="solid",
            borderwidth=1,
            bg=btn_bg,
            fg=btn_fg,
            activebackground=btn_active,
            activeforeground=btn_fg,
            highlightbackground=btn_highlight,
            highlightcolor=btn_highlight,
            highlightthickness=1,
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=2,
            cursor="hand2",
            command=lambda tn=tab_name: self._export_tab_dxf(tn)
        )
        toolbar._btn_dxf = btn_dxf
        btn_dxf.pack(side=tk.LEFT, padx=2, pady=2)
        add_tooltip(btn_dxf, f"Экспортировать чертеж текущей вкладки '{tab_name}' в формат AutoCAD DXF")

        btn_report = tk.Button(
            master=toolbar,
            text="📄 Текстовый отчет",
            relief="solid",
            borderwidth=1,
            bg=btn_bg,
            fg=btn_fg,
            activebackground=btn_active,
            activeforeground=btn_fg,
            highlightbackground=btn_highlight,
            highlightcolor=btn_highlight,
            highlightthickness=1,
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=2,
            cursor="hand2",
            command=self._save_report_to_project
        )
        toolbar._btn_report = btn_report
        btn_report.pack(side=tk.LEFT, padx=(2, 6), pady=2)
        add_tooltip(btn_report, "Сохранить подробный текстовый отчет расчета объема (TXT) в папку проекта")

        # Центральная строка с объемами насыпи и выемки
        lbl_vol = tk.Label(
            master=toolbar,
            text="",
            font=("Segoe UI", 8, "bold"),
            fg=tb_fg,
            bg=tb_bg,
            anchor="center"
        )
        lbl_vol.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        if not hasattr(self, "_toolbar_volume_labels"):
            self._toolbar_volume_labels = []
        self._toolbar_volume_labels.append(lbl_vol)
        self._update_toolbar_volume_labels()

        if hasattr(toolbar, "_message_label"):
            try:
                toolbar._message_label.config(bg=tb_bg, fg=tb_fg, font=("Segoe UI", 8))
            except Exception:
                pass

        return btn_report

    def _open_dxf_export_dialog(self):
        if not self.calc_results or "error" in self.calc_results:
            messagebox.showwarning("Внимание", "Сначала выполните расчёт объема земляных масс.", parent=self)
            return
        try:
            from ui_dialogs import CartogramDxfExportDialog
            CartogramDxfExportDialog(self)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть диалог экспорта DXF:\n{e}", parent=self)

    def _export_tab_dxf(self, tab_name: Optional[str] = None):
        """Экспортирует чертеж активной графической вкладки в формат AutoCAD DXF."""
        if not self.calc_results or "error" in self.calc_results:
            messagebox.showwarning("Внимание", "Сначала выполните расчёт объема земляных масс.", parent=self)
            return

        if not tab_name:
            tab_name = self.tabview.get() if hasattr(self, "tabview") else ""

        # Для картограммы масс открываем подробный диалог настройки картограммы
        if tab_name == self.TAB_DIFF:
            self._open_dxf_export_dialog()
            return

        proj_dir = self._current_project_dir if self._current_project_dir else get_projects_dir()
        try:
            os.makedirs(proj_dir, exist_ok=True)
        except Exception:
            pass
        folder_name = os.path.basename(proj_dir) if self._current_project_dir else "Проект"
        base_name = self._strip_date_from_folder_name(folder_name) or folder_name

        if tab_name == self.TAB_2D:
            clean_name = "2D_План"
            title = "2D плана съёмки"
        elif tab_name == self.TAB_3D:
            clean_name = "3D_Модель"
            title = "3D модели поверхности"
        elif tab_name == self.TAB_TIN:
            clean_name = "TIN_Триангуляция"
            title = "TIN-триангуляции"
        elif tab_name == getattr(self, "TAB_CONTOURS", "Горизонтали"):
            clean_name = "Горизонтали"
            title = "топографических горизонталей"
        else:
            self._open_dxf_export_dialog()
            return

        default_fn = f"{base_name}_{clean_name}.dxf"
        out_path = filedialog.asksaveasfilename(
            parent=self,
            title=f"Экспорт {title} в DXF",
            initialdir=proj_dir,
            initialfile=default_fn,
            defaultextension=".dxf",
            filetypes=[("AutoCAD DXF", "*.dxf"), ("Все файлы", "*.*")]
        )
        if not out_path:
            return

        try:
            from dxf_exporter import (
                export_surface_3d_dxf,
                export_tin_dxf,
                export_contours_dxf,
                export_plan_2d_dxf
            )
            if tab_name == self.TAB_TIN:
                tin_simps = getattr(self, "tin_simplices", None)
                ok, msg = export_tin_dxf(
                    self.calc_results, self.points, self.boundary_indices,
                    tin_simps, out_path, coord_swap=True
                )
            elif tab_name == getattr(self, "TAB_CONTOURS", "Горизонтали"):
                c_step = getattr(self, "contour_step", None)
                ok, msg = export_contours_dxf(
                    self.calc_results, self.points, self.boundary_indices,
                    out_path, contour_step=c_step, coord_swap=True
                )
            elif tab_name == self.TAB_3D:
                ok, msg = export_surface_3d_dxf(
                    self.calc_results, self.points, self.boundary_indices,
                    out_path, coord_swap=True
                )
            elif tab_name == self.TAB_2D:
                ok, msg = export_plan_2d_dxf(
                    self.calc_results, self.points, self.boundary_indices,
                    out_path, coord_swap=True
                )
            else:
                ok, msg = False, "Неизвестный тип вкладки"

            if ok:
                messagebox.showinfo("Успешно", msg, parent=self)
            else:
                messagebox.showerror("Ошибка экспорта", msg, parent=self)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при экспорте в DXF:\n{e}", parent=self)

    def _export_report(self):
        self._save_report_to_project()

    def _export_plot(self):
        curr_tab = self.tabview.get()
        if curr_tab == self.TAB_2D:
            self._save_tab_image(self.fig_2d, self.TAB_2D)
        elif curr_tab == self.TAB_3D:
            self._save_tab_image(self.fig_3d, self.TAB_3D)
        elif curr_tab == self.TAB_DIFF:
            self._save_tab_image(self.fig_diff, self.TAB_DIFF)
        elif curr_tab == self.TAB_TIN:
            self._save_tab_image(self.fig_tin, self.TAB_TIN)
        elif curr_tab == getattr(self, "TAB_CONTOURS", "Горизонтали"):
            self._save_tab_image(self.fig_contours, self.TAB_CONTOURS)

    # ==================== TIN ТРЕУГОЛЬНИКИ ====================

    def _on_tin_canvas_press(self, event):
        """Нажатие кнопки мыши на TIN холсте"""
        try:
            if hasattr(self, "_toolbar_tin") and self._toolbar_tin.mode != "":
                return
        except Exception:
            pass

        if event.x is None or event.y is None or event.inaxes != self.ax_tin:
            return

        self._tin_pan_start = (
            event.x,
            event.y,
            event.xdata,
            event.ydata,
            self.ax_tin.get_xlim(),
            self.ax_tin.get_ylim(),
            event.button
        )
        self._tin_pan_dragged = False

    def _on_tin_canvas_motion(self, event):
        """Перемещение мыши — плавное панорамирование схемы треугольников"""
        if self._tin_pan_start is None or event.x is None or event.y is None:
            return

        start_px_x, start_px_y, _, _, orig_xlim, orig_ylim, btn = self._tin_pan_start
        dx_px = event.x - start_px_x
        dy_px = event.y - start_px_y

        drag_threshold = 3 if btn in (2, 3) else 6
        if abs(dx_px) > drag_threshold or abs(dy_px) > drag_threshold:
            self._tin_pan_dragged = True

        if self._tin_pan_dragged:
            bbox = self.ax_tin.bbox
            if bbox.width > 0 and bbox.height > 0:
                dx_data = dx_px * (orig_xlim[1] - orig_xlim[0]) / bbox.width
                dy_data = dy_px * (orig_ylim[1] - orig_ylim[0]) / bbox.height
                self.ax_tin.set_xlim(orig_xlim[0] - dx_data, orig_xlim[1] - dx_data)
                self.ax_tin.set_ylim(orig_ylim[0] - dy_data, orig_ylim[1] - dy_data)
                self._throttled_canvas_draw(self.canvas_tin, delay_ms=25)

    def _on_tin_canvas_release(self, event):
        """Отпускание кнопки мыши на TIN холсте"""
        if self._tin_pan_start is None:
            return

        btn = self._tin_pan_start[6]
        was_dragged = self._tin_pan_dragged
        self._tin_pan_start = None
        self._tin_pan_dragged = False

        if was_dragged:
            self._flush_canvas_draw(self.canvas_tin)
            return

        if btn == 1:
            self._handle_tin_click(event)
        elif btn == 3:
            self._handle_tin_edge_flip(event)


    def _on_tin_canvas_scroll(self, event):
        """Зумирование колесом мыши вокруг курсора на TIN схеме"""
        if event.xdata is None or event.ydata is None or event.inaxes != self.ax_tin:
            return
        base_scale = 1.25
        scale_factor = 1.0 / base_scale if event.button == "up" else base_scale

        cur_xlim = self.ax_tin.get_xlim()
        cur_ylim = self.ax_tin.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax_tin.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax_tin.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self._throttled_canvas_draw(self.canvas_tin, delay_ms=25)

    def _handle_tin_click(self, event):
        """Клик ЛКМ по треугольнику на TIN холсте — выбор треугольника"""
        if not getattr(self, "_tin_patches", None):
            return
        if self._tin_simplices is None or len(self._tin_simplices) == 0:
            return
        if event.xdata is None or event.ydata is None:
            return

        click_y = event.xdata   # Восток Y
        click_x = event.ydata   # Север X

        def _point_in_triangle(px, py, ax, ay, bx, by, cx, cy):
            d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by)
            d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy)
            d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay)
            has_neg = (d1 < -1e-9) or (d2 < -1e-9) or (d3 < -1e-9)
            has_pos = (d1 > 1e-9) or (d2 > 1e-9) or (d3 > 1e-9)
            return not (has_neg and has_pos)

        found_idx = None
        for i, (p0_idx, p1_idx, p2_idx) in enumerate(self._tin_simplices):
            p0 = self.points[p0_idx]
            p1 = self.points[p1_idx]
            p2 = self.points[p2_idx]
            if _point_in_triangle(click_x, click_y, p0.x, p0.y, p1.x, p1.y, p2.x, p2.y):
                found_idx = i
                break

        if found_idx is not None:
            self._tin_selected_idx = found_idx
            try:
                self._tin_updating_selection = True
                self.tree_tin.selection_set(str(found_idx))
                self.tree_tin.see(str(found_idx))
            except Exception:
                pass
            finally:
                self._tin_updating_selection = False
            self._highlight_tin_selection()

    def _handle_tin_edge_flip(self, event):
        """Переопределение (переброска) грани треугольника по клику ПКМ"""
        if not getattr(self, "_tin_patches", None):
            return
        if self._tin_simplices is None or len(self._tin_simplices) == 0:
            return
        if event.xdata is None or event.ydata is None or event.inaxes != self.ax_tin:
            return

        click_y = event.xdata   # Восток Y (м)
        click_x = event.ydata   # Север X (м)

        # 1. Собираем соответствие рёбер треугольникам
        edge_map = {}  # edge_tuple -> list of simplex_indices
        for s_idx, simplex in enumerate(self._tin_simplices):
            v0, v1, v2 = simplex
            for a, b in [(v0, v1), (v1, v2), (v2, v0)]:
                e = (min(a, b), max(a, b))
                if e not in edge_map:
                    edge_map[e] = []
                edge_map[e].append(s_idx)

        # 2. Находим ближайшее внутреннее ребро (разделяющее ровно 2 треугольника)
        cur_xlim = self.ax_tin.get_xlim()
        cur_ylim = self.ax_tin.get_ylim()
        view_span = max(abs(cur_xlim[1] - cur_xlim[0]), abs(cur_ylim[1] - cur_ylim[0]))
        tol = max(view_span * 0.05, 0.5)

        best_edge = None
        best_dist = float("inf")

        for (u, v), s_indices in edge_map.items():
            if len(s_indices) != 2:
                continue

            p_u = self.points[u]
            p_v = self.points[v]

            ax, ay = p_u.x, p_u.y
            bx, by = p_v.x, p_v.y

            vx, vy = bx - ax, by - ay
            l2 = vx * vx + vy * vy
            if l2 < 1e-12:
                d = float(np.hypot(click_x - ax, click_y - ay))
            else:
                t = ((click_x - ax) * vx + (click_y - ay) * vy) / l2
                t_clamped = max(0.0, min(1.0, t))
                proj_x = ax + t_clamped * vx
                proj_y = ay + t_clamped * vy
                d = float(np.hypot(click_x - proj_x, click_y - proj_y))

            if d < best_dist:
                best_dist = d
                best_edge = (u, v)

        if best_edge is None or best_dist > tol:
            return  # Клик не попал близко ни к одному внутреннему ребру

        u, v = best_edge
        t1, t2 = edge_map[best_edge]
        simp1 = list(self._tin_simplices[t1])
        simp2 = list(self._tin_simplices[t2])

        c_cands = [pt for pt in simp1 if pt != u and pt != v]
        d_cands = [pt for pt in simp2 if pt != u and pt != v]
        if not c_cands or not d_cands:
            return
        c = c_cands[0]
        d = d_cands[0]

        # Проверка: ребро (u, v) не должно быть ребром внешнего контура границы
        if len(self.boundary_indices) >= 3:
            b_len = len(self.boundary_indices)
            is_boundary_edge = False
            for k in range(b_len):
                b1 = self.boundary_indices[k]
                b2 = self.boundary_indices[(k + 1) % b_len]
                if (b1 == u and b2 == v) or (b1 == v and b2 == u):
                    is_boundary_edge = True
                    break
            if is_boundary_edge:
                messagebox.showinfo("Информация",
                                    f"Ребро {self.points[u].id}—{self.points[v].id} является внешней границей контура и не может быть переброшено.")
                return

        # Проверка строгой выпуклости четырёхугольника (диагонали UV и CD должны строго пересекаться)
        pu = self.points[u]
        pv = self.points[v]
        pc = self.points[c]
        pd = self.points[d]

        def _ccw(p1x, p1y, p2x, p2y, p3x, p3y):
            return (p2x - p1x) * (p3y - p1y) - (p2y - p1y) * (p3x - p1x)

        ccw_uv_c = _ccw(pu.x, pu.y, pv.x, pv.y, pc.x, pc.y)
        ccw_uv_d = _ccw(pu.x, pu.y, pv.x, pv.y, pd.x, pd.y)
        ccw_cd_u = _ccw(pc.x, pc.y, pd.x, pd.y, pu.x, pu.y)
        ccw_cd_v = _ccw(pc.x, pc.y, pd.x, pd.y, pv.x, pv.y)

        if not (ccw_uv_c * ccw_uv_d < -1e-9 and ccw_cd_u * ccw_cd_v < -1e-9):
            messagebox.showinfo("Информация",
                                f"Переброска ребра {pu.id}—{pv.id} невозможна: четырёхугольник вокруг него не является строго выпуклым.")
            return

        # Проверка: середина нового ребра должна лежать внутри контура
        mid_x = (pc.x + pd.x) / 2.0
        mid_y = (pc.y + pd.y) / 2.0
        if len(self.boundary_indices) >= 3:
            bound_pts = np.array([[self.points[i].x, self.points[i].y] for i in self.boundary_indices])
            bound_path = MplPath(bound_pts)
            if not bound_path.contains_point((mid_x, mid_y), radius=1e-5):
                messagebox.showinfo("Информация",
                                    f"Переброска ребра {pu.id}—{pv.id} невозможна: новое ребро выходит за границу контура.")
                return

        # Формируем два новых треугольника (c, d, u) и (c, d, v) с сохранением ориентации против часовой стрелки
        if _ccw(pc.x, pc.y, pd.x, pd.y, pu.x, pu.y) > 0:
            new_t1 = (c, d, u)
        else:
            new_t1 = (c, u, d)

        if _ccw(pc.x, pc.y, pd.x, pd.y, pv.x, pv.y) > 0:
            new_t2 = (c, d, v)
        else:
            new_t2 = (c, v, d)

        # Сохраняем в Undo стек
        old_t1 = tuple(simp1)
        old_t2 = tuple(simp2)
        self._undo_stack.append(("tin_flip", t1, t2, old_t1, old_t2, new_t1, new_t2))

        # Обновляем симплексы
        simplices_list = [list(s) for s in self._tin_simplices]
        simplices_list[t1] = list(new_t1)
        simplices_list[t2] = list(new_t2)
        self._tin_simplices = np.array(simplices_list)
        self._tin_custom_simplices = [tuple(s) for s in simplices_list]

        # Если треугольники были исключены — активируем их
        self._tin_excluded.discard(t1)
        self._tin_excluded.discard(t2)

        # Перерисовываем TIN без сброса масштаба
        self._redraw_tin(reset_view=False)
        self.save_project_state(self._current_file_path)

        # Уведомление в статусной строке
        msg = f"Ребро переброшено: {pu.id}—{pv.id} ➔ {pc.id}—{pd.id}"
        self.lbl_tin_stats.configure(text=msg)

        # Автоматический перерасчёт земляных масс
        if len(self.boundary_indices) >= 3:
            self.calculate_volume(use_tin=True)

    def _highlight_tin_selection(self):
        """Быстро обновляет подсветку выбранного треугольника без перерисовки всей сцены и без зацикливания"""

        if not hasattr(self, "_tin_patches") or not self._tin_patches:
            self._redraw_tin(reset_view=False)
            return

        for i, patch in enumerate(self._tin_patches):
            if i == self._tin_selected_idx:
                patch.set_facecolor("#f39c12")
                patch.set_alpha(0.55)
                patch.set_edgecolor("#d35400")
                patch.set_linewidth(2.2)
                patch.set_zorder(4)
            elif i in self._tin_excluded:
                patch.set_facecolor("#e74c3c")
                patch.set_alpha(0.35)
                patch.set_edgecolor("#c0392b")
                patch.set_linestyle("--")
                patch.set_linewidth(1.2)
                patch.set_zorder(2)
            else:
                patch.set_facecolor("#3498db")
                patch.set_alpha(0.15)
                patch.set_edgecolor("#2980b9")
                patch.set_linestyle("-")
                patch.set_linewidth(0.9)
                patch.set_zorder(2)

        self.canvas_tin.draw_idle()

    def _redraw_tin(self, reset_view=False):
        """Перерисовывает TIN-триангуляцию строго внутри контура сшивания"""
        if self.canvas_tin is None:
            return

        old_xlim = None
        old_ylim = None
        if not reset_view:
            try:
                cur_x = self.ax_tin.get_xlim()
                cur_y = self.ax_tin.get_ylim()
                if cur_x != (0.0, 1.0) or cur_y != (0.0, 1.0):
                    old_xlim = cur_x
                    old_ylim = cur_y
            except Exception:
                pass

        try:
            self.ax_tin.clear()
            is_dark = (ctk.get_appearance_mode() == "Dark") if hasattr(ctk, "get_appearance_mode") else False
            title_color = "#e0e0e0" if is_dark else "#212529"
            self.ax_tin.set_xlabel("Восток Y (м)", color=title_color)
            self.ax_tin.set_ylabel("Север X (м)", color=title_color)
            self._apply_axes_theme(self.ax_tin, self.fig_tin, is_dark)
        except Exception:
            return
        self.ax_tin.grid(True, linestyle="--", alpha=0.4)
        self.ax_tin.format_coord = self._format_coord_display
        self.ax_tin.xaxis.set_major_formatter(PlainOffsetFormatter(useOffset=True))
        self.ax_tin.yaxis.set_major_formatter(PlainOffsetFormatter(useOffset=True))

        # Очистка таблицы
        if getattr(self, "tree_tin", None) is not None:
            try:
                self._tin_updating_selection = True
                for item in self.tree_tin.get_children():
                    self.tree_tin.delete(item)
            finally:
                self._tin_updating_selection = False

        if not self.points:
            self.lbl_tin_stats.configure(text="Точки не загружены")
            self.canvas_tin.draw()
            return

        if len(self.boundary_indices) < 3:
            self.ax_tin.text(0.5, 0.5,
                             "Контур не выбран.\nЗадайте контур на вкладке '2D Схема в плане' (минимум 3 точки).",
                             transform=self.ax_tin.transAxes, ha='center', va='center',
                             fontsize=11, color='#7f8c8d')
            self.lbl_tin_stats.configure(text="Контур не задан")
            self.canvas_tin.draw()
            return

        # Граничный контур
        boundary_pts = np.array([[self.points[i].x, self.points[i].y] for i in self.boundary_indices])
        bound_path = MplPath(boundary_pts)
        boundary_set = set(self.boundary_indices)

        mean_bound, split_h, work_type = self._get_split_height()
        is_cut = (work_type == "cut")
        is_two_surfs = self._is_two_surfaces()
        if is_two_surfs:
            target_working_surf = "bottom" if (self._show_bottom.get() and not self._show_top.get()) else "top"
        else:
            target_working_surf = "bottom" if is_cut else "top"

        # Отбираем точки, лежащие на границе или относящиеся к рабочей поверхности контура
        included_indices = []
        surfaces = self._ensure_point_surfaces()
        inside_mask = self._ensure_boundary_inside_mask()
        for i, p in enumerate(self.points):
            surf = surfaces[i] if i < len(surfaces) else "bottom"
            if is_two_surfs:
                # В режиме двух поверхностей в TIN должны попадать ТОЛЬКО точки рабочей поверхности
                if surf == target_working_surf:
                    included_indices.append(i)
            else:
                if i in boundary_set:
                    included_indices.append(i)
                elif bool(inside_mask[i]) if i < len(inside_mask) else False:
                    if surf == target_working_surf:
                        included_indices.append(i)

        if len(included_indices) < 3:
            self.ax_tin.text(0.5, 0.5, "Недостаточно точек внутри контура для триангуляции.",
                             transform=self.ax_tin.transAxes, ha='center', va='center',
                             fontsize=11, color='#7f8c8d')
            self.lbl_tin_stats.configure(text="Недостаточно точек")
            self.canvas_tin.draw()
            return

        # Если есть сохранённая пользовательская триангуляция, используем её
        kept_simplices = None
        if self._tin_custom_simplices is not None and len(self._tin_custom_simplices) > 0:
            n_pts = len(self.points)
            if all(len(s) == 3 and all(0 <= v < n_pts for v in s) for s in self._tin_custom_simplices):
                kept_simplices = [list(s) for s in self._tin_custom_simplices]
            else:
                self._tin_custom_simplices = None

        if kept_simplices is None:
            pts_included = np.array([[self.points[i].x, self.points[i].y] for i in included_indices])
            try:
                tri = Delaunay(pts_included)
            except Exception as e:
                self.ax_tin.text(0.5, 0.5, f"Ошибка построения триангуляции:\n{e}",
                                 transform=self.ax_tin.transAxes, ha='center', va='center',
                                 fontsize=10, color='red')
                self.canvas_tin.draw()
                return

            # Фильтруем треугольники: оставляем ТОЛЬКО те, чей центроид внутри контура
            kept_simplices = []
            for simplex in tri.simplices:
                c0 = pts_included[simplex[0]]
                c1 = pts_included[simplex[1]]
                c2 = pts_included[simplex[2]]
                centroid = (c0 + c1 + c2) / 3.0

                if not is_two_surfs and not bound_path.contains_point(centroid, radius=1e-5):
                    continue

                p0_idx = included_indices[simplex[0]]
                p1_idx = included_indices[simplex[1]]
                p2_idx = included_indices[simplex[2]]
                kept_simplices.append((p0_idx, p1_idx, p2_idx))

        self._tin_simplices = np.array(kept_simplices) if kept_simplices else np.empty((0, 3), dtype=int)
        self._tin_pts_2d = np.array([[p.x, p.y] for p in self.points])

        # 1. Отрисовка треугольников основного TIN с учетом геоморфологии
        self._tin_patches = []
        show_top    = self._show_top.get()
        show_bottom = self._show_bottom.get()
        is_cut = self._is_excavation()

        # В выемке основной TIN (котлован) относится к нижней поверхности (show_bottom).
        # В насыпи основной TIN (насыпь) относится к верхней поверхности (show_top).
        if is_two_surfs:
            show_main_tin = show_bottom if target_working_surf == "bottom" else show_top
        elif show_top and show_bottom:
            show_main_tin = True
        elif is_cut:
            show_main_tin = show_bottom
        else:
            show_main_tin = show_top

        if show_main_tin:
            if is_two_surfs:
                is_bot_active = (target_working_surf == "bottom")
                base_face = "#2980b9" if is_bot_active else "#d35400"
                base_edge = "#1f618d" if is_bot_active else "#a04000"
                base_alpha = 0.15 if is_bot_active else 0.18
            else:
                base_face = "#2980b9" if is_cut else "#d35400"
                base_edge = "#1f618d" if is_cut else "#a04000"
                base_alpha = 0.15 if is_cut else 0.18
            for i, (p0_idx, p1_idx, p2_idx) in enumerate(kept_simplices):
                v0 = (self.points[p0_idx].y, self.points[p0_idx].x)
                v1 = (self.points[p1_idx].y, self.points[p1_idx].x)
                v2 = (self.points[p2_idx].y, self.points[p2_idx].x)
                verts_plot = [v0, v1, v2]

                if i == self._tin_selected_idx:
                    patch = MplPolygon(verts_plot, closed=True,
                                       facecolor="#f39c12", alpha=0.55,
                                       edgecolor="#d35400", linestyle="-", linewidth=2.2, zorder=4)
                elif i in self._tin_excluded:
                    patch = MplPolygon(verts_plot, closed=True,
                                       facecolor="#e74c3c", alpha=0.35,
                                       edgecolor="#c0392b", linestyle="--", linewidth=1.2, zorder=2)
                else:
                    patch = MplPolygon(verts_plot, closed=True,
                                       facecolor=base_face, alpha=base_alpha,
                                       edgecolor=base_edge, linestyle="-", linewidth=0.9, zorder=2)
                self.ax_tin.add_patch(patch)
                self._tin_patches.append(patch)

        # 2. Отрисовка контура границы
        bx = [self.points[i].x for i in self.boundary_indices]
        by = [self.points[i].y for i in self.boundary_indices]
        bx_c = bx + [bx[0]]
        by_c = by + [by[0]]
        self.ax_tin.plot(by_c, bx_c, color="#27ae60", linewidth=2.0, linestyle="-", zorder=4, label="Контур")

        # 3. Векторизованная отрисовка точек (без текстовых подписей) — показываем только точки включенных поверхностей
        if show_top or show_bottom:
            mean_bound = float(np.mean([self.points[b].h for b in self.boundary_indices])) if len(self.boundary_indices) >= 3 else 0.0
            bound_ys, bound_xs = [], []
            upper_ys, upper_xs = [], []
            lower_ys, lower_xs = [], []

            for idx in included_indices:
                p = self.points[idx]
                is_boundary = (idx in boundary_set) if not is_two_surfs else False
                if is_boundary:
                    if show_top and show_bottom:
                        show_pt = True
                    elif is_cut:
                        show_pt = show_top or show_bottom
                    else:
                        show_pt = show_bottom or show_top
                    if show_pt:
                        bound_ys.append(p.y)
                        bound_xs.append(p.x)
                else:
                    pt_is_lower = (p.surface_type == "bottom") or (
                        p.surface_type not in ("top", "boundary") and (
                            (is_cut and p.h <= mean_bound) or (not is_cut and p.h < mean_bound)
                        )
                    )
                    pt_is_upper = not pt_is_lower

                    if pt_is_upper and not show_top:
                        continue
                    if pt_is_lower and not show_bottom:
                        continue

                    if is_cut:
                        if pt_is_lower:
                            lower_ys.append(p.y)
                            lower_xs.append(p.x)
                        else:
                            upper_ys.append(p.y)
                            upper_xs.append(p.x)
                    else:
                        if pt_is_upper:
                            upper_ys.append(p.y)
                            upper_xs.append(p.x)
                        else:
                            lower_ys.append(p.y)
                            lower_xs.append(p.x)

            if bound_ys:
                self.ax_tin.scatter(bound_ys, bound_xs, color="#27ae60", s=18, zorder=6,
                                    edgecolors="#1a5e2a", linewidth=0.7)
            if is_cut:
                if lower_ys:
                    self.ax_tin.scatter(lower_ys, lower_xs, color="#1565c0", s=14, zorder=5,
                                        edgecolors="#0d47a1", linewidth=0.5)
                if upper_ys:
                    self.ax_tin.scatter(upper_ys, upper_xs, color="#784212", s=14, zorder=5,
                                        edgecolors="#4a2800", linewidth=0.5)
            else:
                if upper_ys:
                    self.ax_tin.scatter(upper_ys, upper_xs, color="#d35400", s=14, zorder=5,
                                        edgecolors="#a04000", linewidth=0.5)
                if lower_ys:
                    self.ax_tin.scatter(lower_ys, lower_xs, color="#1565c0", s=14, zorder=5,
                                        edgecolors="#0d47a1", linewidth=0.5)

        # 4. Заполнение статистики и (при наличии) таблицы треугольников
        has_tree = getattr(self, "tree_tin", None) is not None
        try:
            if has_tree:
                self._tin_updating_selection = True
                for item in self.tree_tin.get_children():
                    self.tree_tin.delete(item)

            if show_main_tin:
                n_active = 0
                max_tree_items = 1000
                total = len(kept_simplices)

                for i, (p0_idx, p1_idx, p2_idx) in enumerate(kept_simplices):
                    p0 = self.points[p0_idx]
                    p1 = self.points[p1_idx]
                    p2 = self.points[p2_idx]
                    d0 = (p1.x - p0.x, p1.y - p0.y)
                    d1 = (p2.x - p0.x, p2.y - p0.y)
                    area = 0.5 * abs(d0[0] * d1[1] - d0[1] * d1[0])

                    if i in self._tin_excluded:
                        status = "Исключён"
                        tag = "excluded"
                    else:
                        status = "Активен"
                        tag = "active"
                        n_active += 1

                    if has_tree and (i < max_tree_items or i == self._tin_selected_idx):
                        self.tree_tin.insert("", tk.END, iid=str(i),
                                              values=(i + 1, p0.id, p1.id, p2.id, f"{area:.3f}", status),
                                              tags=(tag,))

                excl = len(self._tin_excluded)
                if has_tree and total > max_tree_items:
                    self.lbl_tin_stats.configure(
                        text=f"Всего: {total} (в таблице первые {max_tree_items}) | Активных: {n_active} | Исключено: {excl}")
                else:
                    self.lbl_tin_stats.configure(
                        text=f"Всего: {total} | Активных: {n_active} | Исключено: {excl}")

                if has_tree and self._tin_selected_idx is not None and str(self._tin_selected_idx) in self.tree_tin.get_children():
                    try:
                        self.tree_tin.selection_set(str(self._tin_selected_idx))
                        self.tree_tin.see(str(self._tin_selected_idx))
                    except Exception:
                        pass
            else:
                if not show_top and not show_bottom:
                    self.lbl_tin_stats.configure(text="Поверхности отключены")
                elif show_top:
                    self.lbl_tin_stats.configure(text="Верхняя поверхность (без треугольников)")
                else:
                    self.lbl_tin_stats.configure(text="Нижняя поверхность (без треугольников)")
        finally:
            if has_tree:
                self._tin_updating_selection = False

        # 5. Масштабирование / сохранение вида
        if not reset_view and getattr(self, "__dict__", {}).get("_tin_view_initialized", False) and old_xlim and old_ylim:
            self.ax_tin.set_xlim(old_xlim)
            self.ax_tin.set_ylim(old_ylim)
            self.canvas_tin.draw()
        else:
            self._fit_tin_view()
        self._tin_dirty = False

    def _open_tin_table_dialog(self):
        """Открывает отдельное диалоговое окно со списком всех треугольников TIN"""
        if (getattr(self, "_tin_simplices", None) is None or len(self._tin_simplices) == 0) and len(getattr(self, "boundary_indices", [])) >= 3:
            try:
                self._redraw_tin(reset_view=False)
            except Exception:
                pass

        pts = getattr(self, "points", None)
        tris = getattr(self, "_tin_simplices", None)
        if tris is None or len(tris) == 0:
            tris = getattr(self, "_tin_calc_triangles", None) or getattr(self, "_tin_triangles", None)

        if pts is None or tris is None or len(tris) == 0:
            from tkinter import messagebox
            messagebox.showinfo("TIN Триангуляция", "Сетка треугольников еще не построена.\nСначала выполните триангуляцию.", parent=self)
            return

        def on_toggle(idx):
            if idx in self._tin_excluded:
                self._tin_excluded.discard(idx)
            else:
                self._tin_excluded.add(idx)
            self._redraw_tin(reset_view=False)

        TINTableDialog(
            parent=self,
            app=self,
            points=pts,
            triangles=tris,
            excluded_set=self._tin_excluded,
            on_toggle_callback=on_toggle
        )

    def _toggle_tin_table(self):
        """Открывает диалоговое окно со списком треугольников TIN"""
        self._open_tin_table_dialog()

    def _fit_tin_view(self):
        """Устанавливает масштаб и границы ax_tin точно в фокус отображаемых данных"""
        if not self.points:
            return

        pts_y = []
        pts_x = []

        if len(self.boundary_indices) >= 3:
            boundary_set = set(self.boundary_indices)
            bound_pts = np.array([[self.points[i].x, self.points[i].y] for i in self.boundary_indices])
            bound_path = MplPath(bound_pts)

            for i, p in enumerate(self.points):
                if i in boundary_set or bound_path.contains_point((p.x, p.y), radius=1e-5):
                    pts_y.append(p.y)
                    pts_x.append(p.x)

        if not pts_y or not pts_x:
            pts_y = [p.y for p in self.points]
            pts_x = [p.x for p in self.points]

        if not pts_y or not pts_x:
            return

        min_y, max_y = min(pts_y), max(pts_y)
        min_x, max_x = min(pts_x), max(pts_x)

        span_y = max(max_y - min_y, 1.0)
        span_x = max(max_x - min_x, 1.0)

        # Отступ 4% с каждой стороны для чистого центрирования
        pad_y = span_y * 0.04
        pad_x = span_x * 0.04

        self.ax_tin.set_xlim(min_y - pad_y, max_y + pad_y)
        self.ax_tin.set_ylim(min_x - pad_x, max_x + pad_x)
        self.ax_tin.set_aspect("equal", adjustable="datalim")
        self._tin_view_initialized = True
        self._apply_fig_layout(self.fig_tin, pad=0.5)
        self.canvas_tin.draw_idle()

    def _on_tabview_change(self):
        """Обработка переключения вкладок — актуализация содержимого по dirty-флагам и центрирование"""
        try:
            selected_tab = self.tabview.get()

            if selected_tab == self.TAB_TIN:
                if getattr(self, "_tin_dirty", False):
                    self._redraw_tin(reset_view=False)
                    self._tin_dirty = False
                elif hasattr(self, "canvas_tin") and self.canvas_tin is not None:
                    self._fit_tin_view()
            elif selected_tab == getattr(self, "TAB_CONTOURS", "Горизонтали"):
                if getattr(self, "_contours_dirty", False):
                    self._redraw_contours()
                    self._contours_dirty = False
                elif not getattr(self, "_contours_view_initialized", False):
                    self._redraw_contours()
                    self._contours_dirty = False
                else:
                    if hasattr(self, "fig_contours") and self.fig_contours is not None:
                        self._apply_fig_layout(self.fig_contours, pad=0.5)
                    if hasattr(self, "canvas_contours") and self.canvas_contours is not None:
                        self.canvas_contours.draw_idle()
            elif selected_tab == self.TAB_TABLE:
                if getattr(self, "_table_dirty", False):
                    self._update_table()
                    self._table_dirty = False
            elif selected_tab == self.TAB_2D:
                if getattr(self, "_2d_dirty", False):
                    self._redraw_2d()
                    self._2d_dirty = False
                else:
                    if hasattr(self, "fig_2d") and self.fig_2d is not None:
                        self._apply_fig_layout(self.fig_2d, pad=0.5)
                    if hasattr(self, "canvas_2d") and self.canvas_2d is not None:
                        self.canvas_2d.draw_idle()
                        if self._has_tk_canvas_widget():
                            self._update_tk_canvas_labels_positions()
            elif selected_tab == self.TAB_DIFF:
                if getattr(self, "_diff_dirty", False) and self.calc_results:
                    self._redraw_diff(self.calc_results)
                    self._diff_dirty = False
                else:
                    if hasattr(self, "fig_diff") and self.fig_diff is not None:
                        self._apply_fig_layout(self.fig_diff, pad=0.5)
                    if hasattr(self, "canvas_diff") and self.canvas_diff is not None:
                        self.canvas_diff.draw_idle()
            elif selected_tab == self.TAB_3D:
                if getattr(self, "_3d_dirty", False) and self.calc_results:
                    self._redraw_3d(self.calc_results)
                    self._3d_dirty = False
                else:
                    if hasattr(self, "fig_3d") and self.fig_3d is not None:
                        self._apply_fig_layout(self.fig_3d, pad=0.5)
                    if hasattr(self, "canvas_3d") and self.canvas_3d is not None:
                        self.canvas_3d.draw_idle()
            self._lift_canvas_overlays()
        except Exception:
            pass

    def _on_tin_table_select(self, event):
        """При выборе строки в таблице TIN — визуально выделяет треугольник"""
        if getattr(self, "_tin_updating_selection", False):
            return
        sel = self.tree_tin.selection()
        if sel:
            try:
                new_idx = int(sel[0])
                if new_idx != self._tin_selected_idx:
                    self._tin_selected_idx = new_idx
                    self._highlight_tin_selection()
            except Exception:
                pass


    def _tin_toggle_selected(self):
        """Переключает статус выбранного треугольника: активен ↔ исключён"""
        sel = self.tree_tin.selection()
        if not sel:
            messagebox.showinfo("Информация", "Выберите треугольник в таблице или кликните по нему на схеме.")
            return
        for item in sel:
            idx = int(item)
            if idx in self._tin_excluded:
                self._tin_excluded.discard(idx)
            else:
                self._tin_excluded.add(idx)
        self._redraw_tin(reset_view=False)
        self.save_project_state(self._current_file_path)

    def _tin_reset(self):
        """Восстанавливает все исключённые треугольники"""
        self._tin_excluded.clear()
        self._redraw_tin(reset_view=False)
        self.save_project_state(self._current_file_path)

    def _tin_reset_to_delaunay(self):
        """Сбрасывает все ручные переброски рёбер обратно к исходной триангуляции Делоне"""
        self._tin_custom_simplices = None
        self._redraw_tin(reset_view=False)
        self.save_project_state(self._current_file_path)
        if len(self.boundary_indices) >= 3:
            self.calculate_volume(use_tin=True)
        if self._tin_simplices is not None:
            self.lbl_tin_stats.configure(text=f"Триангуляция Делоне ({len(self._tin_simplices)} треуг.)")

    def _auto_optimize_tin_edges(self):
        """Автоматическая оптимизация рёбер TIN по критерию максимизации суммарного объёма.

        Алгоритм:
        1. Строит интерполяторы z_top и z_bot из данных calc_results
        2. Итерационно перебирает все внутренние рёбра (не граничные)
        3. Для каждого ребра вычисляет локальный прирост объёма при переброске
        4. Если переброска увеличивает |V_cut| + |V_fill| локально — принимает её
        5. Повторяет до стабилизации (max 15 итераций)
        """
        if self._tin_simplices is None or len(self._tin_simplices) == 0:
            messagebox.showinfo(
                "Авто-оптимизация",
                "Сначала задайте контур на вкладке «2D схема в плане» и произведите расчёт объёма."
            )
            return

        # Нужны оба интерполятора поверхностей для оценки объёма
        if not self.calc_results:
            messagebox.showinfo(
                "Авто-оптимизация",
                "Сначала выполните расчёт объёма, чтобы загрузить данные поверхностей."
            )
            return

        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        r = self.calc_results

        # --- Подготовка интерполяторов z_top и z_bot ---
        top_pts  = r["top_surface_pts"]   # Nx3
        bot_pts  = r["bottom_surface_pts"] # Mx3

        top_2d = top_pts[:, :2]
        bot_2d = bot_pts[:, :2]

        interp_top_lin  = LinearNDInterpolator(top_2d, top_pts[:, 2])
        interp_top_near = NearestNDInterpolator(top_2d, top_pts[:, 2])
        interp_bot_lin  = LinearNDInterpolator(bot_2d, bot_pts[:, 2])
        interp_bot_near = NearestNDInterpolator(bot_2d, bot_pts[:, 2])

        # Выемка: V_cut dominant → правильный TIN УВЕЛИЧИВАЕТ объём
        # Насыпь: V_fill dominant → правильный TIN УМЕНЬШАЕТ объём
        work_type = r.get("work_type")
        if work_type == "fill":
            is_cut_dominant = False
        elif work_type == "cut":
            is_cut_dominant = True
        else:
            v_fill_total = abs(float(r.get("v_fill", 0) or 0))
            v_cut_total  = abs(float(r.get("v_cut",  0) or 0))
            is_cut_dominant = v_cut_total >= v_fill_total

        # maximize_volume=True → принимаем флип, если объём растёт (выемка)
        # maximize_volume=False → принимаем флип, если объём падает (насыпь)
        maximize_volume = is_cut_dominant
        work_type_label = "выемка" if is_cut_dominant else "насыпь"

        def _get_z(idx):
            """Возвращает (z_top, z_bot) для точки по индексу."""
            p = self.points[idx]
            xy = np.array([[p.x, p.y]])

            if not is_cut_dominant:
                # Насыпь: точка p лежит на верхней поверхности
                zt = p.h
                zb = float(interp_bot_lin(xy)[0])
                if np.isnan(zb):
                    zb = float(interp_bot_near(xy)[0])
            else:
                # Выемка: точка p лежит на нижней поверхности
                zb = p.h
                zt = float(interp_top_lin(xy)[0])
                if np.isnan(zt):
                    zt = float(interp_top_near(xy)[0])

            return zt, zb

        def _tri_volume(i0, i1, i2):
            """Локальный вклад треугольника в объём: area_xy × mean(z_top - z_bot)."""
            p0, p1, p2 = self.points[i0], self.points[i1], self.points[i2]
            # Площадь в плане
            cross_z = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)
            area_xy = 0.5 * abs(cross_z)
            # Средняя мощность слоя
            zt0, zb0 = _get_z(i0)
            zt1, zb1 = _get_z(i1)
            zt2, zb2 = _get_z(i2)
            mean_dh = ((zt0 - zb0) + (zt1 - zb1) + (zt2 - zb2)) / 3.0
            return area_xy * abs(mean_dh)   # всегда >= 0, максимизируем сумму

        def _ccw(p1x, p1y, p2x, p2y, p3x, p3y):
            return (p2x - p1x) * (p3y - p1y) - (p2y - p1y) * (p3x - p1x)

        def _min_triangle_angle_deg(i0, i1, i2):
            """Минимальный угол треугольника в градусах."""
            p0, p1, p2 = self.points[i0], self.points[i1], self.points[i2]

            def ang(ax, ay, bx, by, cx, cy):
                # Угол при вершине A
                abx, aby = bx - ax, by - ay
                acx, acy = cx - ax, cy - ay
                dot = abx * acx + aby * acy
                n1 = math.sqrt(abx * abx + aby * aby)
                n2 = math.sqrt(acx * acx + acy * acy)
                if n1 < 1e-12 or n2 < 1e-12:
                    return 0.0
                return math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))

            a0 = ang(p0.x, p0.y, p1.x, p1.y, p2.x, p2.y)
            a1 = ang(p1.x, p1.y, p0.x, p0.y, p2.x, p2.y)
            a2 = 180.0 - a0 - a1
            return min(a0, a1, a2)

        # Минимально допустимый угол — читаем из спинбокса
        try:
            min_angle_deg = max(1.0, min(60.0, float(self._tin_min_angle.get())))
        except (tk.TclError, ValueError):
            min_angle_deg = 7.0

        # Максимальное число итераций — читаем из спинбокса
        try:
            MAX_ITER = max(1, min(50, int(self._tin_max_iter.get())))
        except (tk.TclError, ValueError):
            MAX_ITER = 8

        # Предвычисляем множество рёбер контура для быстрой проверки
        b_len = len(self.boundary_indices)
        boundary_edges: set = set()
        if b_len >= 3:
            for k in range(b_len):
                b1 = self.boundary_indices[k]
                b2 = self.boundary_indices[(k + 1) % b_len]
                boundary_edges.add((min(b1, b2), max(b1, b2)))

        # Предвычисляем bound_path для проверки принадлежности новой середины контуру
        bound_path = None
        if b_len >= 3:
            bound_pts_arr = np.array([[self.points[i].x, self.points[i].y]
                                       for i in self.boundary_indices])
            bound_path = MplPath(bound_pts_arr)

        total_flips = 0
        snap_undo   = []   # список (t1, t2, old_t1, old_t2, new_t1, new_t2) для группового undo

        self.lbl_tin_stats.configure(text="⏳ Авто-оптимизация…")
        self.update_idletasks()

        for iteration in range(MAX_ITER):
            flips_this_iter = 0
            modified_in_iter: set = set()  # треугольники, затронутые в этой итерации

            # Строим карту рёбер заново каждую итерацию (симплексы могут меняться)
            simplices = self._tin_simplices
            edge_map: dict = {}
            for s_idx, simplex in enumerate(simplices):
                v0, v1, v2 = int(simplex[0]), int(simplex[1]), int(simplex[2])
                for a, b in [(v0, v1), (v1, v2), (v2, v0)]:
                    e = (min(a, b), max(a, b))
                    if e not in edge_map:
                        edge_map[e] = []
                    edge_map[e].append(s_idx)

            for (u, v), s_indices in edge_map.items():
                # Только внутренние рёбра (2 смежных треугольника)
                if len(s_indices) != 2:
                    continue
                # Не трогаем рёбра внешнего контура
                if (min(u, v), max(u, v)) in boundary_edges:
                    continue

                t1, t2 = s_indices[0], s_indices[1]

                # Пропускаем треугольники, уже затронутые в этой итерации —
                # чтение их состояния было бы некорректным
                if t1 in modified_in_iter or t2 in modified_in_iter:
                    continue

                simp1 = list(self._tin_simplices[t1])
                simp2 = list(self._tin_simplices[t2])

                c_cands = [pt for pt in simp1 if pt != u and pt != v]
                d_cands = [pt for pt in simp2 if pt != u and pt != v]
                if not c_cands or not d_cands:
                    continue
                c, d = c_cands[0], d_cands[0]

                pu = self.points[u]; pv = self.points[v]
                pc = self.points[c]; pd = self.points[d]

                # Проверка строгой выпуклости четырёхугольника
                ccw_uv_c = _ccw(pu.x, pu.y, pv.x, pv.y, pc.x, pc.y)
                ccw_uv_d = _ccw(pu.x, pu.y, pv.x, pv.y, pd.x, pd.y)
                ccw_cd_u = _ccw(pc.x, pc.y, pd.x, pd.y, pu.x, pu.y)
                ccw_cd_v = _ccw(pc.x, pc.y, pd.x, pd.y, pv.x, pv.y)
                if not (ccw_uv_c * ccw_uv_d < -1e-9 and ccw_cd_u * ccw_cd_v < -1e-9):
                    continue

                # Проверка: середина нового ребра внутри контура
                mid_x = (pc.x + pd.x) / 2.0
                mid_y = (pc.y + pd.y) / 2.0
                if bound_path is not None and not bound_path.contains_point((mid_x, mid_y), radius=1e-5):
                    continue

                # Строим два новых треугольника (c-d вместо u-v)
                if _ccw(pc.x, pc.y, pd.x, pd.y, pu.x, pu.y) > 0:
                    new_t1 = (c, d, u)
                else:
                    new_t1 = (c, u, d)
                if _ccw(pc.x, pc.y, pd.x, pd.y, pv.x, pv.y) > 0:
                    new_t2 = (c, d, v)
                else:
                    new_t2 = (c, v, d)

                # Проверка площади новых треугольников (не создаём вырожденных)
                def _area2d(i0, i1, i2):
                    q0, q1, q2 = self.points[i0], self.points[i1], self.points[i2]
                    return 0.5 * abs((q1.x - q0.x)*(q2.y - q0.y) - (q1.y - q0.y)*(q2.x - q0.x))

                if _area2d(*new_t1) < 1e-10 or _area2d(*new_t2) < 1e-10:
                    continue  # новые треугольники вырожденные — пропускаем

                # Проверка минимального угла новых треугольников
                if (_min_triangle_angle_deg(*new_t1) < min_angle_deg or
                        _min_triangle_angle_deg(*new_t2) < min_angle_deg):
                    continue  # флип создаёт слишком узкий/длинный треугольник — пропускаем

                # Объём текущих двух треугольников
                v_old = _tri_volume(*simp1) + _tri_volume(*simp2)
                # Объём новых двух треугольников
                v_new = _tri_volume(*new_t1) + _tri_volume(*new_t2)

                # Критерий принятия флипа зависит от типа работ:
                # Выемка → максимизируем объём (v_new > v_old)
                # Насыпь → минимизируем объём (v_new < v_old)
                if maximize_volume:
                    flip_improves = v_new > v_old + 1e-9
                else:
                    flip_improves = v_new < v_old - 1e-9

                if flip_improves:
                    old_t1 = tuple(simp1)
                    old_t2 = tuple(simp2)
                    snap_undo.append(("tin_flip", t1, t2, old_t1, old_t2, new_t1, new_t2))

                    simplices_list = [list(s) for s in self._tin_simplices]
                    simplices_list[t1] = list(new_t1)
                    simplices_list[t2] = list(new_t2)
                    self._tin_simplices = np.array(simplices_list)
                    self._tin_custom_simplices = [tuple(s) for s in simplices_list]

                    self._tin_excluded.discard(t1)
                    self._tin_excluded.discard(t2)

                    modified_in_iter.add(t1)
                    modified_in_iter.add(t2)
                    flips_this_iter += 1
                    total_flips += 1

            if flips_this_iter == 0:
                break  # Сходимость

        # Добавляем все флипы как одну группу в undo-стек
        if snap_undo:
            self._undo_stack.append(("auto_flip_group", snap_undo))

        # Перерисовываем TIN
        self._redraw_tin(reset_view=False)

        if total_flips == 0:
            self.lbl_tin_stats.configure(text="⚡ Оптимизация: улучшений не найдено")
            messagebox.showinfo(
                "Авто-оптимизация",
                f"Триангуляция уже оптимальна при заданных ограничениях.\n\n"
                f"• Тип работ: {work_type_label} ({'максимизация' if maximize_volume else 'минимизация'} объёма)\n"
                f"• Мин. угол: ≥ {min_angle_deg:.0f}°"
            )
        else:
            self.lbl_tin_stats.configure(
                text=f"⚡ Оптимизировано: {total_flips} рёбер за {iteration + 1} итер."
            )
            # Пересчёт объёма с новой триангуляцией
            self.calculate_volume(use_tin=True)
            messagebox.showinfo(
                "Авто-оптимизация",
                f"Оптимизация завершена:\n"
                f"• Тип работ: {work_type_label} ({'максимизация' if maximize_volume else 'минимизация'} объёма)\n"
                f"• Переброшено рёбер: {total_flips}\n"
                f"• Итераций: {iteration + 1}\n"
                f"• Мин. угол: ≥ {min_angle_deg:.0f}°\n\n"
                "Объём пересчитан с новой триангуляцией.\n"
                "Для отмены используйте ↩ (Undo)."
            )




if __name__ == "__main__":
    app = VolumeApp()
    app.mainloop()
