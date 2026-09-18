# -*- coding: utf-8 -*-
"""
Модуль вспомогательных пользовательских диалогов и окон GUI:
- PointEditDialog: редактирование точки
- CoordinateRemapDialog: переопределение колонок X/Y/H
- QuickAddPointDialog: добавление точки по клику
- FullScreen3DViewer: полноэкранный 3D просмотр
- ProjectNavigationToolbar: панель инструментов Matplotlib
- PlainOffsetFormatter: форматтер геодезических координат
- set_window_dark_titlebar: оформление заголовка окна в Windows
- calc_3d_stride: вычисление адаптивного шага прореживания для 3D поверхностей
"""
import os
import sys
import re
import math
import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import numpy as np

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import ScalarFormatter


def calc_3d_stride(grid: np.ndarray | None, target_dim: int = 45) -> tuple[int, int]:
    """
    Вычисляет адаптивный шаг прореживания (rstride, cstride) для ax.plot_surface,
    чтобы ограничить число полигонов сетки до ~target_dim x target_dim (по умолчанию ~2000 полигонов).
    Это обеспечивает гладкое интерактивное вращение 3D сцены (30+ FPS) в Matplotlib
    без визуальной деградации геометрии. Для сеток размером <= target_dim возвращает (1, 1).
    """
    if grid is None or not hasattr(grid, "shape") or len(grid.shape) < 2:
        return 2, 2
    rows, cols = grid.shape[:2]
    if rows <= 0 or cols <= 0:
        return 2, 2
    r_step = max(1, int(round(rows / target_dim)))
    c_step = max(1, int(round(cols / target_dim)))
    return r_step, c_step


def set_window_dark_titlebar(window):
    """Безопасно включает темную полосу заголовка окна в Windows 10/11 без скрытия (withdraw)."""
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetAncestor(window.winfo_id(), 2)  # GA_ROOT
        if not hwnd:
            hwnd = window.winfo_id()
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        is_dark = 1 if ctk.get_appearance_mode().lower() == "dark" else 0
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(ctypes.c_int(is_dark)), ctypes.sizeof(ctypes.c_int(is_dark))
        )
    except Exception:
        pass


class ProjectNavigationToolbar(NavigationToolbar2Tk):
    """Минималистичная панель инструментов: только сохранение графика и результатов."""
    toolitems = [
        ("Save", "Сохранить график в проект", "filesave", "save_figure"),
    ]


class PlainOffsetFormatter(ScalarFormatter):
    """Форматтер осей: отображает базовое смещение координат полным числом."""
    def __init__(self, useOffset=True, useMathText=False):
        super().__init__(useOffset=useOffset, useMathText=useMathText)
        self.set_scientific(False)

    def _compute_offset(self):
        if not self._useOffset:
            self.offset = 0
            return
        locs = self._locs
        if not len(locs):
            self.offset = 0
            return
        vmin, vmax = sorted(self.axis.get_view_interval())
        locs = np.asarray(locs)
        locs = locs[(vmin <= locs) & (locs <= vmax)]
        if not len(locs):
            self.offset = 0
            return
        lmin, lmax = locs.min(), locs.max()
        if lmin == lmax or lmin <= 0 <= lmax:
            self.offset = 0
            return

        sign = 1 if lmin > 0 else -1
        abs_min, abs_max = sorted([abs(float(lmin)), abs(float(lmax))])
        span = abs_max - abs_min

        if abs_min < 1000:
            self.offset = 0
            return

        if span <= 10000:
            unit = 1000.0
        elif span <= 100000:
            unit = 10000.0
        else:
            unit = 10.0 ** math.floor(math.log10(span))

        base = math.floor(abs_min / unit) * unit
        if base == 0 or (abs_max - base) >= abs_max:
            self.offset = 0
        else:
            self.offset = sign * base

    def get_offset(self):
        if len(self._locs) == 0:
            return ""
        if self.offset:
            val = self.offset
            if abs(val - round(val)) < 1e-4:
                ival = int(round(val))
                return f"+{ival}" if ival > 0 else f"{ival}"
            else:
                s = f"{val:.3f}".rstrip("0").rstrip(".")
                return f"+{s}" if val > 0 else s
        return super().get_offset()


class PointEditDialog(tk.Toplevel):
    """Модальное диалоговое окно для добавления или редактирования точки."""
    def __init__(self, parent, title="Точка", point=None, default_id=""):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        bg = "#1a1e24" if ctk.get_appearance_mode().lower() == "dark" else "#f5f6f8"
        self.configure(bg=bg)
        set_window_dark_titlebar(self)
        self.transient(parent)
        self.grab_set()

        self.result = None  # (id, x, y, h, surface_type)

        self.geometry("380x310")

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        frame.columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text="ID / Имя точки:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=(0, 10))
        self.ent_id = ctk.CTkEntry(frame, width=200)
        self.ent_id.grid(row=0, column=1, sticky=tk.EW, pady=5)
        if point:
            self.ent_id.insert(0, str(point.id))
        else:
            self.ent_id.insert(0, str(default_id))

        ctk.CTkLabel(frame, text="Север X (м):").grid(row=1, column=0, sticky=tk.W, pady=5, padx=(0, 10))
        self.ent_x = ctk.CTkEntry(frame, width=200)
        self.ent_x.grid(row=1, column=1, sticky=tk.EW, pady=5)
        if point:
            self.ent_x.insert(0, f"{point.x:.3f}")

        ctk.CTkLabel(frame, text="Восток Y (м):").grid(row=2, column=0, sticky=tk.W, pady=5, padx=(0, 10))
        self.ent_y = ctk.CTkEntry(frame, width=200)
        self.ent_y.grid(row=2, column=1, sticky=tk.EW, pady=5)
        if point:
            self.ent_y.insert(0, f"{point.y:.3f}")

        ctk.CTkLabel(frame, text="Высота H (м):").grid(row=3, column=0, sticky=tk.W, pady=5, padx=(0, 10))
        self.ent_h = ctk.CTkEntry(frame, width=200)
        self.ent_h.grid(row=3, column=1, sticky=tk.EW, pady=5)
        if point:
            self.ent_h.insert(0, f"{point.h:.3f}")

        ctk.CTkLabel(frame, text="Тип поверхности:").grid(row=4, column=0, sticky=tk.W, pady=5, padx=(0, 10))
        self.cbo_surf = ctk.CTkComboBox(frame, values=["auto", "top", "bottom", "boundary"], width=200)
        self.cbo_surf.grid(row=4, column=1, sticky=tk.EW, pady=5)
        if point:
            cur_surf = point.surface_type if point.surface_type in ["auto", "top", "bottom", "boundary"] else "auto"
            self.cbo_surf.set(cur_surf)
        else:
            self.cbo_surf.set("auto")

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.grid(row=5, column=0, columnspan=2, pady=(15, 0), sticky=tk.E)

        ctk.CTkButton(btn_frame, text="Отмена", width=100,
                      fg_color="gray40", hover_color="gray30",
                      command=self.destroy).pack(side=tk.RIGHT, padx=(5, 0))
        ctk.CTkButton(btn_frame, text="Сохранить", width=110,
                      command=self._on_save).pack(side=tk.RIGHT)

        self.bind("<Return>", lambda e: self._on_save())
        self.bind("<Escape>", lambda e: self.destroy())

        if point:
            self.ent_x.focus_set()
        else:
            self.ent_id.focus_set()

        self.wait_window(self)

    def destroy(self):
        try:
            self.grab_release()
        except Exception:
            pass
        super().destroy()

    def _on_save(self):
        id_str = self.ent_id.get().strip()
        if not id_str:
            messagebox.showwarning("Внимание", "Укажите ID / имя точки.", parent=self)
            return

        try:
            x_val = float(self.ent_x.get().replace(",", ".").strip())
        except ValueError:
            messagebox.showwarning("Внимание", "Некорректная координата X (Север).", parent=self)
            return

        try:
            y_val = float(self.ent_y.get().replace(",", ".").strip())
        except ValueError:
            messagebox.showwarning("Внимание", "Некорректная координата Y (Восток).", parent=self)
            return

        try:
            h_val = float(self.ent_h.get().replace(",", ".").strip())
        except ValueError:
            messagebox.showwarning("Внимание", "Некорректная высота H.", parent=self)
            return

        surf_type = self.cbo_surf.get()
        self.result = (id_str, x_val, y_val, h_val, surf_type)
        self.destroy()


class CoordinateRemapDialog(tk.Toplevel):
    """Диалог для ручного переопределения колонок координат (ID, X, Y, Z)."""
    def __init__(self, parent, filepath: str, initial_mapping: dict = None):
        super().__init__(parent)
        file_label = os.path.basename(filepath) if filepath else ""
        self.title(f"🔀 Переопределение координат — {file_label}" if file_label else "🔀 Переопределение координат (X, Y, Z)")

        dlg_w, dlg_h = 860, 620
        self.minsize(800, 560)

        try:
            parent.update_idletasks()
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            x = max(0, px + (pw - dlg_w) // 2)
            y = max(0, py + (ph - dlg_h) // 2)
            self.geometry(f"{dlg_w}x{dlg_h}+{x}+{y}")
        except Exception:
            self.geometry(f"{dlg_w}x{dlg_h}")

        bg = "#1a1e24" if ctk.get_appearance_mode().lower() == "dark" else "#f5f6f8"
        self.configure(bg=bg)
        set_window_dark_titlebar(self)

        self.transient(parent)
        self.grab_set()

        self.filepath = filepath
        self.result_mapping = None

        self.sample_rows = []
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8-sig", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("//"):
                        continue
                    if "," in line or ";" in line or "\t" in line:
                        parts = [p.strip() for p in re.split(r"[,;\t]+", line) if p.strip()]
                    else:
                        parts = [p.strip() for p in line.split() if p.strip()]
                    if parts:
                        self.sample_rows.append(parts)
                    if len(self.sample_rows) >= 12:
                        break

        self.num_cols = max([len(r) for r in self.sample_rows]) if self.sample_rows else 4
        self._col_options = [f"Колонка {i+1}" for i in range(self.num_cols)]
        self._init_ui(initial_mapping)
        self.wait_window(self)

    def destroy(self):
        try:
            self.grab_release()
        except Exception:
            pass
        super().destroy()

    def _init_ui(self, initial_mapping):
        is_dark = ctk.get_appearance_mode().lower() == "dark"

        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=14)

        header_lbl = ctk.CTkLabel(
            main_frame,
            text="Укажите, какие колонки файла соответствуют ID, X (Север), Y (Восток) и Z (Высота):",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        header_lbl.pack(anchor=tk.W, pady=(0, 10))

        preview_outer = ctk.CTkFrame(main_frame)
        preview_outer.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        file_label = os.path.basename(self.filepath) if self.filepath else ""
        preview_title = ctk.CTkLabel(
            preview_outer,
            text=f" Предпросмотр исходных данных файла ({file_label}) " if file_label else " Предпросмотр исходных данных файла ",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        preview_title.pack(anchor=tk.W, padx=10, pady=(8, 4))

        preview_frame = ctk.CTkFrame(preview_outer, fg_color="transparent")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        tree_bg = "#25282c" if is_dark else "#ffffff"
        tree_fg = "#f0f0f0" if is_dark else "#202020"
        head_bg = "#343a40" if is_dark else "#e9ecef"
        head_fg = "#ffffff" if is_dark else "#101010"
        sel_bg  = "#1f538d" if is_dark else "#3b8ed0"

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Remap.Treeview",
                        background=tree_bg,
                        foreground=tree_fg,
                        fieldbackground=tree_bg,
                        font=("Consolas", 11),
                        rowheight=28)
        style.configure("Remap.Treeview.Heading",
                        background=head_bg,
                        foreground=head_fg,
                        font=("Segoe UI", 11, "bold"),
                        padding=(6, 6))
        style.map("Remap.Treeview",
                  background=[("selected", sel_bg)],
                  foreground=[("selected", "#ffffff")])

        col_names = [f"col_{i}" for i in range(self.num_cols)]
        self.tree_preview = ttk.Treeview(
            preview_frame,
            columns=col_names,
            show="headings",
            height=8,
            style="Remap.Treeview"
        )
        for i in range(self.num_cols):
            self.tree_preview.heading(f"col_{i}", text=f"Колонка {i+1}")
            self.tree_preview.column(f"col_{i}", width=160, minwidth=130, anchor=tk.CENTER, stretch=True)

        scroll_x = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.tree_preview.xview)
        scroll_y = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.tree_preview.yview)
        self.tree_preview.configure(xscroll=scroll_x.set, yscroll=scroll_y.set)

        self.tree_preview.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        for row in self.sample_rows:
            padded_row = row + [""] * (self.num_cols - len(row))
            self.tree_preview.insert("", tk.END, values=padded_row)

        map_outer = ctk.CTkFrame(main_frame)
        map_outer.pack(fill=tk.X, pady=(0, 12))

        ctk.CTkLabel(
            map_outer,
            text=" Назначение осей и колонок ",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor=tk.W, padx=10, pady=(8, 6))

        map_frame = ctk.CTkFrame(map_outer, fg_color="transparent")
        map_frame.pack(fill=tk.X, padx=12, pady=(0, 10))
        map_frame.columnconfigure(1, weight=1)
        map_frame.columnconfigure(3, weight=1)

        col_options = self._col_options

        def_id = 0
        def_x = 1 if self.num_cols > 1 else 0
        def_y = 2 if self.num_cols > 2 else 0
        def_z = 3 if self.num_cols > 3 else 0

        if self.sample_rows:
            first_row = self.sample_rows[0]
            for idx, p in enumerate(first_row):
                if ":" in p:
                    continue
                clean = p.replace(",", ".")
                try:
                    val = float(clean)
                    int_len = len(clean.lstrip("+-").split(".")[0])
                    if int_len == 6:
                        def_x = idx
                    elif int_len == 7:
                        def_y = idx
                    elif int_len <= 3 and idx not in [def_x, def_y]:
                        def_z = idx
                except ValueError:
                    pass

        if initial_mapping:
            def_id = initial_mapping.get("id", def_id)
            def_x = initial_mapping.get("x", def_x)
            def_y = initial_mapping.get("y", def_y)
            def_z = initial_mapping.get("z", def_z)

        f_lbl = ctk.CTkFont(size=12)
        f_lbl_b = ctk.CTkFont(size=12, weight="bold")
        f_cbo = ctk.CTkFont(size=12)
        f_drop = ctk.CTkFont(size=11)

        ctk.CTkLabel(map_frame, text="ID / Имя:", font=f_lbl).grid(row=0, column=0, sticky=tk.W, padx=(4, 6), pady=6)
        self.cbo_id = ctk.CTkComboBox(map_frame, values=col_options, width=170, font=f_cbo, dropdown_font=f_drop)
        self.cbo_id.grid(row=0, column=1, sticky=tk.EW, padx=(0, 14), pady=6)
        self.cbo_id.set(col_options[min(def_id, self.num_cols - 1)])

        ctk.CTkLabel(map_frame, text="Север X (м):", font=f_lbl_b).grid(row=0, column=2, sticky=tk.W, padx=(6, 6), pady=6)
        self.cbo_x = ctk.CTkComboBox(map_frame, values=col_options, width=170, font=f_cbo, dropdown_font=f_drop)
        self.cbo_x.grid(row=0, column=3, sticky=tk.EW, padx=(0, 16), pady=6)
        self.cbo_x.set(col_options[min(def_x, self.num_cols - 1)])

        ctk.CTkLabel(map_frame, text="Высота Z / H (м):", font=f_lbl).grid(row=1, column=0, sticky=tk.W, padx=(4, 6), pady=6)
        self.cbo_z = ctk.CTkComboBox(map_frame, values=col_options, width=170, font=f_cbo, dropdown_font=f_drop)
        self.cbo_z.grid(row=1, column=1, sticky=tk.EW, padx=(0, 14), pady=6)
        self.cbo_z.set(col_options[min(def_z, self.num_cols - 1)])

        ctk.CTkLabel(map_frame, text="Восток Y (м):", font=f_lbl_b).grid(row=1, column=2, sticky=tk.W, padx=(6, 6), pady=6)
        self.cbo_y = ctk.CTkComboBox(map_frame, values=col_options, width=170, font=f_cbo, dropdown_font=f_drop)
        self.cbo_y.grid(row=1, column=3, sticky=tk.EW, padx=(0, 16), pady=6)
        self.cbo_y.set(col_options[min(def_y, self.num_cols - 1)])

        btn_swap = ctk.CTkButton(
            map_frame,
            text="🔄 Поменять X ↔ Y",
            command=self._swap_xy,
            width=180,
            height=38,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        btn_swap.grid(row=0, column=4, rowspan=2, padx=(6, 4), pady=6, sticky=tk.NSEW)

        btn_box = ctk.CTkFrame(main_frame, fg_color="transparent")
        btn_box.pack(fill=tk.X, pady=(2, 0))

        ctk.CTkButton(
            btn_box,
            text="Отмена",
            width=110,
            height=36,
            font=ctk.CTkFont(size=12),
            fg_color="gray40",
            hover_color="gray30",
            command=self.destroy
        ).pack(side=tk.RIGHT, padx=(8, 0))

        ctk.CTkButton(
            btn_box,
            text="✓ Применить координаты",
            width=210,
            height=36,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_apply
        ).pack(side=tk.RIGHT)

    def _swap_xy(self):
        val_x = self.cbo_x.get()
        val_y = self.cbo_y.get()
        self.cbo_x.set(val_y)
        self.cbo_y.set(val_x)

    def _on_apply(self):
        col_options = self._col_options
        def _idx(cbo):
            val = cbo.get()
            return col_options.index(val) if val in col_options else 0
        self.result_mapping = {
            "id": _idx(self.cbo_id),
            "x":  _idx(self.cbo_x),
            "y":  _idx(self.cbo_y),
            "z":  _idx(self.cbo_z),
        }
        self.destroy()


class QuickAddPointDialog(tk.Toplevel):
    """Быстрый диалог добавления точки по клику на схеме (X, Y уже известны из клика)."""
    def __init__(self, parent, x_val: float, y_val: float, default_id: str = ""):
        super().__init__(parent)
        self.title("➕ Новая точка по клику")
        self.resizable(False, False)
        bg = "#1a1e24" if ctk.get_appearance_mode().lower() == "dark" else "#f5f6f8"
        self.configure(bg=bg)
        set_window_dark_titlebar(self)
        self.transient(parent)
        self.grab_set()

        self.result = None  # (id, x, y, h, surface_type)
        self._x = x_val
        self._y = y_val

        self.geometry("380x280")

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        frame.columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text=f"X (Север): {x_val:.3f}  |  Y (Восток): {y_val:.3f}",
                     font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 12))

        ctk.CTkLabel(frame, text="ID / Имя точки:").grid(row=1, column=0, sticky=tk.W, pady=4, padx=(0, 10))
        self.ent_id = ctk.CTkEntry(frame, width=200)
        self.ent_id.grid(row=1, column=1, sticky=tk.EW, pady=4)
        self.ent_id.insert(0, default_id)

        ctk.CTkLabel(frame, text="Высота H (м):").grid(row=2, column=0, sticky=tk.W, pady=4, padx=(0, 10))
        self.ent_h = ctk.CTkEntry(frame, width=200)
        self.ent_h.grid(row=2, column=1, sticky=tk.EW, pady=4)

        ctk.CTkLabel(frame, text="Тип поверхности:").grid(row=3, column=0, sticky=tk.W, pady=4, padx=(0, 10))
        self.cbo_surf = ctk.CTkComboBox(frame, values=["auto", "top", "bottom", "boundary"], width=200)
        self.cbo_surf.grid(row=3, column=1, sticky=tk.EW, pady=4)
        self.cbo_surf.set("auto")

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.grid(row=4, column=0, columnspan=2, pady=(14, 0), sticky=tk.E)
        ctk.CTkButton(btn_frame, text="Отмена", width=100,
                      fg_color="gray40", hover_color="gray30",
                      command=self.destroy).pack(side=tk.RIGHT, padx=(5, 0))
        ctk.CTkButton(btn_frame, text="Добавить ✓", width=120,
                      command=self._on_save).pack(side=tk.RIGHT)

        self.bind("<Return>", lambda e: self._on_save())
        self.bind("<Escape>", lambda e: self.destroy())
        self.ent_h.focus_set()
        self.wait_window(self)

    def destroy(self):
        try:
            self.grab_release()
        except Exception:
            pass
        super().destroy()

    def _on_save(self):
        id_str = self.ent_id.get().strip()
        if not id_str:
            messagebox.showwarning("Внимание", "Укажите ID / имя точки.", parent=self)
            return
        try:
            h_val = float(self.ent_h.get().replace(",", ".").strip())
        except ValueError:
            messagebox.showwarning("Внимание", "Некорректная высота H.", parent=self)
            return
        self.result = (id_str, self._x, self._y, h_val, self.cbo_surf.get())
        self.destroy()


class FullScreen3DViewer(tk.Toplevel):
    """Полноэкранное окно для детального и эстетичного изучения 3D модели рельефа."""
    def __init__(self, parent, calc_results, points, boundary_indices):
        super().__init__(parent)
        self.title("3D Модель рельефа (Полноэкранный просмотр)")
        self.calc_results = calc_results
        self.points = points
        self.boundary_indices = boundary_indices
        self._is_dark_theme = True
        self._show_axes = False
        self._is_fullscreen = False

        try:
            self.state("zoomed")
        except Exception:
            self.geometry("1200x800")

        self.configure(bg="#1a1e24")
        set_window_dark_titlebar(self)

        self.top_bar = tk.Frame(self, bg="#242b35", height=40)
        self.top_bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(4, 2))

        btn_fs = tk.Button(self.top_bar, text="⛶ Во весь экран (F11)", font=("Segoe UI", 9),
                           bg="#34495e", fg="white", activebackground="#2c3e50", activeforeground="white",
                           relief=tk.FLAT, padx=8, pady=2, command=self._toggle_fullscreen, cursor="hand2")
        btn_fs.pack(side=tk.LEFT, padx=(0, 6))

        btn_reset = tk.Button(self.top_bar, text="🔄 Исходный ракурс", font=("Segoe UI", 9),
                              bg="#34495e", fg="white", activebackground="#2c3e50", activeforeground="white",
                              relief=tk.FLAT, padx=8, pady=2, command=self._reset_view, cursor="hand2")
        btn_reset.pack(side=tk.LEFT, padx=(0, 6))

        btn_top = tk.Button(self.top_bar, text="🔝 Вид сверху (План)", font=("Segoe UI", 9),
                            bg="#34495e", fg="white", activebackground="#2c3e50", activeforeground="white",
                            relief=tk.FLAT, padx=8, pady=2, command=self._view_top, cursor="hand2")
        btn_top.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_axes = tk.Button(self.top_bar, text="👁 Оси и сетка: ВЫКЛ", font=("Segoe UI", 9),
                                  bg="#2c3e50", fg="#bdc3c7", activebackground="#34495e", activeforeground="white",
                                  relief=tk.FLAT, padx=8, pady=2, command=self._toggle_axes, cursor="hand2")
        self.btn_axes.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_theme = tk.Button(self.top_bar, text="🌓 Тема: Тёмная", font=("Segoe UI", 9),
                                   bg="#2c3e50", fg="#bdc3c7", activebackground="#34495e", activeforeground="white",
                                   relief=tk.FLAT, padx=8, pady=2, command=self._toggle_theme, cursor="hand2")
        self.btn_theme.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_save_3d = tk.Button(self.top_bar, text="💾 Сохранить 3D", font=("Segoe UI", 9),
                                     bg="#27ae60", fg="white", activebackground="#2ecc71", activeforeground="white",
                                     relief=tk.FLAT, padx=8, pady=2, command=self._save_3d_image, cursor="hand2")
        self.btn_save_3d.pack(side=tk.LEFT, padx=(0, 10))

        lbl_hint = tk.Label(self.top_bar,
                            text="🖱 ЛКМ: вращение 360° | ПКМ: сдвиг | Колесо: зум | Esc: выход",
                            font=("Segoe UI", 9), bg="#242b35", fg="#7f8c8d")
        lbl_hint.pack(side=tk.RIGHT, padx=5)

        self.fig = Figure(dpi=100, facecolor="#1a1e24")
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<F11>", lambda e: self._toggle_fullscreen())

        self._orig_limits = None
        self._rmb_pan_start = None
        self._pan_timer = None
        self._plot_3d_surfaces()

    def _plot_3d_surfaces(self):
        r = self.calc_results
        if not r:
            return

        self.ax.clear()
        bg_color = "#1a1e24" if self._is_dark_theme else "#ffffff"
        self.fig.patch.set_facecolor(bg_color)
        self.ax.set_facecolor(bg_color)

        if r.get("custom_tin") and r.get("pts_3d") is not None and r.get("active_simplices") is not None and len(r["active_simplices"]) > 0:
            pts_3d = r["pts_3d"]
            active_s = r["active_simplices"]
            edge_color = "#3d2212" if self._is_dark_theme else "#5c3818"
            self.ax.plot_trisurf(pts_3d[:, 1], pts_3d[:, 0], pts_3d[:, 2],
                                 triangles=active_s, cmap="copper", alpha=0.92,
                                 edgecolor=edge_color, linewidth=0.6, antialiased=True)
        else:
            r_step, c_step = calc_3d_stride(r.get("z_top_grid"), target_dim=45)
            self.ax.plot_surface(r["grid_y"], r["grid_x"], r["z_top_grid"],
                                 cmap="copper", alpha=0.88, edgecolor="none", rstride=r_step, cstride=c_step)

        if len(self.boundary_indices) >= 3:
            b_pts = r["boundary"]
            self.ax.plot(list(b_pts[:, 1]) + [b_pts[0, 1]],
                         list(b_pts[:, 0]) + [b_pts[0, 0]],
                         list(b_pts[:, 2]) + [b_pts[0, 2]],
                         color="#2ecc71", linewidth=3.5, label="Контур")

        if not self._show_axes:
            self.ax.set_axis_off()
        else:
            self.ax.set_axis_on()
            text_color = "#ecf0f1" if self._is_dark_theme else "#2c3e50"
            self.ax.set_xlabel("Восток Y (м)", color=text_color)
            self.ax.set_ylabel("Север X (м)", color=text_color)
            self.ax.set_zlabel("Высота H (м)", color=text_color)
            self.ax.tick_params(colors=text_color)

        def _pad(lo, hi, pct):
            d = max((hi - lo) * pct, 0.1)
            return lo - d, hi + d

        grid_y = r.get("grid_y")
        grid_x = r.get("grid_x")
        z_top_g = r.get("z_top_grid")
        z_bot_g = r.get("z_bot_grid")

        if grid_y is not None and grid_x is not None:
            self.ax.set_xlim(*_pad(float(np.nanmin(grid_y)), float(np.nanmax(grid_y)), 0.05))
            self.ax.set_ylim(*_pad(float(np.nanmin(grid_x)), float(np.nanmax(grid_x)), 0.05))

        if z_top_g is not None:
            z_vals = z_top_g[~np.isnan(z_top_g)].ravel()
            if z_bot_g is not None:
                z_vals = np.concatenate([z_vals, z_bot_g[~np.isnan(z_bot_g)].ravel()])
            if len(z_vals) > 0:
                self.ax.set_zlim(*_pad(float(np.min(z_vals)), float(np.max(z_vals)), 0.08))

        if self._orig_limits is None:
            self._orig_limits = (self.ax.get_xlim(), self.ax.get_ylim(), self.ax.get_zlim())

        self.canvas.draw()

    def _on_mouse_press(self, event):
        if event.button == 3:
            self._rmb_pan_start = (
                event.x, event.y,
                self.ax.get_xlim(),
                self.ax.get_ylim(),
            )

    def _on_mouse_motion(self, event):
        if self._rmb_pan_start is None or event.button != 3:
            return
        if event.x is None or event.y is None:
            return

        x0, y0, orig_xlim, orig_ylim = self._rmb_pan_start
        dx_px = event.x - x0
        dy_px = event.y - y0

        canvas_w = max(self.canvas.get_tk_widget().winfo_width(), 1)
        canvas_h = max(self.canvas.get_tk_widget().winfo_height(), 1)

        span_x = orig_xlim[1] - orig_xlim[0]
        span_y = orig_ylim[1] - orig_ylim[0]

        shift_x = -dx_px / canvas_w * span_x * 2.0
        shift_y =  dy_px / canvas_h * span_y * 2.0

        self.ax.set_xlim(orig_xlim[0] + shift_x, orig_xlim[1] + shift_x)
        self.ax.set_ylim(orig_ylim[0] + shift_y, orig_ylim[1] + shift_y)
        if getattr(self, "_pan_timer", None) is None:
            def _delayed():
                self._pan_timer = None
                try:
                    self.canvas.draw_idle()
                except Exception:
                    pass
            self._pan_timer = self.after(25, _delayed)

    def _on_mouse_release(self, event):
        if event.button == 3:
            self._rmb_pan_start = None
            if getattr(self, "_pan_timer", None) is not None:
                try:
                    self.after_cancel(self._pan_timer)
                except Exception:
                    pass
                self._pan_timer = None
            self.canvas.draw_idle()

    def _toggle_fullscreen(self):
        self._is_fullscreen = not self._is_fullscreen
        self.attributes("-fullscreen", self._is_fullscreen)

    def _reset_view(self):
        self.ax.view_init(elev=35, azim=-60)
        if self._orig_limits:
            self.ax.set_xlim(self._orig_limits[0])
            self.ax.set_ylim(self._orig_limits[1])
            self.ax.set_zlim(self._orig_limits[2])
        self.canvas.draw_idle()

    def _view_top(self):
        self.ax.view_init(elev=90, azim=-90)
        self.canvas.draw_idle()

    def _toggle_axes(self):
        self._show_axes = not self._show_axes
        state_text = "ВКЛ" if self._show_axes else "ВЫКЛ"
        self.btn_axes.config(text=f"👁 Оси и сетка: {state_text}")
        self._plot_3d_surfaces()

    def _toggle_theme(self):
        self._is_dark_theme = not self._is_dark_theme
        theme_text = "Тёмная" if self._is_dark_theme else "Светлая"
        self.btn_theme.config(text=f"🌓 Тема: {theme_text}")
        self._plot_3d_surfaces()

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        base_scale = 1.15
        scale = 1.0 / base_scale if event.button == "up" else base_scale

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()

        xm = 0.5 * (xlim[0] + xlim[1])
        ym = 0.5 * (ylim[0] + ylim[1])
        zm = 0.5 * (zlim[0] + zlim[1])

        dx = 0.5 * (xlim[1] - xlim[0]) * scale
        dy = 0.5 * (ylim[1] - ylim[0]) * scale
        dz = 0.5 * (zlim[1] - zlim[0]) * scale

        self.ax.set_xlim(xm - dx, xm + dx)
        self.ax.set_ylim(ym - dy, ym + dy)
        self.ax.set_zlim(zm - dz, zm + dz)
        self.canvas.draw_idle()

    def _save_3d_image(self):
        tab_name = getattr(self.master, "TAB_3D", "3D Поверхности")
        if hasattr(self.master, "_save_tab_image"):
            self.master._save_tab_image(self.fig, tab_name)
        else:
            try:
                out_path = f"{tab_name}.png"
                self.fig.savefig(out_path, dpi=300, bbox_inches="tight")
                messagebox.showinfo("Сохранено", f"Изображение сохранено:\n{out_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить изображение:\n{e}")
