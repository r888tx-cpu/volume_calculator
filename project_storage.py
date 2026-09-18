# -*- coding: utf-8 -*-
"""
Модуль для работы с файлами проектов (.volproj), сериализации и управления папками Projects.
Обеспечивает атомарную запись данных на диск с автоматическим созданием резервных копий (.bak).
"""
import os
import json
import re
import glob
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """Кастомный JSON-энкодер для безопасной сериализации numpy-типов."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


class ProjectStorageService:
    """Сервис для сохранения, загрузки и управления структурой проектов."""

    @staticmethod
    def save_project_file(proj_path: str, data: Dict[str, Any]) -> bool:
        """
        Атомарно сохраняет структуру данных проекта в файл proj_path.
        Создает резервную копию .bak и временный файл .tmp для исключения повреждения данных при сбоях питания.
        """
        temp_path = proj_path + ".tmp"
        bak_path = proj_path + ".bak"

        try:
            os.makedirs(os.path.dirname(proj_path), exist_ok=True)
            indent = None if len(data.get("points", [])) > 1000 else 2
            separators = (',', ':') if indent is None else None
            with open(temp_path, "w", encoding="utf-8") as f:
                if indent is None:
                    json.dump(data, f, ensure_ascii=False, cls=NumpyJSONEncoder, separators=separators)
                else:
                    json.dump(data, f, indent=indent, ensure_ascii=False, cls=NumpyJSONEncoder)
                f.flush()
                os.fsync(f.fileno())

            if os.path.exists(proj_path):
                try:
                    if os.path.exists(bak_path):
                        os.remove(bak_path)
                    os.replace(proj_path, bak_path)
                except Exception:
                    pass

            os.replace(temp_path, proj_path)
            return True
        except Exception as e:
            print(f"Ошибка сохранения проекта: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            return False

    @staticmethod
    def load_project_file(proj_path: str) -> Optional[Dict[str, Any]]:
        """
        Загружает данные проекта из файла proj_path.
        При повреждении основного файла автоматически пытается загрузить резервную копию .bak.
        Возвращает dict с данными проекта или None при ошибке.
        """
        for candidate_path in [proj_path, proj_path + ".bak"]:
            if not os.path.exists(candidate_path):
                continue
            try:
                with open(candidate_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                if raw.strip():
                    return json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue
            except Exception:
                return None
        return None

    @staticmethod
    def strip_date_from_folder_name(folder_name: str) -> str:
        """Возвращает базовое имя папки без даты-префикса ДД.ММ.ГГГГ_ или суффикса _ДД.ММ.ГГГГ."""
        name = re.sub(r"^\d{2}[._]\d{2}[._]\d{4}_?", "", folder_name).strip("_")
        name = re.sub(r"_?\d{2}[._]\d{2}[._]\d{4}(_\d+)?$", "", name).strip("_")
        return name if name else folder_name

    @staticmethod
    def create_project_folder(proj_root: str, base_name: str) -> Tuple[str, str]:
        """
        Создает подпапку проекта в папке Projects в формате «ДД.ММ.ГГГГ_ИмяПроекта».
        Возвращает (имя_папки, полный_путь_к_папке).
        """
        date_str = datetime.now().strftime("%d.%m.%Y")
        clean_base = re.sub(r'[<>:"/\\|?*]', "_", base_name).strip(". ")
        candidate_name = f"{date_str}_{clean_base}"
        target_dir = os.path.join(proj_root, candidate_name)

        counter = 1
        while os.path.exists(target_dir):
            candidate_name = f"{date_str}_{clean_base}_{counter}"
            target_dir = os.path.join(proj_root, candidate_name)
            counter += 1

        os.makedirs(target_dir, exist_ok=True)
        return candidate_name, target_dir

    @staticmethod
    def list_coordinate_files(folder_path: str) -> List[str]:
        """Возвращает список файлов координат внутри папки проекта."""
        coord_files = []
        for ext in ["*.txt", "*.csv", "*.dat", "*.xyz", "*.pts", "*.TXT", "*.CSV", "*.DAT"]:
            for f in glob.glob(os.path.join(folder_path, ext)):
                fn = os.path.basename(f)
                if not fn.endswith("_volume_report.txt") and not fn.endswith("_report.txt"):
                    coord_files.append(fn)
        return sorted(list(set(coord_files)))
