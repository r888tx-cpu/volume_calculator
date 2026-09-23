# -*- coding: utf-8 -*-
"""
Создание релизного zip-архива дистрибутива GeoVolumePro.
Формат имени архива: GeoVolumePro_v{VERSION}_win64.zip
"""
import os
import re
import shutil
import zipfile
from typing import Optional


def get_app_version(base_dir: Optional[str] = None) -> str:
    """Извлекает номер версии из app_utils.py."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    app_utils_path = os.path.join(base_dir, "app_utils.py")
    if os.path.exists(app_utils_path):
        with open(app_utils_path, "r", encoding="utf-8") as f:
            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', f.read())
            if match:
                return match.group(1)
    return "1.0.0"


def create_dist_archive(base_dir: Optional[str] = None, dist_dir: Optional[str] = None) -> Optional[str]:
    """
    Создает релизный zip-архив в папке dist:
    dist/GeoVolumePro_v{VERSION}_win64.zip
    содержит корневую папку GeoVolumePro со всеми исполняемыми файлами и ресурсами.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    if dist_dir is None:
        dist_dir = os.path.join(base_dir, "dist")

    app_dir = os.path.join(dist_dir, "GeoVolumePro")
    if not os.path.isdir(app_dir):
        print(f"Директория приложения не найдена: {app_dir}")
        return None

    version = get_app_version(base_dir)
    archive_basename = f"GeoVolumePro_v{version}_win64"
    zip_path = os.path.join(dist_dir, f"{archive_basename}.zip")

    if os.path.exists(zip_path):
        try:
            os.remove(zip_path)
        except Exception as e:
            print(f"Предупреждение: не удалось удалить старый архив {zip_path}: {e}")

    zip_base = os.path.join(dist_dir, archive_basename)
    out_path = shutil.make_archive(zip_base, "zip", root_dir=dist_dir, base_dir="GeoVolumePro")

    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"Создан релизный архив: {out_path} ({size_mb:.1f} МБ)")
        return out_path
    return None


if __name__ == "__main__":
    create_dist_archive()
