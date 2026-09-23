# -*- coding: utf-8 -*-
import os
import zipfile
import pytest

from dist_packager import get_app_version, create_dist_archive
from app_utils import __version__


def test_get_app_version():
    ver = get_app_version()
    assert ver == __version__
    assert len(ver.split(".")) >= 2


def test_create_dist_archive_success(tmp_path):
    # Подготавливаем тестовую структуру
    base_dir = tmp_path
    app_utils_file = base_dir / "app_utils.py"
    app_utils_file.write_text('__version__ = "1.0.1"\n', encoding="utf-8")

    dist_dir = base_dir / "dist"
    app_dir = dist_dir / "GeoVolumePro"
    app_dir.mkdir(parents=True)

    exe_file = app_dir / "GeoVolumePro.exe"
    exe_file.write_bytes(b"MOCK_EXE_BYTES")

    readme_file = app_dir / "readme.html"
    readme_file.write_text("<h1>Manual</h1>", encoding="utf-8")

    # Создаем архив
    out_zip = create_dist_archive(base_dir=str(base_dir), dist_dir=str(dist_dir))
    assert out_zip is not None
    assert os.path.exists(out_zip)
    assert os.path.basename(out_zip) == "GeoVolumePro_v1.0.1_win64.zip"

    # Проверяем структуру внутри zip-архива
    with zipfile.ZipFile(out_zip, "r") as zf:
        namelist = zf.namelist()
        assert any(name.startswith("GeoVolumePro/") for name in namelist)
        assert "GeoVolumePro/GeoVolumePro.exe" in namelist
        assert "GeoVolumePro/readme.html" in namelist
        assert zf.read("GeoVolumePro/GeoVolumePro.exe") == b"MOCK_EXE_BYTES"


def test_create_dist_archive_missing_app_dir(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    res = create_dist_archive(base_dir=str(tmp_path), dist_dir=str(dist_dir))
    assert res is None


def test_create_dist_archive_overwrites_existing(tmp_path):
    base_dir = tmp_path
    app_utils_file = base_dir / "app_utils.py"
    app_utils_file.write_text('__version__ = "1.0.1"\n', encoding="utf-8")

    dist_dir = base_dir / "dist"
    app_dir = dist_dir / "GeoVolumePro"
    app_dir.mkdir(parents=True)
    (app_dir / "GeoVolumePro.exe").write_bytes(b"NEW_EXE")

    existing_zip = dist_dir / "GeoVolumePro_v1.0.1_win64.zip"
    existing_zip.write_bytes(b"OLD_ZIP_CONTENT")

    out_zip = create_dist_archive(base_dir=str(base_dir), dist_dir=str(dist_dir))
    assert out_zip == str(existing_zip)
    assert os.path.getsize(out_zip) > len(b"OLD_ZIP_CONTENT")
    with zipfile.ZipFile(out_zip, "r") as zf:
        assert "GeoVolumePro/GeoVolumePro.exe" in zf.namelist()
