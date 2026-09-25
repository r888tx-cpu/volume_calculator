# -*- coding: utf-8 -*-
"""
Утилита для корректного определения рабочей папки при запуске из .py и из скомпилированного .exe
"""
import sys
import os

__version__ = "1.1.0"
APP_NAME = "GeoVolume Pro"
APP_TITLE = f"{APP_NAME} v{__version__}"

def get_app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def get_projects_dir() -> str:
    p_dir = os.path.join(get_app_dir(), "Projects")
    os.makedirs(p_dir, exist_ok=True)
    return p_dir
