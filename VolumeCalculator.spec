# -*- mode: python ; coding: utf-8 -*-
import shutil, os
import PyInstaller.building.api as _pyi_api
import PyInstaller.building.utils as _pyi_utils
_orig_clean_dir = _pyi_api._make_clean_directory
def _safe_clean_dir(path):
    try:
        _orig_clean_dir(path)
    except OSError:
        if os.path.isdir(path):
            for item in os.listdir(path):
                p = os.path.join(path, item)
                try:
                    if os.path.isdir(p):
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        os.remove(p)
                except Exception:
                    pass
        os.makedirs(path, exist_ok=True)
_pyi_api._make_clean_directory = _safe_clean_dir
_pyi_utils._make_clean_directory = _safe_clean_dir
from PyInstaller.utils.hooks import collect_data_files

datas = [
    ('geo_parser.py', '.'),
    ('volume_engine.py', '.'),
    ('app_utils.py', '.'),
    ('interactive_app.py', '.'),
    ('ui_dialogs.py', '.'),
    ('spatial_index.py', '.'),
    ('project_storage.py', '.'),
    ('dxf_exporter.py', '.'),
    ('dxf_importer.py', '.'),
    ('app_icon.ico', '.'),
    ('readme.html', '.'),
    ('README.md', '.'),
    ('report_icon.png', '.'),
    ('report_icon_large.png', '.'),
    ('Projects', 'Projects'),
]
datas += collect_data_files('customtkinter')
datas += collect_data_files('tkinterdnd2')
datas += collect_data_files('ezdxf')

a = Analysis(
    ['volume_calc.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['tkinterdnd2', 'customtkinter', 'spatial_index', 'project_storage', 'ui_dialogs', 'dxf_exporter', 'dxf_importer', 'ezdxf'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyinstaller_runtime_hook.py'],
    excludes=[
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'PySide6.QtNetwork',
        'shiboken6',
        'PyQt5',
        'PyQt6',
        'qfluentwidgets',
        'qframelesswindow',
        'ezdxf.addons.xqt',
        'scipy.signal',
        'scipy.stats',
        'scipy.integrate',
        'scipy.cluster',
        'scipy.io',
        'scipy.ndimage',
        'PIL._avif',
        'matplotlib.tests',
        'matplotlib.testing',
        'pytest',
        'IPython',
    ],
    noarchive=False,
    optimize=2,
)

excluded_bin_patterns = [
    'pyside6',
    'shiboken6',
    'qt6',
    'opengl32sw',
    'libcrypto',
    'libssl',
    '_ssl',
]

filtered_binaries = []
for b in a.binaries:
    dest = b[0].lower()
    if any(p in dest for p in excluded_bin_patterns):
        continue
    filtered_binaries.append(b)

excluded_data_patterns = [
    'pyside6',
    'translations/qt',
    'translations\\qt',
]

filtered_datas = []
for d in a.datas:
    dest = d[0].lower()
    if any(p in dest for p in excluded_data_patterns):
        continue
    filtered_datas.append(d)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GeoVolumePro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory='_internal',
    icon='app_icon.ico',
)
coll = COLLECT(
    exe,
    filtered_binaries,
    filtered_datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='GeoVolumePro',
)

# Копируем readme.html, README.md, app_icon.ico и реальные проекты Projects в корень дистрибутива рядом с exe
_dist_root = os.path.join(DISTPATH, 'GeoVolumePro')
os.makedirs(_dist_root, exist_ok=True)
shutil.copy('readme.html', os.path.join(_dist_root, 'readme.html'))
if os.path.exists('README.md'):
    shutil.copy('README.md', os.path.join(_dist_root, 'README.md'))
shutil.copy('app_icon.ico', os.path.join(_dist_root, 'app_icon.ico'))
_dist_projects = os.path.join(_dist_root, 'Projects')
if os.path.exists('Projects'):
    shutil.copytree('Projects', _dist_projects, dirs_exist_ok=True)

# Автоматически создаем релизный zip-архив с номером версии (например, GeoVolumePro_v1.0.1_win64.zip)
try:
    import sys
    if SPECPATH not in sys.path:
        sys.path.insert(0, SPECPATH)
    from dist_packager import create_dist_archive
    create_dist_archive(base_dir=SPECPATH, dist_dir=DISTPATH)
except Exception as _e:
    print(f"Предупреждение при создании zip-архива: {_e}")

