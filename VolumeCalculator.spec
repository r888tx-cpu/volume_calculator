# -*- mode: python ; coding: utf-8 -*-
import shutil, os
from PyInstaller.utils.hooks import collect_data_files

datas = [
    ('geo_parser.py', '.'),
    ('volume_engine.py', '.'),
    ('app_utils.py', '.'),
    ('interactive_app.py', '.'),
    ('ui_dialogs.py', '.'),
    ('spatial_index.py', '.'),
    ('project_storage.py', '.'),
    ('app_icon.ico', '.'),
    ('readme.html', '.'),
    ('report_icon.png', '.'),
    ('report_icon_large.png', '.'),
]
datas += collect_data_files('customtkinter')
datas += collect_data_files('tkinterdnd2')

a = Analysis(
    ['volume_calc.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['tkinterdnd2', 'customtkinter', 'spatial_index', 'project_storage', 'ui_dialogs'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyinstaller_runtime_hook.py'],
    excludes=[
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
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='GeoVolumePro',
)

# Копируем readme.html и app_icon.ico в корень дистрибутива рядом с exe (все ресурсы и иконки находятся в _internal)
_dist_root = os.path.join(DISTPATH, 'GeoVolumePro')
os.makedirs(_dist_root, exist_ok=True)
shutil.copy('readme.html', os.path.join(_dist_root, 'readme.html'))
shutil.copy('app_icon.ico', os.path.join(_dist_root, 'app_icon.ico'))
