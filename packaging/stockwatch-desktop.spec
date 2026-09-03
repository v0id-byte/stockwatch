# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


ROOT = Path(SPECPATH).parent
MODEL_DIR = ROOT / "packaging" / "runtime-models"
datas = []
if MODEL_DIR.is_dir():
    datas.append((str(MODEL_DIR), "models"))
datas += collect_data_files("akshare")

hiddenimports = []
for package in ("akshare", "baostock", "lightgbm"):
    hiddenimports += collect_submodules(package)
hiddenimports += [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "core.agent",
    "dashboard",
    "bot.runner",
]
if sys.platform == "darwin":
    hiddenimports += collect_submodules("ServiceManagement")

a = Analysis(
    [str(ROOT / "desktop" / "app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["sklearn", "pytest", "scripts.train_lgbm", "scripts.train_risk_model"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StockWatch",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="StockWatch",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="StockWatch.app",
        icon=None,
        bundle_identifier="com.melspectrum.stockwatch",
        info_plist={
            "LSUIElement": True,
            "NSHighResolutionCapable": True,
            "NSHumanReadableCopyright": "StockWatch contributors",
        },
    )
