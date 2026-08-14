# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

mujoco_datas, mujoco_binaries, mujoco_hidden = collect_all("mujoco")
glfw_binaries = collect_dynamic_libs("glfw")

analysis = Analysis(
    ["biped_demo.py"],
    pathex=[],
    binaries=mujoco_binaries + glfw_binaries,
    datas=mujoco_datas + [("policy/robot.xml", "policy"), ("agent_checkpoint.pt", ".")],
    hiddenimports=mujoco_hidden + ["glfw"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pygame",
        "tkinter",
        "tensorflow",
        "matplotlib",
        "scipy",
        "IPython",
        "jupyter",
        "notebook",
        "pandas",
        "sklearn",
        "PIL",
        "torchvision",
        "torchaudio",
        "torchtext",
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="BipedDemo",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=True,
)
