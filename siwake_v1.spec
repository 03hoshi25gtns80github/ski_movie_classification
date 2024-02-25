# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['siwake_v1.py'],
    pathex=[],
    binaries=[],
     datas=[
        ('C:\\Users\\03hos\\anaconda3\\envs\\siwake1\\lib\\site-packages\\ipadic\\dicdir', 'ipadic\\dicdir'),
        ('model', 'model')  # ここでmodelディレクトリを追加
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='siwake_v1',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
