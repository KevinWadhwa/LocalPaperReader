# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build of the double-click app: one folder, and a .app on macOS. Entry point: desktop.py.

    pyinstaller --noconfirm bundlebuild/LocalPaperReader.spec     # from grpo/, with pyinstaller and pywebview installed

The language model and the adapter are NOT bundled: the app downloads them on first launch (modelfetch.py).
Built per platform: PyInstaller cannot build a Mac or Windows app from Linux.
"""
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

ROOT = os.path.dirname(SPECPATH)                    # grpo/

# what the app reads at run time, at the same relative paths app.resolve_paths expects
datas = [(os.path.join(ROOT, src), dst) for src, dst in (
    ("configs/reader_4b_gguf.json", "configs"),
    ("icons", "icons"),
    ("readergui/fonts", "readergui/fonts"),
    ("readergui/pdfjs", "readergui/pdfjs"),
    ("runs/tokenizer_4b", "runs/tokenizer_4b"),
    ("runs/embedder_onnx", "runs/embedder_onnx"),
)]
binaries = collect_dynamic_libs("llama_cpp")        # libllama / libggml: loaded by path, no hook ships for them
hiddenimports = ["tokenizers", "regex", "jinja2", "yaml", "certifi"]

# gradio serves its page from data files and reads its own sources
datas += collect_data_files("gradio") + collect_data_files("gradio_client")
# transformers scans its models/ folder for .py files at import and loads tokenizer classes by name,
# so it ships as plain source files (drop this block when the tokenizer no longer needs transformers)
datas += collect_data_files("transformers", include_py_files=True, excludes=["**/__pycache__/*"])
for pkg in ("safehttpx", "groovy"):                 # gradio dependencies PyInstaller does not trace fully
    d, b, h = collect_all(pkg)
    datas, binaries, hiddenimports = datas + d, binaries + b, hiddenimports + h

a = Analysis(
    [os.path.join(ROOT, "desktop.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # the cluster-only path (reader's HF backend, training, evaluation): imported lazily, never used by the app
    excludes=["torch", "torchvision", "torchaudio", "sentence_transformers", "peft", "trl", "datasets", "accelerate",
              "bitsandbytes", "scipy", "sklearn", "sympy", "matplotlib", "pyarrow", "tensorboard", "triton", "nvidia",
              "IPython", "notebook", "pytest", "tkinter"],
    module_collection_mode={"gradio": "py", "transformers": "py"},
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="LocalPaperReader", console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="LocalPaperReader", upx=False)
if sys.platform == "darwin":
    # icons/LocalPaperReader.icns is made on a Mac from icons/app-icon.svg (rsvg-convert, sips, iconutil)
    app = BUNDLE(coll, name="LocalPaperReader.app", bundle_identifier="org.localpaperreader.app",
                 icon=os.path.join(ROOT, "icons", "LocalPaperReader.icns"),
                 info_plist={"NSHighResolutionCapable": True})
