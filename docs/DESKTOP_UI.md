# Desktop UI

The desktop UI is a small PySide6 app for selecting a screen region and running
the existing OCR pipeline against that capture.

## Run

```bash
.venv/bin/python scripts/ocr_desktop_app.py
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\check_ocr_env.py
.\.venv\Scripts\python.exe scripts\ocr_desktop_app.py
```

Flow:

1. Click `选取区域`.
2. Drag a rectangle over the target text.
3. Click `识别`.
4. Read the Chinese result in the text box.

The selected image is saved at `.ocr-cache/ui/selected-region.png`. Generated
OCR variants still use `data/processed/adaptive_ocr/`, which is ignored by Git.

## Platform Notes

- macOS requires Screen Recording permission for screen capture. If the saved
  image is blank or capture fails, enable the permission for the terminal or app
  launcher, then restart the app.
- Windows should create its own `.venv`; do not copy a macOS or Linux virtual
  environment. If Qt or ONNX Runtime reports missing DLLs, install the current
  Microsoft Visual C++ Redistributable and rerun `scripts\check_ocr_env.py`.
- Windows and macOS packages must be built on their target OS. A macOS build
  cannot produce a Windows `.exe`.
- The first recognition loads ddddocr engines. Later recognitions reuse them in
  one background worker thread and should avoid that initialization cost.

## Packaging

Install PyInstaller in the same virtual environment, then build on each target
platform:

```bash
.venv/bin/python -m pip install pyinstaller
.venv/bin/python -m PyInstaller --noconfirm --windowed --name MyCaptchaOCR scripts/ocr_desktop_app.py
```

For reproducible release packages, test the built app on a clean Windows and
macOS machine because `onnxruntime`, OpenCV, Qt, and screen-capture permissions
all have platform-specific behavior.
