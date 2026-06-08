# Desktop UI

The desktop UI is a small PySide6 app for selecting a screen region and running
the existing OCR pipeline against that capture.

## Run

On Windows, double-click `启动.bat` in the project root for one-click launch
(uses the project `.venv` via `pythonw`, no console window).

To run manually:

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

1. Click `选取区域` and drag a rectangle over the target text. The region is
   remembered.
2. Click `识别`. Each click re-captures that same region (so a refreshed captcha
   in the same spot is picked up automatically) and runs OCR — no need to
   re-select.
3. Read the Chinese result in the text box.
4. Click `选取区域` again only when you want to capture a different region.

If the app window overlaps the captured region, it hides itself for a moment
during each capture so it is not photographed; move the window off the captcha
to avoid the flicker.

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
- The first recognition loads ddddocr engines. Later recognitions reuse them and
  fan the candidate inference out across CPU cores (configurable via
  `OCR_WORKERS` in `scripts/ocr_project_env.py`), so a balanced run takes about
  9s on a 6-core CPU. Larger captcha images take longer.

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
