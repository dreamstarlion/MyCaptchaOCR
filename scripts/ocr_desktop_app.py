#!/usr/bin/env python3
"""Mini desktop UI for selecting a screen region and running OCR."""

from __future__ import annotations

import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QObject, QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QGuiApplication, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ocr_project_env import OCR_CACHE_ROOT
from ocr_service import CaptchaOcrService, RecognitionResult


CAPTURE_PATH = OCR_CACHE_ROOT / "ui" / "selected-region.png"
MIN_SELECTION_SIZE = QSize(8, 8)

_SERVICE_LOCK = threading.Lock()
_SERVICE: CaptchaOcrService | None = None


def recognize_with_shared_service(image_path: Path) -> RecognitionResult:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = CaptchaOcrService()
        service = _SERVICE
    return service.recognize(image_path)


def virtual_desktop_geometry() -> QRect:
    screens = QGuiApplication.screens()
    if not screens:
        return QRect(0, 0, 1, 1)
    geometry = QRect(screens[0].geometry())
    for screen in screens[1:]:
        geometry = geometry.united(screen.geometry())
    return geometry


def capture_screen_region(rect: QRect) -> QPixmap:
    screen = QGuiApplication.screenAt(rect.center()) or QGuiApplication.primaryScreen()
    if screen is None:
        return QPixmap()

    screen_rect = screen.geometry()
    local_rect = rect.intersected(screen_rect)
    if local_rect.isEmpty():
        return QPixmap()
    local_rect.translate(-screen_rect.topLeft())
    return screen.grabWindow(0, local_rect.x(), local_rect.y(), local_rect.width(), local_rect.height())


class SelectionOverlay(QWidget):
    selected = Signal(object, QSize)
    canceled = Signal()
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.setMouseTracking(True)
        self._start: QPoint | None = None
        self._current: QPoint | None = None
        self._pending_rect: QRect | None = None
        self.setGeometry(virtual_desktop_geometry())

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self._cancel()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._start = event.globalPosition().toPoint()
        self._current = self._start
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._start is None:
            return
        self._current = event.globalPosition().toPoint()
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._start is None:
            return
        self._current = event.globalPosition().toPoint()
        rect = QRect(self._start, self._current).normalized()
        self._start = None
        self._current = None

        if rect.width() < MIN_SELECTION_SIZE.width() or rect.height() < MIN_SELECTION_SIZE.height():
            self._cancel()
            return

        self._pending_rect = rect
        self.hide()
        QTimer.singleShot(120, self._capture_pending_region)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 92))

        rect = self._selection_rect()
        if rect.isNull():
            return

        local_rect = QRect(rect)
        local_rect.translate(-self.geometry().topLeft())
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.fillRect(local_rect, Qt.GlobalColor.transparent)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setPen(QPen(QColor(22, 119, 255), 2))
        painter.drawRect(local_rect.adjusted(1, 1, -1, -1))

    def _selection_rect(self) -> QRect:
        if self._start is None or self._current is None:
            return QRect()
        return QRect(self._start, self._current).normalized()

    def _capture_pending_region(self) -> None:
        rect = self._pending_rect
        self._pending_rect = None
        if rect is None:
            self._cancel()
            return

        pixmap = capture_screen_region(rect)
        if pixmap.isNull():
            self.failed.emit("截图失败。macOS 请检查“屏幕录制”权限，Windows 请确认目标窗口没有被保护。")
            self.close()
            return

        CAPTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not pixmap.save(str(CAPTURE_PATH), "PNG"):
            self.failed.emit(f"截图保存失败：{CAPTURE_PATH}")
            self.close()
            return

        self.selected.emit(CAPTURE_PATH, pixmap.size())
        self.close()

    def _cancel(self) -> None:
        self.hide()
        self.canceled.emit()
        self.close()


class WorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MyCaptchaOCR")
        self.setMinimumSize(360, 236)
        self.setMaximumWidth(420)
        self._capture_path: Path | None = None
        self._overlay: SelectionOverlay | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="captcha-ocr")
        self._worker_signals = WorkerSignals()
        self._worker_signals.finished.connect(self._on_recognition_finished)
        self._worker_signals.failed.connect(self._on_recognition_failed)

        self._build_ui()
        QShortcut(QKeySequence.StandardKey.Close, self, self.close)

    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        title = QLabel("MyCaptchaOCR")
        title_font = QFont()
        title_font.setPointSize(15)
        title_font.setWeight(QFont.Weight.DemiBold)
        title.setFont(title_font)

        self.status_label = QLabel("请选择识别区域")
        self.status_label.setStyleSheet("color: #4b5563;")

        header = QHBoxLayout()
        header.addWidget(title)
        header.addStretch(1)
        layout.addLayout(header)
        layout.addWidget(self.status_label)

        self.preview = QLabel("未选择区域")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(72)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.preview.setStyleSheet(
            "QLabel { border: 1px solid #d1d5db; border-radius: 6px; color: #6b7280; background: #f9fafb; }"
        )
        layout.addWidget(self.preview)

        button_row = QHBoxLayout()
        self.select_button = QPushButton("选取区域")
        self.select_button.clicked.connect(self._start_selection)
        self.recognize_button = QPushButton("识别")
        self.recognize_button.setEnabled(False)
        self.recognize_button.clicked.connect(self._recognize)
        button_row.addWidget(self.select_button)
        button_row.addWidget(self.recognize_button)
        layout.addLayout(button_row)

        result_frame = QFrame()
        result_layout = QVBoxLayout(result_frame)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(6)
        result_label = QLabel("识别结果")
        result_label.setStyleSheet("color: #374151;")
        self.result_box = QLineEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setPlaceholderText("识别出的汉字会显示在这里")
        self.result_box.setMinimumHeight(34)
        result_layout.addWidget(result_label)
        result_layout.addWidget(self.result_box)
        layout.addWidget(result_frame)

        self.setCentralWidget(root)

    def _start_selection(self) -> None:
        self.status_label.setText("正在选取区域...")
        self.select_button.setEnabled(False)
        self.recognize_button.setEnabled(False)
        self.hide()
        QTimer.singleShot(180, self._show_selection_overlay)

    def _show_selection_overlay(self) -> None:
        self._overlay = SelectionOverlay()
        self._overlay.selected.connect(self._on_region_selected)
        self._overlay.canceled.connect(self._on_selection_canceled)
        self._overlay.failed.connect(self._on_selection_failed)
        self._overlay.show()
        self._overlay.raise_()
        self._overlay.activateWindow()

    def _on_region_selected(self, path: Path, image_size: QSize) -> None:
        self.show()
        self.raise_()
        self.activateWindow()
        self._capture_path = path
        self.status_label.setText(f"已选取 {image_size.width()} x {image_size.height()}")
        self.result_box.clear()
        self.recognize_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self._update_preview(path)

    def _on_selection_canceled(self) -> None:
        self.show()
        self.select_button.setEnabled(True)
        self.recognize_button.setEnabled(self._capture_path is not None)
        self.status_label.setText("已取消选取")

    def _on_selection_failed(self, message: str) -> None:
        self.show()
        self.select_button.setEnabled(True)
        self.recognize_button.setEnabled(self._capture_path is not None)
        self.status_label.setText("截图失败")
        QMessageBox.warning(self, "截图失败", message)

    def _update_preview(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.preview.setText("预览失败")
            return
        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.size() - QSize(16, 16),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _recognize(self) -> None:
        if self._capture_path is None:
            self.status_label.setText("请先选取区域")
            return
        self.status_label.setText("识别中，请等待...")
        self.result_box.clear()
        self.select_button.setEnabled(False)
        self.recognize_button.setEnabled(False)

        future = self._executor.submit(recognize_with_shared_service, self._capture_path)
        future.add_done_callback(self._emit_worker_result)

    def _emit_worker_result(self, future: Future) -> None:
        try:
            self._worker_signals.finished.emit(future.result())
        except Exception as exc:  # noqa: BLE001
            self._worker_signals.failed.emit(str(exc))

    def _on_recognition_finished(self, result: RecognitionResult) -> None:
        self.result_box.setText(result.text)
        score = "" if result.score is None else f"，score={result.score:.1f}"
        self.status_label.setText(f"完成：{result.mode}，{result.elapsed_s:.1f}s{score}")
        self.select_button.setEnabled(True)
        self.recognize_button.setEnabled(True)

    def _on_recognition_failed(self, message: str) -> None:
        self.status_label.setText("识别失败")
        self.select_button.setEnabled(True)
        self.recognize_button.setEnabled(self._capture_path is not None)
        QMessageBox.critical(self, "识别失败", message)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("MyCaptchaOCR")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
