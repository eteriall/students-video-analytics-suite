import os
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QSize, QPoint, QPointF, QRectF, QEvent, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QPixmap, QIcon, QPainter, QPen, QColor, QCursor, QPainterPath, QMovie
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QFileDialog,
    QListWidgetItem,
    QMessageBox,
    QSplitter,
    QMainWindow,
    QComboBox,
    QLineEdit,
    QCompleter,
    QProgressBar,
    QApplication,
    QSplashScreen,
    QStackedLayout,
    QDialog,
    QMenuBar,
    QMenu,
    QAction,
    QStatusBar,
)

from recognition import (
    detect_faces,
    cluster_detections,
    draw_face_boxes,
    cluster_color,
    compute_face_embedding,
    FaceProfileManager,
    FaceOccurrence,
)
from utils import cv_to_qpixmap
from database import FaceDatabase
from projects import ProjectManager, Project, ProjectSettings
from project_dialogs import ProjectDialog, ProjectSelectionDialog, CampusCredentialsDialog


class LoadingSpinner(QLabel):
    """Animated loading spinner widget"""
    def __init__(self, size=100, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(size, size)

        # Style the spinner
        self.setStyleSheet("""
            QLabel {
                background-color: rgba(40, 40, 60, 230);
                border-radius: 15px;
                padding: 40px;
                color: white;
                font-size: 18px;
            }
        """)

        # Create spinning animation using Unicode spinner character
        self.rotation = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_spinner)
        self._update_spinner()  # Show initial state
        self.timer.start(80)  # Update every 80ms

    def _update_spinner(self):
        """Update spinner rotation"""
        spinners = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.rotation = (self.rotation + 1) % len(spinners)
        # Use larger spinner character with better formatting
        self.setText(f'<div style="font-size: 72px; margin-bottom: 20px;">{spinners[self.rotation]}</div>'
                    f'<div style="font-size: 18px;">Loading from database...</div>')
        # Force immediate repaint
        self.repaint()
        QApplication.processEvents()

    def stop(self):
        """Stop the spinner animation"""
        self.timer.stop()


class DatabaseLoadWorker(QThread):
    """Worker thread for loading image from database cache"""
    finished = pyqtSignal(str, object)  # path, cache_entry
    error = pyqtSignal(str, str)  # path, error_message

    def __init__(self, path, image_id, mtime, db_path, profile_manager, parent=None):
        super().__init__(parent)
        self.path = path
        self.image_id = image_id
        self.mtime = mtime
        self.db_path = db_path
        self.profile_manager = profile_manager

    def run(self):
        db = None
        try:
            from recognition import FaceOccurrence, cluster_detections, draw_face_boxes, normalize_face

            # Create a new database connection in this thread
            # SQLite connections cannot be shared across threads
            db = FaceDatabase(self.db_path)

            # Get detections from database WITHOUT loading embeddings or face images
            # This significantly speeds up loading by skipping pickle deserialization
            detections_data = db.get_detections_for_image(
                self.image_id,
                load_embeddings=False,  # Skip embedding deserialization - not needed for display
                load_face_images=False  # Skip face image deserialization - crop from main image instead
            )
            if not detections_data:
                self.error.emit(self.path, "No detections found in database")
                return

            # Load the original image from disk (NOT from database)
            img = cv2.imread(self.path)
            if img is None:
                self.error.emit(self.path, "Failed to load image from disk")
                return

            # Reconstruct detections, faces, profile_ids, and occurrences
            detections = []
            faces = []
            profile_ids = []
            occurrences = []

            for det_data in detections_data:
                # Reconstruct detection dict
                box = (det_data['box_x'], det_data['box_y'], det_data['box_w'], det_data['box_h'])
                x, y, w, h = box

                detection = {
                    "box": box,
                    "center": (x + w / 2.0, y + h / 2.0),
                    "size": max(w, h),
                    "score": det_data.get('score', 0.0),
                }
                detections.append(detection)

                # Crop face from original image (much faster than deserializing from DB)
                if y+h <= img.shape[0] and x+w <= img.shape[1] and w > 0 and h > 0:
                    face_crop = img[y:y+h, x:x+w].copy()
                    # Normalize face to consistent size for display
                    face_img = normalize_face(face_crop, side=320)
                else:
                    face_img = np.empty(0)

                faces.append(face_img)

                # Profile ID
                profile_ids.append(det_data.get('profile_id'))

                # Reconstruct FaceOccurrence with lazy embedding loading
                # Store detection ID for lazy embedding loading if needed later
                occurrence = FaceOccurrence(
                    image_path=self.path,
                    detection_index=det_data['detection_index'],
                    box=box,
                    embedding=None,  # Don't load embedding yet - lazy load when needed
                    face_image=face_img
                )
                # Store detection ID for lazy loading
                occurrence.detection_id = det_data['id']
                occurrence.embedding_blob = det_data.get('embedding_blob')  # Keep blob reference
                occurrences.append(occurrence)

            # Cluster detections
            clusters = cluster_detections(detections)

            # Get labels for annotations
            labels = [self.profile_manager.get_profile(pid).label if pid and self.profile_manager.get_profile(pid) else "" for pid in profile_ids]

            # Draw annotated image
            color_ids = profile_ids if profile_ids else clusters
            annotated = draw_face_boxes(img.copy(), detections, color_ids, labels=labels)

            # Create cache entry
            entry = {
                "mtime": self.mtime,
                "original": img,
                "annotated": annotated,
                "faces": faces,
                "detections": detections,
                "clusters": clusters,
                "profile_ids": profile_ids,
                "occurrences": occurrences,
                "_last_labels": labels,  # Cache labels for optimization in _refresh_entry_annotations
            }

            self.finished.emit(self.path, entry)

        except Exception as e:
            import traceback
            self.error.emit(self.path, f"{str(e)}\n{traceback.format_exc()}")
        finally:
            # Always close the database connection
            if db is not None:
                db.close()


class ImageProcessWorker(QThread):
    """Worker thread for processing a single image"""
    finished = pyqtSignal(str, object)  # image_path, result
    error = pyqtSignal(str, str)  # image_path, error_message

    def __init__(self, image_path, model_name, threshold, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.model_name = model_name
        self.threshold = threshold

    def run(self):
        """Process the image in background thread"""
        try:
            # Import here to avoid issues with threading
            img = cv2.imread(self.image_path)
            if img is None:
                self.error.emit(self.image_path, "Failed to load image")
                return

            # Detect faces (returns tuple: faces, detections)
            faces, detections = detect_faces(img, self.threshold)

            # Compute embeddings
            occurrences = []
            for idx, det in enumerate(detections):
                box = det.get("box")
                if box:
                    x, y, w, h = box
                    face_img = img[y:y+h, x:x+w]
                    embedding = compute_face_embedding(face_img, self.model_name)
                    occurrence = FaceOccurrence(
                        image_path=self.image_path,
                        detection_index=idx,
                        box=box,
                        embedding=embedding,
                        face_image=face_img.copy()
                    )
                    occurrences.append(occurrence)

            result = {
                'original': img,
                'detections': detections,
                'occurrences': occurrences
            }

            self.finished.emit(self.image_path, result)

        except Exception as e:
            import traceback
            self.error.emit(self.image_path, f"{str(e)}\n{traceback.format_exc()}")


class BatchProcessWorker(QThread):
    """Worker thread for batch processing multiple images"""
    progress = pyqtSignal(int, int, str)  # current, total, filename
    image_finished = pyqtSignal(str, object)  # image_path, result
    all_finished = pyqtSignal(int, int)  # successful_count, total_count
    error = pyqtSignal(str, str)  # image_path, error_message

    def __init__(self, image_paths, model_name, threshold, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.model_name = model_name
        self.threshold = threshold

    def run(self):
        """Process all images in background thread"""
        processed_count = 0

        for i, image_path in enumerate(self.image_paths):
            try:
                # Emit progress
                self.progress.emit(i, len(self.image_paths), os.path.basename(image_path))

                # Process image
                img = cv2.imread(image_path)
                if img is None:
                    self.error.emit(image_path, "Failed to load image")
                    continue

                # Detect faces (returns tuple: faces, detections)
                faces, detections = detect_faces(img, self.threshold)

                # Compute embeddings
                occurrences = []
                for idx, det in enumerate(detections):
                    box = det.get("box")
                    if box:
                        x, y, w, h = box
                        face_img = img[y:y+h, x:x+w]
                        embedding = compute_face_embedding(face_img, self.model_name)
                        occurrence = FaceOccurrence(
                            image_path=image_path,
                            detection_index=idx,
                            box=box,
                            embedding=embedding,
                            face_image=face_img.copy()
                        )
                        occurrences.append(occurrence)

                result = {
                    'original': img,
                    'detections': detections,
                    'occurrences': occurrences
                }

                self.image_finished.emit(image_path, result)
                processed_count += 1

            except Exception as e:
                import traceback
                self.error.emit(image_path, f"{str(e)}\n{traceback.format_exc()}")

        self.all_finished.emit(processed_count, len(self.image_paths))


class ZoomableImageView(QWidget):
    hoverFaceChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap()
        self._scale = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._dragging = False
        self._last_pos = QPoint()
        self._user_modified = False
        self._pending_fit = False
        self._base_fit_scale = 1.0
        self._min_scale = 0.1
        self._max_scale = 10.0
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.grabGesture(Qt.PinchGesture)
        self.grabGesture(Qt.PanGesture)
        self._detections = []
        self._clusters = []
        self._highlight_index = -1
        self._external_highlight = -1
        self._external_highlight_active = False

    def set_detections(self, detections, clusters=None):
        self._detections = detections or []
        self._clusters = list(clusters) if clusters is not None else [0] * len(self._detections)
        self._highlight_index = -1
        self._external_highlight = -1
        self._external_highlight_active = False
        self.hoverFaceChanged.emit(-1)
        self.update()
        cursor_pos = self.mapFromGlobal(QCursor.pos())
        self._update_highlight_from_cursor(cursor_pos)

    def sizeHint(self):
        return QSize(640, 480)

    def setPixmap(self, pixmap, reset_view=True):
        new_pixmap = pixmap if pixmap is not None else QPixmap()
        keep_view = (
            not reset_view
            and not self._pixmap.isNull()
            and not new_pixmap.isNull()
            and self._pixmap.size() == new_pixmap.size()
        )
        self._pixmap = new_pixmap
        if keep_view:
            self.update()
            return
        self._pending_fit = True
        self._user_modified = False
        if self.width() > 0 and self.height() > 0:
            self._fit_in_view()
        self.update()

    def event(self, event):
        if event.type() == QEvent.Gesture:
            return self._handle_gesture(event)
        if event.type() == QEvent.NativeGesture:
            return self._handle_native_gesture(event)
        return super().event(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        if self._pixmap.isNull():
            return

        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.save()
        painter.translate(self._offset)
        painter.scale(self._scale, self._scale)
        painter.drawPixmap(0, 0, self._pixmap)
        painter.restore()

        if self._highlight_index >= 0 and self._highlight_index < len(self._detections):
            self._draw_focus_overlay(painter)

        self._draw_preview(painter)

    def wheelEvent(self, event):
        # Disabled scroll wheel zoom - use pinch gesture or double-click to fit
        # Scroll events are ignored to prevent accidental zoom with trackpad
        event.ignore()
        return

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self._pixmap.isNull():
            self._dragging = True
            self._last_pos = event.pos()
            event.accept()
        else:
            super().mousePressEvent(event)


    def leaveEvent(self, event):
        if self._highlight_index != -1:
            self._apply_highlight(-1)
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            event.accept()
            self._update_highlight_from_cursor(event.pos())
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if not self._pixmap.isNull():
            self._fit_in_view()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap.isNull():
            return
        if self._pending_fit or not self._user_modified:
            self._fit_in_view()
        else:
            self._limit_offset()

    def _handle_gesture(self, event):
        handled = False
        pinch = event.gesture(Qt.PinchGesture)
        if pinch:
            self._handle_pinch(pinch)
            handled = True
        pan = event.gesture(Qt.PanGesture)
        if pan:
            self._handle_pan(pan)
            handled = True
        return handled

    def _handle_native_gesture(self, event):
        if self._pixmap.isNull():
            return False
        gtype = event.gestureType()
        handled = False
        if gtype == Qt.ZoomNativeGesture:
            value = event.value()
            factor = 1.0 + value
            if factor > 0:
                previous_scale = self._scale
                self._handle_pinch_like(factor, event.position())
                handled = abs(self._scale - previous_scale) > 1e-4
        elif gtype == Qt.PanNativeGesture:
            delta = event.delta()
            if not delta.isNull():
                self._offset = QPointF(
                    self._offset.x() + float(delta.x()),
                    self._offset.y() + float(delta.y()),
                )
                self._user_modified = True
                self._limit_offset()
                self.update()
                handled = True
        return handled

    def _handle_pinch(self, gesture):
        if self._pixmap.isNull():
            return
        factor = gesture.scaleFactor()
        center = gesture.centerPoint()
        self._handle_pinch_like(factor, center)

    def _handle_pinch_like(self, factor, center_point):
        if factor <= 0 or abs(factor - 1.0) < 1e-3:
            return
        new_scale = self._scale * factor
        new_scale = max(self._min_scale, min(self._max_scale, new_scale))
        if abs(new_scale - self._scale) < 1e-4:
            return
        if isinstance(center_point, QPointF):
            center = center_point
        else:
            center = QPointF(float(self.width()) / 2.0, float(self.height()) / 2.0)
        if center.isNull():
            center = QPointF(float(self.width()) / 2.0, float(self.height()) / 2.0)
        dx = center.x() - self._offset.x()
        dy = center.y() - self._offset.y()
        image_pos = QPointF(dx / self._scale, dy / self._scale)
        self._scale = new_scale
        self._offset = QPointF(
            center.x() - image_pos.x() * self._scale,
            center.y() - image_pos.y() * self._scale,
        )
        self._user_modified = True
        self._limit_offset()
        self.update()
        cursor_pos = self.mapFromGlobal(QCursor.pos())
        self._update_highlight_from_cursor(cursor_pos)

    def _handle_pan(self, gesture):
        if self._pixmap.isNull():
            return
        delta = gesture.delta()
        if delta.isNull():
            return
        self._offset = QPointF(
            self._offset.x() + float(delta.x()),
            self._offset.y() + float(delta.y()),
        )
        self._user_modified = True
        self._limit_offset()
        self.update()
        cursor_pos = self.mapFromGlobal(QCursor.pos())
        self._update_highlight_from_cursor(cursor_pos)

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = event.pos() - self._last_pos
            self._offset = QPointF(
                self._offset.x() + float(delta.x()),
                self._offset.y() + float(delta.y()),
            )
            self._last_pos = event.pos()
            self._user_modified = True
            self._limit_offset()
            self.update()
            cursor_pos = event.pos() if event else None
            self._update_highlight_from_cursor(cursor_pos)
            if event:
                event.accept()
        else:
            self._update_highlight_from_cursor(event.pos())
            super().mouseMoveEvent(event)

    def _fit_in_view(self):
        if self._pixmap.isNull():
            return
        pm_w = self._pixmap.width()
        pm_h = self._pixmap.height()
        if pm_w == 0 or pm_h == 0 or self.width() == 0 or self.height() == 0:
            return
        scale_x = self.width() / pm_w
        scale_y = self.height() / pm_h
        fit_scale = min(scale_x, scale_y)
        if fit_scale <= 0:
            fit_scale = 1.0
        self._scale = fit_scale
        self._base_fit_scale = fit_scale
        self._min_scale = max(fit_scale * 0.25, 0.05)
        self._max_scale = max(fit_scale * 8.0, fit_scale * 1.2)
        self._offset = QPointF(
            (self.width() - pm_w * self._scale) / 2.0,
            (self.height() - pm_h * self._scale) / 2.0,
        )
        self._pending_fit = False
        self._user_modified = False
        self.update()

    def _limit_offset(self):
        if self._pixmap.isNull():
            return
        scaled_w = self._pixmap.width() * self._scale
        scaled_h = self._pixmap.height() * self._scale
        if scaled_w <= 0 or scaled_h <= 0:
            return

        if scaled_w < self.width():
            self._offset.setX((self.width() - scaled_w) / 2.0)
        else:
            max_x = 0.0
            min_x = self.width() - scaled_w
            self._offset.setX(min(max_x, max(min_x, self._offset.x())))

        if scaled_h < self.height():
            self._offset.setY((self.height() - scaled_h) / 2.0)
        else:
            max_y = 0.0
            min_y = self.height() - scaled_h
            self._offset.setY(min(max_y, max(min_y, self._offset.y())))

    def _visible_rect(self):
        if self._pixmap.isNull() or self._scale <= 0:
            return QRectF()
        inv_scale = 1.0 / self._scale
        x = -self._offset.x() * inv_scale
        y = -self._offset.y() * inv_scale
        w = self.width() * inv_scale
        h = self.height() * inv_scale
        rect = QRectF(x, y, w, h)
        pix_rect = QRectF(0.0, 0.0, float(self._pixmap.width()), float(self._pixmap.height()))
        return rect.intersected(pix_rect)

    def _draw_preview(self, painter):
        if self._pixmap.isNull():
            return
        margin = 12
        preview_max = max(120, min(200, self.width() // 4))
        if preview_max <= 0:
            return
        pm_w = self._pixmap.width()
        pm_h = self._pixmap.height()
        if pm_w == 0 or pm_h == 0:
            return
        aspect = pm_w / pm_h
        prev_w = preview_max
        prev_h = int(round(preview_max / aspect))
        if prev_h > preview_max:
            prev_h = preview_max
            prev_w = int(round(preview_max * aspect))
        if prev_w <= 0 or prev_h <= 0:
            return
        if prev_h + 2 * margin > self.height():
            prev_h = max(60, self.height() - 2 * margin)
            prev_w = int(round(prev_h * aspect))
        top_left = QPoint(self.width() - prev_w - margin, margin)
        preview_rect = QRectF(float(top_left.x()), float(top_left.y()), float(prev_w), float(prev_h))

        painter.save()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 140))
        painter.drawRoundedRect(preview_rect.adjusted(-4, -4, 4, 4), 6, 6)
        painter.setClipRect(preview_rect)
        painter.drawPixmap(preview_rect, self._pixmap, QRectF(0.0, 0.0, pm_w, pm_h))
        painter.restore()

        visible = self._visible_rect()
        if visible.isNull() or visible.width() <= 0 or visible.height() <= 0:
            return
        scale_x = preview_rect.width() / pm_w
        scale_y = preview_rect.height() / pm_h
        overlay = QRectF(
            preview_rect.left() + visible.x() * scale_x,
            preview_rect.top() + visible.y() * scale_y,
            visible.width() * scale_x,
            visible.height() * scale_y,
        )
        painter.save()
        pen = QPen(QColor(255, 255, 255, 220), 2)
        painter.setPen(pen)
        painter.setBrush(QColor(255, 255, 255, 60))
        painter.drawRect(overlay)
        painter.restore()

    def _draw_focus_overlay(self, painter):
        if self._pixmap.isNull():
            return
        if self._highlight_index < 0 or self._highlight_index >= len(self._detections):
            return
        det = self._detections[self._highlight_index]
        box = det.get("box")
        if not box or len(box) != 4:
            return
        x, y, w, h = box
        if w <= 0 or h <= 0:
            return
        pix_rect = QRectF(
            self._offset.x(),
            self._offset.y(),
            self._pixmap.width() * self._scale,
            self._pixmap.height() * self._scale,
        )
        highlight_rect = QRectF(
            self._offset.x() + x * self._scale,
            self._offset.y() + y * self._scale,
            w * self._scale,
            h * self._scale,
        )
        painter.save()
        painter.setClipRect(pix_rect)
        outer_path = QPainterPath()
        outer_path.addRect(pix_rect)
        inner_path = QPainterPath()
        inner_path.addRect(highlight_rect)
        mask_path = outer_path.subtracted(inner_path)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 128))
        painter.drawPath(mask_path)
        painter.restore()

        cluster_id = self._clusters[self._highlight_index] if self._highlight_index < len(self._clusters) else -1
        r, g, b = cluster_color(cluster_id)
        painter.save()
        painter.setPen(QPen(QColor(r, g, b, 230), max(3, int(round(self._scale)))))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(highlight_rect)
        painter.restore()

    def _update_highlight_from_cursor(self, pos):
        if self._external_highlight_active and self._external_highlight >= 0:
            return
        if pos is None:
            return
        if self._pixmap.isNull() or not self._detections or self._scale <= 0:
            if self._highlight_index != -1:
                self._apply_highlight(-1)
            return
        if isinstance(pos, QPoint):
            pos_f = QPointF(float(pos.x()), float(pos.y()))
        else:
            pos_f = QPointF(pos.x(), pos.y())
        img_x = (pos_f.x() - self._offset.x()) / self._scale
        img_y = (pos_f.y() - self._offset.y()) / self._scale
        pm_w = self._pixmap.width()
        pm_h = self._pixmap.height()
        if img_x < 0 or img_y < 0 or img_x > pm_w or img_y > pm_h:
            new_index = -1
        else:
            new_index = -1
            for idx, det in enumerate(self._detections):
                box = det.get("box")
                if not box or len(box) != 4:
                    continue
                x, y, w, h = box
                if x <= img_x <= x + w and y <= img_y <= y + h:
                    new_index = idx
                    break
        if new_index != self._highlight_index:
            self._apply_highlight(new_index)

    def reset_view(self):
        if not self._pixmap.isNull():
            self._fit_in_view()

    def set_external_highlight(self, index):
        if index is None:
            index = -1
        if index < 0 or index >= len(self._detections):
            self.clear_external_highlight()
            return
        self._external_highlight = index
        self._external_highlight_active = True
        self._apply_highlight(index)

    def clear_external_highlight(self, index=None):
        if index is not None and self._external_highlight != index:
            return
        self._external_highlight = -1
        self._external_highlight_active = False
        cursor_pos = self.mapFromGlobal(QCursor.pos())
        self._update_highlight_from_cursor(cursor_pos)

    def _apply_highlight(self, index):
        if index is not None and index >= len(self._detections):
            index = -1
        if self._external_highlight_active and self._external_highlight >= 0 and index != self._external_highlight:
            index = self._external_highlight
        if index == self._highlight_index:
            return
        self._highlight_index = index
        self.hoverFaceChanged.emit(index)
        self.update()


class FacePanel(QWidget):
    faceDoubleClicked = pyqtSignal(int)  # Emit profile_id when face is double-clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        # Status label for timing and current action
        self.status_label = QLabel()
        self.status_label.setVisible(False)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #3a3a3a;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        self.status_label.setWordWrap(True)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)

        # Image view container with spinner overlay
        self.image_container = QWidget()
        self.image_container.setStyleSheet("background-color: #2a2a2a;")

        # Use stacked layout for overlay effect
        self.image_container_layout = QStackedLayout(self.image_container)
        self.image_container_layout.setContentsMargins(0, 0, 0, 0)
        self.image_container_layout.setStackingMode(QStackedLayout.StackAll)

        self.image_view = ZoomableImageView()
        self.image_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Loading spinner overlay - centered widget
        spinner_container = QWidget()
        spinner_container.setAttribute(Qt.WA_TranslucentBackground)
        spinner_layout = QVBoxLayout(spinner_container)
        spinner_layout.setAlignment(Qt.AlignCenter)

        self.loading_spinner = LoadingSpinner(size=200)
        self.loading_spinner.setVisible(False)
        spinner_layout.addWidget(self.loading_spinner)

        self.image_container_layout.addWidget(self.image_view)
        self.image_container_layout.addWidget(spinner_container)

        self.faces_container = QWidget()
        self.faces_layout = QHBoxLayout(self.faces_container)
        self.faces_layout.setContentsMargins(8, 8, 8, 8)
        self.faces_layout.setSpacing(8)
        self.face_widgets = []
        self.face_labels = []
        self.face_text_labels = []  # Text labels for names
        self.face_clusters = []
        self.face_profile_ids = []  # Store profile IDs
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.faces_container)
        self.scroll.setFixedHeight(210)  # Increased to accommodate text
        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.image_container)
        layout.addWidget(self.scroll)

    def show_loading(self):
        """Show loading spinner overlay"""
        self.loading_spinner.setVisible(True)
        # Ensure spinner is on top and repaints
        self.loading_spinner.raise_()
        self.loading_spinner.repaint()
        QApplication.processEvents()

    def hide_loading(self):
        """Hide loading spinner"""
        self.loading_spinner.setVisible(False)

    def show_image(self, pixmap, reset_view=True):
        self.image_view.setPixmap(pixmap, reset_view=reset_view)

    def set_faces(self, faces, clusters=None, labels=None, profile_ids=None):
        """
        Set faces to display in the panel.

        Args:
            faces: List of QPixmap face images
            clusters: List of cluster IDs for coloring
            labels: List of person names to display
            profile_ids: List of profile IDs for clicking
        """
        for label in self.face_labels:
            label.removeEventFilter(self)
        while self.faces_layout.count():
            w = self.faces_layout.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.face_widgets = []
        self.face_labels = []
        self.face_text_labels = []
        self.face_clusters = list(clusters) if clusters is not None else []
        self.face_profile_ids = list(profile_ids) if profile_ids is not None else []
        if len(self.face_clusters) < len(faces):
            self.face_clusters.extend([0] * (len(faces) - len(self.face_clusters)))
        if len(self.face_profile_ids) < len(faces):
            self.face_profile_ids.extend([None] * (len(faces) - len(self.face_profile_ids)))

        face_labels = labels if labels else []
        if len(face_labels) < len(faces):
            face_labels.extend([""] * (len(faces) - len(face_labels)))

        for idx, pm in enumerate(faces):
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(2)

            # Face image
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setPixmap(pm.scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            img_label.setFixedSize(170, 170)
            img_label.setProperty("faceIndex", idx)
            img_label.installEventFilter(self)
            vbox.addWidget(img_label)

            # Text label for person name
            text_label = QLabel(face_labels[idx] if idx < len(face_labels) else "")
            text_label.setAlignment(Qt.AlignCenter)
            text_label.setWordWrap(True)
            text_label.setFixedWidth(170)
            text_label.setStyleSheet("font-size: 11px; padding: 2px;")
            text_label.setProperty("faceIndex", idx)
            text_label.installEventFilter(self)
            vbox.addWidget(text_label)

            self.faces_layout.addWidget(container)
            self.face_widgets.append(container)
            self.face_labels.append(img_label)
            self.face_text_labels.append(text_label)
        self.faces_layout.addStretch(1)
        self.highlight_face(-1)

    def highlight_face(self, index, cluster_id=-1):
        valid_index = index is not None and 0 <= index < len(self.face_widgets)
        highlight_active = bool(valid_index)
        for idx, widget in enumerate(self.face_widgets):
            label = self.face_labels[idx] if idx < len(self.face_labels) else None
            if highlight_active and idx == index:
                cid = cluster_id if cluster_id >= 0 else (self.face_clusters[idx] if idx < len(self.face_clusters) else -1)
                r, g, b = cluster_color(cid)
                widget.setStyleSheet(
                    f"border: 4px solid rgb({r}, {g}, {b});"
                    f"background-color: rgba({r}, {g}, {b}, 40);"
                )
                if label:
                    label.setStyleSheet("background-color: transparent;")
            elif highlight_active:
                widget.setStyleSheet("border: 0px solid transparent; background-color: rgba(0, 0, 0, 128);")
                if label:
                    label.setStyleSheet("background-color: rgba(0, 0, 0, 128);")
            else:
                widget.setStyleSheet("")
                if label:
                    label.setStyleSheet("")

    def eventFilter(self, obj, event):
        if obj in self.face_labels or obj in self.face_text_labels:
            idx_var = obj.property("faceIndex")
            idx = int(idx_var) if idx_var is not None else -1

            if event.type() == QEvent.Enter:
                self.image_view.set_external_highlight(idx)
                return False
            if event.type() == QEvent.Leave:
                self.image_view.clear_external_highlight(idx)
                return False
            if event.type() == QEvent.MouseButtonDblClick:
                # Handle double-click to select profile
                if 0 <= idx < len(self.face_profile_ids):
                    profile_id = self.face_profile_ids[idx]
                    if profile_id is not None:
                        self.faceDoubleClicked.emit(profile_id)
                return True
        return super().eventFilter(obj, event)


class ProfilePanel(QWidget):
    profileSelected = pyqtSignal(int)
    occurrenceActivated = pyqtSignal(str, int)
    profileRenamed = pyqtSignal(int, str)  # profile_id, new_name
    seeAllFacesClicked = pyqtSignal()  # Signal to open faces window

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(240)
        self.profile_list = QListWidget()
        self.profile_list.setIconSize(QSize(72, 72))
        self.profile_list.setSelectionMode(QListWidget.SingleSelection)
        self.profile_list.itemSelectionChanged.connect(self._on_profile_selection)
        self.profile_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.occurrence_list = QListWidget()
        self.occurrence_list.setSelectionMode(QListWidget.SingleSelection)
        self.occurrence_list.itemClicked.connect(self._on_occurrence_clicked)
        self.occurrence_list.itemDoubleClicked.connect(self._on_occurrence_clicked)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header with title and button
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        faces_title = QLabel("Faces (Double-click to rename)")
        faces_title.setStyleSheet("font-weight: 600; font-size: 11px;")

        self.see_all_btn = QPushButton("See all faces")
        self.see_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
        """)
        self.see_all_btn.clicked.connect(self._on_see_all_clicked)

        header_layout.addWidget(faces_title)
        header_layout.addStretch()
        header_layout.addWidget(self.see_all_btn)

        layout.addWidget(header_widget)

        # Sorting controls
        sort_widget = QWidget()
        sort_layout = QHBoxLayout(sort_widget)
        sort_layout.setContentsMargins(0, 0, 0, 0)
        sort_layout.setSpacing(8)

        sort_label = QLabel("Sort by:")
        sort_label.setStyleSheet("font-size: 10px; color: #888;")

        self.sort_combo = QComboBox()
        self.sort_combo.addItem("Default", "default")
        self.sort_combo.addItem("Most appearances", "most_appearances")
        self.sort_combo.addItem("Least appearances", "least_appearances")
        self.sort_combo.addItem("Name (A-Z)", "name_asc")
        self.sort_combo.addItem("Name (Z-A)", "name_desc")
        self.sort_combo.setStyleSheet("""
            QComboBox {
                background-color: #4a4a4a;
                color: white;
                border: none;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 10px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #4a4a4a;
                color: white;
                selection-background-color: #5a5a5a;
            }
        """)
        self.sort_combo.currentIndexChanged.connect(self._on_sort_changed)

        sort_layout.addWidget(sort_label)
        sort_layout.addWidget(self.sort_combo, 1)

        layout.addWidget(sort_widget)
        layout.addWidget(self.profile_list, 1)
        self.occurrences_title = QLabel("Appearances")
        self.occurrences_title.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.occurrences_title)
        layout.addWidget(self.occurrence_list, 1)
        self._profiles = {}
        self._suspend_signals = False
        self._editing_item = None
        self._edit_widget = None
        self._all_profiles = []  # Store all profiles for sorting

    def _on_sort_changed(self):
        """Handle sort option change."""
        if not self._all_profiles:
            return
        # Re-apply profiles with current sort
        self.set_profiles(self._all_profiles)

    def _sort_profiles(self, profiles):
        """Sort profiles based on current sort selection."""
        sort_key = self.sort_combo.currentData()

        if sort_key == "default":
            # Default order (by profile_id)
            return sorted(profiles, key=lambda p: p.profile_id)
        elif sort_key == "most_appearances":
            # Sort by occurrence count descending
            return sorted(profiles, key=lambda p: p.occurrence_count, reverse=True)
        elif sort_key == "least_appearances":
            # Sort by occurrence count ascending
            return sorted(profiles, key=lambda p: p.occurrence_count)
        elif sort_key == "name_asc":
            # Sort by label A-Z
            return sorted(profiles, key=lambda p: p.label.lower())
        elif sort_key == "name_desc":
            # Sort by label Z-A
            return sorted(profiles, key=lambda p: p.label.lower(), reverse=True)
        else:
            return profiles

    def set_profiles(self, profiles):
        # Store all profiles for sorting
        self._all_profiles = list(profiles)

        # Apply sorting
        sorted_profiles = self._sort_profiles(profiles)

        selected_id = self.current_profile_id()
        current_occurrence = None
        current_item = self.occurrence_list.currentItem()
        if current_item:
            current_occurrence = current_item.data(Qt.UserRole)
        self._suspend_signals = True
        self.profile_list.blockSignals(True)
        self.profile_list.clear()
        self.profile_list.blockSignals(False)
        self._profiles = {p.profile_id: p for p in sorted_profiles}
        for profile in sorted_profiles:
            # Show profile label with occurrence count
            label_text = f"{profile.label} ({profile.occurrence_count})"
            item = QListWidgetItem(label_text)
            item.setData(Qt.UserRole, profile.profile_id)
            face_img = profile.representative_face
            if face_img is not None and getattr(face_img, "size", 0):
                pm = cv_to_qpixmap(face_img)
                item.setIcon(QIcon(pm))
            self.profile_list.addItem(item)
        if selected_id is not None:
            for idx in range(self.profile_list.count()):
                item = self.profile_list.item(idx)
                if item.data(Qt.UserRole) == selected_id:
                    self.profile_list.setCurrentItem(item)
                    break
        if self.profile_list.count() and self.profile_list.currentRow() == -1:
            self.profile_list.setCurrentRow(0)
        self._suspend_signals = False
        self._populate_occurrences(self.current_profile_id(), selected_data=current_occurrence)

    def current_profile_id(self):
        item = self.profile_list.currentItem()
        if not item:
            return None
        return item.data(Qt.UserRole)

    def _on_profile_selection(self):
        profile_id = self.current_profile_id()
        self._populate_occurrences(profile_id)
        if self._suspend_signals or profile_id is None:
            return
        self.profileSelected.emit(profile_id)
        if self.occurrence_list.count():
            self.occurrence_list.setCurrentRow(0)
            self._emit_current_occurrence()

    def _populate_occurrences(self, profile_id, selected_data=None):
        self.occurrence_list.blockSignals(True)
        self.occurrence_list.clear()
        self.occurrence_list.blockSignals(False)

        if profile_id is None:
            self.occurrences_title.setText("Appearances")
            return

        profile = self._profiles.get(profile_id)
        if not profile:
            self.occurrences_title.setText("Appearances")
            return

        # Update header with person name
        self.occurrences_title.setText(f"Appearances of {profile.label}")

        match_row = -1
        for idx, occ in enumerate(profile.occurrences):
            x, y, w, h = occ.box
            label = f"{os.path.basename(occ.image_path)} - ({x}, {y}) {w}x{h}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, (occ.image_path, occ.detection_index))
            self.occurrence_list.addItem(item)
            if selected_data and selected_data == (occ.image_path, occ.detection_index):
                match_row = idx
        if match_row >= 0:
            self.occurrence_list.setCurrentRow(match_row)

    def _on_occurrence_clicked(self, item):
        if item is None:
            item = self.occurrence_list.currentItem()
        if not item:
            return
        data = item.data(Qt.UserRole)
        if not data:
            return
        image_path, det_idx = data
        self.occurrenceActivated.emit(image_path, det_idx)

    def _emit_current_occurrence(self):
        item = self.occurrence_list.currentItem()
        if item:
            self._on_occurrence_clicked(item)

    def _on_item_double_clicked(self, item):
        """Handle double-click on profile item to enable editing"""
        if not item or self._editing_item is not None:
            return

        self._editing_item = item
        profile_id = item.data(Qt.UserRole)
        current_text = item.text()

        # Get all existing names for autocomplete (excluding current profile)
        all_labels = []
        for pid, profile in self._profiles.items():
            if pid != profile_id:
                all_labels.append(profile.label)

        # Create edit widget with autocomplete
        edit = QLineEdit(self.profile_list)
        edit.setText(current_text)
        edit.selectAll()
        edit.setFocus()

        # Add autocomplete
        completer = QCompleter(sorted(set(all_labels)), edit)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        edit.setCompleter(completer)

        # Connect signals
        edit.editingFinished.connect(lambda: self._finish_editing(item, edit, profile_id))
        edit.returnPressed.connect(lambda: self._finish_editing(item, edit, profile_id))

        # Set as item widget
        self.profile_list.setItemWidget(item, edit)
        self._edit_widget = edit

    def _finish_editing(self, item, edit, profile_id):
        """Finish editing and emit signal with new name"""
        if self._editing_item is None:
            return

        new_name = edit.text().strip()

        # Store the old text before clearing widget
        try:
            old_text = item.text()
        except RuntimeError:
            # Item has been deleted, just clean up
            self._editing_item = None
            self._edit_widget = None
            return

        # Remove widget
        try:
            self.profile_list.setItemWidget(item, None)
        except RuntimeError:
            # Item has been deleted, just clean up
            pass

        self._editing_item = None
        self._edit_widget = None

        # Only emit if name changed and not empty
        if new_name and new_name != old_text:
            self.profileRenamed.emit(profile_id, new_name)

    def get_all_profile_names(self):
        """Get list of all current profile names"""
        return [profile.label for profile in self._profiles.values()]

    def select_profile_by_id(self, profile_id):
        """Select a profile by its ID"""
        for idx in range(self.profile_list.count()):
            item = self.profile_list.item(idx)
            if item and item.data(Qt.UserRole) == profile_id:
                self.profile_list.setCurrentItem(item)
                return True
        return False

    def clear(self):
        self.profile_list.clear()
        self.occurrence_list.clear()
        self._profiles = {}
        self._suspend_signals = False
        self._editing_item = None
        self._edit_widget = None

    def _on_see_all_clicked(self):
        """Handle 'See all faces' button click"""
        self.seeAllFacesClicked.emit()


class FacesWindow(QMainWindow):
    """Window showing all faces in a grid with appearance management"""
    appearanceReassigned = pyqtSignal(int, str, int, str)  # profile_id, image_path, detection_index, new_name
    profileDeleted = pyqtSignal(int)  # profile_id
    profileRenamed = pyqtSignal(int, str)  # profile_id, new_name

    def __init__(self, profile_manager, parent=None):
        super().__init__(parent)
        self.profile_manager = profile_manager
        self.setWindowTitle("Faces and People")
        self.resize(1000, 700)

        # Central widget with splitter
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top bar with Back button (thin)
        top_bar = QWidget()
        top_bar.setStyleSheet("background-color: #2a2a2a;")
        top_bar.setFixedHeight(40)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(8, 4, 8, 4)

        back_btn = QPushButton("← Back")
        back_btn.setFixedHeight(32)
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: none;
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
        """)
        back_btn.clicked.connect(self.close)
        top_bar_layout.addWidget(back_btn)
        top_bar_layout.addStretch()

        main_layout.addWidget(top_bar)

        # Content area with splitter
        content_widget = QWidget()
        layout = QHBoxLayout(content_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Grid of all faces
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("All Faces")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 8px;")
        left_layout.addWidget(title)

        # Scroll area for face grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Container for face grid
        self.grid_container = QWidget()
        self.grid_layout = QVBoxLayout(self.grid_container)
        self.grid_layout.setAlignment(Qt.AlignTop)
        self.grid_layout.setSpacing(8)

        scroll.setWidget(self.grid_container)
        left_layout.addWidget(scroll)

        # Right panel - Appearances of selected face
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)

        self.appearances_title = QLabel("Select a face to view appearances")
        self.appearances_title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 8px;")
        right_layout.addWidget(self.appearances_title)

        # Button container
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(8)

        # Download button
        self.download_btn = QPushButton("Download All Images (ZIP)")
        self.download_btn.setEnabled(False)
        self.download_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.download_btn.clicked.connect(self._on_download_clicked)

        # Delete profile button
        self.delete_profile_btn = QPushButton("Delete Profile")
        self.delete_profile_btn.setEnabled(False)
        self.delete_profile_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:pressed {
                background-color: #c41408;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.delete_profile_btn.clicked.connect(self._on_delete_profile_clicked)

        buttons_layout.addWidget(self.download_btn)
        buttons_layout.addWidget(self.delete_profile_btn)
        right_layout.addWidget(buttons_widget)

        # Scroll area for appearances
        self.appearances_scroll = QScrollArea()
        self.appearances_scroll.setWidgetResizable(True)
        self.appearances_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Container for appearances
        self.appearances_container = QWidget()
        self.appearances_layout = QVBoxLayout(self.appearances_container)
        self.appearances_layout.setAlignment(Qt.AlignTop)
        self.appearances_layout.setSpacing(8)

        self.appearances_scroll.setWidget(self.appearances_container)
        right_layout.addWidget(self.appearances_scroll)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)
        main_layout.addWidget(content_widget)

        # State
        self.current_profile_id = None
        self.face_widgets = {}  # profile_id -> widget

        # Populate faces
        self.refresh_faces()

    def refresh_faces(self):
        """Refresh the grid of all faces"""
        # Clear existing widgets
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.face_widgets.clear()

        profiles = self.profile_manager.profiles()
        if not profiles:
            no_faces = QLabel("No faces detected yet.\nProcess some images first.")
            no_faces.setAlignment(Qt.AlignCenter)
            no_faces.setStyleSheet("color: #888; font-size: 14px; padding: 40px;")
            self.grid_layout.addWidget(no_faces)
            return

        # Create face cards in a flow layout
        for profile in profiles:
            face_widget = self._create_face_card(profile)
            self.face_widgets[profile.profile_id] = face_widget
            self.grid_layout.addWidget(face_widget)

    def _create_face_card(self, profile):
        """Create a clickable face card widget"""
        card = QWidget()
        card.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border-radius: 8px;
                border: 2px solid transparent;
            }
            QWidget:hover {
                border: 2px solid #4CAF50;
            }
        """)
        card.setCursor(QCursor(Qt.PointingHandCursor))
        card.setFixedHeight(120)

        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Face image
        face_label = QLabel()
        if profile.representative_face is not None and profile.representative_face.size > 0:
            pm = cv_to_qpixmap(profile.representative_face)
            face_label.setPixmap(pm.scaled(90, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            face_label.setText("No\nImage")
            face_label.setAlignment(Qt.AlignCenter)
            face_label.setStyleSheet("color: #999; font-size: 10px;")
        face_label.setFixedSize(90, 90)
        layout.addWidget(face_label)

        # Info section
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        name_label = QLabel(profile.label)
        name_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #333;")
        name_label.setWordWrap(True)

        count_label = QLabel(f"{len(profile.occurrences)} appearance(s)")
        count_label.setStyleSheet("font-size: 11px; color: #666;")

        info_layout.addWidget(name_label)
        info_layout.addWidget(count_label)
        info_layout.addStretch()

        layout.addLayout(info_layout, 1)

        # Make card clickable and double-clickable
        card.mousePressEvent = lambda event: self._on_face_clicked(profile.profile_id)
        card.mouseDoubleClickEvent = lambda event: self._on_face_double_clicked(profile.profile_id, name_label)

        # Store label for editing
        card.name_label = name_label

        return card

    def _on_face_clicked(self, profile_id):
        """Handle face card click"""
        self.current_profile_id = profile_id
        profile = self.profile_manager.get_profile(profile_id)
        if not profile:
            return

        # Update title
        self.appearances_title.setText(f"Appearances of {profile.label} ({len(profile.occurrences)})")

        # Enable buttons
        self.download_btn.setEnabled(True)
        self.delete_profile_btn.setEnabled(True)

        # Clear existing appearances
        while self.appearances_layout.count():
            item = self.appearances_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Group appearances by day
        from datetime import datetime
        from collections import defaultdict

        occurrences_by_day = defaultdict(list)
        for occurrence in profile.occurrences:
            # Get file modification time
            try:
                if os.path.isfile(occurrence.image_path):
                    mtime = os.path.getmtime(occurrence.image_path)
                    day = datetime.fromtimestamp(mtime).date()
                    occurrences_by_day[day].append((occurrence, mtime))
            except:
                # If can't get timestamp, put in "Unknown" group
                occurrences_by_day[None].append((occurrence, 0))

        # Sort days in reverse chronological order (newest first)
        sorted_days = sorted([d for d in occurrences_by_day.keys() if d is not None], reverse=True)
        if None in occurrences_by_day:
            sorted_days.append(None)

        # Show appearances grouped by day
        for day in sorted_days:
            # Add day header
            if day is not None:
                day_str = day.strftime("%B %d, %Y")  # e.g., "October 01, 2024"
                day_header = QLabel(day_str)
                day_header.setStyleSheet("""
                    font-size: 13px;
                    font-weight: bold;
                    color: #333;
                    background-color: #e8e8e8;
                    padding: 8px 12px;
                    border-radius: 4px;
                    margin-top: 8px;
                """)
                self.appearances_layout.addWidget(day_header)
            else:
                unknown_header = QLabel("Unknown Date")
                unknown_header.setStyleSheet("""
                    font-size: 13px;
                    font-weight: bold;
                    color: #666;
                    background-color: #f0f0f0;
                    padding: 8px 12px;
                    border-radius: 4px;
                    margin-top: 8px;
                """)
                self.appearances_layout.addWidget(unknown_header)

            # Sort occurrences within the day by time
            day_occurrences = sorted(occurrences_by_day[day], key=lambda x: x[1], reverse=True)

            # Add appearance widgets
            for occurrence, mtime in day_occurrences:
                appearance_widget = self._create_appearance_widget(profile_id, occurrence, mtime)
                self.appearances_layout.addWidget(appearance_widget)

    def _create_appearance_widget(self, profile_id, occurrence, mtime):
        """Create a widget showing one appearance with reassignment button"""
        from datetime import datetime

        widget = QWidget()
        widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 8px;
            }
        """)

        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Face image
        face_label = QLabel()
        if occurrence.face_image is not None and occurrence.face_image.size > 0:
            pm = cv_to_qpixmap(occurrence.face_image)
            face_label.setPixmap(pm.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            face_label.setText("No Image")
            face_label.setStyleSheet("color: #999;")
        face_label.setFixedSize(80, 80)
        layout.addWidget(face_label)

        # Info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        filename_label = QLabel(f"<b>{os.path.basename(occurrence.image_path)}</b>")
        filename_label.setWordWrap(True)

        # Show time
        if mtime > 0:
            time_obj = datetime.fromtimestamp(mtime)
            time_str = time_obj.strftime("%I:%M %p")  # e.g., "02:30 PM"
            time_label = QLabel(f"🕒 {time_str}")
            time_label.setStyleSheet("font-size: 10px; color: #666;")
            info_layout.addWidget(time_label)

        x, y, w, h = occurrence.box
        location_label = QLabel(f"📍 Location: ({x}, {y})  Size: {w}×{h}px")
        location_label.setStyleSheet("font-size: 10px; color: #666;")

        info_layout.addWidget(filename_label)
        info_layout.addWidget(location_label)

        layout.addLayout(info_layout, 1)

        # Reassignment button
        profile = self.profile_manager.get_profile(profile_id)
        profile_name = profile.label if profile else "this person"

        reassign_btn = QPushButton(f"That's not {profile_name}?")
        reassign_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
        """)
        reassign_btn.clicked.connect(lambda: self._on_reassign_appearance(profile_id, occurrence))
        layout.addWidget(reassign_btn)

        return widget

    def _on_reassign_appearance(self, profile_id, occurrence):
        """Handle reassigning an appearance to a different person"""
        profile = self.profile_manager.get_profile(profile_id)
        if not profile:
            return

        # Get all profile names except current one
        all_profiles = self.profile_manager.profiles()
        other_names = [p.label for p in all_profiles if p.profile_id != profile_id]

        # Create dialog for name input with autocomplete
        dialog = QDialog(self)
        dialog.setWindowTitle("Reassign Face")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)

        instruction = QLabel(f"This face is currently assigned to <b>{profile.label}</b>.\nWho is this person actually?")
        instruction.setWordWrap(True)
        layout.addWidget(instruction)

        # Name input with autocomplete
        name_input = QLineEdit()
        name_input.setPlaceholderText("Enter person's name...")

        if other_names:
            completer = QCompleter(sorted(set(other_names)))
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            completer.setCompletionMode(QCompleter.PopupCompletion)
            name_input.setCompleter(completer)

        layout.addWidget(name_input)

        # Buttons
        button_box = QWidget()
        button_layout = QHBoxLayout(button_box)
        button_layout.setContentsMargins(0, 0, 0, 0)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)

        reassign_btn = QPushButton("Reassign")
        reassign_btn.setDefault(True)
        reassign_btn.setAutoDefault(True)
        reassign_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        reassign_btn.clicked.connect(dialog.accept)

        # Connect Enter key in text input to accept dialog
        name_input.returnPressed.connect(dialog.accept)

        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(reassign_btn)

        layout.addWidget(button_box)

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            new_name = name_input.text().strip()
            if new_name and new_name != profile.label:
                # Emit signal to parent window to handle reassignment
                self.appearanceReassigned.emit(profile_id, occurrence.image_path, occurrence.detection_index, new_name)
                # Refresh display
                self.refresh_faces()
                if self.current_profile_id == profile_id:
                    self._on_face_clicked(profile_id)

    def _on_face_double_clicked(self, profile_id, name_label):
        """Handle double-click on face card to rename"""
        profile = self.profile_manager.get_profile(profile_id)
        if not profile:
            return

        # Get all profile names for autocomplete (excluding current)
        all_profiles = self.profile_manager.profiles()
        other_names = [p.label for p in all_profiles if p.profile_id != profile_id]

        # Create inline edit
        dialog = QDialog(self)
        dialog.setWindowTitle("Rename Person")
        dialog.setMinimumWidth(350)

        layout = QVBoxLayout(dialog)

        instruction = QLabel(f"Rename <b>{profile.label}</b>:")
        layout.addWidget(instruction)

        # Name input with autocomplete
        name_input = QLineEdit()
        name_input.setText(profile.label)
        name_input.selectAll()

        if other_names:
            completer = QCompleter(sorted(set(other_names)))
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            completer.setCompletionMode(QCompleter.PopupCompletion)
            name_input.setCompleter(completer)

        layout.addWidget(name_input)

        # Buttons
        button_box = QWidget()
        button_layout = QHBoxLayout(button_box)
        button_layout.setContentsMargins(0, 0, 0, 0)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)

        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.setAutoDefault(True)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        save_btn.clicked.connect(dialog.accept)

        # Connect Enter key in text input to accept dialog
        name_input.returnPressed.connect(dialog.accept)

        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(save_btn)

        layout.addWidget(button_box)

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            new_name = name_input.text().strip()
            if new_name and new_name != profile.label:
                # Emit signal to parent window to handle rename
                self.profileRenamed.emit(profile_id, new_name)
                # Update the label
                name_label.setText(new_name)
                # Refresh if this is the currently selected profile
                if self.current_profile_id == profile_id:
                    self.appearances_title.setText(f"Appearances of {new_name} ({len(profile.occurrences)})")

    def _on_delete_profile_clicked(self):
        """Handle delete profile button click"""
        if self.current_profile_id is None:
            return

        profile = self.profile_manager.get_profile(self.current_profile_id)
        if not profile:
            return

        reply = QMessageBox.question(
            self,
            "Delete Profile",
            f"Are you sure you want to delete the profile for <b>{profile.label}</b>?\n\n"
            f"This will remove all {len(profile.occurrences)} appearance(s).\n"
            f"This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Emit signal to parent window to handle deletion
            self.profileDeleted.emit(self.current_profile_id)
            # Clear selection
            self.current_profile_id = None
            self.download_btn.setEnabled(False)
            self.delete_profile_btn.setEnabled(False)
            self.appearances_title.setText("Select a face to view appearances")
            # Clear appearances
            while self.appearances_layout.count():
                item = self.appearances_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            # Refresh face cards
            self.refresh_faces()

    def _on_download_clicked(self):
        """Handle download all images button"""
        if self.current_profile_id is None:
            return

        profile = self.profile_manager.get_profile(self.current_profile_id)
        if not profile or not profile.occurrences:
            QMessageBox.warning(self, "No Images", "No images to download for this person.")
            return

        # Ask user for save location
        default_name = f"{profile.label.replace(' ', '_')}_images.zip"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Images As",
            default_name,
            "ZIP Archive (*.zip)"
        )

        if not file_path:
            return

        try:
            import zipfile
            import shutil
            import tempfile

            # Create temporary directory for images
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy all images containing this person
                image_paths = list(set(occ.image_path for occ in profile.occurrences))

                with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for idx, img_path in enumerate(image_paths, 1):
                        if os.path.isfile(img_path):
                            # Add to zip with sequential naming
                            ext = os.path.splitext(img_path)[1]
                            zip_name = f"{profile.label.replace(' ', '_')}_{idx:03d}{ext}"
                            zipf.write(img_path, zip_name)

            QMessageBox.information(
                self,
                "Export Complete",
                f"Successfully exported {len(image_paths)} image(s) to:\n{file_path}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to create ZIP archive:\n{str(e)}"
            )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Gallery")
        self.resize(1200, 800)
        self.gallery = QListWidget()
        self.gallery.setIconSize(QSize(64, 64))
        self.gallery.setSelectionMode(QListWidget.ExtendedSelection)  # Enable multiple selection
        self.gallery.itemSelectionChanged.connect(self.on_selection_changed)
        self.gallery.currentRowChanged.connect(self.on_select)
        self.add_btn = QPushButton("Add Images")
        self.add_btn.clicked.connect(self.add_images)
        self.process_all_btn = QPushButton("Process All Images")
        self.process_all_btn.clicked.connect(self.process_all_images)
        self.process_all_btn.setEnabled(False)
        self.blur_btn = QPushButton("Blur Faces")
        self.blur_btn.clicked.connect(self.blur_faces)
        self.blur_btn.setEnabled(False)

        self.unload_btn = QPushButton("Unload Selected")
        self.unload_btn.clicked.connect(self.unload_selected_images)
        self.unload_btn.setEnabled(False)

        # Model selection dropdown
        self.model_label = QLabel("Recognition Model:")
        self.model_selector = QComboBox()
        self.model_selector.addItems([
            "Facenet512",  # Best accuracy, 512-dim
            "ArcFace",     # Excellent for matching, 512-dim
            "Facenet",     # Faster, 128-dim
            "VGG-Face",    # Robust, 2622-dim
            "SFace",       # Fastest, 128-dim
        ])
        self.model_selector.setCurrentText("Facenet512")
        self.model_selector.currentTextChanged.connect(self.on_model_changed)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.add_btn)
        left_layout.addWidget(self.process_all_btn)
        left_layout.addWidget(self.model_label)
        left_layout.addWidget(self.model_selector)
        left_layout.addWidget(self.gallery)
        left_layout.addWidget(self.blur_btn)
        left_layout.addWidget(self.unload_btn)
        left = QWidget()
        left.setLayout(left_layout)
        self.panel = FacePanel()
        self.panel.image_view.hoverFaceChanged.connect(self.on_face_hover)
        self.panel.faceDoubleClicked.connect(self.on_face_double_clicked)
        self.profile_panel = ProfilePanel()
        self.profile_panel.profileSelected.connect(self.on_profile_selected)
        self.profile_panel.occurrenceActivated.connect(self.on_occurrence_activated)
        self.profile_panel.profileRenamed.connect(self.on_profile_renamed)
        self.profile_panel.seeAllFacesClicked.connect(self.on_see_all_faces)
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.panel)
        splitter.addWidget(self.profile_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        container = QWidget()
        root_layout = QHBoxLayout(container)
        root_layout.addWidget(splitter)
        self.setCentralWidget(container)

        # Initialize project management
        self.project_manager = ProjectManager()
        self.current_project = None
        self._is_startup = True  # Flag to track if we're in startup phase

        # Create menu bar
        self._create_menu_bar()

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.project_label = QLabel("No project")
        self.status_bar.addPermanentWidget(self.project_label)

        # Initialize database (will be overridden when project is selected)
        self.db = FaceDatabase()

        # Local storage for imported images
        self.import_dir = Path(__file__).resolve().parent / "imported_images"
        self.import_dir.mkdir(parents=True, exist_ok=True)

        self.images = []
        self.detection_threshold = 0.7
        # DeepFace model configuration
        self.recognition_model = "Facenet512"  # Default: best accuracy
        self.distance_threshold = 0.4  # For cosine distance
        self.current_image_original = None
        self.current_annotated_image = None
        self.current_blurred_image = None
        self.current_blurred_faces = []
        self.current_detections = []
        self.current_clusters = []
        self.current_profile_ids = []
        self.current_color_ids = []
        self.current_faces_original = []
        self.current_faces = []
        self.is_blurred = False
        self.current_image_path = None
        self.analysis_cache = {}

        # Get max_people from current project if available
        max_people = self.current_project.max_people if self.current_project else None

        self.profile_manager = FaceProfileManager(
            distance_threshold=self.distance_threshold,
            model_name=self.recognition_model,
            use_cosine=True,
            max_people=max_people
        )
        self.pending_highlight = None

        # Worker threads
        self.current_worker = None
        self.batch_worker = None

        # Faces window
        self.faces_window = None

        # Show project selection on startup after window is visible
        QTimer.singleShot(100, self._show_startup_project_selection)

    def _show_startup_project_selection(self):
        """Show project selection dialog on application startup."""
        projects = self.project_manager.get_all_projects()

        if projects:
            # Show project selection dialog
            dialog = ProjectSelectionDialog(projects, parent=self)
            dialog.setWindowTitle("Welcome - Select Project")

            # Customize button text
            if hasattr(dialog, 'new_btn'):
                dialog.new_btn.setText("Create New Project")

            result = dialog.exec_()

            if result == QDialog.Accepted:
                project = dialog.get_selected_project()
                if project:
                    self._switch_to_project(project)
                else:
                    # Fallback to default
                    self.load_from_database()
            elif dialog.wants_new_project():
                # User clicked "New Project" button
                self._on_new_project()
                # If still no project after dialog, load default database
                if not self.current_project:
                    self.load_from_database()
            else:
                # User clicked Cancel - ask what to do
                reply = QMessageBox.question(
                    self,
                    "Continue Without Project?",
                    "Would you like to continue without a project or exit?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )

                if reply == QMessageBox.Yes:
                    # Continue without project (use default database)
                    self.load_from_database()
                else:
                    # Exit application
                    QApplication.quit()
                    return
        else:
            # No projects exist - prompt to create first project
            reply = QMessageBox.question(
                self,
                "Welcome to Face Gallery",
                "No projects found. Would you like to create your first project?\n\n"
                "Projects help organize your face recognition work by event or group.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                self._on_new_project()
                # If user canceled project creation, load default database
                if not self.current_project:
                    self.load_from_database()
            else:
                # Continue without project
                self.load_from_database()

        # Startup complete
        self._is_startup = False

    def _create_menu_bar(self):
        """Create menu bar with project management."""
        menubar = self.menuBar()

        # Project menu
        project_menu = menubar.addMenu("&Project")

        new_project_action = QAction("&New Project...", self)
        new_project_action.setShortcut("Ctrl+N")
        new_project_action.triggered.connect(self._on_new_project)
        project_menu.addAction(new_project_action)

        open_project_action = QAction("&Open Project...", self)
        open_project_action.setShortcut("Ctrl+O")
        open_project_action.triggered.connect(self._on_open_project)
        project_menu.addAction(open_project_action)

        project_menu.addSeparator()

        edit_project_action = QAction("&Edit Current Project...", self)
        edit_project_action.triggered.connect(self._on_edit_project)
        project_menu.addAction(edit_project_action)
        self.edit_project_action = edit_project_action

        close_project_action = QAction("&Close Project", self)
        close_project_action.triggered.connect(self._on_close_project)
        project_menu.addAction(close_project_action)
        self.close_project_action = close_project_action

        project_menu.addSeparator()

        delete_project_action = QAction("&Delete Project...", self)
        delete_project_action.triggered.connect(self._on_delete_project)
        project_menu.addAction(delete_project_action)

        # Initially disable project-specific actions
        self.edit_project_action.setEnabled(False)
        self.close_project_action.setEnabled(False)

    def _update_project_ui(self):
        """Update UI to reflect current project state."""
        if self.current_project:
            # Update window title
            self.setWindowTitle(f"Face Gallery - {self.current_project.name}")
            # Update status bar
            self.project_label.setText(f"Project: {self.current_project.name}")
            # Enable project actions
            self.edit_project_action.setEnabled(True)
            self.close_project_action.setEnabled(True)
        else:
            self.setWindowTitle("Face Gallery")
            self.project_label.setText("No project")
            self.edit_project_action.setEnabled(False)
            self.close_project_action.setEnabled(False)

    def _on_new_project(self):
        """Create a new project."""
        dialog = ProjectDialog(parent=self)
        if dialog.exec_() == QDialog.Accepted:
            project = dialog.get_project()
            # Save project to database
            project_id = self.project_manager.create_project(project)
            project.id = project_id
            # Switch to new project
            self._switch_to_project(project)

    def _on_open_project(self):
        """Open an existing project."""
        projects = self.project_manager.get_all_projects()
        if not projects:
            reply = QMessageBox.question(
                self,
                "No Projects",
                "No projects found. Would you like to create a new project?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._on_new_project()
            return

        dialog = ProjectSelectionDialog(projects, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            project = dialog.get_selected_project()
            if project:
                self._switch_to_project(project)

    def _on_edit_project(self):
        """Edit current project."""
        if not self.current_project:
            return

        dialog = ProjectDialog(project=self.current_project, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            updated_project = dialog.get_project()
            # Save updates
            self.project_manager.update_project(updated_project)
            self.current_project = updated_project
            self._update_project_ui()

            # Update settings if they changed
            if self.current_project.settings:
                self.detection_threshold = self.current_project.settings.detection_threshold
                self.distance_threshold = self.current_project.settings.distance_threshold
                self.recognition_model = self.current_project.settings.model_name
                self.model_selector.setCurrentText(self.recognition_model)
                # Update profile manager
                self.profile_manager.distance_threshold = self.distance_threshold
                self.profile_manager.model_name = self.recognition_model
                self.profile_manager.use_cosine = self.current_project.settings.use_cosine
                self.profile_manager.max_people = self.current_project.max_people

    def _on_close_project(self):
        """Close current project."""
        if not self.current_project:
            return

        reply = QMessageBox.question(
            self,
            "Close Project",
            f"Close project '{self.current_project.name}'?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Save any pending changes
            self._save_current_state()
            # Clear current project
            self.current_project = None
            # Reset to default database
            self.db = FaceDatabase()
            # Clear UI
            self.gallery.clear()
            self.images.clear()
            self.analysis_cache.clear()
            self.profile_manager.reset()
            self.profile_panel.set_profiles([])
            self._update_project_ui()

    def _on_delete_project(self):
        """Delete a project."""
        projects = self.project_manager.get_all_projects()
        if not projects:
            QMessageBox.information(self, "No Projects", "No projects to delete.")
            return

        dialog = ProjectSelectionDialog(projects, parent=self)
        dialog.setWindowTitle("Delete Project")
        if dialog.exec_() == QDialog.Accepted:
            project = dialog.get_selected_project()
            if project:
                reply = QMessageBox.question(
                    self,
                    "Delete Project",
                    f"Are you sure you want to delete project '{project.name}'?\n\n"
                    f"This will remove the project metadata but not the database file.\n"
                    f"This action cannot be undone.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    # If deleting current project, close it first
                    if self.current_project and self.current_project.id == project.id:
                        self._on_close_project()

                    # Delete project
                    self.project_manager.delete_project(project.id)
                    QMessageBox.information(self, "Success", f"Project '{project.name}' deleted.")

    def _switch_to_project(self, project: Project):
        """Switch to a different project."""
        # Save current state
        if self.current_project:
            self._save_current_state()

        # Set new project
        self.current_project = project

        # Switch database
        if project.database_path:
            self.db = FaceDatabase(project.database_path)
        else:
            self.db = FaceDatabase()

        # Apply project settings
        if project.settings:
            self.detection_threshold = project.settings.detection_threshold
            self.distance_threshold = project.settings.distance_threshold
            self.recognition_model = project.settings.model_name
            self.model_selector.setCurrentText(self.recognition_model)

        # Update profile manager with project settings
        self.profile_manager = FaceProfileManager(
            distance_threshold=self.distance_threshold,
            model_name=self.recognition_model,
            use_cosine=project.settings.use_cosine if project.settings else True,
            max_people=project.max_people
        )

        # Clear and reload data
        self.gallery.clear()
        self.images.clear()
        self.analysis_cache.clear()
        self.load_from_database()

        # Update UI
        self._update_project_ui()

        self.status_bar.showMessage(f"Switched to project: {project.name}", 3000)

    def _save_current_state(self):
        """Save current application state to database."""
        # This is called when switching or closing projects
        # Database changes are already saved incrementally, so this is mostly a placeholder
        # for any future session state we might want to persist
        pass

    def load_from_database(self):
        """Load profiles and images from database on startup."""
        print("Loading data from database...")

        # Load profiles with all their occurrences
        profiles_data = self.db.get_all_profiles()
        if profiles_data:
            print(f"Loading {len(profiles_data)} profiles from database")
            # Update next_profile_id to avoid conflicts
            max_id = max(p['id'] for p in profiles_data) if profiles_data else 0
            self.profile_manager._next_profile_id = max_id + 1

            # Reconstruct profiles with their occurrences
            for prof_data in profiles_data:
                profile_id = prof_data['id']
                label = prof_data['label']

                # Get all detections for this profile
                detections_data = self.db.get_detections_for_profile(profile_id)

                if detections_data:
                    # Create profile object
                    from recognition import FaceProfile, FaceOccurrence
                    profile = FaceProfile(profile_id=profile_id, label=label)

                    # Add all occurrences to the profile
                    for det_data in detections_data:
                        # Reconstruct FaceOccurrence
                        occurrence = FaceOccurrence(
                            image_path=det_data['image_path'],
                            detection_index=det_data['detection_index'],
                            box=(det_data['box_x'], det_data['box_y'], det_data['box_w'], det_data['box_h']),
                            embedding=det_data['embedding'],
                            face_image=det_data.get('face_image')
                        )
                        profile.add_occurrence(occurrence)

                    # Add profile to manager
                    self.profile_manager._profiles[profile_id] = profile
                    print(f"  Loaded profile {profile_id} ({label}) with {len(detections_data)} occurrences")

        # Load images (but don't load their cache yet - do that lazily when selected)
        images_data = self.db.get_all_images()
        if images_data:
            print(f"Found {len(images_data)} images in database")
            for img_data in images_data:
                path = img_data['path']
                if os.path.isfile(path):
                    # Add to gallery if not already there
                    if path not in self.images:
                        self._add_image_to_gallery(path)

        # Enable Process All button if we have images
        if self.images:
            self.process_all_btn.setEnabled(True)

        print(f"Loaded {len(self.images)} images from database")
        print(f"Loaded {len(self.profile_manager._profiles)} profiles with occurrences")

    def _add_image_to_gallery(self, fpath: str):
        """Add an image to the gallery without analyzing it."""
        item = QListWidgetItem(os.path.basename(fpath))
        item.setData(Qt.UserRole, fpath)
        item.setToolTip(fpath)
        self.gallery.addItem(item)
        self.images.append(fpath)
        return True


    def add_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select images", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if not files:
            return
        added = 0
        for fpath in files:
            source_path = Path(fpath)
            if not source_path.is_file():
                continue

            try:
                local_path = self._store_image_locally(source_path)
            except OSError as exc:
                QMessageBox.warning(self, "Import Error", f"Could not import {source_path.name}:\n{exc}")
                continue

            local_path_str = str(local_path)
            if local_path_str in self.images:
                continue

            # Add to database
            mtime = os.path.getmtime(local_path_str)
            self.db.add_image(local_path_str, mtime, processed=False)

            if self._add_image_to_gallery(local_path_str):
                added += 1

        if added and self.gallery.currentRow() < 0:
            self.gallery.setCurrentRow(0)

        # Enable Process All button if we have images
        if self.images:
            self.process_all_btn.setEnabled(True)

    def _store_image_locally(self, source: Path) -> Path:
        """Copy imported image into the managed storage directory."""
        source = source.resolve()
        dest = self.import_dir / source.name

        try:
            if source.exists() and dest.exists() and source.samefile(dest):
                return dest
        except (FileNotFoundError, OSError):
            # Ignore cases where samefile cannot be evaluated
            pass

        # Avoid name collisions by appending a short UUID suffix
        if dest.exists():
            dest = self.import_dir / f"{source.stem}_{uuid.uuid4().hex[:8]}{source.suffix}"

        shutil.copy2(source, dest)
        return dest

    def process_all_images(self):
        """Process all images with progress bar using background thread."""
        if not self.images:
            return

        reply = QMessageBox.question(
            self,
            "Process All Images",
            f"This will process {len(self.images)} images. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Show status label and progress bar
        self.panel.status_label.setVisible(True)
        self.panel.progress_bar.setVisible(True)
        self.panel.progress_bar.setRange(0, len(self.images))
        self.panel.progress_bar.setValue(0)

        # Track start time
        import time
        self.batch_start_time = time.time()
        self.batch_image_times = []  # Track time per image for estimation

        # Disable buttons during processing
        self.add_btn.setEnabled(False)
        self.process_all_btn.setEnabled(False)
        self.blur_btn.setEnabled(False)
        self.model_selector.setEnabled(False)
        self.gallery.setEnabled(False)

        # Cancel any existing batch worker
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.progress.disconnect()
            self.batch_worker.image_finished.disconnect()
            self.batch_worker.all_finished.disconnect()
            self.batch_worker.error.disconnect()
            self.batch_worker.quit()
            self.batch_worker.wait()

        # Start batch processing worker
        self.batch_worker = BatchProcessWorker(self.images, self.recognition_model, self.detection_threshold, self)
        self.batch_worker.progress.connect(self._on_batch_progress)
        self.batch_worker.image_finished.connect(self._on_batch_image_finished)
        self.batch_worker.all_finished.connect(self._on_batch_all_finished)
        self.batch_worker.error.connect(self._on_batch_error)
        self.batch_worker.start()

    def _format_time(self, seconds):
        """Format seconds into human-readable time string"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def _on_batch_progress(self, current, total, filename):
        """Handle batch processing progress update"""
        import time

        self.panel.progress_bar.setValue(current)
        self.panel.progress_bar.setFormat(f"Processing {current+1}/{total}: {filename}")

        # Calculate timing information
        elapsed_time = time.time() - self.batch_start_time

        if current > 0:
            # Calculate average time per image
            avg_time_per_image = elapsed_time / current

            # Estimate remaining time
            remaining_images = total - current
            estimated_remaining = avg_time_per_image * remaining_images

            # Format times
            elapsed_str = self._format_time(elapsed_time)
            estimated_str = self._format_time(estimated_remaining)

            # Update status label
            status_text = (f"⏱ Elapsed: {elapsed_str}  |  "
                          f"📄 Processing: {filename}  |  "
                          f"⏳ Est. remaining: {estimated_str}")
        else:
            elapsed_str = self._format_time(elapsed_time)
            status_text = f"⏱ Elapsed: {elapsed_str}  |  📄 Starting: {filename}..."

        self.panel.status_label.setText(status_text)

    def _on_batch_image_finished(self, path, result):
        """Handle completion of single image in batch"""
        try:
            # Process the result from the worker thread
            img = result['original']
            detections = result['detections']
            occurrences = result['occurrences']

            # Run profile matching (must be done in main thread)
            profile_ids = []
            for occurrence in occurrences:
                profile = self.profile_manager.assign_profile(occurrence)
                profile_ids.append(profile.profile_id)

            # Get labels and draw annotations
            labels = [self.profile_manager.get_profile(pid).label if pid and self.profile_manager.get_profile(pid) else "" for pid in profile_ids]
            color_ids = profile_ids if profile_ids else [0] * len(detections)
            annotated = draw_face_boxes(img.copy(), detections, color_ids, labels=labels)

            # Extract face images
            faces = []
            for det in detections:
                box = det.get("box")
                if box:
                    x, y, w, h = box
                    face_img = img[y:y+h, x:x+w]
                    faces.append(face_img.copy())

            # Cluster detections
            clusters = cluster_detections(detections)

            # Cache the result
            mtime = os.path.getmtime(path) if os.path.isfile(path) else 0
            entry = {
                "mtime": mtime,
                "original": img,
                "annotated": annotated,
                "faces": faces,
                "detections": detections,
                "clusters": clusters,
                "profile_ids": profile_ids,
                "occurrences": occurrences,
            }
            self.analysis_cache[path] = entry

            # Save to database
            self._save_to_database(path, mtime, occurrences, profile_ids, detections)

        except Exception as e:
            import traceback
            print(f"Error processing batch result for {path}: {e}\n{traceback.format_exc()}")

    def _on_batch_error(self, path, error_msg):
        """Handle error during batch processing"""
        print(f"Batch processing error for {path}: {error_msg}")

    def _on_batch_all_finished(self, processed_count, total_count):
        """Handle completion of all batch processing"""
        import time

        # Calculate total time
        total_time = time.time() - self.batch_start_time
        total_time_str = self._format_time(total_time)

        # Re-enable buttons
        self.add_btn.setEnabled(True)
        self.process_all_btn.setEnabled(True)
        self.blur_btn.setEnabled(True)
        self.model_selector.setEnabled(True)
        self.gallery.setEnabled(True)

        # Update progress bar
        self.panel.progress_bar.setValue(total_count)
        self.panel.progress_bar.setFormat(f"Completed: {processed_count}/{total_count} images processed")

        # Update status label with final time
        self.panel.status_label.setText(f"✅ Completed in {total_time_str}  |  Processed {processed_count}/{total_count} images")

        # Ensure all profiles are saved to database
        self._save_all_profiles()

        # Update display
        self.profile_panel.set_profiles(self.profile_manager.profiles())

        # Show completion message
        QMessageBox.information(
            self,
            "Processing Complete",
            f"Successfully processed {processed_count} of {total_count} images in {total_time_str}"
        )

        # Hide progress bar and status after 3 seconds
        QTimer.singleShot(3000, lambda: self.panel.progress_bar.setVisible(False))
        QTimer.singleShot(3000, lambda: self.panel.status_label.setVisible(False))

    def on_model_changed(self, model_name):
        """Handle recognition model change - reset profiles and clear cache"""
        self.recognition_model = model_name
        # Clear cache since embeddings will be different with new model
        self.analysis_cache.clear()
        # Reset profile manager with new model
        self.profile_manager = FaceProfileManager(
            distance_threshold=self.distance_threshold,
            model_name=self.recognition_model,
            use_cosine=True
        )
        # Re-analyze current image if one is selected
        current_row = self.gallery.currentRow()
        if current_row >= 0:
            self.on_select(current_row)
        print(f"Recognition model changed to: {model_name}")

    def analyze_image(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        mtime = os.path.getmtime(path)
        entry = self.analysis_cache.get(path)
        if entry and entry.get("mtime") == mtime:
            return entry
        if entry:
            self.profile_manager.remove_image_occurrences(path)
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        faces, detections = detect_faces(img, self.detection_threshold)
        clusters = cluster_detections(detections)
        occurrences = []
        profile_ids = []
        for idx, det in enumerate(detections):
            crop = det.get("crop")
            embedding = compute_face_embedding(crop, model_name=self.recognition_model)
            box = det.get("box", (0, 0, 0, 0))
            ix, iy, iw, ih = [int(round(v)) for v in box]
            face_img = None
            if idx < len(faces):
                current_face = faces[idx]
                if isinstance(current_face, np.ndarray):
                    face_img = current_face.copy()
            elif crop is not None and isinstance(crop, np.ndarray):
                face_img = crop.copy()
            occurrence = FaceOccurrence(
                image_path=path,
                detection_index=idx,
                box=(ix, iy, iw, ih),
                embedding=embedding,
                face_image=face_img,
            )
            profile = self.profile_manager.assign_profile(occurrence)
            profile_ids.append(profile.profile_id)
            occurrences.append(occurrence)
        color_ids = profile_ids if profile_ids else clusters
        # Get labels for each detection
        labels = [self.profile_manager.get_profile(pid).label if pid and self.profile_manager.get_profile(pid) else "" for pid in profile_ids]
        annotated = draw_face_boxes(img.copy(), detections, color_ids, labels=labels)
        entry = {
            "mtime": mtime,
            "original": img,
            "annotated": annotated,
            "faces": [face.copy() if isinstance(face, np.ndarray) else face for face in faces],
            "detections": detections,
            "clusters": clusters,
            "profile_ids": profile_ids,
            "occurrences": occurrences,
        }
        self.analysis_cache[path] = entry

        # Save to database
        self._save_to_database(path, mtime, occurrences, profile_ids, detections)

        return entry

    def _save_all_profiles(self):
        """Save all profiles to database."""
        try:
            for profile in self.profile_manager.profiles():
                self.db.save_profile(profile.profile_id, profile.label)
            print(f"Saved {len(self.profile_manager.profiles())} profiles to database")
        except Exception as e:
            print(f"Error saving profiles to database: {e}")

    def _save_to_database(self, path: str, mtime: float, occurrences, profile_ids, detections):
        """Save analysis results to database."""
        try:
            # Get or create image record
            image_id = self.db.get_image_id(path)
            if image_id is None:
                image_id = self.db.add_image(path, mtime, processed=False)

            # Delete old detections for this image
            self.db.delete_detections_for_image(image_id)

            # Save profiles
            for pid in set(profile_ids):
                if pid:
                    profile = self.profile_manager.get_profile(pid)
                    if profile:
                        self.db.save_profile(pid, profile.label)

            # Save detections
            for idx, (occurrence, det) in enumerate(zip(occurrences, detections)):
                if occurrence and det:
                    score = det.get("score", 0.0)
                    emb = occurrence.get_embedding()
                    self.db.save_detection(
                        image_id=image_id,
                        profile_id=emb is not None and profile_ids[idx] or None,
                        detection_index=idx,
                        box=occurrence.box,
                        score=score,
                        embedding=emb,
                        face_image=occurrence.face_image
                    )

            # Mark image as processed
            self.db.mark_image_processed(image_id)

        except Exception as e:
            print(f"Error saving to database: {e}")

    def on_selection_changed(self):
        """Handle gallery selection changes to update unload button state."""
        selected_items = self.gallery.selectedItems()
        # Enable unload button if one or more images are selected
        self.unload_btn.setEnabled(len(selected_items) >= 1)

    def on_select(self, row):
        if row < 0:
            return
        item = self.gallery.item(row)
        if not item:
            return
        path = item.data(Qt.UserRole)

        # Check if already in cache
        mtime = os.path.getmtime(path) if os.path.isfile(path) else 0
        entry = self.analysis_cache.get(path)
        if entry and entry.get("mtime") == mtime:
            # Already processed, display immediately
            self._display_image_entry(path, entry)
            return

        # Not in cache - check if it was processed before (in database)
        image_id = self.db.get_image_id(path)
        if image_id is not None and self.db.is_image_processed(path, mtime):
            # Load from database in background thread (avoids UI freeze)
            print(f"Loading {os.path.basename(path)} from database cache...")
            self.panel.show_loading()

            # Cancel any existing workers
            if self.current_worker and self.current_worker.isRunning():
                self.current_worker.finished.disconnect()
                self.current_worker.error.disconnect()
                self.current_worker.quit()
                self.current_worker.wait()

            # Start database load worker
            self.current_worker = DatabaseLoadWorker(path, image_id, mtime, self.db.db_path, self.profile_manager, self)
            self.current_worker.finished.connect(self._on_database_loaded)
            self.current_worker.error.connect(self._on_database_load_error)
            self.current_worker.start()
            return

        # Not in cache and not in database, process in background thread
        self.panel.show_loading()

        # Cancel any existing worker
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.finished.disconnect()
            self.current_worker.error.disconnect()
            self.current_worker.quit()
            self.current_worker.wait()

        # Start new worker
        self.current_worker = ImageProcessWorker(path, self.recognition_model, self.detection_threshold, self)
        self.current_worker.finished.connect(self._on_image_processed)
        self.current_worker.error.connect(self._on_image_process_error)
        self.current_worker.start()

    def _on_image_processed(self, path, result):
        """Handle completion of image processing"""
        try:
            # Process the result from the worker thread
            img = result['original']
            detections = result['detections']
            occurrences = result['occurrences']

            # Run profile matching (must be done in main thread)
            profile_ids = []
            for occurrence in occurrences:
                profile = self.profile_manager.assign_profile(occurrence)
                profile_ids.append(profile.profile_id)

            # Get labels and draw annotations
            labels = [self.profile_manager.get_profile(pid).label if pid and self.profile_manager.get_profile(pid) else "" for pid in profile_ids]
            color_ids = profile_ids if profile_ids else [0] * len(detections)
            annotated = draw_face_boxes(img.copy(), detections, color_ids, labels=labels)

            # Extract face images
            faces = []
            for det in detections:
                box = det.get("box")
                if box:
                    x, y, w, h = box
                    face_img = img[y:y+h, x:x+w]
                    faces.append(face_img.copy())

            # Cluster detections
            clusters = cluster_detections(detections)

            # Cache the result
            mtime = os.path.getmtime(path) if os.path.isfile(path) else 0
            entry = {
                "mtime": mtime,
                "original": img,
                "annotated": annotated,
                "faces": faces,
                "detections": detections,
                "clusters": clusters,
                "profile_ids": profile_ids,
                "occurrences": occurrences,
            }
            self.analysis_cache[path] = entry

            # Save to database
            self._save_to_database(path, mtime, occurrences, profile_ids, detections)

            # Display the result
            self._display_image_entry(path, entry)

            # Update profile panel
            self.profile_panel.set_profiles(self.profile_manager.profiles())

        except Exception as e:
            import traceback
            print(f"Error processing result: {e}\n{traceback.format_exc()}")
            self.panel.hide_loading()
            QMessageBox.warning(self, "Error", f"Failed to process image: {e}")

    def _on_image_process_error(self, path, error_msg):
        """Handle error during image processing"""
        print(f"Error processing {path}: {error_msg}")
        self.panel.hide_loading()
        QMessageBox.warning(self, "Error", f"Failed to load image:\n{error_msg}")

    def _on_database_loaded(self, path, entry):
        """Handle completion of database cache loading"""
        print(f"Loaded {os.path.basename(path)} from database cache")

        # Store in cache
        self.analysis_cache[path] = entry

        # Display the image
        self._display_image_entry(path, entry)

    def _on_database_load_error(self, path, error_msg):
        """Handle error during database loading - fall back to reprocessing"""
        print(f"Error loading from database: {error_msg}, reprocessing...")

        # Fall back to processing the image from scratch
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.finished.disconnect()
            self.current_worker.error.disconnect()
            self.current_worker.quit()
            self.current_worker.wait()

        # Start new processing worker
        self.current_worker = ImageProcessWorker(path, self.recognition_model, self.detection_threshold, self)
        self.current_worker.finished.connect(self._on_image_processed)
        self.current_worker.error.connect(self._on_image_process_error)
        self.current_worker.start()

    def _display_image_entry(self, path, entry):
        """Display a processed image entry"""
        self.panel.hide_loading()

        annotated, color_ids, labels = self._refresh_entry_annotations(entry)
        pixmap = cv_to_qpixmap(annotated) if annotated is not None else QPixmap()
        self.panel.show_image(pixmap)
        detections = entry.get("detections", [])
        self.panel.image_view.set_detections(detections, color_ids)
        face_pixmaps = [cv_to_qpixmap(face) for face in entry.get("faces", [])]
        face_labels = labels if labels else [""] * len(face_pixmaps)
        self.panel.set_faces(
            face_pixmaps,
            color_ids,
            labels=face_labels,
            profile_ids=entry.get("profile_ids", [])
        )
        original = entry.get("original")
        self.current_image_original = original.copy() if original is not None else None
        self.current_annotated_image = annotated.copy() if annotated is not None else None
        self.current_blurred_image = None
        self.current_blurred_faces = []
        self.current_detections = detections
        self.current_clusters = entry.get("clusters", [])
        self.current_profile_ids = list(entry.get("profile_ids", []))
        self.current_color_ids = list(color_ids)
        faces = entry.get("faces", [])
        self.current_faces_original = [face.copy() for face in faces]
        self.current_faces = [face.copy() for face in faces]
        self.current_image_path = path
        self.is_blurred = False
        has_faces = bool(self.current_detections)
        self.blur_btn.setEnabled(has_faces)
        self.blur_btn.setText("Blur Faces")
        self.profile_panel.set_profiles(self.profile_manager.profiles())
        self._apply_pending_highlight(path)

    def on_profile_selected(self, profile_id):
        if profile_id is None:
            return
        if self.current_image_path and self.current_image_path in self.analysis_cache:
            indices = [
                idx for idx, pid in enumerate(self.current_profile_ids)
                if pid == profile_id
            ]
            if indices:
                idx = indices[0]
                color_id = profile_id if idx >= len(self.current_color_ids) else self.current_color_ids[idx]
                self.panel.highlight_face(idx, color_id)
                self.panel.image_view.set_external_highlight(idx)

    def on_occurrence_activated(self, image_path, detection_index):
        if not image_path:
            return
        self.pending_highlight = (image_path, detection_index)
        row = self._find_gallery_row(image_path)
        if row == -1:
            return
        if self.gallery.currentRow() != row:
            self.gallery.setCurrentRow(row)
        else:
            self._apply_pending_highlight(image_path)

    def on_profile_renamed(self, profile_id, new_name):
        """Handle profile renaming and merge if name already exists"""
        if not new_name or not new_name.strip():
            return

        new_name = new_name.strip()

        # Check if another profile has this name
        existing_profile_id = self.profile_manager.find_profile_by_label(new_name)

        if existing_profile_id is not None and existing_profile_id != profile_id:
            # Merge profiles
            print(f"Merging profile {profile_id} into profile {existing_profile_id} (name: {new_name})")
            success = self.profile_manager.merge_profiles(profile_id, existing_profile_id)

            if success:
                # Update database - delete merged profile and update detections
                self.db.delete_profile(profile_id)

                # Clear cache for all affected images
                profiles_to_update = set()
                for path, entry in self.analysis_cache.items():
                    profile_ids = entry.get("profile_ids", [])
                    if profile_id in profile_ids:
                        profiles_to_update.add(path)

                # Update cache entries and database
                for path in profiles_to_update:
                    entry = self.analysis_cache.get(path)
                    if entry:
                        # Replace source profile_id with target profile_id
                        profile_ids = entry.get("profile_ids", [])
                        entry["profile_ids"] = [existing_profile_id if pid == profile_id else pid for pid in profile_ids]
                        entry["annotated"] = None

                        # Update database detections
                        image_id = self.db.get_image_id(path)
                        if image_id:
                            detections_data = self.db.get_detections_for_image(image_id)
                            for det in detections_data:
                                if det['profile_id'] == profile_id:
                                    self.db.update_detection_profile(det['id'], existing_profile_id)

                # Refresh display
                self.profile_panel.set_profiles(self.profile_manager.profiles())

                # Re-render current image if it's affected
                if self.current_image_path in profiles_to_update:
                    self._redraw_current_image()

                QMessageBox.information(
                    self,
                    "Profiles Merged",
                    f"Profiles merged successfully into '{new_name}'"
                )
        else:
            # Just rename
            success = self.profile_manager.rename_profile(profile_id, new_name)
            if success:
                print(f"Renamed profile {profile_id} to '{new_name}'")
                # Update database
                self.db.save_profile(profile_id, new_name)
                # Refresh display
                self.profile_panel.set_profiles(self.profile_manager.profiles())
                # Mark cached annotations stale so labels refresh
                affected_paths = []
                for path, entry in self.analysis_cache.items():
                    profile_ids = entry.get("profile_ids", [])
                    if profile_id in profile_ids:
                        entry["annotated"] = None
                        affected_paths.append(path)
                # Redraw current image if it contains renamed profile
                if self.current_image_path and self.current_image_path in affected_paths:
                    self._redraw_current_image()

    def _find_gallery_row(self, path):
        for idx in range(self.gallery.count()):
            item = self.gallery.item(idx)
            if item and item.data(Qt.UserRole) == path:
                return idx
        return -1

    def _apply_pending_highlight(self, current_path):
        if not self.pending_highlight:
            return
        path, index = self.pending_highlight
        if path != current_path:
            return
        self.pending_highlight = None
        if index is None or index < 0:
            self.panel.image_view.clear_external_highlight()
            self.panel.highlight_face(-1)
            return
        if index >= len(self.current_detections):
            return
        color_id = self.current_color_ids[index] if index < len(self.current_color_ids) else -1
        self.panel.highlight_face(index, color_id)
        self.panel.image_view.set_external_highlight(index)

    def _get_labels_for_current_detections(self):
        """Get profile labels for current detections"""
        if not self.current_profile_ids:
            return []
        labels = []
        for pid in self.current_profile_ids:
            profile = self.profile_manager.get_profile(pid)
            labels.append(profile.label if profile else "")
        return labels

    def _get_labels_for_profile_ids(self, profile_ids):
        """Get profile labels for given profile IDs"""
        if not profile_ids:
            return []
        labels = []
        for pid in profile_ids:
            profile = self.profile_manager.get_profile(pid)
            labels.append(profile.label if profile else "")
        return labels

    def _refresh_entry_annotations(self, entry):
        """Ensure entry annotations use current labels and return visuals."""
        original = entry.get("original")
        detections = entry.get("detections", [])
        profile_ids = list(entry.get("profile_ids", []) or [])
        clusters = list(entry.get("clusters", []) or [])

        if original is None or not isinstance(original, np.ndarray) or not detections:
            color_ids = profile_ids if profile_ids else clusters
            labels = self._get_labels_for_profile_ids(profile_ids) if profile_ids else []
            return entry.get("annotated"), color_ids, labels

        color_ids = list(profile_ids) if profile_ids else list(clusters)
        if len(color_ids) < len(detections):
            color_ids.extend([0] * (len(detections) - len(color_ids)))
        labels = self._get_labels_for_profile_ids(profile_ids) if profile_ids else []
        if len(labels) < len(detections):
            labels.extend([""] * (len(detections) - len(labels)))

        # OPTIMIZATION: Skip redrawing if annotated image already exists and is valid
        # This eliminates redundant work when loading from database
        existing_annotated = entry.get("annotated")
        if existing_annotated is not None and isinstance(existing_annotated, np.ndarray):
            # Check if we need to redraw (i.e., if labels might have changed)
            # We store the last used labels in the entry to detect changes
            last_labels = entry.get("_last_labels")
            if last_labels == labels:
                # Labels haven't changed, use existing annotated image
                return existing_annotated, color_ids, labels

        # Draw new annotated image
        annotated = draw_face_boxes(original.copy(), detections, color_ids, labels=labels)
        entry["annotated"] = annotated
        entry["_last_labels"] = labels  # Cache labels for next comparison
        return annotated, color_ids, labels

    def _redraw_current_image(self):
        """Redraw current image with updated profile colors"""
        if not self.current_image_path:
            return

        entry = self.analysis_cache.get(self.current_image_path)
        if not entry:
            return

        annotated, color_ids, labels = self._refresh_entry_annotations(entry)
        if annotated is None:
            return

        pixmap = cv_to_qpixmap(annotated)
        self.panel.show_image(pixmap)
        self.panel.image_view.set_detections(entry.get("detections", []), color_ids)
        faces = entry.get("faces", [])
        face_pixmaps = [cv_to_qpixmap(face) for face in faces]
        face_labels = labels if labels else [""] * len(face_pixmaps)
        self.panel.set_faces(face_pixmaps, color_ids, labels=face_labels, profile_ids=entry.get("profile_ids", []))
        self.current_annotated_image = annotated
        self.current_color_ids = list(color_ids)
        self.current_profile_ids = list(entry.get("profile_ids", []))

    def blur_faces(self):
        if self.current_image_original is None or not self.current_detections:
            return
        if not self.is_blurred:
            if self.current_blurred_image is None or not self.current_blurred_faces:
                blurred = self.current_image_original.copy()
                ih, iw = blurred.shape[:2]
                for det in self.current_detections:
                    x, y, w, h = det.get("box", (0, 0, 0, 0))
                    x0 = max(int(round(x)), 0)
                    y0 = max(int(round(y)), 0)
                    x1 = min(int(round(x + w)), iw)
                    y1 = min(int(round(y + h)), ih)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    roi = blurred[y0:y1, x0:x1]
                    if roi.size == 0:
                        continue
                    k = max(48, int(max(w, h) / 4))
                    if k % 2 == 0:
                        k += 1
                    blurred[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)

                blurred_faces = []
                for face in self.current_faces_original:
                    if face is None or face.size == 0:
                        blurred_faces.append(face)
                        continue
                    fh, fw = face.shape[:2]
                    k = max(45, int(max(fh, fw) / 4))
                    if k % 2 == 0:
                        k += 1
                    blurred_faces.append(cv2.GaussianBlur(face, (k, k), 0))

                # Get labels for current detections
                labels = self._get_labels_for_current_detections()
                annotated = draw_face_boxes(blurred, self.current_detections, self.current_color_ids, labels=labels)
                self.current_blurred_image = annotated.copy() if annotated is not None else None
                self.current_blurred_faces = [face.copy() for face in blurred_faces]
            if self.current_blurred_image is None:
                labels = self._get_labels_for_current_detections()
                annotated = draw_face_boxes(self.current_image_original, self.current_detections, self.current_color_ids, labels=labels)
                self.current_blurred_image = annotated.copy() if annotated is not None else None
            pixmap = cv_to_qpixmap(self.current_blurred_image) if self.current_blurred_image is not None else QPixmap()
            self.panel.show_image(pixmap, reset_view=False)
            if not self.current_blurred_faces:
                blurred_faces = []
                for face in self.current_faces_original:
                    if face is None or face.size == 0:
                        blurred_faces.append(face)
                        continue
                    fh, fw = face.shape[:2]
                    k = max(45, int(max(fh, fw) / 4))
                    if k % 2 == 0:
                        k += 1
                    blurred_faces.append(cv2.GaussianBlur(face, (k, k), 0))
                self.current_blurred_faces = [face.copy() for face in blurred_faces]
            face_labels = self._get_labels_for_current_detections()
            self.panel.set_faces([cv_to_qpixmap(face) for face in self.current_blurred_faces], self.current_color_ids, labels=face_labels, profile_ids=self.current_profile_ids)
            self.current_faces = [face.copy() for face in self.current_blurred_faces]
            self.is_blurred = True
            self.blur_btn.setText("Unblur Faces")
        else:
            annotated = self.current_annotated_image
            if annotated is None and self.current_image_original is not None:
                labels = self._get_labels_for_current_detections()
                annotated = draw_face_boxes(self.current_image_original, self.current_detections, self.current_color_ids, labels=labels)
                self.current_annotated_image = annotated.copy() if annotated is not None else None
            pixmap = cv_to_qpixmap(annotated) if annotated is not None else QPixmap()
            self.panel.show_image(pixmap, reset_view=False)
            face_labels = self._get_labels_for_current_detections()
            self.panel.set_faces([cv_to_qpixmap(face) for face in self.current_faces_original], self.current_color_ids, labels=face_labels, profile_ids=self.current_profile_ids)
            self.current_faces = [face.copy() for face in self.current_faces_original]
            self.is_blurred = False
            self.blur_btn.setText("Blur Faces")

    def on_face_double_clicked(self, profile_id):
        """Handle double-click on face thumbnail - select the profile"""
        if profile_id is not None:
            # Select the profile in the profile panel
            self.profile_panel.select_profile_by_id(profile_id)

    def on_face_hover(self, index):
        if not self.current_faces:
            self.panel.highlight_face(-1)
            return
        if index < 0 or index >= len(self.current_faces):
            self.panel.highlight_face(-1)
        else:
            cluster_id = self.current_color_ids[index] if index < len(self.current_color_ids) else -1
            self.panel.highlight_face(index, cluster_id)

    def on_see_all_faces(self):
        """Open the Faces and People window"""
        if self.faces_window is None or not self.faces_window.isVisible():
            self.faces_window = FacesWindow(self.profile_manager, self)
            self.faces_window.appearanceReassigned.connect(self.on_appearance_reassigned)
            self.faces_window.profileDeleted.connect(self.on_profile_deleted)
            self.faces_window.profileRenamed.connect(self.on_profile_renamed_from_faces_window)

        # Refresh window data in case profiles changed
        self.faces_window.refresh_faces()
        self.faces_window.show()
        self.faces_window.raise_()
        self.faces_window.activateWindow()

    def on_appearance_reassigned(self, old_profile_id, image_path, detection_index, new_name):
        """Handle reassignment of an appearance to a different person"""
        old_profile = self.profile_manager.get_profile(old_profile_id)
        if not old_profile:
            return

        # Find the occurrence to reassign
        occurrence_to_reassign = None
        for occ in old_profile.occurrences:
            if occ.image_path == image_path and occ.detection_index == detection_index:
                occurrence_to_reassign = occ
                break

        if not occurrence_to_reassign:
            return

        # Check if new_name matches an existing profile
        target_profile_id = self.profile_manager.find_profile_by_label(new_name)

        if target_profile_id:
            # Reassign to existing profile
            target_profile = self.profile_manager.get_profile(target_profile_id)
            if target_profile:
                # Remove from old profile
                old_profile.occurrences.remove(occurrence_to_reassign)
                if old_profile.occurrences:
                    old_profile.embedding_sum = np.sum([occ.embedding for occ in old_profile.occurrences], axis=0).astype(np.float32)
                    old_profile.occurrence_count = len(old_profile.occurrences)
                else:
                    # Delete empty profile
                    del self.profile_manager._profiles[old_profile_id]
                    self.db.delete_profile(old_profile_id)

                # Add to target profile
                target_profile.add_occurrence(occurrence_to_reassign)

                # Update database
                image_id = self.db.get_image_id(image_path)
                if image_id:
                    detections = self.db.get_detections_for_image(image_id)
                    for det in detections:
                        if (det['profile_id'] == old_profile_id and
                            det['detection_index'] == detection_index):
                            self.db.update_detection_profile(det['id'], target_profile_id)
                            break

                # Save profiles
                self.db.save_profile(target_profile_id, target_profile.label)
        else:
            # Create new profile with new_name
            from recognition import FaceProfile
            new_profile_id = self.profile_manager._next_profile_id
            self.profile_manager._next_profile_id += 1

            new_profile = FaceProfile(profile_id=new_profile_id, label=new_name)
            self.profile_manager._profiles[new_profile_id] = new_profile

            # Remove from old profile
            old_profile.occurrences.remove(occurrence_to_reassign)
            if old_profile.occurrences:
                old_profile.embedding_sum = np.sum([occ.embedding for occ in old_profile.occurrences], axis=0).astype(np.float32)
                old_profile.occurrence_count = len(old_profile.occurrences)
            else:
                # Delete empty profile
                del self.profile_manager._profiles[old_profile_id]
                self.db.delete_profile(old_profile_id)

            # Add to new profile
            new_profile.add_occurrence(occurrence_to_reassign)

            # Update database
            self.db.save_profile(new_profile_id, new_profile.label)
            image_id = self.db.get_image_id(image_path)
            if image_id:
                detections = self.db.get_detections_for_image(image_id)
                for det in detections:
                    if (det['profile_id'] == old_profile_id and
                        det['detection_index'] == detection_index):
                        self.db.update_detection_profile(det['id'], new_profile_id)
                        break

        # Clear cache for this image
        if image_path in self.analysis_cache:
            del self.analysis_cache[image_path]

        # Update profile panel
        self.profile_panel.set_profiles(self.profile_manager.profiles())

        # If this is the current image, refresh display
        if self.current_image_path == image_path:
            self.on_select(self.gallery.currentRow())

        print(f"Reassigned appearance from {image_path} to {new_name}")

    def on_profile_deleted(self, profile_id):
        """Handle deletion of an entire profile"""
        profile = self.profile_manager.get_profile(profile_id)
        if not profile:
            return

        # Get all images that need to be refreshed
        image_paths = list(set(occ.image_path for occ in profile.occurrences))

        # Delete from profile manager
        del self.profile_manager._profiles[profile_id]

        # Delete from database
        self.db.delete_profile(profile_id)

        # Clear caches for affected images
        for path in image_paths:
            if path in self.analysis_cache:
                del self.analysis_cache[path]

        # Update profile panel
        self.profile_panel.set_profiles(self.profile_manager.profiles())

        # If current image is affected, refresh display
        if self.current_image_path in image_paths:
            self.on_select(self.gallery.currentRow())

        print(f"Deleted profile {profile_id} ({profile.label})")

    def on_profile_renamed_from_faces_window(self, profile_id, new_name):
        """Handle profile rename from faces window"""
        # Reuse the existing on_profile_renamed handler
        self.on_profile_renamed(profile_id, new_name)

    def on_appearance_removed(self, profile_id, image_path, detection_index):
        """Handle removal of an appearance from a profile"""
        profile = self.profile_manager.get_profile(profile_id)
        if not profile:
            return

        # Find and remove the occurrence
        occurrence_to_remove = None
        for occ in profile.occurrences:
            if occ.image_path == image_path and occ.detection_index == detection_index:
                occurrence_to_remove = occ
                break

        if occurrence_to_remove:
            # Remove from profile
            profile.occurrences.remove(occurrence_to_remove)

            # Recalculate embedding sum
            if profile.occurrences:
                profile.embedding_sum = np.sum([occ.embedding for occ in profile.occurrences], axis=0).astype(np.float32)
                profile.occurrence_count = len(profile.occurrences)

                # Update representative face to last occurrence
                last_occ = profile.occurrences[-1]
                if last_occ.face_image is not None and last_occ.face_image.size > 0:
                    x, y, w, h = last_occ.box
                    profile.representative_face = last_occ.face_image.copy()
                    profile.representative_area = w * h
            else:
                # No more occurrences, remove profile
                del self.profile_manager._profiles[profile_id]
                # Also remove from database
                self.db.delete_profile(profile_id)

            # Update database - remove this specific detection
            image_id = self.db.get_image_id(image_path)
            if image_id:
                # Get all detections for this image and remove the specific one
                detections = self.db.get_detections_for_image(image_id)
                for det in detections:
                    if (det['profile_id'] == profile_id and
                        det['detection_index'] == detection_index):
                        # Mark as unassigned (set profile_id to None)
                        self.db.update_detection_profile(det['id'], None)
                        break

            # Clear cache for this image so it gets regenerated
            if image_path in self.analysis_cache:
                del self.analysis_cache[image_path]

            # Update profile panel
            self.profile_panel.set_profiles(self.profile_manager.profiles())

            # If this is the current image, refresh display
            if self.current_image_path == image_path:
                self.on_select(self.gallery.currentRow())

            print(f"Removed appearance from {image_path} for profile {profile_id}")

    def unload_selected_images(self):
        """Unload selected images from the gallery and database."""
        selected_items = self.gallery.selectedItems()

        if len(selected_items) < 1:
            return

        # Confirm unload
        image_word = "image" if len(selected_items) == 1 else "images"
        reply = QMessageBox.question(
            self,
            "Unload Images",
            f"Are you sure you want to unload {len(selected_items)} selected {image_word}?\n"
            "They will be removed from the gallery but files will remain on disk.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Collect paths to unload
        paths_to_unload = []
        for item in selected_items:
            path = item.data(Qt.UserRole)
            if path:
                paths_to_unload.append(path)

        # Remove from cache, database, and profile manager
        profiles_to_check = set()

        for path in paths_to_unload:
            # Remove from cache
            if path in self.analysis_cache:
                del self.analysis_cache[path]

            # Track which profiles are affected
            for profile in self.profile_manager.profiles():
                for occ in profile.occurrences:
                    if occ.image_path == path:
                        profiles_to_check.add(profile.profile_id)
                        break

            # Remove from profile manager occurrences
            self.profile_manager.remove_image_occurrences(path)

            # Delete image and all its detections from database (CASCADE will handle detections)
            self.db.delete_image(path)

            # Remove from images list
            if path in self.images:
                self.images.remove(path)

        # Check profiles and delete empty ones
        empty_profile_ids = []
        for profile_id in profiles_to_check:
            profile = self.profile_manager.get_profile(profile_id)
            if profile and profile.occurrence_count == 0:
                empty_profile_ids.append(profile_id)

        # Delete empty profiles
        for profile_id in empty_profile_ids:
            if profile_id in self.profile_manager._profiles:
                del self.profile_manager._profiles[profile_id]
                self.db.delete_profile(profile_id)
                print(f"Deleted empty profile {profile_id}")

        # Remove items from gallery list widget
        for item in selected_items:
            row = self.gallery.row(item)
            self.gallery.takeItem(row)

        # Clear current display if the current image was unloaded
        if self.current_image_path in paths_to_unload:
            self.current_image_path = None
            self.current_image_original = None
            self.current_annotated_image = None
            self.panel.show_image(QPixmap())
            self.panel.set_faces([])

        # Update profile panel
        self.profile_panel.set_profiles(self.profile_manager.profiles())

        # Disable buttons if no images left
        if not self.images:
            self.process_all_btn.setEnabled(False)

        # Update unload button state
        self.on_selection_changed()

        image_word = "image" if len(paths_to_unload) == 1 else "images"
        QMessageBox.information(
            self,
            "Images Unloaded",
            f"Successfully unloaded {len(paths_to_unload)} {image_word}"
        )
