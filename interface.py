import os

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QSize, QPoint, QPointF, QRectF, QEvent, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon, QPainter, QPen, QColor, QCursor, QPainterPath
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
        if self._pixmap.isNull():
            return
        angle = event.angleDelta().y()
        if angle == 0:
            return
        factor = 1.2 if angle > 0 else 1.0 / 1.2
        new_scale = self._scale * factor
        new_scale = max(self._min_scale, min(self._max_scale, new_scale))
        if abs(new_scale - self._scale) < 1e-4:
            return

        pos = event.pos()
        cursor_pos = QPointF(float(pos.x()), float(pos.y()))
        dx = cursor_pos.x() - self._offset.x()
        dy = cursor_pos.y() - self._offset.y()
        image_pos = QPointF(dx / self._scale, dy / self._scale)
        self._scale = new_scale
        self._offset = QPointF(
            cursor_pos.x() - image_pos.x() * self._scale,
            cursor_pos.y() - image_pos.y() * self._scale,
        )
        self._user_modified = True
        self._limit_offset()
        self._update_highlight_from_cursor(event.pos())

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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_view = ZoomableImageView()
        self.image_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.faces_container = QWidget()
        self.faces_layout = QHBoxLayout(self.faces_container)
        self.faces_layout.setContentsMargins(8, 8, 8, 8)
        self.faces_layout.setSpacing(8)
        self.face_widgets = []
        self.face_labels = []
        self.face_clusters = []
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.faces_container)
        self.scroll.setFixedHeight(180)
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_view)
        layout.addWidget(self.scroll)

    def show_image(self, pixmap, reset_view=True):
        self.image_view.setPixmap(pixmap, reset_view=reset_view)

    def set_faces(self, faces, clusters=None):
        for label in self.face_labels:
            label.removeEventFilter(self)
        while self.faces_layout.count():
            w = self.faces_layout.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.face_widgets = []
        self.face_labels = []
        self.face_clusters = list(clusters) if clusters is not None else []
        if len(self.face_clusters) < len(faces):
            self.face_clusters.extend([0] * (len(faces) - len(self.face_clusters)))
        for idx, pm in enumerate(faces):
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(0)
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setPixmap(pm.scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            img_label.setFixedSize(170, 170)
            img_label.setProperty("faceIndex", idx)
            img_label.installEventFilter(self)
            vbox.addWidget(img_label)
            self.faces_layout.addWidget(container)
            self.face_widgets.append(container)
            self.face_labels.append(img_label)
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
        if obj in self.face_labels:
            idx_var = obj.property("faceIndex")
            idx = int(idx_var) if idx_var is not None else -1
            if event.type() == QEvent.Enter:
                self.image_view.set_external_highlight(idx)
                return False
            if event.type() == QEvent.Leave:
                self.image_view.clear_external_highlight(idx)
                return False
        return super().eventFilter(obj, event)


class ProfilePanel(QWidget):
    profileSelected = pyqtSignal(int)
    occurrenceActivated = pyqtSignal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(240)
        self.profile_list = QListWidget()
        self.profile_list.setIconSize(QSize(72, 72))
        self.profile_list.setSelectionMode(QListWidget.SingleSelection)
        self.profile_list.itemSelectionChanged.connect(self._on_profile_selection)
        self.occurrence_list = QListWidget()
        self.occurrence_list.setSelectionMode(QListWidget.SingleSelection)
        self.occurrence_list.itemClicked.connect(self._on_occurrence_clicked)
        self.occurrence_list.itemDoubleClicked.connect(self._on_occurrence_clicked)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        faces_title = QLabel("Faces")
        faces_title.setStyleSheet("font-weight: 600;")
        layout.addWidget(faces_title)
        layout.addWidget(self.profile_list, 1)
        occurrences_title = QLabel("Appearances")
        occurrences_title.setStyleSheet("font-weight: 600;")
        layout.addWidget(occurrences_title)
        layout.addWidget(self.occurrence_list, 1)
        self._profiles = {}
        self._suspend_signals = False

    def set_profiles(self, profiles):
        selected_id = self.current_profile_id()
        current_occurrence = None
        current_item = self.occurrence_list.currentItem()
        if current_item:
            current_occurrence = current_item.data(Qt.UserRole)
        self._suspend_signals = True
        self.profile_list.blockSignals(True)
        self.profile_list.clear()
        self.profile_list.blockSignals(False)
        self._profiles = {p.profile_id: p for p in profiles}
        for profile in profiles:
            item = QListWidgetItem(profile.label)
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
            return
        profile = self._profiles.get(profile_id)
        if not profile:
            return
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

    def clear(self):
        self.profile_list.clear()
        self.occurrence_list.clear()
        self._profiles = {}
        self._suspend_signals = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Gallery")
        self.resize(1200, 800)
        self.gallery = QListWidget()
        self.gallery.setIconSize(QSize(64, 64))
        self.gallery.currentRowChanged.connect(self.on_select)
        self.add_btn = QPushButton("Add Images")
        self.add_btn.clicked.connect(self.add_images)
        self.blur_btn = QPushButton("Blur Faces")
        self.blur_btn.clicked.connect(self.blur_faces)
        self.blur_btn.setEnabled(False)
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.add_btn)
        left_layout.addWidget(self.blur_btn)
        left_layout.addWidget(self.gallery)
        left = QWidget()
        left.setLayout(left_layout)
        self.panel = FacePanel()
        self.panel.image_view.hoverFaceChanged.connect(self.on_face_hover)
        self.profile_panel = ProfilePanel()
        self.profile_panel.profileSelected.connect(self.on_profile_selected)
        self.profile_panel.occurrenceActivated.connect(self.on_occurrence_activated)
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
        self.images = []
        self.detection_threshold = 0.7
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
        self.profile_manager = FaceProfileManager()
        self.pending_highlight = None

    def add_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select images", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if not files:
            return
        added = 0
        for fpath in files:
            if not os.path.isfile(fpath):
                continue
            if fpath in self.images:
                continue
            img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            th = 64
            h, w = img.shape[:2]
            if w == 0:
                continue
            scale = th / max(h, w)
            nw, nh = int(w * scale), int(h * scale)
            thumb = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
            item = QListWidgetItem(os.path.basename(fpath))
            item.setData(Qt.UserRole, fpath)
            item.setIcon(QIcon(cv_to_qpixmap(thumb)))
            self.gallery.addItem(item)
            self.images.append(fpath)
            added += 1
        if added and self.gallery.currentRow() < 0:
            self.gallery.setCurrentRow(0)

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
            embedding = compute_face_embedding(crop)
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
        annotated = draw_face_boxes(img.copy(), detections, color_ids)
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
        return entry

    def on_select(self, row):
        if row < 0:
            return
        item = self.gallery.item(row)
        if not item:
            return
        path = item.data(Qt.UserRole)
        try:
            entry = self.analyze_image(path)
        except Exception:
            QMessageBox.warning(self, "Error", "Failed to load image.")
            return
        annotated = entry.get("annotated")
        pixmap = cv_to_qpixmap(annotated) if annotated is not None else QPixmap()
        self.panel.show_image(pixmap)
        color_ids = entry.get("profile_ids") or entry.get("clusters") or []
        self.panel.image_view.set_detections(entry.get("detections", []), color_ids)
        face_pixmaps = [cv_to_qpixmap(face) for face in entry.get("faces", [])]
        self.panel.set_faces(face_pixmaps, color_ids)
        original = entry.get("original")
        self.current_image_original = original.copy() if original is not None else None
        self.current_annotated_image = annotated.copy() if annotated is not None else None
        self.current_blurred_image = None
        self.current_blurred_faces = []
        self.current_detections = entry.get("detections", [])
        self.current_clusters = entry.get("clusters", [])
        self.current_profile_ids = list(color_ids)
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

                annotated = draw_face_boxes(blurred, self.current_detections, self.current_color_ids)
                self.current_blurred_image = annotated.copy() if annotated is not None else None
                self.current_blurred_faces = [face.copy() for face in blurred_faces]
            if self.current_blurred_image is None:
                annotated = draw_face_boxes(self.current_image_original, self.current_detections, self.current_color_ids)
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
            self.panel.set_faces([cv_to_qpixmap(face) for face in self.current_blurred_faces], self.current_color_ids)
            self.current_faces = [face.copy() for face in self.current_blurred_faces]
            self.is_blurred = True
            self.blur_btn.setText("Unblur Faces")
        else:
            annotated = self.current_annotated_image
            if annotated is None and self.current_image_original is not None:
                annotated = draw_face_boxes(self.current_image_original, self.current_detections, self.current_color_ids)
                self.current_annotated_image = annotated.copy() if annotated is not None else None
            pixmap = cv_to_qpixmap(annotated) if annotated is not None else QPixmap()
            self.panel.show_image(pixmap, reset_view=False)
            self.panel.set_faces([cv_to_qpixmap(face) for face in self.current_faces_original], self.current_color_ids)
            self.current_faces = [face.copy() for face in self.current_faces_original]
            self.is_blurred = False
            self.blur_btn.setText("Blur Faces")

    def on_face_hover(self, index):
        if not self.current_faces:
            self.panel.highlight_face(-1)
            return
        if index < 0 or index >= len(self.current_faces):
            self.panel.highlight_face(-1)
        else:
            cluster_id = self.current_color_ids[index] if index < len(self.current_color_ids) else -1
            self.panel.highlight_face(index, cluster_id)
