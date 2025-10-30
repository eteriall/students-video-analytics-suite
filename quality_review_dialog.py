"""
Quality Review Dialog UI

Provides a GUI for reviewing and fixing quality issues in face recognition results.
"""

import cv2
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QListWidget, QListWidgetItem, QGroupBox,
                             QSpinBox, QDoubleSpinBox, QProgressBar, QTextEdit,
                             QMessageBox, QSplitter, QComboBox, QFormLayout,
                             QScrollArea, QWidget, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QColor, QFont, QPixmap, QImage

from quality_review import QualityReviewer, QualityFixer, generate_quality_report, QualityIssue
from database import FaceDatabase


class QualityReviewThread(QThread):
    """Background thread for running quality checks"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)  # List of QualityIssue

    def __init__(self, db_path, min_detection_score, outlier_distance, merge_threshold, min_occurrences):
        super().__init__()
        self.db_path = db_path
        self.min_detection_score = min_detection_score
        self.outlier_distance = outlier_distance
        self.merge_threshold = merge_threshold
        self.min_occurrences = min_occurrences

    def run(self):
        # Create a new database connection in this thread
        thread_db = FaceDatabase(self.db_path)
        reviewer = QualityReviewer(thread_db)

        self.progress.emit(20)
        reviewer.check_low_confidence_detections(self.min_detection_score)

        self.progress.emit(40)
        reviewer.check_profile_outliers(self.outlier_distance)

        self.progress.emit(60)
        reviewer.check_duplicate_profiles(self.merge_threshold)

        self.progress.emit(80)
        reviewer.check_invalid_embeddings()
        reviewer.check_small_profiles(self.min_occurrences)

        self.progress.emit(100)
        self.finished.emit(reviewer.issues)


class QualityReviewDialog(QDialog):
    """Dialog for reviewing and fixing quality issues"""

    # Signals
    qualityImproved = pyqtSignal()  # Emitted when fixes are applied

    def __init__(self, database: FaceDatabase, parent=None):
        super().__init__(parent)
        self.db = database
        self.fixer = QualityFixer(database)
        self.issues = []

        self.setWindowTitle("Face Recognition Quality Review")
        self.setMinimumSize(900, 700)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("Quality Review & Cleanup")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Settings group
        settings_group = QGroupBox("Analysis Settings")
        settings_layout = QFormLayout()

        self.min_score_spin = QDoubleSpinBox()
        self.min_score_spin.setRange(0.5, 0.99)
        self.min_score_spin.setValue(0.95)
        self.min_score_spin.setSingleStep(0.01)
        self.min_score_spin.setDecimals(2)
        settings_layout.addRow("Min Detection Confidence:", self.min_score_spin)

        self.outlier_spin = QDoubleSpinBox()
        self.outlier_spin.setRange(0.3, 1.0)
        self.outlier_spin.setValue(0.6)
        self.outlier_spin.setSingleStep(0.05)
        self.outlier_spin.setDecimals(2)
        settings_layout.addRow("Outlier Distance Threshold:", self.outlier_spin)

        self.merge_spin = QDoubleSpinBox()
        self.merge_spin.setRange(0.1, 0.5)
        self.merge_spin.setValue(0.3)
        self.merge_spin.setSingleStep(0.05)
        self.merge_spin.setDecimals(2)
        settings_layout.addRow("Profile Merge Threshold:", self.merge_spin)

        self.min_occurrences_spin = QSpinBox()
        self.min_occurrences_spin.setRange(1, 20)
        self.min_occurrences_spin.setValue(2)
        self.min_occurrences_spin.setSingleStep(1)
        self.min_occurrences_spin.setToolTip("Profiles with fewer faces than this are considered 'small'")
        settings_layout.addRow("Min Faces Per Profile:", self.min_occurrences_spin)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Run analysis button
        self.run_btn = QPushButton("Run Quality Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        layout.addWidget(self.run_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Splitter for issues list and details
        splitter = QSplitter(Qt.Horizontal)

        # Issues list
        issues_widget = QGroupBox("Issues Found")
        issues_layout = QVBoxLayout()

        self.issue_list = QListWidget()
        self.issue_list.currentItemChanged.connect(self.on_issue_selected)
        issues_layout.addWidget(self.issue_list)

        issues_widget.setLayout(issues_layout)
        splitter.addWidget(issues_widget)

        # Issue details and actions
        details_widget = QGroupBox("Issue Details")
        details_layout = QVBoxLayout()

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(200)
        details_layout.addWidget(self.details_text)

        # Face images section
        images_label = QLabel("Affected Face Images:")
        images_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        details_layout.addWidget(images_label)

        # Scroll area for face images
        self.images_scroll = QScrollArea()
        self.images_scroll.setWidgetResizable(True)
        self.images_scroll.setMinimumHeight(150)

        self.images_widget = QWidget()
        self.images_layout = QGridLayout(self.images_widget)
        self.images_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.images_scroll.setWidget(self.images_widget)

        details_layout.addWidget(self.images_scroll)

        # Action buttons
        action_layout = QHBoxLayout()

        self.fix_btn = QPushButton("Apply Suggested Fix")
        self.fix_btn.setEnabled(False)
        self.fix_btn.clicked.connect(self.apply_fix)
        action_layout.addWidget(self.fix_btn)

        self.skip_btn = QPushButton("Skip")
        self.skip_btn.setEnabled(False)
        self.skip_btn.clicked.connect(self.skip_issue)
        action_layout.addWidget(self.skip_btn)

        details_layout.addLayout(action_layout)
        details_widget.setLayout(details_layout)
        splitter.addWidget(details_widget)

        splitter.setSizes([300, 600])
        layout.addWidget(splitter)

        # Summary
        self.summary_label = QLabel("Run analysis to find quality issues")
        layout.addWidget(self.summary_label)

        # Bottom buttons
        button_layout = QHBoxLayout()

        self.export_btn = QPushButton("Export Report")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_report)
        button_layout.addWidget(self.export_btn)

        button_layout.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def run_analysis(self):
        """Run quality analysis in background thread"""
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.issue_list.clear()

        self.thread = QualityReviewThread(
            self.db.db_path,
            self.min_score_spin.value(),
            self.outlier_spin.value(),
            self.merge_spin.value(),
            self.min_occurrences_spin.value()
        )
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.on_analysis_complete)
        self.thread.start()

    def on_analysis_complete(self, issues):
        """Handle completion of quality analysis"""
        self.issues = issues
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)

        # Populate issue list
        for issue in issues:
            item = QListWidgetItem()
            item.setText(f"[{issue.severity.upper()}] {issue.issue_type}: {issue.description}")
            item.setData(Qt.UserRole, issue)

            # Color code by severity
            if issue.severity == 'high':
                item.setBackground(QColor(255, 200, 200))
            elif issue.severity == 'medium':
                item.setBackground(QColor(255, 240, 200))
            else:
                item.setBackground(QColor(230, 230, 230))

            self.issue_list.addItem(item)

        # Update summary
        high = sum(1 for i in issues if i.severity == 'high')
        medium = sum(1 for i in issues if i.severity == 'medium')
        low = sum(1 for i in issues if i.severity == 'low')

        self.summary_label.setText(
            f"Found {len(issues)} issue(s): {high} high, {medium} medium, {low} low severity"
        )

        self.export_btn.setEnabled(len(issues) > 0)

        if len(issues) == 0:
            QMessageBox.information(self, "Quality Check",
                                   "No quality issues found! Your face recognition data looks good.")

    def on_issue_selected(self, current, previous):
        """Display details when issue is selected"""
        if not current:
            self.details_text.clear()
            self._clear_face_images()
            self.fix_btn.setEnabled(False)
            self.skip_btn.setEnabled(False)
            return

        issue = current.data(Qt.UserRole)

        details = []
        details.append(f"<h3>{issue.issue_type.replace('_', ' ').title()}</h3>")
        details.append(f"<p><b>Severity:</b> {issue.severity.upper()}</p>")
        details.append(f"<p><b>Description:</b> {issue.description}</p>")
        details.append(f"<p><b>Affected items:</b> {len(issue.affected_ids)}</p>")
        details.append(f"<p><b>Suggested action:</b> {issue.suggested_action}</p>")

        if issue.metadata:
            details.append("<p><b>Details:</b></p>")
            details.append("<ul>")
            for key, value in issue.metadata.items():
                if isinstance(value, float):
                    details.append(f"<li>{key}: {value:.4f}</li>")
                else:
                    details.append(f"<li>{key}: {value}</li>")
            details.append("</ul>")

        self.details_text.setHtml("\n".join(details))

        # Load and display face images for affected detections
        self._load_face_images(issue)

        self.fix_btn.setEnabled(True)
        self.skip_btn.setEnabled(True)

    def _clear_face_images(self):
        """Clear all face images from the display"""
        while self.images_layout.count():
            item = self.images_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _load_face_images(self, issue: QualityIssue):
        """Load and display face images for the affected detections"""
        import pickle
        import numpy as np

        # Clear previous images
        self._clear_face_images()

        # Determine if affected_ids are detection IDs or profile IDs
        if issue.issue_type == 'small_profile':
            # For small profiles, load detections from each profile
            detection_ids = []
            for prof_id in issue.affected_ids:
                cursor = self.db.conn.cursor()
                cursor.execute("SELECT id FROM detections WHERE profile_id = ? LIMIT 10", (prof_id,))
                detection_ids.extend([row[0] for row in cursor.fetchall()])
        elif issue.issue_type == 'duplicate_profile':
            # For duplicate profiles, load detections from both profiles
            prof1 = issue.metadata.get('profile1_id')
            prof2 = issue.metadata.get('profile2_id')
            detection_ids = []
            if prof1 and prof2:
                cursor = self.db.conn.cursor()
                cursor.execute("SELECT id FROM detections WHERE profile_id IN (?, ?) LIMIT 20", (prof1, prof2))
                detection_ids = [row[0] for row in cursor.fetchall()]
        else:
            # For other issue types, affected_ids are detection IDs
            detection_ids = issue.affected_ids[:20]  # Limit to first 20 images

        if not detection_ids:
            no_images_label = QLabel("No face images available")
            no_images_label.setStyleSheet("color: gray; font-style: italic;")
            self.images_layout.addWidget(no_images_label, 0, 0)
            return

        # Query database for face images
        cursor = self.db.conn.cursor()
        placeholders = ','.join('?' * len(detection_ids))
        cursor.execute(f"""
            SELECT id, face_image, profile_id
            FROM detections
            WHERE id IN ({placeholders}) AND face_image IS NOT NULL
            ORDER BY id
        """, detection_ids)

        rows = cursor.fetchall()

        if not rows:
            no_images_label = QLabel("No face images available for these detections")
            no_images_label.setStyleSheet("color: gray; font-style: italic;")
            self.images_layout.addWidget(no_images_label, 0, 0)
            return

        # Display face images in a grid (5 columns)
        cols = 5
        for idx, (det_id, face_blob, profile_id) in enumerate(rows):
            try:
                # Deserialize face image
                face_array = pickle.loads(face_blob)

                # Convert numpy array to QPixmap
                if len(face_array.shape) == 2:
                    # Grayscale
                    height, width = face_array.shape
                    bytes_per_line = width
                    q_image = QImage(face_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                else:
                    # RGB - but OpenCV stores as BGR, so convert
                    import cv2
                    face_array_rgb = cv2.cvtColor(face_array, cv2.COLOR_BGR2RGB)
                    height, width, channels = face_array_rgb.shape
                    bytes_per_line = channels * width
                    q_image = QImage(face_array_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

                pixmap = QPixmap.fromImage(q_image)

                # Scale to reasonable size (80x80)
                pixmap = pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                # Create label with image
                image_label = QLabel()
                image_label.setPixmap(pixmap)
                image_label.setToolTip(f"Detection ID: {det_id}\nProfile ID: {profile_id}")
                image_label.setStyleSheet("border: 1px solid #ccc; padding: 2px;")

                # Add to grid
                row = idx // cols
                col = idx % cols
                self.images_layout.addWidget(image_label, row, col)

            except Exception as e:
                print(f"Failed to load face image for detection {det_id}: {e}")
                continue

        # Add count label if there are more images than displayed
        if len(detection_ids) > len(rows):
            info_label = QLabel(f"Showing {len(rows)} of {len(detection_ids)} face images")
            info_label.setStyleSheet("color: gray; font-style: italic; margin-top: 5px;")
            row = (len(rows) // cols) + 1
            self.images_layout.addWidget(info_label, row, 0, 1, cols)

    def apply_fix(self):
        """Apply the suggested fix for current issue"""
        item = self.issue_list.currentItem()
        if not item:
            return

        issue = item.data(Qt.UserRole)

        # Confirm action
        reply = QMessageBox.question(
            self,
            "Confirm Fix",
            f"Apply suggested fix?\n\n{issue.suggested_action}\n\n"
            f"This will affect {len(issue.affected_ids)} item(s).",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Apply fix based on issue type
        try:
            if issue.issue_type == 'low_confidence':
                count = self.fixer.delete_detections(issue.affected_ids)
                QMessageBox.information(self, "Fix Applied",
                                       f"Deleted {count} low-confidence detection(s)")

            elif issue.issue_type == 'outlier':
                count = self.fixer.remove_detections_from_profile(issue.affected_ids)
                QMessageBox.information(self, "Fix Applied",
                                       f"Removed {count} outlier(s) from profile")

            elif issue.issue_type == 'duplicate_profile':
                # Merge profiles
                prof1 = issue.metadata['profile1_id']
                prof2 = issue.metadata['profile2_id']

                # Ask which one to keep
                label1 = issue.metadata['profile1_label']
                label2 = issue.metadata['profile2_label']

                keep = QMessageBox.question(
                    self,
                    "Select Profile to Keep",
                    f"Which profile should we keep?\n\n"
                    f"Yes: Keep '{label1}'\n"
                    f"No: Keep '{label2}'",
                    QMessageBox.Yes | QMessageBox.No
                )

                if keep == QMessageBox.Yes:
                    self.fixer.merge_profiles(prof2, prof1)
                    QMessageBox.information(self, "Merged",
                                           f"Merged '{label2}' into '{label1}'")
                else:
                    self.fixer.merge_profiles(prof1, prof2)
                    QMessageBox.information(self, "Merged",
                                           f"Merged '{label1}' into '{label2}'")

            elif issue.issue_type == 'invalid_embedding':
                count = self.fixer.delete_detections(issue.affected_ids)
                QMessageBox.information(self, "Fix Applied",
                                       f"Deleted {count} detection(s) with invalid embeddings")

            elif issue.issue_type == 'small_profile':
                # Reallocate small profiles to bigger profiles
                min_occurrences = self.min_occurrences_spin.value()
                result = self.fixer.reallocate_small_profiles(issue.affected_ids, min_occurrences)

                # Build detailed message
                if result['reallocated'] > 0:
                    message = f"Successfully reallocated {result['reallocated']} small profile(s) to bigger profiles:\n\n"
                    for detail in result['details']:
                        if detail['status'] == 'success':
                            message += f"✓ '{detail['small_profile']}' → '{detail['merged_into']}' (distance: {detail['distance']:.3f})\n"
                        elif detail['status'] == 'skipped':
                            message += f"⊗ '{detail['small_profile']}' - {detail['reason']}\n"
                        elif detail['status'] == 'failed':
                            message += f"✗ '{detail['small_profile']}' - {detail.get('error', 'Unknown error')}\n"

                    QMessageBox.information(self, "Profiles Reallocated", message)
                else:
                    error_msg = result.get('error', 'Unknown error')
                    QMessageBox.warning(self, "Reallocation Failed",
                                       f"Could not reallocate small profiles.\n\n{error_msg}")

            # Remove from list
            row = self.issue_list.row(item)
            self.issue_list.takeItem(row)
            self.issues.remove(issue)

            # Update summary
            high = sum(1 for i in self.issues if i.severity == 'high')
            medium = sum(1 for i in self.issues if i.severity == 'medium')
            low = sum(1 for i in self.issues if i.severity == 'low')

            self.summary_label.setText(
                f"Found {len(self.issues)} issue(s): {high} high, {medium} medium, {low} low severity"
            )

            # Emit signal
            self.qualityImproved.emit()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply fix: {str(e)}")

    def skip_issue(self):
        """Skip current issue without fixing"""
        current_row = self.issue_list.currentRow()
        if current_row < self.issue_list.count() - 1:
            self.issue_list.setCurrentRow(current_row + 1)
        else:
            self.issue_list.setCurrentRow(0)

    def export_report(self):
        """Export quality report to text"""
        if not self.issues:
            return

        report = generate_quality_report(self.issues)

        # Show in dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Quality Report")
        dialog.setMinimumSize(700, 500)

        layout = QVBoxLayout()

        text_edit = QTextEdit()
        text_edit.setPlainText(report)
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Courier", 10))
        layout.addWidget(text_edit)

        btn_layout = QHBoxLayout()
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(lambda: self.copy_to_clipboard(report))
        btn_layout.addWidget(copy_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)
        dialog.setLayout(layout)
        dialog.exec_()

    def copy_to_clipboard(self, text):
        """Copy text to clipboard"""
        from PyQt5.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        QMessageBox.information(self, "Copied", "Report copied to clipboard")
