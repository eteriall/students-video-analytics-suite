"""
Face Analysis Results Dialog

Shows emotion and gaze statistics for a person's profile.
"""

import cv2
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QProgressBar, QMessageBox, QGroupBox,
                             QTextEdit, QScrollArea, QWidget, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage
from collections import Counter

from recognition import FaceProfile
from database import FaceDatabase
from face_analysis import analyze_face


def cv_to_qpixmap(cv_img):
    """Convert OpenCV image to QPixmap"""
    if cv_img is None or cv_img.size == 0:
        return QPixmap()

    # Convert BGR to RGB
    if len(cv_img.shape) == 3:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv_img

    h, w = rgb.shape[:2]
    if len(rgb.shape) == 3:
        bytes_per_line = 3 * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    else:
        bytes_per_line = w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_Grayscale8)

    return QPixmap.fromImage(q_img)


class FaceAnalysisThread(QThread):
    """Background thread for analyzing face images"""
    progress = pyqtSignal(int, int)  # current, total
    face_analyzed = pyqtSignal(object, str, str)  # face_image, emotion, gaze_direction
    finished = pyqtSignal(dict)  # results dictionary
    error = pyqtSignal(str)

    def __init__(self, profile: FaceProfile, db_path: str, use_deepface: bool = False):
        super().__init__()
        self.profile = profile
        self.db_path = db_path
        self.use_deepface = use_deepface

    def run(self):
        """Analyze all face occurrences for the profile"""
        try:
            # Create thread-local database connection
            thread_db = FaceDatabase(self.db_path)

            results = {
                'emotions': [],
                'gaze_directions': [],
                'processed': 0,
                'skipped': 0,
                'errors': 0,
                'individual_results': []
            }

            total = len(self.profile.occurrences)

            for idx, occurrence in enumerate(self.profile.occurrences):
                self.progress.emit(idx + 1, total)

                # Get face image
                face_image = occurrence.face_image

                if face_image is None or face_image.size == 0:
                    results['errors'] += 1
                    continue

                # Check if already analyzed
                if occurrence.emotion and occurrence.gaze_direction:
                    results['skipped'] += 1
                    results['emotions'].append(occurrence.emotion)
                    results['gaze_directions'].append(occurrence.gaze_direction)

                    # Emit for display
                    self.face_analyzed.emit(face_image, occurrence.emotion, occurrence.gaze_direction)

                    results['individual_results'].append({
                        'face_image': face_image,
                        'emotion': occurrence.emotion,
                        'gaze_direction': occurrence.gaze_direction,
                        'was_cached': True
                    })
                    continue

                # Analyze
                try:
                    analysis = analyze_face(face_image, use_deepface=self.use_deepface)

                    emotion = analysis.get('emotion')
                    gaze = analysis.get('gaze_direction')

                    if emotion:
                        results['emotions'].append(emotion)
                    if gaze:
                        results['gaze_directions'].append(gaze)

                    # Update occurrence
                    occurrence.emotion = emotion
                    occurrence.gaze_direction = gaze

                    # Save to database using thread-local connection
                    if occurrence.detection_id:
                        thread_db.update_detection_analysis(
                            occurrence.detection_id,
                            emotion,
                            gaze
                        )

                    # Emit for display
                    self.face_analyzed.emit(face_image, emotion, gaze)

                    results['individual_results'].append({
                        'face_image': face_image,
                        'emotion': emotion,
                        'gaze_direction': gaze,
                        'was_cached': False
                    })

                    results['processed'] += 1

                except Exception as e:
                    print(f"Error analyzing occurrence: {e}")
                    results['errors'] += 1

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class FaceAnalysisDialog(QDialog):
    """Dialog showing face analysis results"""

    def __init__(self, profile: FaceProfile, db: FaceDatabase, parent=None):
        super().__init__(parent)
        self.profile = profile
        self.db = db
        self.thread = None

        self.setWindowTitle(f"Face Analysis - {profile.label}")
        self.setMinimumSize(700, 600)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel(f"Analyzing: {self.profile.label}")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Info
        info = QLabel(f"Total images: {len(self.profile.occurrences)}")
        layout.addWidget(info)

        # Options
        options_group = QGroupBox("Analysis Options")
        options_layout = QVBoxLayout()

        # Check if DeepFace is available
        try:
            import deepface
            deepface_available = True
            note_text = "Note: Using DeepFace for enhanced emotion detection."
        except ImportError:
            deepface_available = False
            note_text = "Note: Using basic emotion detection. Install DeepFace for better accuracy: pip install deepface"

        self.deepface_available = deepface_available

        note_label = QLabel(note_text)
        note_label.setStyleSheet("color: #666; font-size: 10px;")
        note_label.setWordWrap(True)
        options_layout.addWidget(note_label)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Progress
        progress_label = QLabel("Processing images...")
        layout.addWidget(progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Individual results area with scrolling
        self.results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()

        # Scroll area for face images
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(300)

        self.scroll_widget = QWidget()
        self.scroll_layout = QGridLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll_area.setWidget(self.scroll_widget)

        results_layout.addWidget(self.scroll_area)

        # Summary statistics text
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(200)
        results_layout.addWidget(QLabel("Summary Statistics:"))
        results_layout.addWidget(self.summary_text)

        self.results_group.setLayout(results_layout)
        self.results_group.setVisible(False)
        layout.addWidget(self.results_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Analysis")
        self.start_btn.clicked.connect(self.start_analysis)
        button_layout.addWidget(self.start_btn)

        button_layout.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Track face count for grid layout
        self.face_count = 0

    def start_analysis(self):
        """Start the analysis process"""
        self.start_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.results_group.setVisible(True)
        self.face_count = 0

        # Clear previous results
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Start thread with db_path instead of db object
        self.thread = FaceAnalysisThread(
            self.profile,
            self.db.db_path,
            use_deepface=self.deepface_available
        )
        self.thread.progress.connect(self.on_progress)
        self.thread.face_analyzed.connect(self.on_face_analyzed)
        self.thread.finished.connect(self.on_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def on_progress(self, current, total):
        """Update progress"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def on_face_analyzed(self, face_image, emotion, gaze_direction):
        """Display individual face analysis result"""
        # Create widget for this face
        face_widget = QWidget()
        face_layout = QVBoxLayout(face_widget)
        face_layout.setContentsMargins(5, 5, 5, 5)
        face_layout.setSpacing(3)

        # Face image
        pixmap = cv_to_qpixmap(face_image)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        face_label = QLabel()
        face_label.setPixmap(pixmap)
        face_label.setAlignment(Qt.AlignCenter)
        face_label.setStyleSheet("border: 1px solid #ccc; padding: 2px;")
        face_layout.addWidget(face_label)

        # Emotion label
        emotion_label = QLabel(f"<b>Emotion:</b><br>{emotion if emotion else 'Unknown'}")
        emotion_label.setAlignment(Qt.AlignCenter)
        emotion_label.setStyleSheet("font-size: 9px;")
        emotion_label.setWordWrap(True)
        face_layout.addWidget(emotion_label)

        # Gaze label
        gaze_text = gaze_direction.replace('_', ' ').title() if gaze_direction else 'Unknown'
        gaze_label = QLabel(f"<b>Gaze:</b><br>{gaze_text}")
        gaze_label.setAlignment(Qt.AlignCenter)
        gaze_label.setStyleSheet("font-size: 9px;")
        gaze_label.setWordWrap(True)
        face_layout.addWidget(gaze_label)

        # Add to grid (5 columns)
        cols = 5
        row = self.face_count // cols
        col = self.face_count % cols
        self.scroll_layout.addWidget(face_widget, row, col)

        self.face_count += 1

    def on_finished(self, results):
        """Display results"""
        self.progress_bar.setVisible(False)
        self.start_btn.setEnabled(True)
        self.results_group.setVisible(True)

        # Build results text
        text = []

        text.append("<h2>Analysis Complete</h2>")
        text.append(f"<p><b>Processed:</b> {results['processed']} images</p>")
        text.append(f"<p><b>Already analyzed:</b> {results['skipped']} images</p>")
        if results['errors'] > 0:
            text.append(f"<p><b>Errors:</b> {results['errors']} images</p>")

        # Emotion statistics
        if results['emotions']:
            emotion_counts = Counter(results['emotions'])
            text.append("<h3>Emotion Distribution</h3>")
            text.append("<table border='1' cellpadding='5'>")
            text.append("<tr><th>Emotion</th><th>Count</th><th>Percentage</th></tr>")

            total_emotions = len(results['emotions'])
            for emotion, count in emotion_counts.most_common():
                percentage = (count / total_emotions) * 100
                text.append(f"<tr><td>{emotion.capitalize()}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>")

            text.append("</table>")
        else:
            text.append("<p><i>No emotion data available</i></p>")

        # Gaze statistics
        if results['gaze_directions']:
            gaze_counts = Counter(results['gaze_directions'])
            text.append("<h3>Gaze Direction Distribution</h3>")
            text.append("<table border='1' cellpadding='5'>")
            text.append("<tr><th>Direction</th><th>Count</th><th>Percentage</th></tr>")

            total_gaze = len(results['gaze_directions'])
            for gaze, count in gaze_counts.most_common():
                percentage = (count / total_gaze) * 100
                gaze_label = gaze.replace('_', ' ').title()
                text.append(f"<tr><td>{gaze_label}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>")

            text.append("</table>")
        else:
            text.append("<p><i>No gaze direction data available</i></p>")

        self.summary_text.setHtml("\n".join(text))

        QMessageBox.information(
            self,
            "Analysis Complete",
            f"Analyzed {results['processed']} new images + "
            f"{results['skipped']} cached for {self.profile.label}"
        )

    def on_error(self, error_msg):
        """Handle error"""
        self.progress_bar.setVisible(False)
        self.start_btn.setEnabled(True)

        QMessageBox.critical(
            self,
            "Analysis Error",
            f"An error occurred during analysis:\n{error_msg}"
        )
