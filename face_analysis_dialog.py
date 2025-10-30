"""
Face Analysis Results Dialog

Shows emotion and gaze statistics for a person's profile.
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QProgressBar, QTableWidget, QTableWidgetItem,
                             QMessageBox, QGroupBox, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from collections import Counter
from typing import List, Dict
import os

from recognition import FaceProfile
from database import FaceDatabase
from face_analysis import analyze_face


class FaceAnalysisThread(QThread):
    """Background thread for analyzing face images"""
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(dict)  # results dictionary
    error = pyqtSignal(str)

    def __init__(self, profile: FaceProfile, db: FaceDatabase, use_deepface: bool = False):
        super().__init__()
        self.profile = profile
        self.db = db
        self.use_deepface = use_deepface

    def run(self):
        """Analyze all face occurrences for the profile"""
        try:
            results = {
                'emotions': [],
                'gaze_directions': [],
                'processed': 0,
                'skipped': 0,
                'errors': 0
            }

            total = len(self.profile.occurrences)

            for idx, occurrence in enumerate(self.profile.occurrences):
                self.progress.emit(idx + 1, total)

                # Skip if already analyzed
                if occurrence.emotion and occurrence.gaze_direction:
                    results['skipped'] += 1
                    results['emotions'].append(occurrence.emotion)
                    results['gaze_directions'].append(occurrence.gaze_direction)
                    continue

                # Get face image
                face_image = occurrence.face_image

                if face_image is None or face_image.size == 0:
                    results['errors'] += 1
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

                    # Save to database
                    if occurrence.detection_id:
                        self.db.update_detection_analysis(
                            occurrence.detection_id,
                            emotion,
                            gaze
                        )

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

        self.use_deepface_checkbox = QLabel("Note: Using basic emotion detection. Install DeepFace for better accuracy.")
        self.use_deepface_checkbox.setStyleSheet("color: #666; font-size: 10px;")
        options_layout.addWidget(self.use_deepface_checkbox)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Progress
        progress_label = QLabel("Processing images...")
        layout.addWidget(progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Results area
        self.results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(300)
        results_layout.addWidget(self.results_text)

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

    def start_analysis(self):
        """Start the analysis process"""
        self.start_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.results_group.setVisible(False)

        # Start thread
        self.thread = FaceAnalysisThread(self.profile, self.db, use_deepface=False)
        self.thread.progress.connect(self.on_progress)
        self.thread.finished.connect(self.on_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def on_progress(self, current, total):
        """Update progress"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

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

        self.results_text.setHtml("\n".join(text))

        QMessageBox.information(
            self,
            "Analysis Complete",
            f"Analyzed {results['processed']} images for {self.profile.label}"
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
