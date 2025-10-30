"""
Histogram Dialog - Face Count Distribution

Shows a histogram of how many faces are detected in each image,
with interactive range selection and export functionality.
"""

import os
import shutil
import zipfile
from typing import Dict, List, Tuple
from collections import Counter

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from database import FaceDatabase


class HistogramCanvas(FigureCanvas):
    """Canvas for displaying the histogram with interactive selection"""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        self.face_counts = {}  # {image_path: face_count}
        self.selection_rect = None
        self.selection_range = None  # (min_faces, max_faces)
        self.is_selecting = False
        self.start_x = None

        # Connect mouse events
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('button_release_event', self.on_release)

        self.parent_dialog = parent

    def load_data(self, db: FaceDatabase):
        """Load face count data from database"""
        # Get all images
        images_data = db.get_all_images()

        self.face_counts = {}
        for img_data in images_data:
            image_path = img_data['path']
            if not os.path.isfile(image_path):
                continue

            # Count faces for this image
            detections = db.get_detections_for_image(db.get_image_id(image_path))
            face_count = len(detections) if detections else 0
            self.face_counts[image_path] = face_count

        self.draw_histogram()

    def draw_histogram(self):
        """Draw the histogram"""
        self.ax.clear()

        # Reset rectangle reference since we're clearing
        self.selection_rect = None

        if not self.face_counts:
            self.ax.text(0.5, 0.5, 'No data available',
                        ha='center', va='center', transform=self.ax.transAxes)
            self.draw()
            return

        # Count occurrences
        count_distribution = Counter(self.face_counts.values())

        # Prepare data for histogram
        if not count_distribution:
            self.ax.text(0.5, 0.5, 'No faces detected in any images',
                        ha='center', va='center', transform=self.ax.transAxes)
            self.draw()
            return

        max_faces = max(count_distribution.keys())
        x_values = list(range(0, max_faces + 1))
        y_values = [count_distribution.get(x, 0) for x in x_values]

        # Draw bars
        self.ax.bar(x_values, y_values, color='steelblue', edgecolor='black', alpha=0.7)

        # Labels and title
        self.ax.set_xlabel('Number of Faces Detected', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        self.ax.set_title('Face Count Distribution Across Images', fontsize=14, fontweight='bold')
        self.ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Set integer ticks on x-axis
        self.ax.set_xticks(x_values)

        # Draw selection rectangle if exists
        if self.selection_range:
            self._draw_selection()

        self.fig.tight_layout()
        self.draw()

    def _draw_selection(self):
        """Draw the selection rectangle"""
        if not self.selection_range:
            return

        min_faces, max_faces = self.selection_range

        # Get y-axis limits
        y_max = self.ax.get_ylim()[1]

        # Draw new rectangle (old one was cleared with ax.clear())
        width = max_faces - min_faces + 1
        self.selection_rect = Rectangle(
            (min_faces - 0.5, 0), width, y_max,
            facecolor='yellow', alpha=0.3, edgecolor='orange', linewidth=2
        )
        self.ax.add_patch(self.selection_rect)

    def on_press(self, event):
        """Handle mouse press"""
        if event.inaxes != self.ax or event.button != 1:
            return

        self.is_selecting = True
        self.start_x = event.xdata

    def on_motion(self, event):
        """Handle mouse motion"""
        if not self.is_selecting or event.inaxes != self.ax:
            return

        if self.start_x is None:
            return

        # Update selection range
        current_x = event.xdata
        min_x = min(self.start_x, current_x)
        max_x = max(self.start_x, current_x)

        # Round to integers (face counts)
        min_faces = max(0, int(round(min_x)))
        max_faces = int(round(max_x))

        self.selection_range = (min_faces, max_faces)
        self.draw_histogram()

    def on_release(self, event):
        """Handle mouse release"""
        if not self.is_selecting:
            return

        self.is_selecting = False

        if self.start_x is None or event.xdata is None:
            return

        # Finalize selection
        current_x = event.xdata
        min_x = min(self.start_x, current_x)
        max_x = max(self.start_x, current_x)

        min_faces = max(0, int(round(min_x)))
        max_faces = int(round(max_x))

        self.selection_range = (min_faces, max_faces)
        self.draw_histogram()

        # Notify parent
        if self.parent_dialog:
            self.parent_dialog.update_selection_info()

    def get_selected_images(self) -> List[str]:
        """Get list of images in the selected range"""
        if not self.selection_range:
            return []

        min_faces, max_faces = self.selection_range
        selected = []

        for image_path, face_count in self.face_counts.items():
            if min_faces <= face_count <= max_faces:
                selected.append(image_path)

        return selected

    def clear_selection(self):
        """Clear the current selection"""
        self.selection_range = None
        self.selection_rect = None
        self.draw_histogram()


class FaceCountHistogramDialog(QDialog):
    """Dialog for viewing face count histogram and exporting images"""

    def __init__(self, db: FaceDatabase, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("Face Count Distribution")
        self.setMinimumSize(1000, 700)

        self.init_ui()
        self.canvas.load_data(self.db)

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("Face Count Distribution")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Instructions
        instructions = QLabel(
            "Click and drag on the histogram to select a range of face counts.\n"
            "Then click 'Export as Zip' to save those images to a zip file."
        )
        instructions.setStyleSheet("padding: 5px; color: #666;")
        layout.addWidget(instructions)

        # Canvas
        self.canvas = HistogramCanvas(self)
        layout.addWidget(self.canvas)

        # Selection info
        self.selection_label = QLabel("No range selected")
        self.selection_label.setStyleSheet("padding: 10px; font-weight: bold;")
        layout.addWidget(self.selection_label)

        # Buttons
        button_layout = QHBoxLayout()

        self.clear_btn = QPushButton("Clear Selection")
        self.clear_btn.clicked.connect(self.clear_selection)
        self.clear_btn.setEnabled(False)
        button_layout.addWidget(self.clear_btn)

        self.export_btn = QPushButton("Export as Zip")
        self.export_btn.clicked.connect(self.export_images)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)

        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def update_selection_info(self):
        """Update the selection info label"""
        if not self.canvas.selection_range:
            self.selection_label.setText("No range selected")
            self.clear_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            return

        min_faces, max_faces = self.canvas.selection_range
        selected_images = self.canvas.get_selected_images()
        count = len(selected_images)

        if count == 0:
            self.selection_label.setText(
                f"Selected range: {min_faces}-{max_faces} faces (no images in range)"
            )
            self.clear_btn.setEnabled(True)
            self.export_btn.setEnabled(False)
        else:
            self.selection_label.setText(
                f"Selected range: {min_faces}-{max_faces} faces ({count} image{'s' if count != 1 else ''})"
            )
            self.clear_btn.setEnabled(True)
            self.export_btn.setEnabled(True)

    def clear_selection(self):
        """Clear the selection"""
        self.canvas.clear_selection()
        self.update_selection_info()

    def export_images(self):
        """Export selected images to a zip file"""
        selected_images = self.canvas.get_selected_images()

        if not selected_images:
            QMessageBox.warning(self, "No Images", "No images in the selected range.")
            return

        # Generate default filename based on range
        min_faces, max_faces = self.canvas.selection_range
        default_filename = f"{min_faces}-{max_faces}-photos.zip"

        # Ask for save location
        zip_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Images as Zip",
            default_filename,
            "Zip Files (*.zip)"
        )

        if not zip_path:
            return

        # Ensure .zip extension
        if not zip_path.endswith('.zip'):
            zip_path += '.zip'

        # Export images to zip
        try:
            exported_count = 0
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for image_path in selected_images:
                    if not os.path.isfile(image_path):
                        continue

                    filename = os.path.basename(image_path)

                    # Add file to zip with just the filename (no path)
                    zipf.write(image_path, filename)
                    exported_count += 1

            QMessageBox.information(
                self,
                "Export Complete",
                f"Successfully exported {exported_count} image{'s' if exported_count != 1 else ''} to:\n{zip_path}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export images:\n{str(e)}"
            )
