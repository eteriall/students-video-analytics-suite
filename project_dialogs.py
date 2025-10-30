"""
Project management UI dialogs.

Provides:
- Project creation/editing dialog
- Project selection dialog
- CAMPUS credentials dialog
- Participant management widgets
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QDateEdit, QListWidget, QListWidgetItem,
    QComboBox, QCheckBox, QMessageBox, QGroupBox, QDialogButtonBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QWidget, QSplitter, QSpinBox
)
from PyQt5.QtGui import QIcon
from datetime import datetime

from projects import Project, Participant, ProjectSettings, CampusAPI


class ParticipantTableWidget(QWidget):
    """Widget for managing participants list."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.participants = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Table for participants
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Name", "Gender", "Notes"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        layout.addWidget(self.table)

        # Buttons
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("Add Participant")
        self.add_btn.clicked.connect(self._add_participant)
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self._remove_participant)

        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)

    def _add_participant(self):
        """Add a new empty row for participant."""
        row = self.table.rowCount()
        self.table.insertRow(row)

        # Name field
        self.table.setItem(row, 0, QTableWidgetItem(""))

        # Gender dropdown
        gender_combo = QComboBox()
        gender_combo.addItems(["", "Male", "Female", "Other"])
        self.table.setCellWidget(row, 1, gender_combo)

        # Notes field
        self.table.setItem(row, 2, QTableWidgetItem(""))

    def _remove_participant(self):
        """Remove selected participant row."""
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)

    def set_participants(self, participants: list):
        """Load participants into table."""
        self.table.setRowCount(0)
        for participant in participants:
            row = self.table.rowCount()
            self.table.insertRow(row)

            # Name
            self.table.setItem(row, 0, QTableWidgetItem(participant.name))

            # Gender
            gender_combo = QComboBox()
            gender_combo.addItems(["", "Male", "Female", "Other"])
            if participant.gender:
                index = gender_combo.findText(participant.gender.capitalize())
                if index >= 0:
                    gender_combo.setCurrentIndex(index)
            self.table.setCellWidget(row, 1, gender_combo)

            # Notes
            self.table.setItem(row, 2, QTableWidgetItem(participant.notes or ""))

    def get_participants(self) -> list:
        """Get participants from table."""
        participants = []
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            notes_item = self.table.item(row, 2)

            if not name_item:
                continue

            name = name_item.text().strip()
            if not name:
                continue

            gender_combo = self.table.cellWidget(row, 1)
            gender = gender_combo.currentText().lower() if gender_combo and gender_combo.currentText() else None

            notes = notes_item.text().strip() if notes_item else ""

            participants.append(Participant(name=name, gender=gender, notes=notes))

        return participants


class ProjectDialog(QDialog):
    """Dialog for creating or editing a project."""

    def __init__(self, project: Project = None, parent=None):
        super().__init__(parent)
        self.project = project
        self.is_edit_mode = project is not None
        self.campus_api = None

        self.setWindowTitle("Edit Project" if self.is_edit_mode else "New Project")
        self.setMinimumSize(700, 600)
        self._init_ui()

        if self.is_edit_mode:
            self._load_project_data()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Fetch from CAMPUS checkbox at the top
        self.fetch_campus_check = QCheckBox("Fetch from CAMPUS")
        self.fetch_campus_check.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.fetch_campus_check.stateChanged.connect(self._on_fetch_campus_changed)
        layout.addWidget(self.fetch_campus_check)

        layout.addSpacing(10)

        # Create form layout for basic fields
        self.form_layout = QFormLayout()

        # Project name (always visible)
        self.name_label = QLabel("Project Name:*")
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., Summer Camp 2024")
        self.form_layout.addRow(self.name_label, self.name_edit)

        # Dates (hidden when fetch_from_campus is checked)
        self.dates_label = QLabel("Dates:")
        date_layout = QHBoxLayout()
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(datetime.now().date())
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(datetime.now().date())
        date_layout.addWidget(QLabel("From:"))
        date_layout.addWidget(self.start_date_edit)
        date_layout.addWidget(QLabel("To:"))
        date_layout.addWidget(self.end_date_edit)
        date_layout.addStretch()
        self.dates_widget = QWidget()
        self.dates_widget.setLayout(date_layout)
        self.form_layout.addRow(self.dates_label, self.dates_widget)

        # Location (hidden when fetch_from_campus is checked)
        self.location_label = QLabel("Location:")
        self.location_edit = QLineEdit()
        self.location_edit.setPlaceholderText("e.g., New York, USA")
        self.form_layout.addRow(self.location_label, self.location_edit)

        # Max people (only shown when fetch_from_campus is NOT checked)
        self.max_people_label = QLabel("Maximum People:")
        self.max_people_spin = QSpinBox()
        self.max_people_spin.setMinimum(1)
        self.max_people_spin.setMaximum(10000)
        self.max_people_spin.setValue(50)
        self.max_people_spin.setSpecialValueText("Unlimited")
        self.max_people_spin.setPrefix("")
        self.form_layout.addRow(self.max_people_label, self.max_people_spin)

        layout.addLayout(self.form_layout)

        # CAMPUS Integration Group (shown only when fetch_from_campus is checked)
        self.campus_group = QGroupBox("CAMPUS Platform Integration")
        campus_layout = QVBoxLayout(self.campus_group)

        campus_form = QFormLayout()
        self.campus_link_edit = QLineEdit()
        self.campus_link_edit.setPlaceholderText("https://campus.example.com/event/123")
        campus_form.addRow("Event Link:", self.campus_link_edit)

        campus_layout.addLayout(campus_form)

        # CAMPUS buttons
        campus_btn_layout = QHBoxLayout()
        self.credentials_btn = QPushButton("Set Credentials")
        self.credentials_btn.clicked.connect(self._show_credentials_dialog)
        self.fetch_details_btn = QPushButton("Fetch Event Details")
        self.fetch_details_btn.clicked.connect(self._fetch_event_details)
        self.fetch_participants_btn = QPushButton("Fetch Participants")
        self.fetch_participants_btn.clicked.connect(self._fetch_participants)
        self.download_images_btn = QPushButton("Download Images")
        self.download_images_btn.clicked.connect(self._download_images)

        campus_btn_layout.addWidget(self.credentials_btn)
        campus_btn_layout.addWidget(self.fetch_details_btn)
        campus_btn_layout.addWidget(self.fetch_participants_btn)
        campus_btn_layout.addWidget(self.download_images_btn)
        campus_btn_layout.addStretch()

        campus_layout.addLayout(campus_btn_layout)

        layout.addWidget(self.campus_group)

        # Participants section (shown only when fetch_from_campus is checked)
        self.participants_group = QGroupBox("Participants")
        participants_layout = QVBoxLayout(self.participants_group)
        self.participants_table = ParticipantTableWidget()
        participants_layout.addWidget(self.participants_table)
        layout.addWidget(self.participants_group)

        # Settings section (collapsible)
        settings_group = QGroupBox("Advanced Settings")
        settings_layout = QFormLayout(settings_group)

        self.threshold_edit = QLineEdit("0.9")
        settings_layout.addRow("Detection Threshold:", self.threshold_edit)

        self.distance_edit = QLineEdit("0.4")
        settings_layout.addRow("Distance Threshold:", self.distance_edit)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["Facenet512", "Facenet", "VGG-Face", "ArcFace", "SFace"])
        settings_layout.addRow("Model:", self.model_combo)

        self.cosine_check = QCheckBox("Use Cosine Distance")
        self.cosine_check.setChecked(True)
        settings_layout.addRow("", self.cosine_check)

        layout.addWidget(settings_group)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._save)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Set initial visibility
        self._on_fetch_campus_changed()

    def _on_fetch_campus_changed(self):
        """Toggle field visibility based on fetch_from_campus checkbox."""
        is_campus = self.fetch_campus_check.isChecked()

        # When fetching from CAMPUS:
        # - Hide: name, dates, location, max_people
        # - Show: CAMPUS integration, participants

        # When NOT fetching from CAMPUS:
        # - Show: name, dates, location, max_people
        # - Hide: CAMPUS integration, participants

        # Toggle basic fields (hide when CAMPUS mode is ON)
        self.name_label.setVisible(not is_campus)
        self.name_edit.setVisible(not is_campus)

        self.dates_label.setVisible(not is_campus)
        self.dates_widget.setVisible(not is_campus)

        self.location_label.setVisible(not is_campus)
        self.location_edit.setVisible(not is_campus)

        # Max people only shown when NOT fetching from CAMPUS
        self.max_people_label.setVisible(not is_campus)
        self.max_people_spin.setVisible(not is_campus)

        # CAMPUS group only shown when fetching from CAMPUS
        self.campus_group.setVisible(is_campus)

        # Participants only shown when fetching from CAMPUS
        self.participants_group.setVisible(is_campus)

    def _load_project_data(self):
        """Load project data into form fields."""
        if not self.project:
            return

        # Load fetch_from_campus checkbox
        self.fetch_campus_check.setChecked(self.project.fetch_from_campus)

        self.name_edit.setText(self.project.name)

        if self.project.start_date:
            self.start_date_edit.setDate(datetime.fromisoformat(self.project.start_date).date())

        if self.project.end_date:
            self.end_date_edit.setDate(datetime.fromisoformat(self.project.end_date).date())

        self.location_edit.setText(self.project.location or "")
        self.campus_link_edit.setText(self.project.campus_event_link or "")

        # Load max_people
        if self.project.max_people:
            self.max_people_spin.setValue(self.project.max_people)

        self.participants_table.set_participants(self.project.participants)

        # Load settings
        self.threshold_edit.setText(str(self.project.settings.detection_threshold))
        self.distance_edit.setText(str(self.project.settings.distance_threshold))
        self.model_combo.setCurrentText(self.project.settings.model_name)
        self.cosine_check.setChecked(self.project.settings.use_cosine)

        # Update field visibility
        self._on_fetch_campus_changed()

    def _show_credentials_dialog(self):
        """Show CAMPUS credentials dialog."""
        dialog = CampusCredentialsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            credentials = dialog.get_credentials()
            self.campus_api = CampusAPI(credentials)
            if self.campus_api.validate_credentials():
                QMessageBox.information(self, "Success", "Credentials validated successfully!")
            else:
                QMessageBox.warning(self, "Invalid Credentials", "Could not validate CAMPUS credentials.")

    def _fetch_event_details(self):
        """Fetch event details from CAMPUS platform."""
        if not self.campus_api:
            QMessageBox.warning(self, "No Credentials", "Please set CAMPUS credentials first.")
            return

        event_link = self.campus_link_edit.text().strip()
        if not event_link:
            QMessageBox.warning(self, "No Link", "Please enter a CAMPUS event link.")
            return

        # TODO: This is a placeholder - actual implementation in projects.py
        details = self.campus_api.fetch_event_details(event_link)
        if details:
            # Populate form with fetched details
            self.name_edit.setText(details.get('title', ''))
            self.location_edit.setText(details.get('location', ''))
            # ... more fields
            QMessageBox.information(self, "Success", "Event details fetched successfully!")
        else:
            QMessageBox.information(self, "TODO", "CAMPUS API integration is not yet implemented.\nSee projects.py for placeholders.")

    def _fetch_participants(self):
        """Fetch participants from CAMPUS platform."""
        if not self.campus_api:
            QMessageBox.warning(self, "No Credentials", "Please set CAMPUS credentials first.")
            return

        event_link = self.campus_link_edit.text().strip()
        if not event_link:
            QMessageBox.warning(self, "No Link", "Please enter a CAMPUS event link.")
            return

        # TODO: This is a placeholder
        participants = self.campus_api.fetch_participants(event_link)
        if participants:
            self.participants_table.set_participants(participants)
            QMessageBox.information(self, "Success", f"Fetched {len(participants)} participants!")
        else:
            QMessageBox.information(self, "TODO", "CAMPUS API integration is not yet implemented.\nSee projects.py for placeholders.")

    def _download_images(self):
        """Download images from CAMPUS event."""
        if not self.campus_api:
            QMessageBox.warning(self, "No Credentials", "Please set CAMPUS credentials first.")
            return

        event_link = self.campus_link_edit.text().strip()
        if not event_link:
            QMessageBox.warning(self, "No Link", "Please enter a CAMPUS event link.")
            return

        # TODO: This is a placeholder
        QMessageBox.information(self, "TODO", "CAMPUS image download is not yet implemented.\nSee projects.py CampusAPI.download_event_images() for placeholder.")

    def _save(self):
        """Validate and save project."""
        is_campus = self.fetch_campus_check.isChecked()

        # Validation based on mode
        if not is_campus:
            name = self.name_edit.text().strip()
            if not name:
                QMessageBox.warning(self, "Validation Error", "Please enter a project name.")
                return
        else:
            # For CAMPUS mode, name will be fetched or set later
            name = self.name_edit.text().strip() or "CAMPUS Project"

        # Create or update project
        if not self.project:
            self.project = Project(id=None, name=name)

        # Save fetch_from_campus state
        self.project.fetch_from_campus = is_campus

        # Save fields based on mode
        self.project.name = name

        if not is_campus:
            # Manual mode - save basic fields and max_people
            self.project.start_date = self.start_date_edit.date().toString(Qt.ISODate)
            self.project.end_date = self.end_date_edit.date().toString(Qt.ISODate)
            self.project.location = self.location_edit.text().strip()
            self.project.max_people = self.max_people_spin.value()
            self.project.campus_event_link = None
            self.project.participants = []
        else:
            # CAMPUS mode - save CAMPUS fields and participants
            self.project.campus_event_link = self.campus_link_edit.text().strip()
            self.project.participants = self.participants_table.get_participants()
            self.project.max_people = None  # Not used in CAMPUS mode

        self.project.campus_credentials_valid = self.campus_api is not None and self.campus_api.authenticated

        # Save settings
        try:
            self.project.settings.detection_threshold = float(self.threshold_edit.text())
            self.project.settings.distance_threshold = float(self.distance_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Validation Error", "Invalid threshold values.")
            return

        self.project.settings.model_name = self.model_combo.currentText()
        self.project.settings.use_cosine = self.cosine_check.isChecked()

        self.accept()

    def get_project(self) -> Project:
        """Get the created/edited project."""
        return self.project


class ProjectSelectionDialog(QDialog):
    """Dialog for selecting/opening a project."""

    # Custom result code for "New Project" button
    NewProject = QDialog.Rejected + 1

    def __init__(self, projects: list, parent=None):
        super().__init__(parent)
        self.projects = projects
        self.selected_project = None
        self._want_new_project = False

        self.setWindowTitle("Select Project")
        self.setMinimumSize(600, 400)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Select a project to open:")
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # Project list
        self.project_list = QListWidget()
        self.project_list.itemDoubleClicked.connect(self._on_project_double_clicked)

        for project in self.projects:
            # Format: "Project Name (Start - End) [Location]"
            date_str = ""
            if project.start_date and project.end_date:
                date_str = f" ({project.start_date} to {project.end_date})"
            elif project.start_date:
                date_str = f" (from {project.start_date})"

            location_str = f" [{project.location}]" if project.location else ""

            item_text = f"{project.name}{date_str}{location_str}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, project.id)
            self.project_list.addItem(item)

        layout.addWidget(self.project_list)

        # Project details
        details_group = QGroupBox("Project Details")
        details_layout = QFormLayout(details_group)
        self.details_label = QLabel("Select a project to view details")
        self.details_label.setWordWrap(True)
        details_layout.addRow(self.details_label)
        layout.addWidget(details_group)

        self.project_list.currentItemChanged.connect(self._on_selection_changed)

        # Buttons
        button_layout = QHBoxLayout()

        self.new_btn = QPushButton("New Project")
        self.new_btn.clicked.connect(self._on_new_project)

        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(self._on_open)
        self.open_btn.setDefault(True)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(self.new_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.open_btn)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

    def _on_selection_changed(self, current, previous):
        """Update details when selection changes."""
        if not current:
            return

        project_id = current.data(Qt.UserRole)
        project = next((p for p in self.projects if p.id == project_id), None)

        if project:
            details = f"<b>{project.name}</b><br>"
            if project.start_date:
                details += f"Start: {project.start_date}<br>"
            if project.end_date:
                details += f"End: {project.end_date}<br>"
            if project.location:
                details += f"Location: {project.location}<br>"
            details += f"Participants: {len(project.participants)}<br>"
            if project.campus_event_link:
                details += f"<a href='{project.campus_event_link}'>CAMPUS Event Link</a><br>"

            self.details_label.setText(details)

    def _on_project_double_clicked(self, item):
        """Open project on double-click."""
        self._on_open()

    def _on_new_project(self):
        """Handle new project button."""
        self._want_new_project = True
        self.done(self.NewProject)

    def _on_open(self):
        """Open selected project."""
        current = self.project_list.currentItem()
        if not current:
            QMessageBox.warning(self, "No Selection", "Please select a project to open.")
            return

        project_id = current.data(Qt.UserRole)
        self.selected_project = next((p for p in self.projects if p.id == project_id), None)
        self.accept()

    def get_selected_project(self):
        """Get the selected project."""
        return self.selected_project

    def wants_new_project(self):
        """Check if user clicked New Project button."""
        return self._want_new_project


class CampusCredentialsDialog(QDialog):
    """Dialog for entering CAMPUS API credentials."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CAMPUS Credentials")
        self.setMinimumWidth(400)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Info label
        info = QLabel("Enter your CAMPUS platform credentials:")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Form
        form = QFormLayout()

        self.username_edit = QLineEdit()
        form.addRow("Username:", self.username_edit)

        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        form.addRow("Password:", self.password_edit)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("Optional")
        form.addRow("API Key:", self.api_key_edit)

        layout.addLayout(form)

        # Note
        note = QLabel("<i>Note: Credentials are not stored persistently. "
                      "You'll need to re-enter them each session.</i>")
        note.setWordWrap(True)
        note.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(note)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_credentials(self):
        """Get entered credentials."""
        credentials = {
            'username': self.username_edit.text().strip(),
            'password': self.password_edit.text().strip(),
        }

        api_key = self.api_key_edit.text().strip()
        if api_key:
            credentials['api_key'] = api_key

        return credentials
