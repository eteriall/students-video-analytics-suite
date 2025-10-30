"""
Project management module for organizing face recognition work.

Supports:
- Multiple projects with metadata
- Per-project settings and databases
- CAMPUS platform integration (placeholders)
- Participant lists with demographics
"""

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path


@dataclass
class Participant:
    """Participant in a project/event."""
    name: str
    gender: Optional[str] = None  # "male", "female", "other", or None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Participant':
        return cls(**data)


@dataclass
class ProjectSettings:
    """Per-project settings."""
    detection_threshold: float = 0.9
    distance_threshold: float = 0.4
    model_name: str = "Facenet512"
    use_cosine: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'ProjectSettings':
        return cls(**data)


@dataclass
class Project:
    """Project/Event data model."""
    id: Optional[int]
    name: str
    fetch_from_campus: bool = False  # If True, fetch data from CAMPUS platform
    max_people: Optional[int] = None  # Maximum number of people (only used when fetch_from_campus=False)
    start_date: Optional[str] = None  # ISO format YYYY-MM-DD
    end_date: Optional[str] = None    # ISO format YYYY-MM-DD
    location: str = ""
    participants: List[Participant] = field(default_factory=list)
    campus_event_link: Optional[str] = None
    campus_credentials_valid: bool = False
    settings: ProjectSettings = field(default_factory=ProjectSettings)
    database_path: Optional[str] = None  # Path to project's database
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert settings and participants to dict
        data['participants'] = [p.to_dict() for p in self.participants]
        data['settings'] = self.settings.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'Project':
        """Create from dictionary."""
        # Parse participants
        participants_data = data.pop('participants', [])
        participants = [Participant.from_dict(p) for p in participants_data]

        # Parse settings
        settings_data = data.pop('settings', {})
        settings = ProjectSettings.from_dict(settings_data)

        return cls(
            participants=participants,
            settings=settings,
            **data
        )


class ProjectManager:
    """Manages projects and their metadata."""

    def __init__(self, db_path: str = "projects.db"):
        """Initialize project manager with database."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create projects table if it doesn't exist."""
        cursor = self.conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                fetch_from_campus BOOLEAN DEFAULT 0,
                max_people INTEGER,
                start_date TEXT,
                end_date TEXT,
                location TEXT,
                participants TEXT,  -- JSON array of participants
                campus_event_link TEXT,
                campus_credentials_valid BOOLEAN DEFAULT 0,
                settings TEXT,  -- JSON settings
                database_path TEXT,  -- Path to project's face_recognition.db
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create index on name for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_projects_name
            ON projects(name)
        ''')

        # Migrate existing projects table if needed (add new columns)
        self._migrate_schema()

        self.conn.commit()

    def _migrate_schema(self):
        """Add new columns to existing projects table if they don't exist."""
        cursor = self.conn.cursor()

        # Check if columns exist and add them if they don't
        cursor.execute("PRAGMA table_info(projects)")
        columns = [col[1] for col in cursor.fetchall()]

        if 'fetch_from_campus' not in columns:
            cursor.execute('ALTER TABLE projects ADD COLUMN fetch_from_campus BOOLEAN DEFAULT 0')

        if 'max_people' not in columns:
            cursor.execute('ALTER TABLE projects ADD COLUMN max_people INTEGER')

    def create_project(self, project: Project) -> int:
        """Create a new project and return its ID."""
        cursor = self.conn.cursor()

        # Generate database path for this project
        project_db_name = f"project_{project.name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        project.database_path = project_db_name

        # Serialize participants and settings to JSON
        participants_json = json.dumps([p.to_dict() for p in project.participants])
        settings_json = json.dumps(project.settings.to_dict())

        cursor.execute('''
            INSERT INTO projects
            (name, fetch_from_campus, max_people, start_date, end_date, location, participants,
             campus_event_link, campus_credentials_valid, settings, database_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            project.name,
            project.fetch_from_campus,
            project.max_people,
            project.start_date,
            project.end_date,
            project.location,
            participants_json,
            project.campus_event_link,
            project.campus_credentials_valid,
            settings_json,
            project.database_path
        ))

        self.conn.commit()
        return cursor.lastrowid

    def get_project(self, project_id: int) -> Optional[Project]:
        """Get a project by ID."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM projects WHERE id = ?', (project_id,))
        row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_project(row)

    def get_all_projects(self) -> List[Project]:
        """Get all projects ordered by updated_at DESC."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM projects ORDER BY updated_at DESC')
        return [self._row_to_project(row) for row in cursor.fetchall()]

    def update_project(self, project: Project) -> bool:
        """Update an existing project."""
        if project.id is None:
            return False

        cursor = self.conn.cursor()

        # Serialize participants and settings to JSON
        participants_json = json.dumps([p.to_dict() for p in project.participants])
        settings_json = json.dumps(project.settings.to_dict())

        cursor.execute('''
            UPDATE projects SET
                name = ?,
                fetch_from_campus = ?,
                max_people = ?,
                start_date = ?,
                end_date = ?,
                location = ?,
                participants = ?,
                campus_event_link = ?,
                campus_credentials_valid = ?,
                settings = ?,
                database_path = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (
            project.name,
            project.fetch_from_campus,
            project.max_people,
            project.start_date,
            project.end_date,
            project.location,
            participants_json,
            project.campus_event_link,
            project.campus_credentials_valid,
            settings_json,
            project.database_path,
            project.id
        ))

        self.conn.commit()
        return cursor.rowcount > 0

    def delete_project(self, project_id: int) -> bool:
        """Delete a project (does not delete associated database file)."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM projects WHERE id = ?', (project_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def _row_to_project(self, row: sqlite3.Row) -> Project:
        """Convert database row to Project object."""
        # Parse JSON fields
        participants_data = json.loads(row['participants']) if row['participants'] else []
        participants = [Participant.from_dict(p) for p in participants_data]

        settings_data = json.loads(row['settings']) if row['settings'] else {}
        settings = ProjectSettings.from_dict(settings_data)

        return Project(
            id=row['id'],
            name=row['name'],
            fetch_from_campus=bool(row.get('fetch_from_campus', False)),
            max_people=row.get('max_people'),
            start_date=row['start_date'],
            end_date=row['end_date'],
            location=row['location'],
            participants=participants,
            campus_event_link=row['campus_event_link'],
            campus_credentials_valid=bool(row['campus_credentials_valid']),
            settings=settings,
            database_path=row['database_path'],
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# CAMPUS API Integration (Placeholder functions with TODOs)

class CampusAPI:
    """
    CAMPUS platform API integration.

    TODO: Implement actual API integration with CAMPUS platform.
    This is a placeholder structure for future implementation.
    """

    def __init__(self, credentials: Optional[Dict[str, str]] = None):
        """
        Initialize CAMPUS API client.

        Args:
            credentials: Dict with 'username', 'password', 'api_key', etc.

        TODO: Implement authentication with CAMPUS API
        """
        self.credentials = credentials or {}
        self.authenticated = False

    def validate_credentials(self) -> bool:
        """
        Validate CAMPUS credentials.

        Returns:
            True if credentials are valid, False otherwise

        TODO: Implement actual credential validation against CAMPUS API
        """
        # Placeholder implementation
        if not self.credentials:
            return False

        # TODO: Make actual API call to validate credentials
        # For now, just check if credentials dict has required fields
        required_fields = ['username', 'password']  # Update based on actual CAMPUS API
        return all(field in self.credentials for field in required_fields)

    def fetch_event_details(self, event_link: str) -> Optional[Dict]:
        """
        Fetch event details from CAMPUS platform.

        Args:
            event_link: URL to the event on CAMPUS platform

        Returns:
            Dict with event details (title, dates, location, etc.) or None

        TODO: Implement actual API call to fetch event details
        """
        # Placeholder implementation
        # TODO: Parse event_link to extract event ID
        # TODO: Make API call to CAMPUS to get event details
        # TODO: Return structured data

        return None

    def fetch_participants(self, event_link: str) -> List[Participant]:
        """
        Fetch nominated participants list from CAMPUS event.

        Args:
            event_link: URL to the event on CAMPUS platform

        Returns:
            List of Participant objects

        TODO: Implement actual API call to fetch participants
        """
        # Placeholder implementation
        # TODO: Parse event_link to extract event ID
        # TODO: Make API call to CAMPUS to get participants
        # TODO: Parse response and create Participant objects

        return []

    def download_event_images(self, event_link: str, output_dir: str) -> List[str]:
        """
        Download new images from CAMPUS event.

        Args:
            event_link: URL to the event on CAMPUS platform
            output_dir: Directory to save downloaded images

        Returns:
            List of paths to downloaded images

        TODO: Implement actual image download from CAMPUS
        """
        # Placeholder implementation
        # TODO: Parse event_link to extract event ID
        # TODO: Make API call to get list of images for event
        # TODO: Download each image to output_dir
        # TODO: Return list of downloaded file paths

        return []

    def upload_results(self, event_link: str, results: Dict) -> bool:
        """
        Upload analysis results back to CAMPUS platform.

        Args:
            event_link: URL to the event on CAMPUS platform
            results: Analysis results to upload

        Returns:
            True if upload successful, False otherwise

        TODO: Implement result upload to CAMPUS
        """
        # Placeholder implementation
        # TODO: Format results according to CAMPUS API requirements
        # TODO: Make API call to upload results
        # TODO: Handle response and errors

        return False
