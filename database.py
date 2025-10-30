"""
Database module for storing face recognition data persistently.

Stores:
- Images (paths and metadata)
- Profiles (person identities)
- Detections (face locations in images)
- Embeddings (face recognition vectors)
"""

import sqlite3
import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class FaceDatabase:
    def __init__(self, db_path: str = "face_recognition.db"):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()

    def create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Images table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                mtime REAL NOT NULL,
                processed BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Profiles table (person identities)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY,
                label TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Detections table (face locations)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                profile_id INTEGER,
                detection_index INTEGER NOT NULL,
                box_x INTEGER NOT NULL,
                box_y INTEGER NOT NULL,
                box_w INTEGER NOT NULL,
                box_h INTEGER NOT NULL,
                score REAL,
                embedding BLOB,
                face_image BLOB,
                emotion TEXT,
                gaze_direction TEXT,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE SET NULL
            )
        ''')

        # Create indexes for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_detections_image
            ON detections(image_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_detections_profile
            ON detections(profile_id)
        ''')

        # Migration: Add emotion and gaze_direction columns if they don't exist
        cursor.execute("PRAGMA table_info(detections)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'emotion' not in columns:
            cursor.execute('ALTER TABLE detections ADD COLUMN emotion TEXT')

        if 'gaze_direction' not in columns:
            cursor.execute('ALTER TABLE detections ADD COLUMN gaze_direction TEXT')

        self.conn.commit()

    def add_image(self, path: str, mtime: float, processed: bool = False) -> int:
        """Add or update an image record. Returns image_id."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO images (path, mtime, processed)
            VALUES (?, ?, ?)
        ''', (path, mtime, processed))
        self.conn.commit()
        return cursor.lastrowid

    def get_image_id(self, path: str) -> Optional[int]:
        """Get image ID by path."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT id FROM images WHERE path = ?', (path,))
        row = cursor.fetchone()
        return row['id'] if row else None

    def is_image_processed(self, path: str, mtime: float) -> bool:
        """Check if image is already processed and up-to-date."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT processed, mtime FROM images WHERE path = ?
        ''', (path,))
        row = cursor.fetchone()
        if not row:
            return False
        return row['processed'] and row['mtime'] == mtime

    def get_all_images(self) -> List[Dict]:
        """Get all stored images."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM images ORDER BY created_at DESC')
        return [dict(row) for row in cursor.fetchall()]

    def save_profile(self, profile_id: int, label: str):
        """Save or update a profile."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO profiles (id, label, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (profile_id, label))
        self.conn.commit()

    def get_all_profiles(self) -> List[Dict]:
        """Get all profiles."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM profiles ORDER BY id')
        return [dict(row) for row in cursor.fetchall()]

    def delete_profile(self, profile_id: int):
        """Delete a profile."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM profiles WHERE id = ?', (profile_id,))
        self.conn.commit()

    def save_detection(self, image_id: int, profile_id: int, detection_index: int,
                       box: Tuple[int, int, int, int], score: float,
                       embedding: np.ndarray, face_image: Optional[np.ndarray] = None,
                       emotion: Optional[str] = None, gaze_direction: Optional[str] = None):
        """Save a face detection with embedding."""
        cursor = self.conn.cursor()

        # Serialize embedding
        embedding_blob = pickle.dumps(embedding)

        # Serialize face image if provided
        face_blob = pickle.dumps(face_image) if face_image is not None else None

        box_x, box_y, box_w, box_h = box

        cursor.execute('''
            INSERT INTO detections
            (image_id, profile_id, detection_index, box_x, box_y, box_w, box_h,
             score, embedding, face_image, emotion, gaze_direction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (image_id, profile_id, detection_index, box_x, box_y, box_w, box_h,
              score, embedding_blob, face_blob, emotion, gaze_direction))
        self.conn.commit()

    def update_detection_analysis(self, detection_id: int, emotion: Optional[str] = None,
                                  gaze_direction: Optional[str] = None):
        """Update emotion and gaze direction for a detection."""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE detections
            SET emotion = ?, gaze_direction = ?
            WHERE id = ?
        ''', (emotion, gaze_direction, detection_id))
        self.conn.commit()

    def get_detections_for_image(self, image_id: int, load_embeddings: bool = False, load_face_images: bool = False) -> List[Dict]:
        """
        Get all detections for an image with optional lazy loading.

        Args:
            image_id: ID of the image
            load_embeddings: If True, deserialize embeddings (default False for performance)
            load_face_images: If True, deserialize face images (default False for performance)

        Returns:
            List of detection dictionaries. Embeddings and face_images will be None unless explicitly loaded.
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM detections WHERE image_id = ?
            ORDER BY detection_index
        ''', (image_id,))

        detections = []
        for row in cursor.fetchall():
            det = dict(row)
            # Keep blob data for lazy loading, but don't deserialize by default
            if load_embeddings and det['embedding']:
                det['embedding'] = pickle.loads(det['embedding'])
            elif not load_embeddings:
                # Keep the blob reference for later lazy loading
                det['embedding_blob'] = det['embedding']
                det['embedding'] = None

            if load_face_images and det['face_image']:
                det['face_image'] = pickle.loads(det['face_image'])
            else:
                # Don't load face images - we'll crop from main image instead
                det['face_image'] = None

            detections.append(det)
        return detections

    def get_embedding(self, detection_id: int) -> Optional[np.ndarray]:
        """
        Get embedding for a specific detection (lazy loading).

        Args:
            detection_id: ID of the detection

        Returns:
            Deserialized embedding numpy array or None
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT embedding FROM detections WHERE id = ?', (detection_id,))
        row = cursor.fetchone()
        if row and row['embedding']:
            return pickle.loads(row['embedding'])
        return None

    def get_detections_for_profile(self, profile_id: int) -> List[Dict]:
        """Get all detections for a profile (all occurrences of a person)."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT d.*, i.path as image_path
            FROM detections d
            JOIN images i ON d.image_id = i.id
            WHERE d.profile_id = ?
            ORDER BY d.id
        ''', (profile_id,))

        detections = []
        for row in cursor.fetchall():
            det = dict(row)
            # Deserialize embedding
            if det['embedding']:
                det['embedding'] = pickle.loads(det['embedding'])
            # Deserialize face image
            if det['face_image']:
                det['face_image'] = pickle.loads(det['face_image'])
            detections.append(det)
        return detections

    def delete_detections_for_image(self, image_id: int):
        """Delete all detections for an image."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM detections WHERE image_id = ?', (image_id,))
        self.conn.commit()

    def update_detection_profile(self, detection_id: int, profile_id: int):
        """Update the profile assignment for a detection."""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE detections SET profile_id = ? WHERE id = ?
        ''', (profile_id, detection_id))
        self.conn.commit()

    def mark_image_processed(self, image_id: int):
        """Mark an image as processed."""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE images SET processed = 1 WHERE id = ?
        ''', (image_id,))
        self.conn.commit()

    def delete_image(self, path: str):
        """Delete an image and all its detections."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM images WHERE path = ?', (path,))
        self.conn.commit()

    def get_next_profile_id(self) -> int:
        """Get the next available profile ID."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT MAX(id) as max_id FROM profiles')
        row = cursor.fetchone()
        max_id = row['max_id'] if row['max_id'] is not None else 0
        return max_id + 1

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
