"""
Quality Review Module for Face Recognition

Provides tools to:
- Filter low-confidence detections
- Find and remove outliers within profiles
- Identify profiles that should be merged
- Validate embedding quality
- Clean up bad recognitions
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from recognition import cosine_distance
from database import FaceDatabase


@dataclass
class QualityIssue:
    """Represents a quality issue found during review"""
    issue_type: str  # 'low_confidence', 'outlier', 'duplicate_profile', 'invalid_embedding'
    severity: str  # 'low', 'medium', 'high'
    description: str
    affected_ids: List[int]  # detection_ids or profile_ids
    suggested_action: str
    metadata: dict = None


class QualityReviewer:
    """Analyzes face recognition results and identifies quality issues"""

    def __init__(self, database: FaceDatabase):
        self.db = database
        self.issues = []

    def run_all_checks(self,
                      min_detection_score=0.95,
                      outlier_distance=0.6,
                      merge_threshold=0.3) -> List[QualityIssue]:
        """Run all quality checks and return issues found"""
        self.issues = []

        self.check_low_confidence_detections(min_detection_score)
        self.check_profile_outliers(outlier_distance)
        self.check_duplicate_profiles(merge_threshold)
        self.check_invalid_embeddings()
        self.check_small_profiles()

        return self.issues

    def check_low_confidence_detections(self, min_score=0.95):
        """Find detections with low confidence scores"""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT id, image_id, score
            FROM detections
            WHERE score < ?
            ORDER BY score ASC
        """, (min_score,))

        low_conf = cursor.fetchall()
        if low_conf:
            det_ids = [row[0] for row in low_conf]
            avg_score = sum(row[2] for row in low_conf) / len(low_conf)

            self.issues.append(QualityIssue(
                issue_type='low_confidence',
                severity='medium' if min_score - avg_score > 0.1 else 'low',
                description=f"Found {len(low_conf)} detections below {min_score} confidence",
                affected_ids=det_ids,
                suggested_action='Delete these detections',
                metadata={'avg_score': avg_score, 'min_score': min(row[2] for row in low_conf)}
            ))

    def check_profile_outliers(self, max_distance=0.6):
        """Find faces within profiles that don't match well"""
        cursor = self.db.conn.cursor()

        # Get all profiles with embeddings
        cursor.execute("""
            SELECT p.id, p.label, d.id, d.embedding
            FROM profiles p
            JOIN detections d ON d.profile_id = p.id
            WHERE d.embedding IS NOT NULL
        """)

        # Group by profile
        profiles = {}
        for row in cursor.fetchall():
            prof_id, label, det_id, emb_blob = row
            if prof_id not in profiles:
                profiles[prof_id] = {'label': label, 'detections': []}

            # Unpickle embedding
            import pickle
            embedding = pickle.loads(emb_blob)
            profiles[prof_id]['detections'].append((det_id, embedding))

        # Check each profile for outliers
        for prof_id, data in profiles.items():
            if len(data['detections']) < 3:
                continue  # Need at least 3 to identify outliers

            # Calculate average embedding
            embeddings = [emb for _, emb in data['detections']]
            avg_emb = np.mean(embeddings, axis=0)

            # Find outliers
            outliers = []
            for det_id, emb in data['detections']:
                dist = cosine_distance(emb, avg_emb)
                if dist > max_distance:
                    outliers.append((det_id, dist))

            if outliers:
                self.issues.append(QualityIssue(
                    issue_type='outlier',
                    severity='high' if len(outliers) > len(data['detections']) * 0.3 else 'medium',
                    description=f"Profile '{data['label']}' has {len(outliers)} outlier face(s)",
                    affected_ids=[det_id for det_id, _ in outliers],
                    suggested_action='Remove outliers from profile or reassign to new profile',
                    metadata={
                        'profile_id': prof_id,
                        'profile_label': data['label'],
                        'distances': {det_id: dist for det_id, dist in outliers}
                    }
                ))

    def check_duplicate_profiles(self, merge_threshold=0.3):
        """Find profiles that are likely the same person"""
        cursor = self.db.conn.cursor()

        # Get average embedding for each profile
        cursor.execute("""
            SELECT p.id, p.label, d.embedding
            FROM profiles p
            JOIN detections d ON d.profile_id = p.id
            WHERE d.embedding IS NOT NULL
        """)

        # Calculate average embeddings
        import pickle
        profile_embeddings = {}

        for row in cursor.fetchall():
            prof_id, label, emb_blob = row
            embedding = pickle.loads(emb_blob)

            if prof_id not in profile_embeddings:
                profile_embeddings[prof_id] = {
                    'label': label,
                    'embeddings': []
                }
            profile_embeddings[prof_id]['embeddings'].append(embedding)

        # Compute average for each profile
        profiles = []
        for prof_id, data in profile_embeddings.items():
            avg_emb = np.mean(data['embeddings'], axis=0)
            profiles.append((prof_id, data['label'], avg_emb))

        # Find similar pairs
        duplicates = []
        for i, (id1, label1, emb1) in enumerate(profiles):
            for id2, label2, emb2 in profiles[i+1:]:
                dist = cosine_distance(emb1, emb2)
                if dist < merge_threshold:
                    duplicates.append((id1, id2, label1, label2, dist))

        if duplicates:
            for id1, id2, label1, label2, dist in duplicates:
                self.issues.append(QualityIssue(
                    issue_type='duplicate_profile',
                    severity='high' if dist < 0.15 else 'medium',
                    description=f"Profiles '{label1}' and '{label2}' are very similar (distance: {dist:.3f})",
                    affected_ids=[id1, id2],
                    suggested_action='Merge these profiles into one',
                    metadata={
                        'profile1_id': id1,
                        'profile2_id': id2,
                        'profile1_label': label1,
                        'profile2_label': label2,
                        'distance': dist
                    }
                ))

    def check_invalid_embeddings(self):
        """Find detections with missing or corrupted embeddings"""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT id, image_id, embedding
            FROM detections
        """)

        invalid = []
        import pickle

        for row in cursor.fetchall():
            det_id, img_id, emb_blob = row

            if emb_blob is None:
                invalid.append(det_id)
                continue

            try:
                emb = pickle.loads(emb_blob)
                if emb is None or len(emb) == 0 or not isinstance(emb, np.ndarray):
                    invalid.append(det_id)
            except:
                invalid.append(det_id)

        if invalid:
            self.issues.append(QualityIssue(
                issue_type='invalid_embedding',
                severity='high',
                description=f"Found {len(invalid)} detections with invalid embeddings",
                affected_ids=invalid,
                suggested_action='Delete these detections or reprocess images',
                metadata={'count': len(invalid)}
            ))

    def check_small_profiles(self, min_occurrences=2):
        """Find profiles with very few face occurrences"""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT p.id, p.label, COUNT(d.id) as count
            FROM profiles p
            LEFT JOIN detections d ON d.profile_id = p.id
            GROUP BY p.id
            HAVING count < ?
        """, (min_occurrences,))

        small_profiles = cursor.fetchall()
        if small_profiles:
            prof_ids = [row[0] for row in small_profiles]

            self.issues.append(QualityIssue(
                issue_type='small_profile',
                severity='low',
                description=f"Found {len(small_profiles)} profiles with fewer than {min_occurrences} faces",
                affected_ids=prof_ids,
                suggested_action='Reallocate these small profiles to the closest bigger profiles',
                metadata={
                    'profiles': [(row[0], row[1], row[2]) for row in small_profiles]
                }
            ))


class QualityFixer:
    """Applies fixes to quality issues"""

    def __init__(self, database: FaceDatabase):
        self.db = database

    def delete_detections(self, detection_ids: List[int]) -> int:
        """Delete detections by IDs"""
        if not detection_ids:
            return 0

        cursor = self.db.conn.cursor()
        placeholders = ','.join('?' * len(detection_ids))
        cursor.execute(f"""
            DELETE FROM detections
            WHERE id IN ({placeholders})
        """, detection_ids)
        self.db.conn.commit()

        return cursor.rowcount

    def remove_detections_from_profile(self, detection_ids: List[int]) -> int:
        """Unassign detections from their profiles"""
        if not detection_ids:
            return 0

        cursor = self.db.conn.cursor()
        placeholders = ','.join('?' * len(detection_ids))
        cursor.execute(f"""
            UPDATE detections
            SET profile_id = NULL
            WHERE id IN ({placeholders})
        """, detection_ids)
        self.db.conn.commit()

        return cursor.rowcount

    def merge_profiles(self, source_profile_id: int, target_profile_id: int):
        """Merge source profile into target profile"""
        cursor = self.db.conn.cursor()

        # Move all detections from source to target
        cursor.execute("""
            UPDATE detections
            SET profile_id = ?
            WHERE profile_id = ?
        """, (target_profile_id, source_profile_id))

        # Delete source profile
        cursor.execute("""
            DELETE FROM profiles
            WHERE id = ?
        """, (source_profile_id,))

        self.db.conn.commit()

    def delete_profile(self, profile_id: int):
        """Delete a profile and unassign its detections"""
        cursor = self.db.conn.cursor()

        # Unassign detections
        cursor.execute("""
            UPDATE detections
            SET profile_id = NULL
            WHERE profile_id = ?
        """, (profile_id,))

        # Delete profile
        cursor.execute("""
            DELETE FROM profiles
            WHERE id = ?
        """, (profile_id,))

        self.db.conn.commit()

    def reassign_detections(self, detection_ids: List[int], new_profile_id: int):
        """Reassign detections to a different profile"""
        if not detection_ids:
            return 0

        cursor = self.db.conn.cursor()
        placeholders = ','.join('?' * len(detection_ids))
        cursor.execute(f"""
            UPDATE detections
            SET profile_id = ?
            WHERE id IN ({placeholders})
        """, [new_profile_id] + detection_ids)
        self.db.conn.commit()

        return cursor.rowcount

    def create_new_profile_from_detections(self, detection_ids: List[int], label: str) -> int:
        """Create new profile and assign detections to it"""
        if not detection_ids:
            return None

        cursor = self.db.conn.cursor()

        # Create new profile
        cursor.execute("""
            INSERT INTO profiles (label, created_at, updated_at)
            VALUES (?, datetime('now'), datetime('now'))
        """, (label,))

        new_profile_id = cursor.lastrowid

        # Assign detections
        self.reassign_detections(detection_ids, new_profile_id)

        return new_profile_id

    def reallocate_small_profiles(self, small_profile_ids: List[int], min_occurrences: int = 2) -> Dict:
        """
        Reallocate small profiles to bigger profiles by finding closest match.

        Args:
            small_profile_ids: List of profile IDs to reallocate
            min_occurrences: Threshold for what constitutes a "big" profile

        Returns:
            Dict with 'reallocated' count and 'details' list
        """
        import pickle

        if not small_profile_ids:
            return {'reallocated': 0, 'details': []}

        cursor = self.db.conn.cursor()

        # Get all profiles with their detection counts and embeddings
        cursor.execute("""
            SELECT p.id, p.label, COUNT(d.id) as count
            FROM profiles p
            LEFT JOIN detections d ON d.profile_id = p.id
            GROUP BY p.id
        """)

        all_profiles = cursor.fetchall()

        # Separate small and big profiles
        small_profiles = [p for p in all_profiles if p[0] in small_profile_ids]
        big_profiles = [p for p in all_profiles if p[2] >= min_occurrences and p[0] not in small_profile_ids]

        if not big_profiles:
            # No big profiles to merge into - cannot reallocate
            return {'reallocated': 0, 'details': [], 'error': 'No big profiles available for reallocation'}

        # Compute average embeddings for all profiles
        def get_avg_embedding(profile_id):
            cursor.execute("""
                SELECT embedding
                FROM detections
                WHERE profile_id = ? AND embedding IS NOT NULL
            """, (profile_id,))

            embeddings = []
            for row in cursor.fetchall():
                try:
                    emb = pickle.loads(row[0])
                    if emb is not None and len(emb) > 0:
                        embeddings.append(emb)
                except:
                    continue

            if not embeddings:
                return None

            return np.mean(embeddings, axis=0)

        # Precompute embeddings for big profiles
        big_profile_embeddings = {}
        for prof_id, label, count in big_profiles:
            avg_emb = get_avg_embedding(prof_id)
            if avg_emb is not None:
                big_profile_embeddings[prof_id] = {'label': label, 'embedding': avg_emb, 'count': count}

        if not big_profile_embeddings:
            return {'reallocated': 0, 'details': [], 'error': 'No valid embeddings in big profiles'}

        # Process each small profile
        reallocated_count = 0
        details = []

        for small_prof_id, small_label, small_count in small_profiles:
            small_avg_emb = get_avg_embedding(small_prof_id)

            if small_avg_emb is None:
                details.append({
                    'small_profile': small_label,
                    'status': 'skipped',
                    'reason': 'No valid embeddings'
                })
                continue

            # Find closest big profile
            best_match_id = None
            best_distance = float('inf')

            for big_prof_id, big_data in big_profile_embeddings.items():
                dist = cosine_distance(small_avg_emb, big_data['embedding'])
                if dist < best_distance:
                    best_distance = dist
                    best_match_id = big_prof_id

            if best_match_id is not None:
                # Merge small profile into big profile
                big_label = big_profile_embeddings[best_match_id]['label']

                try:
                    self.merge_profiles(small_prof_id, best_match_id)
                    reallocated_count += 1
                    details.append({
                        'small_profile': small_label,
                        'merged_into': big_label,
                        'distance': float(best_distance),
                        'status': 'success'
                    })

                    # Update the count for the big profile
                    big_profile_embeddings[best_match_id]['count'] += small_count

                except Exception as e:
                    details.append({
                        'small_profile': small_label,
                        'status': 'failed',
                        'error': str(e)
                    })

        return {
            'reallocated': reallocated_count,
            'details': details
        }


def generate_quality_report(issues: List[QualityIssue]) -> str:
    """Generate human-readable quality report"""
    if not issues:
        return "✓ No quality issues found!"

    report = []
    report.append("=" * 60)
    report.append("FACE RECOGNITION QUALITY REPORT")
    report.append("=" * 60)
    report.append("")

    # Group by severity
    high = [i for i in issues if i.severity == 'high']
    medium = [i for i in issues if i.severity == 'medium']
    low = [i for i in issues if i.severity == 'low']

    report.append(f"Summary: {len(issues)} issue(s) found")
    report.append(f"  - High severity: {len(high)}")
    report.append(f"  - Medium severity: {len(medium)}")
    report.append(f"  - Low severity: {len(low)}")
    report.append("")

    # Detail each issue
    for priority, issue_list in [("HIGH", high), ("MEDIUM", medium), ("LOW", low)]:
        if not issue_list:
            continue

        report.append(f"\n{priority} PRIORITY ISSUES:")
        report.append("-" * 60)

        for issue in issue_list:
            report.append(f"\n• {issue.description}")
            report.append(f"  Type: {issue.issue_type}")
            report.append(f"  Affected items: {len(issue.affected_ids)}")
            report.append(f"  Suggested action: {issue.suggested_action}")

            if issue.metadata:
                report.append(f"  Details: {issue.metadata}")

    report.append("\n" + "=" * 60)

    return "\n".join(report)
