import colorsys
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from retinaface import RetinaFace
from deepface import DeepFace


def placeholder_face(side: int = 320):
    canvas = np.full((side, side, 3), 240, dtype=np.uint8)
    cv2.putText(canvas, "No faces", (60, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 80), 2, cv2.LINE_AA)
    return canvas


def normalize_face(crop, side: int = 320):
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return placeholder_face(side)
    scale = side / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
    canvas = np.full((side, side, 3), 240, dtype=np.uint8)
    y0 = (side - nh) // 2
    x0 = (side - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def detect_faces(img, threshold: float):
    crops = []
    detections = []
    if img is None or img.size == 0:
        crops.append(placeholder_face())
        return [normalize_face(c) for c in crops], detections

    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        temp_path = temp_file.name
        with open(temp_path, 'wb') as f:
            f.write(buffer)
        try:
            results = RetinaFace.detect_faces(img_path=temp_path, threshold=threshold)
        except Exception as exc:  # noqa: F841
            print(f"Detection error: {exc}")
            results = {}
        finally:
            os.unlink(temp_path)

    if isinstance(results, dict) and len(results) > 0:
        ih, iw = img.shape[:2]
        for data in results.values():
            if not isinstance(data, dict):
                continue
            area = data.get("facial_area")
            if not area or len(area) != 4:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in area]
            x0 = max(x0, 0)
            y0 = max(y0, 0)
            x1 = min(x1, iw)
            y1 = min(y1, ih)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            crop = img[y0:y1, x0:x1].copy()
            if crop.size == 0:
                continue
            score = data.get("score", 0.0)
            bbox_w = x1 - x0
            bbox_h = y1 - y0
            crops.append(crop)
            detections.append({
                "box": (x0, y0, bbox_w, bbox_h),
                "center": (x0 + bbox_w / 2.0, y0 + bbox_h / 2.0),
                "size": max(bbox_w, bbox_h),
                "score": float(score),
                "crop": crop,
            })

    if not crops:
        crops.append(placeholder_face())
    return [normalize_face(c) for c in crops], detections


def cluster_detections(detections: List[Dict]):
    n = len(detections)
    if n <= 1:
        return [0] * n if n == 1 else []

    sizes = [d["size"] for d in detections]
    avg_size = sum(sizes) / n
    threshold = max(avg_size * 1.6, 80.0)

    parent = list(range(n))

    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    centers = [d["center"] for d in detections]
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = centers[i], centers[j]
            dist = ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5
            if dist <= threshold:
                union(i, j)

    roots = [find(i) for i in range(n)]
    counts = {}
    for r in roots:
        counts[r] = counts.get(r, 0) + 1

    for i in range(n):
        if counts[roots[i]] > 1:
            continue
        ci = centers[i]
        min_dist = float('inf')
        nearest_root = None
        for j in range(n):
            if i == j or roots[j] == roots[i]:
                continue
            cj = centers[j]
            dist = ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_root = roots[j]
        if nearest_root is not None and min_dist <= threshold * 1.5:
            union(i, nearest_root)

    final_roots = [find(i) for i in range(n)]
    cluster_map = {r: idx for idx, r in enumerate(set(final_roots))}
    return [cluster_map.get(r, -1) for r in final_roots]


def cluster_color(cluster_id: int) -> Tuple[int, int, int]:
    if cluster_id < 0:
        return (200, 200, 200)
    hue = (cluster_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
    return (int(255 * r), int(255 * g), int(255 * b))


def draw_face_boxes(img, detections, clusters, scale: float = 1.0, labels: Optional[List[str]] = None):
    """
    Draw bounding boxes around detected faces with optional labels.

    Args:
        img: Input image
        detections: List of face detections
        clusters: Cluster IDs for coloring
        scale: Scaling factor for coordinates
        labels: Optional list of labels to display below boxes

    Returns:
        Annotated image with bounding boxes and labels
    """
    if img is None:
        return img
    if not detections:
        return img
    annotated = img.copy()
    thickness = max(1, min(2, int(round(2 / scale))))

    cluster_nodes: Dict[int, List[Dict]] = {}
    for idx, det in enumerate(detections):
        x, y, w, h = det["box"]
        cluster_id = clusters[idx] if clusters else 0
        color_rgb = cluster_color(cluster_id)
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        x0 = int(round(x * scale))
        y0 = int(round(y * scale))
        x1 = int(round((x + w) * scale))
        y1 = int(round((y + h) * scale))
        cv2.rectangle(annotated, (x0, y0), (x1, y1), color_bgr, thickness)

        # Draw label below bounding box
        if labels and idx < len(labels):
            label_text = labels[idx]
            if label_text:
                # Calculate text size for background - much larger for visibility
                font = cv2.FONT_HERSHEY_SIMPLEX
                # Much larger font scale (1.2-2.0 range) for better visibility on large images
                font_scale = max(1.2, 1.5 * scale if scale > 0.5 else 1.2)
                # Thicker text (2-4 pixels) for better readability
                font_thickness = max(2, int(2 * scale))
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, font, font_scale, font_thickness
                )

                # Position text below the box with more padding
                text_x = x0
                text_y = y1 + text_height + 10

                # Draw background rectangle for text with padding (black background)
                padding = 6
                bg_x1 = text_x - padding // 2
                bg_y1 = y1 + 4
                bg_x2 = text_x + text_width + padding
                bg_y2 = text_y + baseline + padding

                # Black background for better contrast
                cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)

                # Draw text with padding (white text on black)
                cv2.putText(
                    annotated,
                    label_text,
                    (text_x + padding // 4, text_y + padding // 4),
                    font,
                    font_scale,
                    (255, 255, 255),  # White text
                    font_thickness,
                    cv2.LINE_AA
                )

        cx = int(round((x + w / 2.0) * scale))
        cy = int(round((y + h / 2.0) * scale))
        cluster_nodes.setdefault(cluster_id, []).append({"index": idx, "center": (cx, cy), "color": color_bgr})

    drawn_pairs = set()
    for nodes in cluster_nodes.values():
        if len(nodes) < 2:
            continue
        centers = {n["index"]: n["center"] for n in nodes}
        for node in nodes:
            node_idx = node["index"]
            cx, cy = node["center"]
            color = node["color"]
            nearest_dist = float('inf')
            nearest_idx = None
            for other in nodes:
                if other["index"] == node_idx:
                    continue
                ox, oy = other["center"]
                dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx = other["index"]
            if nearest_idx is not None:
                pair = tuple(sorted((node_idx, nearest_idx)))
                if pair not in drawn_pairs:
                    drawn_pairs.add(pair)
                    ocx, ocy = centers[nearest_idx]
                    cv2.line(annotated, (cx, cy), (ocx, ocy), color, thickness)

    return annotated


def compute_face_embedding(face_img: Optional[np.ndarray], model_name: str = "Facenet512") -> np.ndarray:
    """
    Compute face embedding using DeepFace with specified model.

    Args:
        face_img: Face image crop (BGR format)
        model_name: Model to use. Options:
            - "Facenet512" (default, 512-dim, best accuracy)
            - "Facenet" (128-dim, faster)
            - "VGG-Face" (2622-dim, robust)
            - "ArcFace" (512-dim, excellent for matching)
            - "SFace" (128-dim, fastest)

    Returns:
        Normalized embedding vector
    """
    if face_img is None or face_img.size == 0:
        # Return zero embedding with appropriate dimension
        dim_map = {
            "Facenet512": 512,
            "Facenet": 128,
            "VGG-Face": 2622,
            "ArcFace": 512,
            "SFace": 128,
        }
        dim = dim_map.get(model_name, 512)
        return np.zeros(dim, dtype=np.float32)

    try:
        # DeepFace expects RGB format
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Generate embedding using DeepFace
        # enforce_detection=False since we already have a detected face
        embedding_objs = DeepFace.represent(
            img_path=face_rgb,
            model_name=model_name,
            enforce_detection=False,
            detector_backend="skip",  # Skip detection since we already have the face
            align=True,  # Align face for better accuracy
        )

        # DeepFace.represent returns a list of dicts, get the first one
        if embedding_objs and len(embedding_objs) > 0:
            embedding = np.array(embedding_objs[0]["embedding"], dtype=np.float32)
            # Normalize the embedding for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 1e-12:
                embedding /= norm
            return embedding
        else:
            # Fallback to zero embedding
            dim_map = {
                "Facenet512": 512,
                "Facenet": 128,
                "VGG-Face": 2622,
                "ArcFace": 512,
                "SFace": 128,
            }
            dim = dim_map.get(model_name, 512)
            return np.zeros(dim, dtype=np.float32)

    except Exception as e:
        print(f"Embedding computation error: {e}")
        # Fallback to zero embedding
        dim_map = {
            "Facenet512": 512,
            "Facenet": 128,
            "VGG-Face": 2622,
            "ArcFace": 512,
            "SFace": 128,
        }
        dim = dim_map.get(model_name, 512)
        return np.zeros(dim, dtype=np.float32)


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.

    Returns:
        Similarity score (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)
    """
    if emb1.size == 0 or emb2.size == 0:
        return 0.0

    # Normalize if not already normalized
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0

    emb1_norm = emb1 / norm1
    emb2_norm = emb2 / norm2

    return float(np.dot(emb1_norm, emb2_norm))


def cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine distance (1 - cosine_similarity).

    Returns:
        Distance score (0.0 = identical, 2.0 = opposite)
    """
    return 1.0 - cosine_similarity(emb1, emb2)


@dataclass
class FaceOccurrence:
    image_path: str
    detection_index: int
    box: Tuple[int, int, int, int]
    embedding: np.ndarray = field(repr=False)
    face_image: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    # Optional fields for lazy loading
    detection_id: Optional[int] = field(default=None, repr=False)
    embedding_blob: Optional[bytes] = field(default=None, repr=False)
    # Emotion and gaze analysis
    emotion: Optional[str] = field(default=None)
    gaze_direction: Optional[str] = field(default=None)

    def get_embedding(self):
        """
        Get embedding, loading lazily from blob if needed.

        Returns:
            Embedding numpy array
        """
        # If embedding is already loaded, return it
        if self.embedding is not None and (isinstance(self.embedding, np.ndarray) and self.embedding.size > 0):
            return self.embedding

        # Try to load from blob if available
        if self.embedding_blob is not None:
            import pickle
            self.embedding = pickle.loads(self.embedding_blob)
            # Clear blob to free memory after deserialization
            self.embedding_blob = None
            return self.embedding

        # No embedding available
        return None


@dataclass
class FaceProfile:
    profile_id: int
    label: str
    representative_face: Optional[np.ndarray] = field(default=None, repr=False)
    representative_area: int = 0
    embedding_sum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32), repr=False)
    occurrence_count: int = 0
    occurrences: List[FaceOccurrence] = field(default_factory=list, repr=False)

    def add_occurrence(self, occurrence: FaceOccurrence):
        # Check if this occurrence already exists (same image path and detection index)
        for existing_occ in self.occurrences:
            if (existing_occ.image_path == occurrence.image_path and
                existing_occ.detection_index == occurrence.detection_index):
                # Already exists, don't add duplicate
                return

        self.occurrences.append(occurrence)
        # Get embedding (lazy load if needed)
        emb = occurrence.get_embedding()
        if emb is not None:
            emb = emb.astype(np.float32)
            if self.embedding_sum.size == 0:
                self.embedding_sum = emb.copy()
            else:
                self.embedding_sum += emb
        self.occurrence_count += 1
        # Always use the last (most recent) occurrence as representative face
        if occurrence.face_image is not None and occurrence.face_image.size > 0:
            x, y, w, h = occurrence.box
            area = w * h
            self.representative_face = occurrence.face_image.copy()
            self.representative_area = area

    def average_embedding(self) -> np.ndarray:
        if self.occurrence_count == 0:
            return np.zeros_like(self.embedding_sum)
        return self.embedding_sum / max(1, self.occurrence_count)

    def remove_occurrences_for_image(self, image_path: str) -> bool:
        if not self.occurrences:
            return False
        remaining = []
        removed = []
        for occ in self.occurrences:
            if occ.image_path == image_path:
                removed.append(occ)
            else:
                remaining.append(occ)
        if not removed:
            return False
        self.occurrences = remaining
        if not remaining:
            self.embedding_sum = np.zeros(0, dtype=np.float32)
            self.occurrence_count = 0
            self.representative_face = None
            self.representative_area = 0
            return True
        self.embedding_sum = np.sum([occ.get_embedding() for occ in remaining if occ.get_embedding() is not None], axis=0).astype(np.float32)
        self.occurrence_count = len(remaining)
        # Use the last (most recent) occurrence as representative face
        if remaining:
            last_occ = remaining[-1]
            if last_occ.face_image is not None and last_occ.face_image.size > 0:
                x, y, w, h = last_occ.box
                self.representative_face = last_occ.face_image.copy()
                self.representative_area = w * h
            else:
                self.representative_face = None
                self.representative_area = 0
        return True


class FaceProfileManager:
    def __init__(
        self,
        distance_threshold: float = 0.4,
        model_name: str = "Facenet512",
        use_cosine: bool = True,
        max_people: Optional[int] = None
    ):
        """
        Initialize FaceProfileManager with improved matching.

        Args:
            distance_threshold: Maximum distance for profile matching
                - For cosine distance: 0.3-0.5 (default 0.4)
                - For L2 distance: depends on embedding dimension
            model_name: DeepFace model to use for embeddings
            use_cosine: Use cosine distance (True) or L2 distance (False)
            max_people: Maximum number of people/profiles allowed (None = unlimited)
        """
        self.distance_threshold = distance_threshold
        self.model_name = model_name
        self.use_cosine = use_cosine
        self.max_people = max_people
        self._profiles: Dict[int, FaceProfile] = {}
        self._next_profile_id = 1
        # Cache for normalized embeddings to speed up matching
        self._embedding_cache: Dict[int, np.ndarray] = {}

    def profiles(self) -> List[FaceProfile]:
        return list(self._profiles.values())

    def get_profile(self, profile_id: int) -> Optional[FaceProfile]:
        return self._profiles.get(profile_id)

    def _find_best_match(self, embedding: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Find the best matching profile for an embedding using optimized distance calculation.

        Returns:
            Tuple of (profile_id, distance) or (None, inf) if no match
        """
        if embedding is None or embedding.size == 0:
            return None, float("inf")

        best_id = None
        best_dist = float("inf")

        # Vectorize distance calculation for better performance
        if self._profiles:
            profile_ids = []
            embeddings = []

            for pid, profile in self._profiles.items():
                # Use cached normalized embedding if available
                if pid not in self._embedding_cache:
                    avg_emb = profile.average_embedding()
                    if avg_emb.size == 0:
                        continue
                    # Normalize and cache
                    norm = np.linalg.norm(avg_emb)
                    if norm > 1e-12:
                        self._embedding_cache[pid] = avg_emb / norm
                    else:
                        continue

                profile_ids.append(pid)
                embeddings.append(self._embedding_cache[pid])

            if embeddings:
                embeddings_matrix = np.array(embeddings)

                if self.use_cosine:
                    # Cosine distance: more robust for deep learning embeddings
                    # Normalize query embedding
                    norm = np.linalg.norm(embedding)
                    if norm > 1e-12:
                        embedding_norm = embedding / norm
                        # Compute cosine similarity with all profiles at once
                        similarities = np.dot(embeddings_matrix, embedding_norm)
                        distances = 1.0 - similarities
                        best_idx = np.argmin(distances)
                        best_dist = distances[best_idx]
                        best_id = profile_ids[best_idx]
                else:
                    # L2 distance: traditional Euclidean distance
                    distances = np.linalg.norm(embeddings_matrix - embedding, axis=1)
                    best_idx = np.argmin(distances)
                    best_dist = distances[best_idx]
                    best_id = profile_ids[best_idx]

        return best_id, float(best_dist)

    def assign_profile(self, occurrence: FaceOccurrence) -> FaceProfile:
        # Get embedding (lazy load if needed)
        emb = occurrence.get_embedding()
        match_id, dist = self._find_best_match(emb)

        # Check if we've reached max_people limit
        current_profile_count = len(self._profiles)
        max_reached = self.max_people is not None and current_profile_count >= self.max_people

        if match_id is None or dist > self.distance_threshold:
            if max_reached:
                # Max people reached - force assignment to best existing profile
                # even if distance is above threshold
                if match_id is not None:
                    # Assign to the best match we found
                    profile = self._profiles[match_id]
                elif self._profiles:
                    # No match at all, assign to first available profile
                    profile = next(iter(self._profiles.values()))
                else:
                    # Edge case: no profiles exist yet AND we're at limit
                    # Only create profile if limit allows at least one
                    if self.max_people is None or self.max_people > 0:
                        profile = self._create_profile()
                    else:
                        # max_people is 0, cannot create any profiles
                        raise ValueError("Cannot create profile: max_people limit is 0")
            else:
                # Under limit - create new profile
                profile = self._create_profile()
        else:
            # Good match found
            profile = self._profiles[match_id]

        # Invalidate cache for this profile since it's being updated
        if profile.profile_id in self._embedding_cache:
            del self._embedding_cache[profile.profile_id]

        profile.add_occurrence(occurrence)
        return profile

    def _create_profile(self) -> FaceProfile:
        pid = self._next_profile_id
        self._next_profile_id += 1
        profile = FaceProfile(profile_id=pid, label=f"Person {pid}")
        self._profiles[pid] = profile
        return profile

    def reset(self):
        self._profiles.clear()
        self._next_profile_id = 1
        self._embedding_cache.clear()

    def remove_image_occurrences(self, image_path: str) -> bool:
        changed = False
        empty_profiles = []
        for pid, profile in self._profiles.items():
            if profile.remove_occurrences_for_image(image_path):
                changed = True
                # Invalidate cache for this profile
                if pid in self._embedding_cache:
                    del self._embedding_cache[pid]
                if profile.occurrence_count == 0:
                    empty_profiles.append(pid)
        for pid in empty_profiles:
            del self._profiles[pid]
            if pid in self._embedding_cache:
                del self._embedding_cache[pid]
        return changed

    def rename_profile(self, profile_id: int, new_label: str) -> bool:
        """
        Rename a profile with a new label.

        Args:
            profile_id: ID of profile to rename
            new_label: New label/name for the profile

        Returns:
            True if renamed successfully, False otherwise
        """
        profile = self._profiles.get(profile_id)
        if not profile:
            return False
        profile.label = new_label
        return True

    def merge_profiles(self, source_id: int, target_id: int) -> bool:
        """
        Merge source profile into target profile.

        All occurrences from source profile are moved to target profile,
        and source profile is deleted.

        Args:
            source_id: Profile ID to merge from (will be deleted)
            target_id: Profile ID to merge into (will be kept)

        Returns:
            True if merged successfully, False otherwise
        """
        if source_id == target_id:
            return False

        source = self._profiles.get(source_id)
        target = self._profiles.get(target_id)

        if not source or not target:
            return False

        # Move all occurrences from source to target
        for occurrence in source.occurrences:
            target.add_occurrence(occurrence)

        # Delete source profile
        del self._profiles[source_id]

        # Invalidate caches for both profiles
        if source_id in self._embedding_cache:
            del self._embedding_cache[source_id]
        if target_id in self._embedding_cache:
            del self._embedding_cache[target_id]

        return True

    def find_profile_by_label(self, label: str) -> Optional[int]:
        """
        Find a profile ID by its label.

        Args:
            label: Label to search for

        Returns:
            Profile ID if found, None otherwise
        """
        for pid, profile in self._profiles.items():
            if profile.label == label:
                return pid
        return None

    def get_all_labels(self) -> List[str]:
        """
        Get all unique profile labels.

        Returns:
            List of all profile labels
        """
        return [profile.label for profile in self._profiles.values()]
