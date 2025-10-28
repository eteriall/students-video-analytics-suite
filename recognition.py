import colorsys
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from retinaface import RetinaFace


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


def draw_face_boxes(img, detections, clusters, scale: float = 1.0):
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


def compute_face_embedding(face_img: Optional[np.ndarray]) -> np.ndarray:
    if face_img is None or face_img.size == 0:
        return np.zeros(1024 + 96, dtype=np.float32)
    resized = cv2.resize(face_img, (112, 112), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    hists = []
    for channel in cv2.split(lab):
        hist = cv2.calcHist([channel], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hists.append(hist.astype(np.float32))
    gray_small = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray_small, (32, 32), interpolation=cv2.INTER_AREA)
    gray_norm = gray_small.astype(np.float32) / 255.0
    features = np.concatenate([gray_norm.flatten(), *hists]).astype(np.float32)
    norm = np.linalg.norm(features)
    if norm > 1e-12:
        features /= norm
    return features


@dataclass
class FaceOccurrence:
    image_path: str
    detection_index: int
    box: Tuple[int, int, int, int]
    embedding: np.ndarray = field(repr=False)
    face_image: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))


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
        self.occurrences.append(occurrence)
        emb = occurrence.embedding.astype(np.float32)
        if self.embedding_sum.size == 0:
            self.embedding_sum = emb.copy()
        else:
            self.embedding_sum += emb
        self.occurrence_count += 1
        x, y, w, h = occurrence.box
        area = w * h
        if area > self.representative_area and occurrence.face_image is not None and occurrence.face_image.size > 0:
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
        self.embedding_sum = np.sum([occ.embedding for occ in remaining], axis=0).astype(np.float32)
        self.occurrence_count = len(remaining)
        self.representative_face = None
        self.representative_area = 0
        for occ in remaining:
            x, y, w, h = occ.box
            area = w * h
            if area > self.representative_area and occ.face_image is not None and occ.face_image.size > 0:
                self.representative_face = occ.face_image.copy()
                self.representative_area = area
        return True


class FaceProfileManager:
    def __init__(self, distance_threshold: float = 0.38):
        self.distance_threshold = distance_threshold
        self._profiles: Dict[int, FaceProfile] = {}
        self._next_profile_id = 1

    def profiles(self) -> List[FaceProfile]:
        return list(self._profiles.values())

    def get_profile(self, profile_id: int) -> Optional[FaceProfile]:
        return self._profiles.get(profile_id)

    def _find_best_match(self, embedding: np.ndarray) -> Tuple[Optional[int], float]:
        if embedding is None or embedding.size == 0:
            return None, float("inf")
        best_id = None
        best_dist = float("inf")
        for pid, profile in self._profiles.items():
            avg_emb = profile.average_embedding()
            if avg_emb.size == 0:
                continue
            dist = np.linalg.norm(avg_emb - embedding)
            if dist < best_dist:
                best_dist = dist
                best_id = pid
        return best_id, best_dist

    def assign_profile(self, occurrence: FaceOccurrence) -> FaceProfile:
        emb = occurrence.embedding
        match_id, dist = self._find_best_match(emb)
        if match_id is None or dist > self.distance_threshold:
            profile = self._create_profile()
        else:
            profile = self._profiles[match_id]
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

    def remove_image_occurrences(self, image_path: str) -> bool:
        changed = False
        empty_profiles = []
        for pid, profile in self._profiles.items():
            if profile.remove_occurrences_for_image(image_path):
                changed = True
                if profile.occurrence_count == 0:
                    empty_profiles.append(pid)
        for pid in empty_profiles:
            del self._profiles[pid]
        return changed
