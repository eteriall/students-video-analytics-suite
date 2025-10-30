"""
Face Analysis Module

Provides emotion detection and gaze direction analysis for face images.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict
import dlib


# Initialize dlib face detector and shape predictor
try:
    detector = dlib.get_frontal_face_detector()
    # You may need to download shape_predictor_68_face_landmarks.dat
    # from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    DLIB_AVAILABLE = True
except Exception as e:
    print(f"Warning: dlib landmarks not available: {e}")
    DLIB_AVAILABLE = False


def detect_emotion_simple(face_image: np.ndarray) -> Optional[str]:
    """
    Detect emotion using a simple heuristic-based approach.

    This is a basic implementation. For production use, consider:
    - DeepFace library
    - FER (Facial Expression Recognition) library
    - Custom trained models

    Args:
        face_image: Face crop as numpy array (BGR format)

    Returns:
        Emotion label: 'happy', 'sad', 'angry', 'surprised', 'neutral', 'fear', 'disgust'
    """
    try:
        # Convert to grayscale
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image

        # Simple approach: try to detect basic facial features
        # This is a placeholder - you should use a proper emotion detection model

        # For now, use a very basic heuristic based on image brightness and contrast
        mean_brightness = np.mean(gray)
        std_contrast = np.std(gray)

        # This is just a demonstration - NOT accurate emotion detection
        # Replace with actual model inference
        if mean_brightness > 150:
            return 'happy'
        elif mean_brightness < 80:
            return 'sad'
        elif std_contrast > 60:
            return 'surprised'
        else:
            return 'neutral'

    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return None


def detect_emotion_deepface(face_image: np.ndarray) -> Optional[str]:
    """
    Detect emotion using DeepFace library.

    Requires: pip install deepface

    Args:
        face_image: Face crop as numpy array (BGR format)

    Returns:
        Emotion label
    """
    try:
        from deepface import DeepFace

        # Analyze emotion
        result = DeepFace.analyze(
            face_image,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )

        # Get dominant emotion
        if isinstance(result, list):
            result = result[0]

        emotion = result.get('dominant_emotion', 'neutral')
        return emotion

    except ImportError:
        print("DeepFace not installed. Using simple detection instead.")
        return detect_emotion_simple(face_image)
    except Exception as e:
        print(f"Error with DeepFace: {e}")
        return detect_emotion_simple(face_image)


def detect_gaze_direction(face_image: np.ndarray) -> Optional[str]:
    """
    Detect if person is looking down or up/forward.

    Uses facial landmarks to estimate gaze direction.

    Args:
        face_image: Face crop as numpy array (BGR format)

    Returns:
        'looking_down', 'looking_forward', 'looking_up', or None
    """
    if not DLIB_AVAILABLE:
        return detect_gaze_simple(face_image)

    try:
        # Convert to grayscale
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image

        # Detect faces
        faces = detector(gray, 1)

        if len(faces) == 0:
            return None

        # Get landmarks for first face
        face = faces[0]
        landmarks = predictor(gray, face)

        # Get eye landmarks
        # Left eye: points 36-41
        # Right eye: points 42-47
        # Nose tip: point 33
        # Chin: point 8

        left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        nose_tip = (landmarks.part(33).x, landmarks.part(33).y)
        chin = (landmarks.part(8).x, landmarks.part(8).y)

        # Calculate average eye position
        left_eye_center = np.mean(left_eye_points, axis=0)
        right_eye_center = np.mean(right_eye_points, axis=0)
        eye_center = (left_eye_center + right_eye_center) / 2

        # Calculate eye-to-nose distance ratio
        # If eyes are high relative to nose, person is looking up
        # If eyes are low relative to nose, person is looking down

        eye_nose_distance = nose_tip[1] - eye_center[1]
        face_height = chin[1] - face.top()

        if face_height == 0:
            return None

        ratio = eye_nose_distance / face_height

        # Thresholds (may need tuning)
        if ratio < 0.15:
            return 'looking_up'
        elif ratio > 0.35:
            return 'looking_down'
        else:
            return 'looking_forward'

    except Exception as e:
        print(f"Error detecting gaze: {e}")
        return detect_gaze_simple(face_image)


def detect_gaze_simple(face_image: np.ndarray) -> Optional[str]:
    """
    Simple gaze detection fallback using image analysis.

    Args:
        face_image: Face crop as numpy array

    Returns:
        Gaze direction estimate
    """
    try:
        h, w = face_image.shape[:2]

        # Analyze top half vs bottom half brightness
        # Eyes are typically in the top half
        top_half = face_image[:h//2, :]
        bottom_half = face_image[h//2:, :]

        top_brightness = np.mean(top_half)
        bottom_brightness = np.mean(bottom_half)

        # Very rough heuristic
        diff = top_brightness - bottom_brightness

        if diff > 20:
            return 'looking_down'
        elif diff < -20:
            return 'looking_up'
        else:
            return 'looking_forward'

    except Exception as e:
        print(f"Error in simple gaze detection: {e}")
        return None


def analyze_face(face_image: np.ndarray, use_deepface: bool = False) -> Dict[str, Optional[str]]:
    """
    Perform complete face analysis: emotion and gaze direction.

    Args:
        face_image: Face crop as numpy array (BGR format)
        use_deepface: Whether to use DeepFace for emotion detection (slower but more accurate)

    Returns:
        Dictionary with 'emotion' and 'gaze_direction' keys
    """
    result = {
        'emotion': None,
        'gaze_direction': None
    }

    if face_image is None or face_image.size == 0:
        return result

    # Detect emotion
    if use_deepface:
        result['emotion'] = detect_emotion_deepface(face_image)
    else:
        result['emotion'] = detect_emotion_simple(face_image)

    # Detect gaze
    result['gaze_direction'] = detect_gaze_direction(face_image)

    return result
