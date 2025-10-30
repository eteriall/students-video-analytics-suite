# Face Analysis Feature

This feature adds emotion detection and gaze direction tracking to face profiles.

## Features

- **Emotion Detection**: Detects emotions (happy, sad, angry, surprised, neutral, fear, disgust)
- **Gaze Direction**: Tracks where the person is looking (looking_down, looking_forward, looking_up)
- **Per-Profile Analysis**: Process all images for a specific person
- **Statistics Dashboard**: View emotion and gaze distribution for each profile

## How to Use

1. **Process Images for a Profile**:
   - Go to the "Faces & People" panel (right sidebar)
   - Select a profile (person)
   - Click the **"Process Images"** button
   - Click **"Start Analysis"** in the dialog

2. **View Results**:
   - After processing, you'll see statistics showing:
     - Emotion distribution (percentage of each emotion)
     - Gaze direction distribution (how often they look down/up/forward)

3. **Re-process**:
   - Already analyzed images are skipped automatically
   - You can re-run analysis if you want to update results

## Installation

### Basic Installation (Simple Detection)

The feature works out of the box with basic heuristic-based detection.

### Advanced Installation (Better Accuracy)

For better emotion detection, install DeepFace:

```bash
pip install deepface
```

For better gaze detection, download dlib face landmarks:

```bash
# Install dlib if not already installed
pip install dlib

# Download shape predictor (required for accurate gaze detection)
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

Place `shape_predictor_68_face_landmarks.dat` in the project directory.

## Implementation Details

### Files Modified

1. **database.py**: Added `emotion` and `gaze_direction` columns to detections table
2. **recognition.py**: Updated `FaceOccurrence` dataclass with new fields
3. **interface.py**:
   - Added "Process Images" button to profile panel
   - Added handler for face analysis
   - Updated database loading to include emotion/gaze data
4. **face_analysis.py** (NEW): Core emotion and gaze detection functions
5. **face_analysis_dialog.py** (NEW): UI dialog for analysis and results

### Database Schema

```sql
ALTER TABLE detections ADD COLUMN emotion TEXT;
ALTER TABLE detections ADD COLUMN gaze_direction TEXT;
```

Migration is automatic when you open the app.

### Emotion Detection Methods

1. **Simple (Default)**: Basic heuristic-based detection using image analysis
2. **DeepFace (Optional)**: ML-based detection with better accuracy

### Gaze Detection Methods

1. **dlib Landmarks (Preferred)**: Uses facial landmarks to estimate gaze angle
2. **Simple Fallback**: Basic image analysis if dlib is not available

## Limitations

- **Basic detection** has limited accuracy - it's a placeholder
- **DeepFace** requires internet connection on first use (downloads models)
- **Gaze detection** works best with frontal faces
- Processing many images may take time depending on method used

## Future Improvements

- [ ] Support for more emotions
- [ ] Confidence scores for emotions
- [ ] Eye gaze tracking (not just up/down)
- [ ] Age and gender detection
- [ ] Batch processing across all profiles
- [ ] Export analysis results to CSV/Excel

## Troubleshooting

**"No face images available for these detections"**
- Make sure you've imported and processed images first

**"DeepFace not installed"**
- Falls back to simple detection automatically
- Install DeepFace for better results: `pip install deepface`

**Slow processing**
- DeepFace is slower but more accurate
- Basic detection is faster but less accurate
- Consider processing profiles one at a time

## API Reference

### analyze_face(face_image, use_deepface=False)

Analyzes a face image for emotion and gaze direction.

**Parameters:**
- `face_image` (np.ndarray): Face crop as BGR numpy array
- `use_deepface` (bool): Use DeepFace for emotion detection

**Returns:**
- dict: `{'emotion': str, 'gaze_direction': str}`

### detect_emotion_simple(face_image)

Simple emotion detection using heuristics.

### detect_emotion_deepface(face_image)

Advanced emotion detection using DeepFace library.

### detect_gaze_direction(face_image)

Detects gaze direction using facial landmarks or fallback method.
