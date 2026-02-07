# Online Proctoring System

An advanced AI-powered online exam proctoring system with face verification, head pose monitoring, eye gaze tracking, and object detection capabilities to ensure exam integrity.

## 📋 Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Output Interpretation](#output-interpretation)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Documentation](#documentation)

## ✨ Features

- **Face Verification**: Compares the test taker's face with a reference image using FaceNet embeddings to ensure identity
- **Adaptive Head Pose Estimation**: Calibrates to user's natural sitting position, then monitors head orientation (yaw, pitch, roll) to detect suspicious behavior and looking away
- **Baseline Calibration**: Automatically establishes neutral head position from initial frames, eliminating false positives from varied sitting positions
- **Eye Gaze Tracking**: Tracks iris movement to detect when students look away from the screen
- **Multiple Person Detection**: Uses YOLO to detect if multiple people are present in the frame
- **Prohibited Object Detection**: Detects cell phones, books, laptops, and other unauthorized materials using YOLOv8
- **Real-time Analysis**: Processes video frames with multiple AI models for comprehensive monitoring
- **Intelligent Verdict System**: Provides detailed analysis with overall exam validity assessment
- **Frame-by-Frame Reporting**: Detailed logging of all detection events for review

## 🤖 AI Models Used

This system integrates multiple state-of-the-art AI models:

1. **FaceNet** - Face recognition and verification using 128-dimensional embeddings
2. **MediaPipe Face Detection** - Fast and accurate face localization
3. **MediaPipe Face Mesh** - 468 facial landmarks for pose and gaze estimation with adaptive baseline calibration
4. **YOLOv8 Nano** - Real-time object detection from COCO dataset (80 classes)

### Adaptive Features

- **Baseline Calibration**: Automatically adapts to each user's natural sitting position
- **Relative Angle Measurement**: Detects movement from baseline, not absolute angles
- **Frame-based Statistics**: Separates calibration and analysis phases for accurate reporting

## 🖥️ System Requirements

### Hardware
- Webcam or video recording device
- Minimum 4GB RAM (8GB recommended)
- GPU (optional, for faster processing)

### Software
- Python 3.8 or higher
- Windows/Linux/macOS

## 📦 Installation

### Step 1: Clone or Download the Project

```bash
cd "C:\Users\bhara\Downloads\Mini Project"
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

Create a `requirements.txt` file with the following content:

```txt
opencv-python==4.8.1.78
numpy==1.24.3
mediapipe==0.10.8
keras-facenet==0.3.2
tensorflow==2.15.0
ultralytics>=8.0.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Alternative: Install Packages Individually

```bash
pip install opencv-python
pip install numpy
pip install mediapipe
pip install keras-facenet
pip install tensorflow
pip install ultralytics
```

## ⚙️ Configuration

### 1. Prepare Reference Image

- Take a clear frontal photo of the authorized person
- Save it in a known location
- Supported formats: `.jpg`, `.png`, `.jpeg`

### 2. Prepare Test Video

- Record a test video or use webcam feed
- Ensure good lighting conditions
- Video should show the face clearly

### 3. Update File Paths

Edit `app.py` and update the following variables with your file paths:

```python
# Update these paths in the PATHS section
video_path = "path/to/your/test/video.mp4"
reference_image = "path/to/your/reference/image.jpg"
```

**Example:**
```python
video_path = r"C:\Users\bhara\Videos\test_exam.mp4"
reference_image = r"C:\Users\bhara\Pictures\student_reference.jpg"
```

**Note:** Use raw strings (prefix with `r`) to avoid escape character issues in Windows paths.

### 4. First-Time YOLO Setup

On first run, YOLOv8 will automatically download the model weights (~6MB):
- Model: `yolov8n.pt` (Nano variant - optimized for speed)
- Location: Cached in your home directory
- Internet connection required for first run only

## 🚀 Usage

### Run the Application

```bash
# Make sure virtual environment is activated
python app.py
```

### Expected Output

The script will display:

1. **Initialization Messages:**
   ```
   >>> Script started
   >>> Imports done
   >>> FaceNet loaded
   >>> MediaPipe models loaded
   >>> YOLO model loaded
   ```

2. **Baseline Calibration Phase:**
   ```
   Frame 0: Authorized | Pose: Calibrating (1/3) | Gaze: Normal (Looking Center) | Persons: Single | Objects: None
   Frame 10: Authorized | Pose: Calibrating (2/3) | Gaze: Normal (Looking Center) | Persons: Single | Objects: None
   Frame 20: Authorized | Pose: Calibrating (3/3) | Gaze: Normal (Looking Center) | Persons: Single | Objects: None
   
   >>> Baseline calibrated: Yaw=5.3°, Pitch=110.2°, Roll=-2.1°
   ```

3. **Frame-by-Frame Analysis:**
   ```
   Frame 30: Authorized | Pose: Normal (ΔY=+2.1°, ΔP=-0.8°, ΔR=+1.2°) | Gaze: Normal (Looking Center) | Persons: Single | Objects: None
   Frame 40: Authorized | Pose: Deviating (ΔY=+18.5°, ΔP=-3.2°, ΔR=+0.6°) | Gaze: Suspicious (Looking Right) | Persons: Single | Objects: None
   Frame 50: Authorized | Pose: Normal (ΔY=+1.2°, ΔP=+0.5°, ΔR=-0.3°) | Gaze: Normal (Looking Center) | Persons: Multiple (2) | Objects: Found: Cell Phone(1)
   ```

4. **Final Results:**
   ```
   ========== FINAL RESULT ==========
   Identity Result     : AUTHORIZED PERSON
   Deviation Result    : NOT DEVIATED
   Eye Gaze Result     : NORMAL EYE MOVEMENT
   Person Count Result : SINGLE PERSON
   Object Detection    : NO PROHIBITED OBJECTS
   ==================================
   Baseline Head Pose  : Yaw=5.3°, Pitch=110.2°, Roll=-2.1°
   Detection Thresholds: ΔYaw=±15°, ΔPitch=±10°, ΔRoll=±10°
   ==================================
   Total frames processed: 50
   Calibration frames: 3
   Analyzed frames: 47
   Head pose deviation: 5 (10.6%)
   Gaze deviation: 8 (17.0%)
   Multiple persons: 0 (0.0%)
   Prohibited objects: 0 (0.0%)
   ==================================

   ==================================================
   FINAL VERDICT: ✓ EXAM VALID - No violations detected
   ==================================================
   ```

## 🔍 How It Works

**Note:** For detailed flow diagrams, complete algorithms, and step-by-step execution timeline, see [flow.txt](flow.txt).

### 1. Face Detection & Verification

- Uses **MediaPipe Face Detection** to locate faces in frames
- Extracts face embeddings using **FaceNet** (128-dimensional vectors)
- Compares embeddings using Euclidean distance
- Threshold: Distance < 1.0 = Same Person

### 2. Adaptive Head Pose Estimation with Baseline Calibration

**Calibration Phase (First 3 frames):**
- Collects head angles from initial frames
- Calculates average yaw, pitch, and roll as baseline
- Establishes user's natural sitting position as "zero point"
- Handles any starting angle (e.g., pitch = 110° is treated as neutral)

**Detection Phase:**
- Uses **MediaPipe Face Mesh** (468 facial landmarks)
- Applies **PnP (Perspective-n-Point)** algorithm
- Calculates 3D head orientation angles relative to baseline:
  - **Relative Yaw** (ΔYaw): Left/Right rotation from baseline (threshold: ±15°)
  - **Relative Pitch** (ΔPitch): Up/Down rotation from baseline (threshold: ±10°)
  - **Relative Roll** (ΔRoll): Tilt rotation from baseline (threshold: ±10°)
- Deviation = |current_angle - baseline_angle| > threshold
- Eliminates false positives from users not perfectly centered

### 3. Eye Gaze Tracking

- Tracks iris landmarks (468 for left eye, 473 for right eye)
- Calculates iris position relative to eye corners
- Determines gaze direction:
  - **Looking Left**: Gaze ratio > 0.65
  - **Looking Right**: Gaze ratio < 0.35
  - **Looking Center**: Gaze ratio 0.35-0.65 (Normal)
- Flags suspicious behavior when looking away from screen

### 4. YOLO Object Detection

- Uses **YOLOv8 Nano** model for real-time detection
- Detects persons (COCO class 0)
- Detects prohibited objects:
  - **Cell Phone** (class 67)
  - **Book** (class 73)
  - **Laptop** (class 63)
- Counts multiple persons in frame
- Tracks prohibited object appearances

### 5. Frame Processing

- Processes every 10th frame for efficiency
- Maximum 50 frames analyzed (configurable via `MAX_FRAMES`)
- First 3 processed frames used for baseline calibration
- Remaining frames analyzed for violations
- Reduces computation time while maintaining accuracy
- Runs multiple AI models in parallel for comprehensive analysis

### 6. Decision Logic

**Identity Verdict:**
- Authorized if `same_person_frames > different_person_frames`
- Otherwise, Unauthorized

**Deviation Verdict:**
- Deviated if `deviation_frames >= 25%` of analyzed frames (excluding calibration)
- Otherwise, Not Deviated

**Eye Gaze Verdict:**
- Suspicious if `gaze_deviation_frames >= 30%` of analyzed frames (excluding calibration)
- Otherwise, Normal

**Person Count Verdict:**
- Multiple persons detected if `multiple_person_frames >= 20%` of analyzed frames
- Otherwise, Single person

**Object Detection Verdict:**
- Prohibited objects detected if `prohibited_object_frames >= 15%` of total frames
- Otherwise, No prohibited objects

**Overall Verdict:**
- ✓ **EXAM VALID**: No violations detected
- ⚠️ **EXAM SUSPICIOUS**: Deviations or suspicious gaze detected (review required)
- ⚠️ **EXAM INVALID**: Unauthorized person, multiple persons, or prohibited objects detected

## 📊 Output Interpretation

### Frame Output During Calibration

```
Frame 10: Authorized | Pose: Calibrating (2/3) | Gaze: Normal (Looking Center) | Persons: Single | Objects: None
```

- **Frame 10**: Frame number being analyzed
- **Pose: Calibrating (2/3)**: Collecting calibration data, 2 of 3 frames complete
- No deviation detection occurs during calibration phase

### Frame Output After Calibration

```
Frame 40: Authorized | Pose: Deviating (ΔY=+18.5°, ΔP=-3.2°, ΔR=+0.6°) | Gaze: Suspicious (Looking Right) | Persons: Multiple (2) | Objects: Found: Cell Phone(1)
```

- **Frame 40**: Frame number being analyzed
- **Authorized**: Face matches reference (distance < 1.0)
- **Pose: Deviating**: Head movement exceeds threshold from baseline
- **ΔY=+18.5°**: Yaw changed +18.5° from baseline (exceeds ±15° threshold)
- **ΔP=-3.2°**: Pitch changed -3.2° from baseline (within ±10° threshold)
- **ΔR=+0.6°**: Roll changed +0.6° from baseline (within ±10° threshold)
- **Gaze: Suspicious (Looking Right)**: Eyes not looking at center (screen)
- **Persons: Multiple (2)**: Two persons detected in frame
- **Objects: Found**: Prohibited object detected (Cell Phone with count)

### Final Verdict

**Note:** All percentage thresholds are calculated based on analyzed frames (excluding the 3 calibration frames).

| Result | Meaning | Threshold |
|--------|---------|-----------|
| AUTHORIZED PERSON | Majority frames matched reference face | > 50% frames |
| UNAUTHORIZED PERSON | Majority frames did NOT match reference | > 50% frames |
| NOT DEVIATED | Acceptable head movement from baseline | < 25% analyzed frames |
| DEVIATED | Excessive head movement from baseline detected | ≥ 25% analyzed frames |
| NORMAL EYE MOVEMENT | Eyes focused on screen | < 30% analyzed frames |
| SUSPICIOUS EYE MOVEMENT | Frequent looking away detected | ≥ 30% analyzed frames |
| SINGLE PERSON | Only one person in frame | < 20% analyzed frames |
| MULTIPLE PERSONS DETECTED | Additional people present | ≥ 20% analyzed frames |
| NO PROHIBITED OBJECTS | No unauthorized items found | < 15% analyzed frames |
| PROHIBITED OBJECTS DETECTED | Unauthorized items found | ≥ 15% analyzed frames |

### Overall Verdict Categories

| Verdict | Icon | Description |
|---------|------|-------------|
| EXAM VALID | ✓ | All checks passed, no violations |
| EXAM SUSPICIOUS | ⚠️ | Minor violations (head pose/gaze), requires review |
| EXAM INVALID | ⚠️ | Major violations (wrong person/multiple people/prohibited objects) |

## 🔧 Troubleshooting

### Issue: "No face detected in reference image"

**Solution:**
- Ensure reference image has clear, frontal face
- Check image path is correct
- Try different image with better lighting

### Issue: ImportError for TensorFlow/Keras

**Solution:**
```bash
pip install --upgrade tensorflow
pip install keras-facenet
```

### Issue: ImportError for Ultralytics/YOLO

**Solution:**
```bash
pip install ultralytics
# First run will automatically download YOLOv8n.pt model
```

### Issue: Video file not opening

**Solution:**
- Check video path uses raw string: `r"C:\path\to\video.mp4"`
- Verify video file exists and is not corrupted
- Try different video format (mp4, avi, mov)

### Issue: Slow processing

**Solutions:**
- Increase frame skip: Change `frame_count % 10` to higher value (e.g., `% 15`)
- Reduce `MAX_FRAMES` to process fewer frames
- Use GPU-enabled TensorFlow/PyTorch if available
- YOLO nano model (yolov8n.pt) is already optimized for speed

### Issue: High false positives/negatives

**Solutions:**
- Adjust distance threshold (default: 1.0) for face matching
- Adjust head pose thresholds relative to baseline (Δyaw: ±15°, Δpitch: ±10°, Δroll: ±10°)
- Adjust gaze ratio thresholds (0.35-0.65 for center)
- Adjust YOLO confidence thresholds for object detection
- Ensure consistent lighting between reference and test video

### Issue: Baseline calibration seems incorrect

**Solutions:**
- Ensure user maintains neutral position during first 3 processed frames
- Check that face is clearly visible in calibration frames
- Increase `CALIBRATION_FRAMES` (default: 3) for more stable baseline
- User should look straight ahead during calibration phase
- Avoid head movement during calibration

### Issue: YOLO not detecting objects correctly

**Solutions:**
- Ensure good lighting and clear view of objects
- Check object is in YOLO's training classes (COCO dataset)
- Adjust confidence threshold in `detect_persons_and_objects()` function
- Objects must be clearly visible and not occluded

## 📁 Project Structure

```
Online-Proctoring/
│
├── app.py                 # Main application script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── flow.txt              # Complete flow documentation with diagrams
│
└── (Optional folders)
    ├── videos/           # Store test videos
    ├── references/       # Store reference images
    └── logs/            # Store output logs
```

## 🎯 Customization Options

### Adjust Detection Sensitivity

```python
# Face detection confidence
min_detection_confidence=0.6  # Lower = more detections, higher false positives

# Face matching threshold
if distance < 1.0:  # Lower = stricter matching, Higher = more lenient
```

### Adjust Baseline Calibration

```python
# Number of frames to average for baseline calculation
CALIBRATION_FRAMES = 3  # Increase for more stable baseline (e.g., 5 or 7)
# Higher value = more stable but requires user to stay still longer
# Lower value = faster calibration but potentially less accurate
```

### Adjust Head Pose Thresholds (Relative to Baseline)

```python
# Deviation detection based on relative movement from baseline
relative_yaw = yaw - baseline_yaw
relative_pitch = pitch - baseline_pitch
relative_roll = roll - baseline_roll

deviating = abs(relative_yaw) > 15 or abs(relative_pitch) > 10 or abs(relative_roll) > 10
# Increase values for more lenient detection
# Decrease values for stricter detection
# Note: These are RELATIVE angles (change from baseline), not absolute
```

### Adjust Eye Gaze Thresholds

```python
# Gaze direction determination
if avg_gaze_ratio < 0.35:
    gaze_direction = "Looking Right"
elif avg_gaze_ratio > 0.65:
    gaze_direction = "Looking Left"
# Adjust 0.35 and 0.65 to change sensitivity
```

### Adjust YOLO Detection Settings

```python
# Person detection confidence
if cls == PERSON_CLASS and conf > 0.5:  # Adjust 0.5 threshold

# Object detection confidence
if cls in PROHIBITED_CLASSES and conf > 0.4:  # Adjust 0.4 threshold

# Add more prohibited objects
PROHIBITED_CLASSES = {
    67: 'Cell Phone',
    73: 'Book',
    63: 'Laptop',
    # Add more COCO class IDs as needed
}
```

### Adjust Final Verdict Thresholds

```python
# Note: analyzed_frames = processed_frames - CALIBRATION_FRAMES
# Deviation threshold (default: 25% of analyzed frames)
if deviation_frames >= (0.25 * analyzed_frames):  # Change 0.25

# Gaze deviation threshold (default: 30% of analyzed frames)
if gaze_deviation_frames >= (0.30 * analyzed_frames):  # Change 0.30

# Multiple persons threshold (default: 20% of analyzed frames)
if multiple_person_frames >= (0.20 * analyzed_frames):  # Change 0.20

# Prohibited objects threshold (default: 15% of analyzed frames)
if prohibited_object_frames >= (0.15 * analyzed_frames):  # Change 0.15
```

### Process More/Fewer Frames

```python
# Frame skipping
if frame_count % 10 != 0:  # Change 10 to process more/fewer frames

# Total frames to process (includes calibration frames)
MAX_FRAMES = 50  # Increase for longer analysis
# Note: First CALIBRATION_FRAMES are used for baseline, remaining for analysis
```

## 💡 Important Notes

### Why Baseline Calibration?

The baseline calibration system provides several crucial advantages:

1. **Adaptability**: Works with any sitting position - no need for users to be perfectly centered
2. **Accuracy**: Eliminates false positives from natural variations in posture
3. **Real-world Application**: Handles different desk heights, chair positions, and monitor placements
4. **Individual Differences**: Accounts for personal ergonomic preferences and physical limitations
5. **Example**: If a user naturally sits with pitch = 110°, this becomes their "zero" - only deviations of ±10° from 110° trigger alerts

### Best Practices

- **During Calibration**: User should maintain their natural, comfortable exam-taking position
- **After Calibration**: System detects movement away from this established baseline
- **First 3 Frames**: Critical for accurate baseline - ensure face is clearly visible
- **Longer Videos**: Consider increasing `CALIBRATION_FRAMES` for more stable baseline

## � Documentation

### Complete Flow Documentation

For a comprehensive understanding of the system's internal workings, refer to [flow.txt](flow.txt), which includes:

1. **Main Flow Diagram**: ASCII visualization of the complete execution flow
2. **Detailed Phase Breakdown**: 
   - Initialization Phase
   - Reference Loading Phase
   - Video Initialization Phase
   - Baseline Calibration Phase (first 3 frames)
   - Main Processing Loop
   - Final Decision Phase
3. **Module-wise Flow Diagrams**:
   - Face Verification (FaceNet)
   - Head Pose Estimation (with baseline calibration)
   - Eye Gaze Tracking (iris detection)
   - YOLO Object Detection
4. **Decision Logic Flow**: Complete verdict determination process
5. **Data Flow**: Input → Processing → Output pipeline
6. **Step-by-Step Execution**: Timeline with real examples from Frame 0 to 50
7. **Key Algorithms & Formulas**: Mathematical foundations
   - Euclidean distance for face matching
   - PnP algorithm for head pose
   - Euler angle extraction
   - Gaze ratio calculation
   - Baseline calibration mathematics
8. **Performance Metrics**: Processing time, accuracy, resource usage
9. **Error Handling**: All edge cases and failure scenarios
10. **Customization Points**: All configurable parameters

### Quick Reference

- **System Architecture**: See [flow.txt](flow.txt) Section 2 (Main Flow Diagram)
- **Algorithm Details**: See [flow.txt](flow.txt) Section 8 (Key Algorithms)
- **Execution Timeline**: See [flow.txt](flow.txt) Section 7 (Step-by-Step)
- **Troubleshooting**: See above section and [flow.txt](flow.txt) Section 9

## �📝 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to fork, modify, and submit pull requests.

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

## 🛠️ Technologies Used

- **OpenCV**: Video processing and computer vision
- **MediaPipe**: Face detection, facial landmarks, and face mesh
- **FaceNet**: Face recognition and embedding generation
- **YOLOv8**: Real-time object detection (persons and prohibited items)
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations

---

**Made with ❤️ using Python, OpenCV, MediaPipe, FaceNet, and YOLOv8**
