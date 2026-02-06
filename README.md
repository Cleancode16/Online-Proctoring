# Online Proctoring System

A real-time face verification and head pose monitoring system for online exam proctoring using AI/ML technologies.

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

## ✨ Features

- **Face Verification**: Compares the test taker's face with a reference image to ensure authorized person
- **Head Pose Estimation**: Monitors head orientation (yaw, pitch, roll) to detect suspicious behavior
- **Real-time Analysis**: Processes video frames to detect deviations
- **Final Decision Making**: Provides comprehensive verdict on authorization and deviation status

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

Edit `app.py` and update the following variables:

```python
# Line 91-92
video_path = "path/to/your/test/video.mp4"
reference_image = "path/to/your/reference/image.jpg"
```

**Example:**
```python
video_path = r"C:\Users\bhara\Videos\test_exam.mp4"
reference_image = r"C:\Users\bhara\Pictures\student_reference.jpg"
```

**Note:** Use raw strings (prefix with `r`) to avoid escape character issues in Windows paths.

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
   ```

2. **Frame-by-Frame Analysis:**
   ```
   Frame 0: Authorized | Normal | Yaw=2.3, Pitch=1.5, Roll=0.8
   Frame 10: Authorized | Deviating | Yaw=18.2, Pitch=3.1, Roll=2.1
   Frame 20: Unauthorized | Normal | Yaw=3.5, Pitch=2.0, Roll=1.2
   ```

3. **Final Results:**
   ```
   ========== FINAL RESULT ==========
   Identity Result : AUTHORIZED PERSON
   Deviation Result: NOT DEVIATED
   ==================================
   ```

## 🔍 How It Works

### 1. Face Detection & Verification

- Uses **MediaPipe Face Detection** to locate faces in frames
- Extracts face embeddings using **FaceNet** (128-dimensional vectors)
- Compares embeddings using Euclidean distance
- Threshold: Distance < 1.0 = Same Person

### 2. Head Pose Estimation

- Uses **MediaPipe Face Mesh** (468 facial landmarks)
- Applies **PnP (Perspective-n-Point)** algorithm
- Calculates 3D head orientation angles:
  - **Yaw**: Left/Right rotation (threshold: ±15°)
  - **Pitch**: Up/Down rotation (threshold: ±10°)
  - **Roll**: Tilt rotation (threshold: ±10°)

### 3. Frame Processing

- Processes every 10th frame for efficiency
- Maximum 50 frames analyzed (configurable via `MAX_FRAMES`)
- Reduces computation time while maintaining accuracy

### 4. Decision Logic

**Identity Verdict:**
- Authorized if `same_person_frames > different_person_frames`
- Otherwise, Unauthorized

**Deviation Verdict:**
- Deviated if `deviation_frames >= 25%` of total processed frames
- Otherwise, Not Deviated

## 📊 Output Interpretation

### Frame Output

```
Frame 10: Authorized | Deviating | Yaw=18.2, Pitch=3.1, Roll=2.1
```

- **Frame 10**: Frame number being analyzed
- **Authorized**: Face matches reference (distance < 1.0)
- **Deviating**: Head pose exceeds threshold
- **Angles**: Current head orientation in degrees

### Final Verdict

| Result | Meaning |
|--------|---------|
| AUTHORIZED PERSON | Majority frames matched reference face |
| UNAUTHORIZED PERSON | Majority frames did NOT match reference |
| NOT DEVIATED | Less than 25% frames showed suspicious head movement |
| DEVIATED | 25% or more frames showed suspicious head movement |

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

### Issue: Video file not opening

**Solution:**
- Check video path uses raw string: `r"C:\path\to\video.mp4"`
- Verify video file exists and is not corrupted
- Try different video format (mp4, avi, mov)

### Issue: Slow processing

**Solutions:**
- Increase frame skip: Change `frame_count % 10` to higher value (e.g., `% 15`)
- Reduce `MAX_FRAMES` to process fewer frames
- Use GPU-enabled TensorFlow if available

### Issue: High false positives/negatives

**Solutions:**
- Adjust distance threshold in line 185 (default: 1.0)
- Adjust head pose thresholds in line 203
- Ensure consistent lighting between reference and test video

## 📁 Project Structure

```
Online-Proctoring/
│
├── app.py                 # Main application script
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
└── (Optional folders)
    ├── videos/           # Store test videos
    ├── references/       # Store reference images
    └── logs/            # Store output logs
```

## 🎯 Customization Options

### Adjust Detection Sensitivity

```python
# Line 17-19: Face detection confidence
min_detection_confidence=0.6  # Lower = more detections, higher false positives

# Line 185: Face matching threshold
if distance < 1.0:  # Lower = stricter matching
```

### Adjust Head Pose Thresholds

```python
# Line 203: Deviation detection
deviating = abs(yaw) > 15 or abs(pitch) > 10 or abs(roll) > 10
# Increase values for more lenient detection
```

### Process More/Fewer Frames

```python
# Line 137: Frame skipping
if frame_count % 10 != 0:  # Change 10 to process more/fewer frames

# Line 143: Total frames to process
MAX_FRAMES = 50  # Increase for longer analysis
```

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to fork, modify, and submit pull requests.

##  Contact

For questions or issues, please open an issue in the repository.

---

**Made with  using Python, OpenCV, MediaPipe, and FaceNet**
