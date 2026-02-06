print(">>> Script started")

import cv2
import numpy as np
import mediapipe as mp
from keras_facenet import FaceNet

print(">>> Imports done")

# ================= INITIALIZATION =================
embedder = FaceNet()
print(">>> FaceNet loaded")

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.6
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

print(">>> MediaPipe models loaded")

# ================= STATIC 3D FACE MODEL =================
model_3d = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -63.6, -12.5),      # Chin
    (-43.3, 32.7, -26.0),     # Left eye corner
    (43.3, 32.7, -26.0),      # Right eye corner
    (-28.9, -28.9, -24.1),    # Left mouth corner
    (28.9, -28.9, -24.1)      # Right mouth corner
], dtype=np.float64)

# MediaPipe landmark indices
# nose, chin, left eye, right eye, left mouth, right mouth
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# ================= FACE EMBEDDING =================
def get_face_embedding(frame, bbox):
    x, y, w, h = bbox
    face_crop = frame[y:y+h, x:x+w]

    if face_crop.size == 0:
        return None

    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_crop = cv2.resize(face_crop, (160, 160))
    return embedder.embeddings([face_crop])[0]

# ================= HEAD POSE =================
def get_head_pose(frame, landmarks):
    h, w = frame.shape[:2]

    image_2d = np.array([
        (int(landmarks[i].x * w), int(landmarks[i].y * h))
        for i in LANDMARK_IDS
    ], dtype=np.float64)

    cam_matrix = np.array([
        [w, 0, w / 2],
        [0, w, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    _, rvec, _ = cv2.solvePnP(
        model_3d,
        image_2d,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    rot_matrix, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot_matrix[0, 0]**2 + rot_matrix[1, 0]**2)

    pitch = np.degrees(np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2]))
    yaw   = np.degrees(np.arctan2(-rot_matrix[2, 0], sy))
    roll  = np.degrees(np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0]))
    
    return yaw, pitch, roll

# ================= PATHS =================
video_path = r"C:\Users\bhara\OneDrive\Pictures\Camera Roll\WIN_20260206_20_33_41_Pro.mp4"
reference_image = r"C:\Users\bhara\OneDrive\Pictures\Camera Roll\WIN_20260206_21_26_58_Pro.jpg"

# ================= LOAD REFERENCE =================
print(">>> Loading reference image")
ref_img = cv2.imread(reference_image)
ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

ref_result = face_detection.process(ref_rgb)

if not ref_result.detections:
    raise ValueError("No face detected in reference image")

d = ref_result.detections[0].location_data.relative_bounding_box
h, w = ref_img.shape[:2]
ref_bbox = (
    int(d.xmin * w),
    int(d.ymin * h),
    int(d.width * w),
    int(d.height * h)
)

ref_embedding = get_face_embedding(ref_img, ref_bbox)
print(">>> Reference embedding created")

# ================= VIDEO LOOP =================
cap = cv2.VideoCapture(video_path)
print(">>> Video opened")

frame_count = 0
processed = 0

same_person_frames = 0
different_person_frames = 0
deviation_frames = 0

MAX_FRAMES = 50

while processed < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 10 != 0:
        frame_count += 1
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det_result = face_detection.process(rgb)

    if not det_result.detections:
        frame_count += 1
        continue

    d = det_result.detections[0].location_data.relative_bounding_box
    h, w = frame.shape[:2]
    bbox = (
        int(d.xmin * w),
        int(d.ymin * h),
        int(d.width * w),
        int(d.height * h)
    )

    # -------- FACE VERIFICATION --------
    embedding = get_face_embedding(frame, bbox)
    if embedding is None:
        frame_count += 1
        continue

    distance = np.linalg.norm(embedding - ref_embedding)

    if distance < 1.0:
        same_person_frames += 1
        identity = "Authorized"
    else:
        different_person_frames += 1
        identity = "Unauthorized"

    # -------- HEAD POSE --------
    mesh_result = face_mesh.process(rgb)
    if not mesh_result.multi_face_landmarks:
        frame_count += 1
        continue

    landmarks = mesh_result.multi_face_landmarks[0].landmark
    yaw, pitch, roll = get_head_pose(frame, landmarks)

    deviating = abs(yaw) > 15 or abs(pitch) > 10 or abs(roll) > 10
    pose_status = "Deviating" if deviating else "Normal"

    if deviating:
        deviation_frames += 1

    print(
        f"Frame {frame_count}: {identity} | {pose_status} | "
        f"Yaw={yaw:.1f}, Pitch={pitch:.1f}, Roll={roll:.1f}"
    )

    processed += 1
    frame_count += 1

cap.release()

# ================= FINAL DECISION =================
print("\n========== FINAL RESULT ==========")

final_identity = (
    "AUTHORIZED PERSON"
    if same_person_frames > different_person_frames
    else "UNAUTHORIZED PERSON"
)

final_deviation = (
    "DEVIATED"
    if deviation_frames >= (0.25 * processed)
    else "NOT DEVIATED"
)

print("Identity Result :", final_identity)
print("Deviation Result:", final_deviation)
print("==================================")
