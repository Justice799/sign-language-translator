import os
import pickle
import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# Paths
WORDS_DIR = './words'
AVERAGE_SEQ_FILE = 'average_sequences.pkl'

# Mediapipe drawing setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Load average sequences
if os.path.exists(AVERAGE_SEQ_FILE):
    with open(AVERAGE_SEQ_FILE, 'rb') as f:
        average_sequences = pickle.load(f)
    print(f"[INFO] Loaded average sequences for {len(average_sequences)} words.")
else:
    average_sequences = {}
    print("[WARN] No average_sequences.pkl found! Only videos will be used.")

def draw_pose_frame(pose_frame):
    """Draw pose landmarks on a blank image."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    left_poses = pose_frame[:63].reshape(21, 3)
    right_poses = pose_frame[63:].reshape(21, 3)

    # Draw left hand
    landmarks_left = landmark_pb2.LandmarkList()
    for p in left_poses:
        lm = landmark_pb2.Landmark()
        lm.x = int((p[0] * 300) + 100)
        lm.y = int((p[1] * 400) + 40)
        lm.z = p[2] * 100
        landmarks_left.landmark.append(lm)

    # Draw right hand
    landmarks_right = landmark_pb2.LandmarkList()
    for p in right_poses:
        lm = landmark_pb2.Landmark()
        lm.x = int((p[0] * 300) + 340)
        lm.y = int((p[1] * 400) + 40)
        lm.z = p[2] * 100
        landmarks_right.landmark.append(lm)

    mp_drawing.draw_landmarks(frame, landmarks_left, mp_hands.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, landmarks_right, mp_hands.HAND_CONNECTIONS)

    return frame

def yield_sign_frames(text):
    """Yields frames from videos in 'words' folder, or average poses."""
    print(f"[DEBUG] Requested text: {text}")

    words = text.lower().split()
    print(f"[DEBUG] Split into words: {words}")

    for word in words:
        played = False

        # 1️⃣ Try video from 'words' folder
        video_path = os.path.join(WORDS_DIR, f"{word}.mp4")
        if os.path.exists(video_path):
            print(f"[DEBUG] Found video for '{word}' → {video_path}")
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                ret, buffer = cv2.imencode('.jpg', frame)  # Use original frame
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                           buffer.tobytes() + b'\r\n')
                time.sleep(1/30)  # 30 FPS
            cap.release()
            played = True

        # 2️⃣ Try average sequence
        if not played and word in average_sequences:
            print(f"[DEBUG] Using average pose for '{word}'")
            sequence = average_sequences[word]
            for pose_frame in sequence:
                frame = draw_pose_frame(pose_frame)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                           buffer.tobytes() + b'\r\n')
                time.sleep(1/30)
            played = True

        # 3️⃣ If no data at all, fallback to black frame
        if not played:
            print(f"[WARN] No data for '{word}' → showing black frame")
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', black_frame)
            if ret:
                for _ in range(15):  # 0.5s
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                           buffer.tobytes() + b'\r\n')
                    time.sleep(1/30)

        # Small pause between words
        pause_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ret, buffer = cv2.imencode('.jpg', pause_frame)
        if ret:
            for _ in range(5):  # 0.15s
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')
                time.sleep(1/30)