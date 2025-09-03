import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.1, max_num_hands=2)

DATA_DIR = './data_words'
data = {}  # Dict: word -> list of sequences (each sequence: list of [frames, 126 features])
words_to_process = ['hello', 'my', 'name', 'is', 'thank you', 'book', 'eat', 'drink', 'computer', 'chair', 'go', 'come', 'yes', 'no', 'please', 'sorry', 'love', 'like', 'hate', 'want', 'need', 'see', 'hear', 'speak', 'walk', 'run', 'sit', 'stand', 'help', 'stop', 'start', 'finish', 'all', 'some', 'none', 'what', 'where', 'when', 'why', 'who', 'how', 'good', 'bad', 'happy', 'sad', 'big', 'small', 'hot', 'cold', 'new', 'old']

for dir_ in words_to_process:
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue
    print(f"Processing word: {dir_}")
    sequences = []  # List of sequences for this word
    for video_path in os.listdir(dir_path):
        video_full_path = os.path.join(dir_path, video_path)
        cap = cv2.VideoCapture(video_full_path)
        sequence = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            data_aux_list = [0.0] * 126
            if results.multi_hand_landmarks:
                data_aux_list = []
                for hand_landmarks in results.multi_hand_landmarks:
                    x_ = [lm.x for lm in hand_landmarks.landmark]
                    y_ = [lm.y for lm in hand_landmarks.landmark]
                    z_ = [lm.z for lm in hand_landmarks.landmark]
                    min_x, min_y, min_z = min(x_), min(y_), min(z_)
                    data_aux = [lm.x - min_x for lm in hand_landmarks.landmark] + [lm.y - min_y for lm in hand_landmarks.landmark] + [lm.z - min_z for lm in hand_landmarks.landmark]
                    data_aux_list.extend(data_aux)
                if len(results.multi_hand_landmarks) == 1:
                    data_aux_list.extend([0.0] * 63)
                data_aux_list = data_aux_list[:126]  # Ensure fixed size
            sequence.append(data_aux_list)
        cap.release()
        if sequence:
            sequences.append(sequence)
    if sequences:
        data[dir_] = sequences

print(f"Processed {len(data)} words")
with open('text_to_sign_data.pickle', 'wb') as f:
    pickle.dump(data, f)