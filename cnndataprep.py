import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import gc

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.1, max_num_hands=2)

DATA_DIR = './data_words'
data = {'inputs': [], 'outputs': []}

words_to_process = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]  # Auto-detect your 12 words

label_encoder = LabelEncoder()
label_encoder.fit(words_to_process)

for dir_ in words_to_process:
    print(f"Starting processing for word: {dir_}")
    dir_path = os.path.join(DATA_DIR, dir_)
    word_idx = label_encoder.transform([dir_])[0]
    word_onehot = to_categorical([word_idx], num_classes=len(words_to_process))[0]

    video_files = [v for v in os.listdir(dir_path) if v.endswith('.mp4')]
    for idx, video_path in enumerate(video_files):
        print(f"Processing video {idx+1}/{len(video_files)} for {dir_}: {video_path}")
        video_full_path = os.path.join(dir_path, video_path)
        cap = cv2.VideoCapture(video_full_path)
        sequence = []
        frame_count = 0
        while cap.isOpened() and frame_count < 90:
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
                data_aux_list = data_aux_list[:126]
            sequence.append(data_aux_list)
            frame_count += 1

        cap.release()
        if len(sequence) == 90:
            data['inputs'].append(word_onehot)
            data['outputs'].append(np.array(sequence))

        gc.collect()  # Cleanup after each video

print(f"Processed {len(data['outputs'])} samples")
with open('cnn_data.pickle', 'wb') as f:
    pickle.dump(data, f)
gc.collect()