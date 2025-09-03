# save as build_average_sequences.py
import os
import pickle
import numpy as np
import cv2
import mediapipe as mp

DATA_DIR = './data_words'
OUTPUT_FILE = 'average_sequences.pkl'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.1, max_num_hands=2)

def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    while cap.isOpened() and len(sequence) < 90:  # limit to 90 frames
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Default zero vector
        data_aux_list = [0.0] * 126

        if results.multi_hand_landmarks:
            data_aux_list = []
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]
                z_ = [lm.z for lm in hand_landmarks.landmark]

                min_x, min_y, min_z = min(x_), min(y_), min(z_)
                data_aux = (
                    [lm.x - min_x for lm in hand_landmarks.landmark] +
                    [lm.y - min_y for lm in hand_landmarks.landmark] +
                    [lm.z - min_z for lm in hand_landmarks.landmark]
                )
                data_aux_list.extend(data_aux)

            # Pad if only one hand detected
            if len(results.multi_hand_landmarks) == 1:
                data_aux_list.extend([0.0] * 63)

            # Keep only 126 values
            data_aux_list = data_aux_list[:126]

        sequence.append(data_aux_list)

    cap.release()

    # Pad sequence to exactly 90 frames
    while len(sequence) < 90:
        sequence.append([0.0] * 126)

    return np.array(sequence)

def build_average_sequences():
    average_sequences = {}

    for word in os.listdir(DATA_DIR):
        word_path = os.path.join(DATA_DIR, word)
        if not os.path.isdir(word_path):
            continue

        print(f"[INFO] Processing word: {word}")
        all_sequences = []

        for video_file in os.listdir(word_path):
            if not video_file.lower().endswith('.mp4'):
                continue
            video_path = os.path.join(word_path, video_file)
            seq = extract_landmarks_from_video(video_path)
            all_sequences.append(seq)

        if all_sequences:
            avg_seq = np.mean(all_sequences, axis=0)
            average_sequences[word.lower()] = avg_seq

    # Save dictionary
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(average_sequences, f)

    print(f"[DONE] Saved average sequences for {len(average_sequences)} words to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_average_sequences()