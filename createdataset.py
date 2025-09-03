import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt  # Remove if unused

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.1, max_num_hands=2)

DATA_DIR = './data_words'
data = []
labels = []
words_to_process = ['hello', 'my', 'name', 'is', 'thank you', 'book', 'eat', 'drink', 'computer', 'chair', 'go', 'come', 'yes', 'no', 'please', 'sorry', 'love', 'like', 'hate', 'want', 'need', 'see', 'hear', 'speak', 'walk', 'run', 'sit', 'stand', 'help', 'stop', 'start', 'finish', 'all', 'some', 'none', 'what', 'where', 'when', 'why', 'who', 'how', 'good', 'bad', 'happy', 'sad', 'big', 'small', 'hot', 'cold', 'new', 'old']

for dir_ in words_to_process:
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue
    print(f"Processing word: {dir_}")
    video_count = 0
    for video_path in os.listdir(dir_path):
        video_full_path = os.path.join(dir_path, video_path)
        cap = cv2.VideoCapture(video_full_path)
        sequence = []  # Sequence of landmarks for the video
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            data_aux_list = [0.0] * 126  # Default zeros for no hands
            if results.multi_hand_landmarks:
                data_aux_list = []  # Reset for detected
                for hand_landmarks in results.multi_hand_landmarks:
                    if len(hand_landmarks.landmark) != 21:
                        print(f"Warning: Irregular landmarks ({len(hand_landmarks.landmark)}) in {video_full_path}, frame {frame_count}")
                        continue  # Skip malformed
                    x_ = [lm.x for lm in hand_landmarks.landmark]
                    y_ = [lm.y for lm in hand_landmarks.landmark]
                    z_ = [lm.z for lm in hand_landmarks.landmark]
                    min_x, min_y, min_z = min(x_), min(y_), min(z_)
                    data_aux = []
                    for lm in hand_landmarks.landmark:
                        data_aux.extend([lm.x - min_x, lm.y - min_y, lm.z - min_z])
                    data_aux_list.extend(data_aux)
                
                # Pad if only one hand
                if len(results.multi_hand_landmarks) == 1:
                    data_aux_list.extend([0.0] * 63)
                
                # Ensure exactly 126
                if len(data_aux_list) != 126:
                    print(f"Warning: Features len {len(data_aux_list)} !=126 in {video_full_path}, frame {frame_count}. Padding.")
                    data_aux_list += [0.0] * (126 - len(data_aux_list))  # Pad short
                    data_aux_list = data_aux_list[:126]  # Truncate long (rare)
            
            sequence.append(data_aux_list)
            frame_count += 1
        
        cap.release()
        if sequence:
            data.append(sequence)
            labels.append(dir_)
            video_count += 1
        else:
            print(f"No frames in: {video_full_path}")
    print(f"Processed {video_count} videos for {dir_}")

print(f"Total samples: {len(data)}, Total labels: {len(labels)}")
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()