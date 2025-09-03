import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import tensorflow as tf
import warnings
import time  # For timeouts
import logging  # For debugging

# Setup logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for more info; change to ERROR for less

# Suppress warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide TF/TFLite logs
warnings.filterwarnings('ignore', category=UserWarning)

class SignRecognizer:
    def __init__(self, model_path='model.h5', encoder_path='label_encoder.pkl', sequence_length=90, confidence_threshold=0.8):  # Increased threshold
        self.model = load_model(model_path)
        # Recompile to build metrics and silence warning
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)  # Increased for better filtering
        self.frame_buffer = deque(maxlen=sequence_length)
        self.sentence = []
        self.confidence_threshold = confidence_threshold
        self.no_hand_counter = 0  # For buffer timeout
        self.max_no_hand_frames = 60  # Increased tolerance for movement

    def get_sentence(self):
        return ' '.join(self.sentence)

    def generate_frames(self):
        cap = cv2.VideoCapture(0)
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(error_frame, "Camera read failed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    _, buffer = cv2.imencode('.jpg', error_frame)
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    continue

                # Optional: Resize frame to "push back" (simulate zoom-out for better distant detection)
                frame = cv2.resize(frame, (int(frame.shape[1] * 0.8), int(frame.shape[0] * 0.8)))  # Scale down 20%; adjust as needed

                H, W, _ = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)

                data_aux_list = []  # For multi-hand landmarks
                if results.multi_hand_landmarks:
                    self.no_hand_counter = 0
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                                       self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                                       self.mp_drawing_styles.get_default_hand_connections_style())
                        x_ = [lm.x for lm in hand_landmarks.landmark]
                        y_ = [lm.y for lm in hand_landmarks.landmark]
                        z_ = [lm.z for lm in hand_landmarks.landmark]
                        min_x, min_y, min_z = min(x_), min(y_), min(z_)
                        data_aux = []
                        for lm in hand_landmarks.landmark:
                            data_aux.extend([lm.x - min_x, lm.y - min_y, lm.z - min_z])
                        data_aux_list.extend(data_aux)

                    if len(results.multi_hand_landmarks) == 1:
                        data_aux_list.extend([0.0] * (21 * 3))

                    self.frame_buffer.append(data_aux_list)

                    if results.multi_hand_landmarks:
                        hand_lm = results.multi_hand_landmarks[0]
                        x_ = [lm.x for lm in hand_lm.landmark]
                        y_ = [lm.y for lm in hand_lm.landmark]
                        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                        x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Predict when buffer full
                    if len(self.frame_buffer) == self.frame_buffer.maxlen:
                        input_seq = np.array([self.frame_buffer])
                        prediction = self.model.predict(input_seq)
                        max_prob = np.max(prediction)
                        logging.debug(f"Prediction prob: {max_prob}")  # Log for debugging
                        if max_prob > self.confidence_threshold:
                            predicted_word = self.label_encoder.inverse_transform([np.argmax(prediction)])[0]
                            self.sentence.append(predicted_word)
                            logging.debug(f"Appended: {predicted_word}")  # Log successful appends
                        self.frame_buffer.clear()

                else:
                    self.no_hand_counter += 1
                    if self.no_hand_counter > self.max_no_hand_frames:
                        self.frame_buffer.clear()
                        self.no_hand_counter = 0

                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                else:
                    raise ValueError("Frame encoding failed")

            except Exception as e:
                logging.error(f"Full exception: {str(e)}")  # Log full error
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"Error: {str(e)[:50]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                _, buffer = cv2.imencode('.jpg', error_frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        cap.release()

# Instantiate the recognizer
recognizer = SignRecognizer()

# Expose functions for app.py import
def generate_frames():
    return recognizer.generate_frames()

def get_sentence():
    return recognizer.get_sentence()