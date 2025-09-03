import os
import cv2
import numpy as np  # For augmentation
import time  # For timeouts

DATA_DIR = './data_words'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

unique_clips = 10  # Number of unique videos to capture per word
variations = 5  # Number of variations per unique video
dataset_size = unique_clips * variations  # Total 50

print("Instructions:")
print("Enter the word/phrase to capture (e.g., 'hello').")
print("A 5-second preview countdown will start; recording begins automatically after.")
print("You can press 'q' during preview to start recording early.")
print("Enter 'q' to quit when prompted for a word.")

while True:
    word = input("Enter the word/phrase to capture (or 'q' to quit): ").strip().lower()
    if word == 'q':
        break
    if not word:  # Skip empty input
        print("No word entered. Try again.")
        continue

    word_dir = os.path.join(DATA_DIR, word)
    if not os.path.exists(word_dir):
        os.makedirs(word_dir)

    existing_videos = len(os.listdir(word_dir))
    print(f"Collecting videos for word '{word}'. Existing videos: {existing_videos}")

    # Open camera fresh for each word
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        continue  # Skip to next prompt instead of exit

    count = 0
    unique_videos = []  # Store paths of unique captures
    while count < unique_clips:
        print(f"Prepare to sign '{word}'. Preview starting for clip {count + 1}...")

        # Preview with 5-second timeout
        start_time = time.time()
        preview_duration = 5  # Seconds
        while time.time() - start_time < preview_duration:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame during preview.")
                break
            remaining = int(preview_duration - (time.time() - start_time))
            cv2.putText(frame, f"Sign '{word}' - Starting in {remaining}s (or press 'q')", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Preview', frame)
            key = cv2.waitKey(25)
            if key == ord('q'):
                print("Manual start triggered.")
                break
            elif key == 27:  # Esc to abort clip
                print("Aborted clip.")
                break

        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Process close

        # Record 3-second video (90 frames at 30fps)
        try:
            video_path = os.path.join(word_dir, f'{existing_videos + count}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

            frame_count = 0
            while frame_count < 90:
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to capture frame during recording at frame {frame_count}.")
                    break
                out.write(frame)
                cv2.putText(frame, f"Recording: {frame_count + 1}/90", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Recording', frame)
                cv2.waitKey(1)
                frame_count += 1

            out.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Process close

            if frame_count < 90:
                print(f"Warning: Clip for '{word}' stopped early at {frame_count} frames.")
            else:
                unique_videos.append(video_path)
                count += 1
                print(f"Captured unique video clip {count} for '{word}'.")
        except Exception as e:
            print(f"Error during recording: {e}")

    # Release camera after captures
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # Duplicate/augment to reach dataset_size
    for idx, unique_path in enumerate(unique_videos):
        try:
            cap_aug = cv2.VideoCapture(unique_path)
            frames = []
            while cap_aug.isOpened():
                ret, frame = cap_aug.read()
                if not ret:
                    break
                frames.append(frame)
            cap_aug.release()

            if not frames:
                print(f"Warning: No frames read from {unique_path}. Skipping augmentation.")
                continue

            for v in range(variations):
                aug_frames = frames.copy()
                # Variation 1: Flip horizontally
                if v == 0:
                    aug_frames = [cv2.flip(f, 1) for f in aug_frames]
                # Variation 2: Add noise
                elif v == 1:
                    aug_frames = [np.clip(f + np.random.normal(0, 10, f.shape), 0, 255).astype(np.uint8) for f in aug_frames]
                # Variation 3: Brightness/contrast adjustment
                elif v == 2:
                    alpha = np.random.uniform(0.8, 1.2)  # Contrast
                    beta = np.random.uniform(-20, 20)   # Brightness
                    aug_frames = [cv2.convertScaleAbs(f, alpha=alpha, beta=beta) for f in aug_frames]
                # Variation 4: Rotation (±5 degrees)
                elif v == 3:
                    angle = np.random.uniform(-5, 5)
                    h, w = frames[0].shape[:2]
                    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                    aug_frames = [cv2.warpAffine(f, M, (w, h)) for f in aug_frames]
                # Variation 5: Speed variation (subsample for faster, pad if short)
                elif v == 4:
                    aug_frames = aug_frames[::2]  # Skip every other frame
                    while len(aug_frames) < 90:
                        aug_frames.append(aug_frames[-1])  # Pad with last frame

                aug_path = os.path.join(word_dir, f'{existing_videos + (idx * variations) + v + unique_clips}.mp4')  # Unique naming
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(aug_path, fourcc, 30.0, (aug_frames[0].shape[1], aug_frames[0].shape[0]))
                for frame in aug_frames:
                    out.write(frame)
                out.release()
                print(f"Created variation {v+1} for unique clip {idx+1}.")

        except Exception as e:
            print(f"Error during augmentation of {unique_path}: {e}")

    print(f"Augmented to {dataset_size} videos for '{word}'.")
    print(f"Completed '{word}'. Ready for next word or 'q' to quit.")  # Feedback

# Final release if loop exits
cv2.destroyAllWindows()