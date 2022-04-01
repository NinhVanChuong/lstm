import cv2
import mediapipe as mp
import pandas as pd
import time
import os
import numpy as np

# Đọc ảnh từ webcam
cap = cv2.VideoCapture(0)

# Khởi tạo thư viện mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Hand_Data')
# DATA_PATH_CSV = os.path.join('Hand_Data_CSV')

# Actions that we try to detect
actions = np.array(['dua'])

# Thirty videos worth of data
no_sequences = 20

# Videos are going to be 30 frames in length
sequence_length = 20

# Folder start
start_folder = 50

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

lm_lists = []
lm_list = []

prev_frame_time = 0
new_frame_time = 0


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def make_landmark_timestep(results):
    c_lm = []
    for handLms in results.multi_hand_landmarks:
        for id, lm in enumerate(handLms.landmark):
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
            # c_lm.append(lm.visibility)
    #Muon luu file numpy thi can phai flatten()
    c_lm = np.array(c_lm).flatten()
    return c_lm


def draw_landmark_on_image(results, image):
    # Vẽ các đường nối
    image.flags.writeable = False
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    return image

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    for action in actions:
        ret, image = cap.read()
        if ret:
        # Loop through sequences aka videos
            while cv2.waitKey(1000) & 0xFF != ord(' '):
                cv2.putText(image, 'Press Space - {}'.format(action), (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)

        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            keypoints=[]
            for frame_num in range(sequence_length):
                # Read feed
                ret, frame = cap.read()
                if ret:
                    # Make detections
                    frame=cv2.flip(frame,1)
                    image, results = mediapipe_detection(frame, hands)

                    if results.multi_hand_landmarks is None:
                        new_frame_time = time.time()
                        fps = 1 / (new_frame_time - prev_frame_time)
                        prev_frame_time = new_frame_time

                        cv2.putText(image, str(int(fps)), (400, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 2,
                                    cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)

                        # Break gracefully
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                        continue

                    # Draw landmarks
                    draw_landmark_on_image(results,image)

                    # NEW Apply wait logic
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, '{} {}'.format(action, sequence), (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                        # Show to screenqqqqqqqq
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, '{} {}'.format(action, sequence), (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                        # Show to screen
                        cv2.putText(image, str(frame_num), (200, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)

                    # NEW Export keypoints
                    keypoints = make_landmark_timestep(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    # Break gracefully
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    cap.release()
    cv2.destroyAllWindows()

