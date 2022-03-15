import cv2
import mediapipe as mp
import pandas as pd
import tensorflow as tf
import numpy as np
import threading

# Đọc ảnh từ webcam
cap = cv2.VideoCapture(0)

# Khởi tạo thư viện mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

lm_list = []
label = "Unknow"

model = tf.keras.models.load_model("model.h5")

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results = model.predict(lm_list)
    print(results)
    '''[1, 0, 0]=traiphai -tron    0, 1, 0=lenxuong  -vuong      0, 0, 1=dung -tamgiac'''
    if results[0][0]>0.8:
        label='tron'
    elif results[0][1]>0.8:
        label='vuong'
    elif results[0][2]>0.8:
        label='tamgiac'
    else:
        label='Unknow'
    return label

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def make_landmark_timestep(results):
    c_lm = []
    for handLms in results.multi_hand_landmarks:

        for id, lm in enumerate(handLms.landmark):
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
            c_lm.append(lm.visibility)
        return c_lm

def draw_landmark_on_image(results, image):
    # Vẽ các đường nối
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    return image

i = 0
warmup_frames = 60

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while True:
        i = i + 1
        ret, frame = cap.read()
        if ret:
            # Nhận diện hand
            frame.flags.writeable = False
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            if i > warmup_frames:
                if results.multi_hand_landmarks:
                    # Ghi nhận thông số khung xương
                    lm = make_landmark_timestep(results)
                    lm_list.append(lm)
                    if len(lm_list) == 10:
                        # predict
                        label = 'Unknow'
                        t1 = threading.Thread(target=detect, args=(model, lm_list,))
                        t1.start()
                        lm_list = []

                    frame = draw_landmark_on_image(results, frame)

            draw_class_on_image(label,frame)
            cv2.imshow("image", frame)
            if cv2.waitKey(1) == ord('q'):
                break

# Write vào file csv
cap.release()
cv2.destroyAllWindows()