import cv2
import mediapipe as mp
import pandas as pd

# Đọc ảnh từ webcam
cap = cv2.VideoCapture(0)

# Khởi tạo thư viện mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

lm_list = []
label = "tamgiac"
no_of_frames = 200

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
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while len(lm_list) <= no_of_frames:
        ret, frame = cap.read()
        if ret:
            # Nhận diện hand
            frame.flags.writeable = False
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_hand_landmarks:
                # Ghi nhận thông số khung xương
                lm = make_landmark_timestep(results)
                print(len(lm))
                lm_list.append(lm)
                # Vẽ khung xương lên ảnh
                frame = draw_landmark_on_image(results, frame)

            cv2.imshow("image", frame)
            if cv2.waitKey(1) == ord('q'):
                break

# Write vào file csv
df = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")
cap.release()
cv2.destroyAllWindows()