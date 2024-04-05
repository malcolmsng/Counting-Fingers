import cv2
import mediapipe as mp
from VideoDataset import VideoDataset
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
# while True:
# # in range(10):
#     ret, frame = cap.read()
#     # print(frame.shape)
    
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         [mp_drawing.draw_landmarks(
#                 frame,  # image to draw
#                 hand_landmarks,  # model output
#                 mp_hands.HAND_CONNECTIONS,  # hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style()) for hand_landmarks in results.multi_hand_landmarks]
    
#     cv2.imshow('frame', frame)
#     key = cv2.waitKey(1)
#     if key == ord("p"):
#         cv2.waitKey(0)
#     elif key == ord("e"):
#         cap.release()
#         break

dataset = VideoDataset(class_names=['one','two','three'], n_classes=3, n_frames=10)
# dataset.capture_video()
data,labels = dataset.get_landmark_dataset()
import numpy as np
print(np.array(data).shape)
