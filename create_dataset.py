import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# initializes a mp Hands object --> processes an rgb image and returns hand landmarks
#                (self,
#                static_image_mode=False,
#                max_num_hands=2,
#                model_complexity=1,
#                min_detection_confidence=0.5,
#                min_tracking_confidence=0.5):
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
# each class landmark data
data = []
labels = []
# list class folders
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        # list of landmarks for each frame in 1 class
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # colors in open cv are normally BGR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # process RGB image and returns handland marks of each detected hand
        results = hands.process(img_rgb)
        # if hand detected
        if results.multi_hand_landmarks:
            # for each hand
            for hand_landmarks in results.multi_hand_landmarks:
                # for every landmark on each hand
                # each landmark has coordinates (x,y,z)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    # recentering
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
