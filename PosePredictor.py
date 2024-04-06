import pickle
import VideoDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader 
import cv2
import mediapipe as mp
import numpy as np

class PosePredictor():

    def __init__(self, dataset: VideoDataset):
        self.dataset = dataset
    def predict(model: object):
        
        cap = cv2.VideoCapture(2)

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        # TODO: account for labels
        # TODO: account for 
        labels_dict = {0: 'A', 1: 'B', 2: 'L'}
        while True:

            data_aux = []
            x_ = []
            y_ = []

            ret, frame = cap.read()
            # H W Channel shape
            H, W, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # (
            # image: np.ndarray,
            # landmark_list: landmark_pb2.NormalizedLandmarkList,
            # connections: Optional[List[Tuple[int, int]]] = None,
            # landmark_drawing_spec: Union[DrawingSpec,
            #                              Mapping[int, DrawingSpec]] = DrawingSpec(
            #                                  color=RED_COLOR),
            # connection_drawing_spec: Union[DrawingSpec,
            #                                Mapping[Tuple[int, int],
            #                                        DrawingSpec]] = DrawingSpec(),
            # is_drawing_landmarks: bool = True):
                    mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                except:
                    prediction =  torch.argmax(torch.softmax(model(torch.tensor(data_aux))))
                predicted_character = labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

            cv2.imshow('frame', frame)
            cv2.waitKey(1)
