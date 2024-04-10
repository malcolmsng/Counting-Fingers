import pickle
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader 
import cv2
import mediapipe as mp
import numpy as np
from typing import Union

class PosePredictor():

    def __init__(self):
        print('its predictin time')
    def predict(self, model, labels: list[Union[str, int]], label_map: dict[int,str]):
        
        cap = cv2.VideoCapture(0)

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        # TODO: account for exception properly
        
        while True:

            landmark_coords = []
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
                        landmark_coords.append(x - min(x_))
                        landmark_coords.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10
                try:
                    
                    prediction = model.predict([np.asarray(landmark_coords)])
                    print(prediction)
                    predicted_character = label_map[int(prediction[0])]
                    print(predicted_character)
                except Exception as e:
                    device = 'cuda'if torch.cuda.is_available()  else 'cpu'
                    x = torch.tensor(landmark_coords).to(device) 
                    prediction =  torch.argmax(torch.softmax(model(x), dim = -1))
                    predicted_character = label_map[prediction.item()]
                
                
                

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if  key == ord('q'):
                break
            elif key == ord("p"):
                cv2.waitKey(0)
            elif key == ord("e"):
                cap.release()
                break
            # create directory
