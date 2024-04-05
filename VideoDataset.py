import os
import pickle

import mediapipe as mp
import cv2

class VideoDataset():
    def __init__(self, class_names: list[str], n_classes:int, n_frames:int,
                 path: str = './data', ):
        self.path = path
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.class_names = class_names

    def capture_video(self, camera_id: int = 0, video_width: 
                      int = 640, video_height: int = 480):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        # Video capture window
        cap = cv2.VideoCapture(camera_id)
        # setting size of the capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,video_height)
        
        for j in range(self.n_classes):
            
            print(f'Collecting data for class {self.class_names[j]}')

            while True:
                # grab and retrieve image frame --> display text
                ret, frame = cap.read()
                # display text on image 
                # (image, text, point, fontFace, fontScale, color, thickness, linetype)
                cv2.putText(frame, 'Press Q to start video capture for', 
                            (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(frame, f'{self.class_names[j]}', 
                            (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3,
                            cv2.LINE_AA)
                # displays an image in the specified window
                # (window name, image)
                cv2.imshow('frame', frame)
                key = cv2.waitKey(1)
                if  key == ord('q'):
                    break
                elif key == ord("p"):
                    cv2.waitKey(0)
                elif key == ord("e"):
                    cap.release()
                    break
            # create directory for each class
            if not os.path.exists(os.path.join(self.path, self.class_names[j])):
                os.makedirs(os.path.join(self.path, self.class_names[j]))
            counter = 0
            while counter < self.n_frames:
                ret, frame = cap.read()
                cv2.imshow('frame', frame)
                # (delay) -> code of pressed key or -1 if no key was pressed before delay

                cv2.waitKey(25)
                # creating jpg
                cv2.imwrite(os.path.join(self.path, str(self.class_names[j]), f'{counter}.jpg'), frame)
                counter += 1

    def create_landmark_dataset(self):
        mp_hands = mp.solutions.hands
        # initializes a mp Hands object --> processes an rgb image and returns hand landmarks
        #                (self,
        #                static_image_mode=False,
        #                max_num_hands=2,
        #                model_complexity=1,
        #                min_detection_confidence=0.5,
        #                min_tracking_confidence=0.5):
        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        data = []
        labels = []
        # list class folders
        for dir_ in os.listdir(self.path):
            for img_path in os.listdir(os.path.join(self.path, dir_)):
                # list of landmarks for each frame in 1 class
                landmark_coords = []

                x_ = []
                y_ = []

                img = cv2.imread(os.path.join(self.path, dir_, img_path))
                # colors in open cv are normally BGR
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # process RGB image and returns handland marks of each detected hand
                results = hands.process(img_rgb)
                # if hand detected
                if results.multi_hand_landmarks:
                    # for each hand
                    for hand_landmarks in results.multi_hand_landmarks:
                        # for every handmark on each hand
                        # each handmark has coordinates (x,y,z)
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            
                            x_.append(x)
                            y_.append(y)

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            # recentering
                            landmark_coords.append(x - min(x_))
                            landmark_coords.append(y - min(y_))

                    data.append(landmark_coords)
                    labels.append(dir_)
        with open('data.pickle', 'wb') as file:
            pickle.dump({'data': data, 'labels': labels}, file)
            file.close()
    def get_landmark_dataset(self):
        with open('data.pickle', 'rb') as file:
            file = pickle.load(file)
            data = file['data']
            labels = file['labels']
        return data,labels
dataset = VideoDataset(class_names=['one','two','three'], n_classes=3, n_frames=10)
# # dataset.capture_video()
# data,labels = dataset.get_landmark_dataset()
# print(data)

