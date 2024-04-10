# Hand Pose Estimation

# Example Usage

### Create video dataset with class names, number of classes, and number of frames per class

```

dataset = VideoDataset(class_names=['one','two','three'],n_classes= 3, n_frames=10)
```

### Record video to be used in dataset

```
dataset.capture_video()
dataset.create_landmark_dataset()
```

### Get Coordinates and Labels

```
data,labels = dataset.get_landmark_dataset()
```

### Train model

```
trainer = PoseTrainer(dataset=dataset)
res = trainer.train_svm()
```

### Predict Pose

```
model = res['model']
labels = res['labels']
label_map = res['label_map']
predictor = PosePredictor()
predictor.predict(model=model, labels = labels, label_map= label_map)

```
