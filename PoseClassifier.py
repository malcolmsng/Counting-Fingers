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


class PoseClassifier():

    def __init__(self, dataset: VideoDataset, model: int = 0):
        self.dataset = dataset
        # 0 - all
        # 1 - randomforest only
        # 2 - SVM only
        # 3 - MLP only
        self.model = model
        self.model_list = []
    def train_randomforest(self):
        dataset = self.dataset.get_landmark_dataset()
        
        data = np.asarray(dataset['data'])
        labels = np.asarray(dataset['labels'])

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

        model = RandomForestClassifier()

        train_acc = model.fit(X_train, y_train).score(X_train, y_train)

        y_predict = model.predict(X_test)

        test_acc = accuracy_score(y_predict, y_test)
        print(f'training accuracy: {train_acc}\ntesting accuracy: {test_acc}')
        self.save_model('randomforest', (model,test_acc))
        
    def train_svm(self):
        dataset = self.dataset.get_landmark_dataset()
        
        data = np.asarray(dataset['data'])
        labels = np.asarray(dataset['labels'])

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

        model = SVC(gamma = 'auto')

        train_acc = model.fit(X_train, y_train).score(X_train, y_train)

        y_predict = model.predict(X_test)

        test_acc = accuracy_score(y_predict, y_test)
        print(f'training accuracy: {train_acc}\ntesting accuracy: {test_acc}')
        # print('{}% of samples were classified correctly !'.format(score * 100))

        self.save_model('svm', model)
    
    def train_mlp(self, model: nn.Module, epochs:int = 5):
        device = 'cuda'if torch.cuda.is_available()  else 'cpu' 
        dataset = self.dataset.get_landmark_dataset()
        results = {
      'train_loss': [],
      'train_acc': [],
      'test_loss': [],
      'test_acc': []
  }
        data = np.asarray(dataset['data'])
        labels = np.asarray(dataset['labels'])

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        train_data = []
        test_data = []
        for i in range(len(X_train)):
            train_data.append([X_train[i], y_train[i]])
        for i in range(len(X_test)):
            test_data.append([X_test[i], y_test[i]])
        train_dataloader = []
        test_dataloader = DataLoader(train_data, batch_size=1)

        n_class = np.unique(labels)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam()
        model = model
        model.train()
        train_loss, train_acc = 0,0
        
        for epoch in range(epochs):
            for batch, (X,y) in enumerate(train_dataloader):
            # Send data to target device
                X, y = X.to(device), y.to(device)
                # forward
                y_pred = model(X)
                # calc loss
                loss = loss_fn(y_pred,y)
                train_loss += loss.item()
                # reset optimizer
                optimizer.zero_grad()
                # backprop
                loss.backward()
                # next
                optimizer.step()
                # calc accuracy
                y_pred_class = torch.argmax(torch.softmax(y_pred,dim = 1), dim =1)
                train_acc += (y==y_pred_class).sum().item()/len(y_pred)
            train_loss = train_loss/len(train_dataloader)
            train_acc = train_acc/len(train_dataloader)
            model.eval()

            test_loss, test_acc = 0,0
            # iterate through data loader
            with torch.inference_mode():
                for batch, (X,y) in enumerate(test_dataloader):
                # Send data to target device
                    X, y = X.to(device), y.to(device)
                    # forward
                    test_pred = model(X)
                    # calc loss
                    loss = loss_fn(test_pred,y)
                    test_loss += loss.item()
                    # calc accuracy
                    test_pred_labels = test_pred.argmax(dim=1)
                    test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels)


            test_loss = test_loss/len(test_dataloader)
            test_acc = test_acc/len(test_dataloader)
            print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
            # update res
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
    # TODO: Try out transformer
    # TODO: Try out voting classifier     
    def create_dl_model(n_classes: int,model_type: str = 'mlp') -> nn.Module: 
        if model_type =='mlp':
            class BabyEncoder(nn.Module):
                def __init__(self, n_classes:int):
                    # always will have 42 features --> 21 each hand
                    super().__init__()
                    in_features = 42
                    self.layer_1 = nn.Sequential(
                        nn.Linear(in_features= in_features, out_features= in_features//2),
                        nn.LeakyReLU(),
                        nn.Dropout()
                    )
                    in_features //= 2
                    self.layer_2 = nn.Sequential(
                        nn.Linear(in_features= in_features, out_features= in_features//2),
                        nn.LeakyReLU(),
                        nn.Dropout()
                    )
                    in_features //= 2
                    self.layer_3 = nn.Sequential(
                        nn.Linear(in_features= in_features, out_features= n_classes),
                        nn.LeakyReLU()
                    )
                def forward(self, x: torch.Tensor):
                    x = self.layer_1(x)
                    x = self.layer_2(x)
                    x = self.layer_3(x)
                    return x
            model = BabyEncoder(n_classes = n_classes)
            return model


    def train_all(self):
        return
    def predict(self):
        return
    def save_model(self, name:str,model):
        with open('model.pickle', 'wb') as file:
            pickle.dump({name: model}, file)
            file.close()