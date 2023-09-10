#!/usr/bin/env python3
# pylint: disable=C0415,E0401,R0914
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import logging
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
#import intel_extension_for_pytorch as ipex
# from ipex.training.MvtecAdDataset import MvtecAdDataset
# from ipex.utils.base_model import AbstractModelInference, AbstractModelTraining
# from ipex.utils.utils import data_augmentation, plot_confusion_matrix, get_bbox_from_heatmap
#from torchipex.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score
#import simulatedDataset
#import torch.nn as nn
from torchipex.training.SimulatedDataset import SimulatedDataset
from torchipex.unet.unet_model import UNet
# return none
# def simulatedDataset():
#     return None 
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

class TrainModel(): 
    # add an argument called mode 
    def __init__(self) -> None:
        self.train_loader = []
        self.test_loader = []
        self.model = None
    
    # def load_model(self, n_channels, n_classes, img_dim):
    #     model = UNet(n_channels=n_channels, n_classes=n_classes, img_dim =img_dim)
    def load_model(self, n_channels, n_classes, img_dim):
        self.model = UNet(n_channels=n_channels, 
                          n_classes=n_classes, 
                          img_dim = img_dim)
        
    def load_data(self, n_channels, 
                  n_samples, 
                  img_dim, 
                  percent_test,
                  batch_size,
                  num_workers):
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.percent_test = percent_test  
        self.batch_size = batch_size    
        self.num_workers = num_workers  
        self.img_dim = img_dim
        
        n_train_samples = int(self.n_samples * (1 - self.percent_test))
        n_val_samples = int(self.n_samples * (self.percent_test))
        n_test_samples = n_val_samples
        
        train_dataset = SimulatedDataset(img_dim = self.img_dim,
                                         n_channels = n_channels,
                                         n_samples = n_train_samples,
                                         defect_coverage=0.75,
                                         random_seed=0)
        val_dataset = SimulatedDataset(img_dim = self.img_dim,
                                         n_channels = n_channels,
                                         n_samples = n_val_samples,
                                         defect_coverage=0.75,
                                         random_seed=2)
        test_dataset = SimulatedDataset(img_dim = self.img_dim,
                                         n_channels = n_channels,
                                         n_samples = n_test_samples,
                                         defect_coverage=0.75,
                                         random_seed=4)
        loader_args = dict(batch_size=self.batch_size, num_workers=self.num_workers)
        print(loader_args)  
        #print(train_dataset)
        
        train_loader = DataLoader(train_dataset,  drop_last=True, **loader_args)
        val_loader = DataLoader(val_dataset,  drop_last=True, **loader_args) 
        test_loader = DataLoader(test_dataset, shuffle=True, drop_last=True, **loader_args)
        
        # assign them to class instances 
        (self.train_dataset , self.val_dataset , self.test_dataset) = (train_dataset , val_dataset , test_dataset)
        (self.n_train, self.n_val, self.n_test) = (len(train_dataset), len(val_dataset), len(test_dataset)) 
        (self.train_loader, self.val_loader, self.test_loader) = (train_loader, val_loader, test_loader)
        
        logging.info(f'''Loaded data:
            Training size:   {self.n_train}
            Validation size: {self.n_val}
            Test size : {self.n_test},
        ''')


    def train(self, n_epochs=5, target_accuracy=None, learning_rate= 0.0001, data_aug=False):
        threshold = 0.5        
        #criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # try:
        #     model, optimizer = ipex.optimize(model=self.model, optimizer=optimizer, dtype=torch.float32)
        # except Exception as e:
        #     print("IPEX error : Ignoring IPEX optimization")
        #     model = self.model
        #     optimizer = optimizer
            
        self.model.train()
        for epoch in range(1, n_epochs + 1):
            print(f"Epoch {epoch}/{n_epochs}:", end=" ")
            running_loss = 0
            running_corrects = 0
            n_samples = 0
            # for inputs1, labels1 in self.train_loader:
            #     inputs, labels = inputs1, labels1
            #     if data_aug:
            #         print("Applying DataAugmentation----> Flipping/Rotation/"
            #         "Enhancing/Cropping and keeping the Regular images as well")
            #         inputs, labels = data_augmentation(inputs1, labels1)
            #     inputs = inputs.to(self.device)
            #     labels = labels.to(self.device)
            # batch iteration
            for batch in self.train_loader:
                inputs = batch['data']
                labels = batch['mask']
                # change input shape to (batch_size, n_channels, img_dim, img_dim)
                inputs = torch.swapaxes(inputs, 1, 3)
                
                #print("Inputs shape is : ", inputs.shape)
                #print("Mask shape is : ", labels.shape)
                
                optimizer.zero_grad()
                masks = self.model(inputs).squeeze()
                loss = criterion(masks, labels)
                loss.backward()
                optimizer.step()
                print(masks)
                print("predictions sum is : ", torch.sum(masks))
                print("predictions std is : ", torch.std(masks))
                print("predictions mean is : ", torch.mean(masks))
                
                # loss can take unthresholded masks
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum((masks>threshold) == labels)
                n_samples += inputs.shape[0]*inputs.shape[2]*inputs.shape[3] # n_channels*img_dim*img_dim - since we are computing accuracy on a pixel level 

            epoch_loss = running_loss / n_samples
            epoch_acc = running_corrects.double() / n_samples
            print("Loss = {:.4f}, Accuracy = {:.4f}".format(epoch_loss, epoch_acc))

            if target_accuracy is not None:
                if epoch_acc > target_accuracy:
                    print("Early Stopping")
                    break
        return epoch_loss, epoch_acc
    
    def evaluate(self):
        # Create empty dictionaries to store evaluation metrics
        metrics_train = {}
        metrics_val = {}
        metrics_test = {}

        # Evaluate on the training dataset
        train_preds, train_labels = self.predict(self.train_loader)
        train_preds = train_preds.reshape(-1)
        train_labels = train_labels.reshape(-1)
        print("train_preds shape is : ", train_preds.shape)
        print("train_labels shape is : ", train_labels.shape)
        #print(train_preds)
        #print(train_labels)
        metrics_train["Precision"] = precision_score(train_preds, train_labels)
        metrics_train["Recall"] = recall_score(train_preds, train_labels)
        metrics_train["Accuracy"] = accuracy_score(train_preds, train_labels)
        metrics_train["F1-Score"] = f1_score(train_preds, train_labels)

        # Evaluate on the validation dataset
        val_preds, val_labels = self.predict(self.val_loader)
        val_preds = val_preds.reshape(-1)
        val_labels = val_labels.reshape(-1)
        metrics_val["Precision"] = precision_score(val_preds, val_labels)
        metrics_val["Recall"] = recall_score(val_preds, val_labels)
        metrics_val["Accuracy"] = accuracy_score(val_preds, val_labels)
        metrics_val["F1-Score"] = f1_score(val_preds, val_labels)
        
        # Evaluate on the test dataset
        test_preds, test_labels = self.predict(self.test_loader)
        test_preds = test_preds.reshape(-1)
        test_labels = test_labels.reshape(-1)
        metrics_test["Precision"] = precision_score(test_preds, test_labels)
        metrics_test["Recall"] = recall_score(test_preds, test_labels)
        metrics_test["Accuracy"] = accuracy_score(test_preds, test_labels)
        metrics_test["F1-Score"] = f1_score(test_preds, test_labels)
        
        # Create dataframes for the evaluation metrics
        df_train = pd.DataFrame(metrics_train, index=["Train"])
        df_val = pd.DataFrame(metrics_val, index=["Validation"])
        df_test = pd.DataFrame(metrics_test, index=["Test"])

        # Concatenate the dataframes to create a single dataframe
        evaluation_df = pd.concat([df_train, df_val, df_test])

        return evaluation_df

    def predict(self, dataloader):
        # Function to generate predictions from the model for a given dataloader
        predictions = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['data']
                # change input shape to (batch_size, n_channels, img_dim, img_dim)
                inputs = torch.swapaxes(inputs, 1, 3)
                masks = self.model(inputs).squeeze()
                predictions.extend(masks.cpu().numpy())
                labels.extend(batch['mask'].cpu().numpy())
        return (np.array(predictions) > 0.5).astype(int), np.array(labels)  # Apply threshold for binary prediction

        
   
    # # PRACTICALLY DONT NEED THIS FUNCTION IN ACTIVE INFERENCE 
    # def evaluate(self):
    #     """
    #     This module will be responsible for evaluating the trained model and calculate the accuracy
    #     Script to evaluate a model after training.
    #     Outputs accuracy and balanced accuracy, draws confusion matrix.
    #     """

    #     self.model.to(self.device)
    #     self.model.eval()

    #     y_true = np.empty(shape=(0,))
    #     y_pred = np.empty(shape=(0,))

    #     for inputs, labels in self.test_loader:
    #         inputs = inputs.to()
    #         labels = labels.to()
    #         start_time = time.time()
    #         preds_probs = self.model(inputs)[0]
    #         infer_time = time.time()-start_time
    #         print('infer_time_per_sample=', infer_time)
    #         preds_class = torch.argmax(preds_probs, dim=-1)
    #         labels = labels.to("cpu").numpy()
    #         preds_class = preds_class.detach().to("cpu").numpy()
    #         y_true = np.concatenate((y_true, labels))
    #         y_pred = np.concatenate((y_pred, preds_class))

    #     accuracy = f1_score(y_true, y_pred)
    #     balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    #     print("f1 Accuracy Score: ", accuracy)
    #     print("Balanced Accuracy: ", balanced_accuracy)

    #     return accuracy, balanced_accuracy
    
    # # NOT LOCALIZING ANYTHING IN ACTIVE INFERENCE
    # def predict_localize(self, thres=0.8, n_samples=9, show_heatmap=False):
    #     """
    #     Runs predictions for the samples in the dataloader.
    #     Shows image, its true label, predicted label and probability.
    #     If an anomaly is predicted, draws bbox around defected region and heatmap.
    #     """
    #     self.model.to(self.device)
    #     self.model.eval()

    #     transform_to_pil = transforms.ToPILImage()

    #     n_cols = 4
    #     n_rows = int(np.ceil(n_samples / n_cols))
    #     plt.figure(figsize=[n_cols * 5, n_rows * 5])

    #     counter = 0
    #     for inputs, _ in self.test_loader:
    #         inputs = inputs.to(self.device)
    #         out = self.model(inputs)
    #         _, class_preds = torch.max(out[0], dim=-1)
    #         feature_maps = out[1].to(self.device)

    #         for img_i in range(inputs.size(0)):
    #             img = transform_to_pil(inputs[img_i])
    #             class_pred = class_preds[img_i]
    #             heatmap = feature_maps[img_i][self.neg_class].detach().cpu().numpy()

    #             counter += 1
    #             plt.subplot(n_rows, n_cols, counter)
    #             plt.imshow(img)
    #             plt.axis("off")

    #             if class_pred == self.neg_class:
    #                 x_0, y_0, x_1, y_1 = get_bbox_from_heatmap(heatmap, thres)
    #                 rectangle = Rectangle(
    #                     (x_0, y_0),
    #                     x_1 - x_0,
    #                     y_1 - y_0,
    #                     edgecolor="red",
    #                     facecolor="none",
    #                     lw=3,
    #                 )
    #                 plt.gca().add_patch(rectangle)
    #                 if show_heatmap:
    #                     plt.imshow(heatmap, cmap="Reds", alpha=0.3)

    #             if counter == n_samples:
    #                 plt.tight_layout()
    #                 plt.show()
    #                 return
    
    # ADD ADDITIONAL LAYERS HERE IF USING BOTH ACTIVE AND SUPERVISED LEARNING
    def save_model(self, model_path):
        torch.save(self.model, model_path)