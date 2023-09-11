#!/usr/bin/env python3
# pylint: disable=C0415,E0401,R0914
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import logging
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import intel_extension_for_pytorch as ipex
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score

from torchipex.training.SimulatedDataset import SimulatedDataset
from torchipex.unet.unet_model import UNet

import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

SCAN_DICT = {"low_defect_scan": {"defect_coverage": 0.15, "random_seed": 9}, 
             "medium_defect_scan": {"defect_coverage": 0.5, "random_seed": 9}, 
             "high_defect_scan": {"defect_coverage": 0.8, "random_seed": 9},
             "random": {"defect_coverage": np.random.randn(), "random_seed": int(np.random.uniform(10,1000))}}
class TrainModel(): 
    # add an argument called mode 
    def __init__(self, active_learning=False, al_threshold=0.5) -> None:
        self.train_loader = []
        self.test_loader = []
        self.model = None
        self.img_dim = None
        self.n_channels = None
        self.n_samples = None
        self.n_classes = None
        self.percent_test = None
        self.batch_size = None
        self.num_workers = None
        self.n_train = None
        self.n_val = None
        self.n_test = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.active_learning = active_learning
        self.al_threshold = al_threshold
    

    def load_model(self, n_channels, n_classes, img_dim):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.img_dim = img_dim
        self.model = UNet(n_channels=self.n_channels, 
                          n_classes=self.n_classes, 
                          img_dim = self.img_dim)
        
    def save_model(self, model_name, model_path):
        torch.save(self.model, os.path.join(model_path,model_name + '.pt'))
        
    def load_model_from_file(self, model_name, model_path):
        if os.path.exists(os.path.join(model_path,model_name + '.pt')):
            self.model = torch.load(os.path.join(model_path,model_name + '.pt'))
        else:
            print("Model does not exist")
            self.load_model(n_channels=self.n_channels,
                            n_classes=self.n_classes,
                            img_dim = self.img_dim)
        
    def load_data(self, 
                  n_samples, 
                  percent_test,
                  batch_size,
                  num_workers):
        self.n_samples = n_samples
        self.percent_test = percent_test  
        self.batch_size = batch_size    
        self.num_workers = num_workers  
        
        n_train_samples = int(self.n_samples * (1 - self.percent_test))
        n_val_samples = int(self.n_samples * (self.percent_test))
        n_test_samples = n_val_samples
        
        train_dataset = SimulatedDataset(img_dim = self.img_dim,
                                         n_channels = self.n_channels,
                                         n_samples = n_train_samples,
                                         defect_coverage=0.75,
                                         random_seed=0)
        val_dataset = SimulatedDataset(img_dim = self.img_dim,
                                         n_channels = self.n_channels,
                                         n_samples = n_val_samples,
                                         defect_coverage=0.75,
                                         random_seed=2)
        test_dataset = SimulatedDataset(img_dim = self.img_dim,
                                         n_channels = self.n_channels,
                                         n_samples = n_test_samples,
                                         defect_coverage=0.75,
                                         random_seed=4)
        loader_args = dict(batch_size=self.batch_size, num_workers=self.num_workers)
        print(loader_args)  
        
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
        if self.active_learning:
            threshold = self.al_threshold
        else:
            threshold = 0.5
        #criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        try:
            self.model, optimizer = ipex.optimize(model=self.model, optimizer=optimizer, dtype=torch.float32)
        except Exception as e:
            print("IPEX error : Ignoring IPEX optimization")
        #     optimizer = optimizer
        
        self.model.train()
        for epoch in range(1, n_epochs + 1):
            print(f"Epoch {epoch}/{n_epochs}:", end=" ")
            running_loss = 0
            running_corrects = 0
            n_samples = 0

            for batch in self.train_loader:
                inputs = batch['data']
                labels = batch['mask']
                # change input shape to (batch_size, n_channels, img_dim, img_dim)
                inputs = torch.swapaxes(inputs, 1, 3)
                
                optimizer.zero_grad()
                masks = self.model(inputs).squeeze()
                
                        # Critical component of active learning
                if self.active_learning:
                    pred_threshold_mask = (masks>threshold)
                    labels = torch.where(pred_threshold_mask, labels, torch.zeros_like(labels)) 
                
                loss = criterion(masks, labels)
                loss.backward()
                optimizer.step()

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
        train_preds, train_labels,_ = self.predict(self.train_loader)
        train_preds = train_preds.reshape(-1)
        train_labels = train_labels.reshape(-1)

        metrics_train["Precision"] = precision_score(train_preds, train_labels, zero_division=0)
        metrics_train["Recall"] = recall_score(train_preds, train_labels, zero_division=0)
        metrics_train["Accuracy"] = accuracy_score(train_preds, train_labels )
        metrics_train["F1-Score"] = f1_score(train_preds, train_labels, average='weighted', zero_division=0)

        # Evaluate on the validation dataset
        val_preds, val_labels,_ = self.predict(self.val_loader)
        val_preds = val_preds.reshape(-1)
        val_labels = val_labels.reshape(-1)
        metrics_val["Precision"] = precision_score(val_preds, val_labels, zero_division=0)
        metrics_val["Recall"] = recall_score(val_preds, val_labels, zero_division=0)
        metrics_val["Accuracy"] = accuracy_score(val_preds, val_labels )
        metrics_val["F1-Score"] = f1_score(val_preds, val_labels, average='weighted', zero_division=0)
    
        # Evaluate on the test dataset
        test_preds, test_labels,_ = self.predict(self.test_loader)
        test_preds = test_preds.reshape(-1)
        test_labels = test_labels.reshape(-1)
        metrics_test["Precision"] = precision_score(test_preds, test_labels, zero_division=0)
        metrics_test["Recall"] = recall_score(test_preds, test_labels, zero_division=0)
        metrics_test["Accuracy"] = accuracy_score(test_preds, test_labels)
        metrics_test["F1-Score"] = f1_score(test_preds, test_labels, average='weighted', zero_division=0)
        
        # Create dataframes for the evaluation metrics
        df_train = pd.DataFrame(metrics_train, index=["Train"])
        df_val = pd.DataFrame(metrics_val, index=["Validation"])
        df_test = pd.DataFrame(metrics_test, index=["Test"])

        # Concatenate the dataframes to create a single dataframe
        evaluation_df = pd.concat([df_train, df_val, df_test])
        evaluation_df.fillna(0, inplace=True)
        return evaluation_df

    def load_single_scan(self, 
                         scan_name):
        self.img_dim = self.img_dim
        self.n_channels = self.n_channels
        self.n_samples = 1
        # did not separate the synthetic data function, so just load dataset
        # and provide the appropriate parameters
        scan_params = SCAN_DICT[scan_name]
        single_dataset = SimulatedDataset(img_dim = self.img_dim,
                                    n_channels = self.n_channels,
                                    n_samples = 1,
                                    defect_coverage=scan_params["defect_coverage"],
                                    random_seed=scan_params["random_seed"])
        single_loader = DataLoader(single_dataset,  drop_last=True, batch_size=1, num_workers=1)
        return single_loader
    
    def predict(self, dataloader=None):
        # Function to generate predictions from the model for a given dataloader
        # active inference and active learning
        threshold = self.al_threshold

        predictions = []
        labels = []
        inputs = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['data']
                # change input shape to (batch_size, n_channels, img_dim, img_dim)
                inputs = torch.swapaxes(inputs, 1, 3)
                masks = self.model(inputs).squeeze()
                predictions.extend(masks.cpu().numpy())
                labels.extend(batch['mask'].cpu().numpy())
                inputs = inputs.cpu().numpy()
        return (np.array(predictions) > threshold).astype(int), np.array(labels), inputs  # Apply threshold for binary prediction
