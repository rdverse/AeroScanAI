#!/usr/bin/env python3
# pylint: disable=C0415,E0401,R0914
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import logging
import intel_extension_for_pytorch as ipex
# from ipex.training.MvtecAdDataset import MvtecAdDataset
# from ipex.utils.base_model import AbstractModelInference, AbstractModelTraining
# from ipex.utils.utils import data_augmentation, plot_confusion_matrix, get_bbox_from_heatmap
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score

class TrainModel(): 
    def __init__(self, model, img_dim, n_samples,percenttest, mode) -> None:
        self.train_loader = []
        self.test_loader = []
        self.model = model
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    def load_data(self): 
        n_train_samples = int(self.n_samples * (1 - self.percenttest))
        n_val_samples = int(self.n_samples * (self.percenttest))
        n_test_samples = n_val_samples
        
        train_dataset = simulatedDataset()
        val_dataset = simulatedDataset()
        test_dataset = simulatedDataset()
        loader_args = dict(batch_size=self.batch_size, num_workers=self.num_workers)
        print(loader_args)  
        print(train_dataset)
        
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


    def train(self, epochs=5, target_accuracy=None, learning_rate= 0.0001, data_aug=False):
        
        #criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
        criterion = torch.nn.BCEwithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        try:
            model, optimizer = ipex.optimize(model=self.model, optimizer=optimizer, dtype=torch.float32)
        except Exception as e:
            print("IPEX error : Ignoring IPEX optimization")
            model = self.model
            optimizer = optimizer
            
        self.model.train()

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}:", end=" ")
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
            for inputs, labels in self.train_loader:
                optimizer.zero_grad()
                masks = self.model(inputs)
                loss = criterion(masks, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(masks == labels)
                n_samples += inputs.size(0)

            epoch_loss = running_loss / n_samples
            epoch_acc = running_corrects.double() / n_samples
            print("Loss = {:.4f}, Accuracy = {:.4f}".format(epoch_loss, epoch_acc))

            if target_accuracy is not None:
                if epoch_acc > target_accuracy:
                    print("Early Stopping")
                    break
        
        return epoch_loss, epoch_acc
   
   
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