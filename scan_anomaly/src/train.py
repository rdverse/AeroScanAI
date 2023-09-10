# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
'''
Module to train and prediction using XGBoost Classifier
'''
# !/usr/bin/env python
# coding: utf-8
# pylint: disable=import-error
import sys
import numpy as np
import pandas as pd 
import logging
import warnings
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV  # pylint: disable=C0415
from sklearn.ensemble import RandomForestClassifier  # pylint: disable=C0415
from sklearn import metrics  # pylint: disable=C0415
from sklearnex import patch_sklearn
patch_sklearn()
from data_utils import synthetic_defects

 
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class WaveformProbe():
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.file = ''
        self.X_train = ''
        self.X_test = ''
        self.y_train = ''
        self.y_test = ''
        self.bench_dict = {}
        self.model_path = ''
        self.hypparameters = ''
        
   # def process_data(self, file: str, test_size: int = .25):    
    def process_data(self, img_dim : int, n_channels : int, train_scan : str, test_scan : str, append_path: str): 
        """_summary_
        Parameters
        ----------
        file : str
            _description_
        test_size : int, optional
            _description_, by default .25
        """
        # Generating our data
        logger.info('Reading the dataset from %s...', append_path)
        # try:
        #     data = pd.read_pickle(file)
        # except FileNotFoundError:
        #     # pass
        #     sys.exit(f'Data loading error, file not found at {file}')

        #data = np.random.randint(low=0, high=20, size=(1000, 10))
        #X = data
        #y = np.random.randint(low=0, high=2, size=(1000, 1))
        
        #X = data.drop('defect', axis=1)
        #y = data.defect

#        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)
        self.X_train, self.y_train, pixelmap_train, columns = synthetic_defects(img_dim, n_channels, train_scan, stat_features=False)
        self.X_test, self.y_test, pixelmap_test, columns = synthetic_defects(img_dim, n_channels, test_scan, stat_features=False)
                
    def grid_search(self, ncpu: int = 1):
        """_summary_
        Returns
        -------
        _type_
            _description_
        """
        #Hyperparameters
        params = {
            'n_estimators': [10 ],
            'max_leaf_nodes': [None, 3],
            'max_features': [None, 'sqrt'],
            'max_depth': [None, 3, ]
        }
        self.bench_dict = {}
        # Run grid search
        start_time = time.time()
        model_rf = GridSearchCV(RandomForestClassifier(
                criterion='gini', max_depth=None, n_jobs=ncpu, oob_score=True, random_state=42), param_grid=params, cv=5)
        # fit model
        model_rf.fit(self.X_train, self.y_train)
        # save grid search time
        self.bench_dict["grid_search_time"] = time.time()-start_time
        # get best estimator
        self.model_rf = model_rf.best_estimator_
        # save hyperparameters
        self.hypparameters = self.model_rf.get_params()
        return self.bench_dict
    
    def train(self, ncpu: int = 1):
        """_summary_

        Parameters
        ----------
        ncpu : int, optional
            _description_, by default 1
        """        
        # Set xgboost parameters
        # Run grid search first to load the model
        self.grid_search(ncpu= ncpu)
        # Train the model
        start_time = time.time()
        self.model_rf.fit(self.X_train, self.y_train)
        # save training time        
        self.bench_dict["training_time"] = time.time()-start_time
        return self.bench_dict


    def validate(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        individual_stream_times = []
        preds = {}
        # Run inference on 10 samples
        for i in range(10):  # pylint: disable=C0415,W0612
            print(self.X_test)
            print(len(self.X_test))
            sample_x = self.X_test.loc[i].to_numpy()#[np.random.randint(0, len(self.X_test), 1)]
            sample_x = np.array([sample_x])
            start_time = time.time()
            y_pred = self.model_rf.predict(sample_x)
            individual_stream_times.append(time.time()-start_time)
        avg_stream_time = sum(individual_stream_times) / len(individual_stream_times)
        self.bench_dict["Average_Inference_latency"] = avg_stream_time

        start_time = time.time()
        y_pred = self.model_rf.predict(self.X_test)
        self.bench_dict["Inference_time_test"] = time.time()-start_time
        self.bench_dict["Accuracy_test"] = metrics.accuracy_score(self.y_test, y_pred)
        self.bench_dict["F1_test"] = metrics.f1_score(self.y_test, y_pred, average='macro')
        # dont need to return the predictions
        preds = dict(zip(np.arange(len(y_pred)), y_pred))
        print(preds)
        return self.bench_dict
    
    def save(self, model_path):
        """_summary_

        Parameters
        ----------
        model_path : _type_
            _description_
        """
        self.model_path = model_path +  self.model_name + '.joblib'
        
        logger.info("Saving model")
        with open( self.model_path, "wb") as fh:
            joblib.dump(self.model_rf, fh.name)
        
        return self.model_path
    