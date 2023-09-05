'''
Module to train and prediction using XGBoost Classifier
'''
# !/usr/bin/env python
# coding: utf-8
# pylint: disable=import-error
import daal4py as d4p
import sys
import numpy as np
import xgboost as xgb
import pandas as pd 
import logging
import warnings
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearnex import patch_sklearn
patch_sklearn()

from data_utils import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class DefectClassify():
    
    def __init__(self):
        self.append_file = ''
        self.y_train = ''
        self.y_test = ''
        self.X_train_scaled_transformed = ''
        self.X_test_scaled_transformed = ''
        self.d4p_model = ''
        self.accuracy_score = ''
        self.model_path = ''
        self.parameters = ''
        self.robust_scaler = ''
        
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
        
        if append_path != '':
            print("Append file is specified")
            try:
                data = pd.read_pickle(append_path)
            except FileNotFoundError:
                print("Append file not found")
                #sys.exit(f'Data loading error, file not found at {file}')
       
        # TRAINING DATA
        X_train, self.y_train, pixelmap_train, columns = synthetic_defects(img_dim, n_channels, train_scan)
        X_test, self.y_test, pixelmap_test, columns = synthetic_defects(img_dim, n_channels, test_scan)
        # synthetic data
            
        df_num_train = X_train.select_dtypes(['float', 'int', 'int32'])
        df_num_test = X_test.select_dtypes(['float', 'int', 'int32'])
        self.robust_scaler = RobustScaler()
        X_train_scaled = self.robust_scaler.fit_transform(df_num_train)
        X_test_scaled = self.robust_scaler.transform(df_num_test)

        # Making them pandas dataframes
        self.X_train_scaled_transformed = pd.DataFrame(X_train_scaled,
                                                  index=df_num_train.index,
                                                  columns=df_num_train.columns)
        self.X_test_scaled_transformed = pd.DataFrame(X_test_scaled,
                                                 index=df_num_test.index,
                                                 columns=df_num_test.columns)
        
    def train(self, ncpu: int = 1):
        """_summary_

        Parameters
        ----------
        ncpu : int, optional
            _description_, by default 1
        """
        # Set xgboost parameters
        self.parameters = {
        'max_bin': 256,
        'scale_pos_weight': 2,
        # l2 regularization - reduces overfitting
        'lambda_l2': 100,
        # alpha is l1 - it will encourage sparsity
        'alpha': 0.0,
        'max_depth': 8,
        'num_leaves': 2**4,
        'verbosity': 2,
        'objective': 'multi:softmax',
        'learning_rate': 0.1,
        'num_class': 3,
        'nthread': 2,
       # 'feature_importance_type': 'gain',
        #'feature_fraction': 0.9, 
        'rate_dropout':0.2
        }
        
        xgb_train = xgb.DMatrix(self.X_train_scaled_transformed, label=np.array(self.y_train))
        xgb_model = xgb.train(self.parameters, xgb_train, num_boost_round=50)
        self.d4p_model = d4p.get_gbt_model_from_xgboost(xgb_model)
        print(xgb_model.get_fscore())
        return xgb_model.get_fscore()

    def validate(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        daal_predict_algo = d4p.gbt_classification_prediction( 
            nClasses=self.parameters["num_class"],
            resultsToEvaluate="computeClassLabels",
            fptype='float')
            
        daal_prediction = daal_predict_algo.compute(self.X_test_scaled_transformed, self.d4p_model)
        
        daal_errors_count  = np.count_nonzero(daal_prediction.prediction[:, 0] - np.ravel(self.y_test))
        self.d4p_acc = abs((daal_errors_count  / daal_prediction.prediction.shape[0]) - 1)
        

        print('=====> XGBoost Daal accuracy score %f', self.d4p_acc)
        print('DONE')
        #print(f"model predictions are {daal_prediction.prediction}")
        print(self.y_test)
        print(self.y_train)
        return self.d4p_acc
    
    def save(self, model_path, model_name):
        """_summary_

        Parameters
        ----------
        model_path : _type_
            _description_
        """
        self.append_path = model_path +  model_name + '.joblib'
        self.scaler_path = model_path +  model_name + '_scaler.joblib'
        
        logger.info("Saving model")
        with open( self.append_path, "wb") as fh:
            joblib.dump(self.d4p_model, fh.name)
        
        logger.info("Saving Scaler")
        with open( self.scaler_path, "wb") as fh:
            joblib.dump(self.robust_scaler, fh.name)
    
        return self.model_path, self.scaler_path
    