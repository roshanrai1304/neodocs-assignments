import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle

#abstract class
class InverseDataStrategy(ABC):
    
    """
    Abstract class for Inverse processing data
    """
    
    @abstractmethod
    def inverse_data(self, X_test: np.ndarray, y_true:np.ndarray, y_predict:np.ndarray, scaler_X:StandardScaler, scaler_y:StandardScaler) -> pd.DataFrame:
        pass
    
    
class InverseDataProcessing(InverseDataStrategy):
    
    """
    Strategy for inverse processing data
    """
    
    def inverse_data(self, X_test: pd.DataFrame, y_true:np.ndarray, y_pred:np.ndarray) -> pd.DataFrame:
        
        """
        Inverse process the data to its original values
        
        Args:
         data: np.ndarray
        
        Returns:
             data (pd.DataFrame) -> Unscaled data
        """
        
        try:
            
            model_cols = X_test.iloc[:, :8].reset_index(drop=True)
            features = X_test.iloc[:, 8:]
            features_cols = features.columns
            with open("scaler/scaling_X.pkl", "rb") as f:
                scaling_X = pickle.load(f)
            with open("scaler/scaling_y.pkl", "rb") as f:
                scaling_y = pickle.load(f)
                
            features_unscaled = pd.DataFrame(scaling_X.inverse_transform(features), columns=features_cols).reset_index(drop=True)
            data = pd.concat([model_cols, features_unscaled], axis=1)
            
            y_true = pd.Series(scaling_y.inverse_transform(y_true.reshape(-1, 1)).squeeze()).reset_index(drop=True)
            y_pred = pd.Series(scaling_y.inverse_transform(y_pred.reshape(-1, 1)).squeeze()).reset_index(drop=True)
            
            data['hbvalues_true'] = y_true
            data['hbvalues_pred'] = y_pred
            data.to_csv("predicted.csv")
            return data
            
        except Exception as e:
            logging.error("Error in inverse processing the data {}".format(e))     