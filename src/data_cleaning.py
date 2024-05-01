import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from pandas.core.api import Series as Series

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

"""
    The purpose of abstract methods is to define a method in an abstract base class 
    without providing an implementation. Subclasses are then required to provide their 
    own implementation of the abstract method.
"""

#abstract class
class DataStrategy(ABC): 
    
    """
    Abstract class defining strategy for handling data
    """
    
    """
    The type hint Union[pd.DataFrame, pd.Series] is specifying that a function or 
    method should return either a Pandas DataFrame or a Pandas Series.
    """
    
    @abstractmethod
    def handle_data(self, data: Union[pd.DataFrame, Series]) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        pass
    
    
class DataPreProcessingStrategy(DataStrategy):
    
    """
    Startegy for processing data

    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame :
        """
        Preprocess data

        Args:
            data (pd.DataFrame)

        Returns:
            pd.DataFrame
            
        """
        try :
            data.drop(["image_name"], axis=1, inplace=True)
            categorical_col = 'model'
            features_col = list(data.columns)
            features_col.remove(categorical_col)
            features_col.remove('hbvalues')
            
            categorical = pd.get_dummies(data[categorical_col], prefix=categorical_col, drop_first=True).astype(int)
            
            scaler_X = StandardScaler()
            scaler_X.fit(data[features_col])
            with open("scaler/scaling_X.pkl", "wb") as f:
                pickle.dump(scaler_X, f)
            
            features_value = pd.DataFrame(scaler_X.transform(data[features_col]), columns=features_col) 
            
            features = pd.concat([categorical, features_value], axis=1)
            
            target = data['hbvalues'].values.reshape(-1, 1)
            # print(target)
            scaler_y = StandardScaler()
            scaler_y.fit(target)
            with open("scaler/scaling_y.pkl", "wb") as f:
                pickle.dump(scaler_y, f)
            target = scaler_y.transform(target)
            target = pd.Series(target.squeeze())
            
            features['hbvalues'] = target
            
            # features.to_csv("data.csv")
            
            return features
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e
    
    
class DataDivideStrategy(DataStrategy):
    
    """
    Strategy for dividing data into train and test
    """
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train and test set

        Args:
            data (pd.DataFrame)

        Returns:
            Union[pd.DataFrame, pd.Series]
        """
        
        try:
            
            X_train, X_test, y_train, y_test = train_test_split(data.drop(['hbvalues'], axis=1), data['hbvalues'], test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data {}".format(e))
            raise e
        
        
class DataCleaning:
    
    """
    Class for cleaning data which processes the data and divides it into train and test
    """
    
    def __init__(self, data:pd.DataFrame, strategy: DataStrategy):
        self.data =  data
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        """
        handle data

        Returns:
            Union[pd.DataFrame, pd.Series, np.ndarray]
        """
        
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e