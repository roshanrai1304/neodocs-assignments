import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin,clone
import numpy as np



class Model(ABC, BaseEstimator, RegressorMixin):
    """
    Abstract class for all models

    """
    @abstractmethod
    def fit(self, X_train, y_train):
        """
        Train the model

        Args:
            X_train : Traininig data
            y_train : Traininig label
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Makes predictions using the trained model

        Args:
            X (type): Input data for prediction
            
        Returns:
            type: Predictions
        """
        pass


class LinearRegressionModel(Model):
    
    """
    Linear Regression Model
    
    Args:
      X_train: Training features
      y_train: Training labels
    """
    
    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)
        
        
    def fit(self, X_train, y_train):
        """
        Trains the model

        Args:
            X_train : Training features
            y_train : Training labels
        """
        try:
            self.model.fit(X_train, y_train)
            logging.info("Model training completed")
            return self
        except Exception as e:
            logging.error("Error in training model LinearRegression: {}".format(e))
            raise e
        
    def predict(self, X_test):
        """
        predicts for test_data

        Args:
            X_test
        """
        return self.model.predict(X_test)
   

class RandomForestRegressorModel(Model):
    
    """
    Random Forest Regressor Model
    
    Args:
      X_train: Training features
      y_train: Training labels
    """
    
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)
        
        
    def fit(self, X_train, y_train):
        """
        Trains the model

        Args:
            X_train : Training features
            y_train : Training labels
        """
        try:
            self.model.fit(X_train, y_train)
            logging.info("Model training completed")
            return self
        except Exception as e:
            logging.error("Error in training model RandomForestRegressor: {}".format(e))
            raise e
        
    def predict(self, X_test):
        """
        predicts for test_data

        Args:
            X_test
        """
        return self.model.predict(X_test)
    
        
class GradientBoostingRegressorModel(Model):
    
    """
    GradientBoostingRegressor Model
    
    Args:
      X_train: Training features
      y_train: Training labels
    """
    
    def __init__(self, **kwargs):
        self.model = GradientBoostingRegressor(**kwargs)
        
        
    def fit(self, X_train, y_train):
        """
        Trains the model

        Args:
            X_train : Training features
            y_train : Training labels
        """
        try:
            self.model.fit(X_train, y_train)
            logging.info("Model training completed")
            return self
        except Exception as e:
            logging.error("Error in training model GradientBoostingRegressor: {}".format(e))
            raise e
        
    def predict(self, X_test):
        """
        predicts for test_data

        Args:
            X_test
        """
        return self.model.predict(X_test)
     