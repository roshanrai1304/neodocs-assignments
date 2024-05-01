import logging
import pandas as pd
import numpy as np

from sklearn.base import RegressorMixin
from src.model_dev import LinearRegressionModel, RandomForestRegressorModel, GradientBoostingRegressorModel


def train_model(
    model: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RegressorMixin:
    
    """
    Trains the model on the ingested data.
    
    Args:
    df: the ingested data
    
    """
    
    try:
        
        if model == "LinearRegression":
            linear = LinearRegressionModel()
            linear.fit(X_train, y_train.ravel())
            return linear
        
        elif model == "RandomForest":
            random_forest = RandomForestRegressorModel()
            random_forest.fit(X_train, y_train)
            return random_forest
        
        elif model == "GradientBoost":
            gradient = GradientBoostingRegressorModel()
            gradient.fit(X_train, y_train)
            return gradient

    except Exception as e:
        logging.error("Error in training model: {}, {}".format(model, e))
        raise e