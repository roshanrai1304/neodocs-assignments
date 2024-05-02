import logging
import pandas as pd
import numpy as np

from sklearn.base import RegressorMixin
from src.evaluation import MSE, R2, RMSE
from src.inverse_preprocess_data import InverseDataProcessing

from typing import Tuple
from typing_extensions import Annotated

def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.Series
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"], Annotated[np.ndarray, "y_pred"]] :
    
    """
    Args:
     model: RegressionMixin
     X_test: pd.DataFrame
     y_test: pd.Series
     
    Return:
     r2_score: float
     rmse: float
     y_pred: np.ndarray
    """
    
    try:
        prediction = model.predict(X_test)
        inverse_data = InverseDataProcessing()
        data = inverse_data.inverse_data(X_test, y_test.to_numpy(), prediction)
        
        y_true = data['hbvalues_true']
        y_pred = data['hbvalues_pred']
        
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_true, y_pred)
        print(f"The MSE score is {mse}")
        
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_true, y_pred)
        print(f"The R2 score is {r2}")
        
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_true, y_pred)
        print(f"The RMSE score is {rmse}")
        
        
    except Exception as e:
      logging.error("Error in evaluating model: {}".format(e))
      raise e