import logging
import pandas as pd
from sklearn.decomposition import PCA



def combine_features(
   data: pd.DataFrame
) -> pd.DataFrame:
    
    """
    It combines the model's into one model
    
    Args:
      rows (pd.DataFrame): rows of different models
      
    Return:
      
      data (pd.DataFrame)
    
    """
    feature_cols = data.columns
    pca = PCA(n_components=1)  # Use number of features as components
    reduced_data = pca.fit_transform(data)
    return pd.DataFrame(pca.components_, columns=feature_cols)