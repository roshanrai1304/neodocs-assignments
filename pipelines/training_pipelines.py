import logging
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation_prediction import evaluate_model
from steps.combine_value import combine_features
from config import model_names
import pickle


# model to predict the value
def training_pipeline(data_path: str):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model_name = model_names[1]
    model = train_model(model_name, X_train, y_train)
    with open(f"saved_models/{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)
    evaluate_model(model, X_test, y_test)
    
# model capable of calibrating a new phone model to predict hemoglobin value
def combine_value_model(data_path: str):
    
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    value = combine_features(X_train.iloc[:2])
    with open("saved_models/RandomForest.pkl", "rb") as f:
        model = pickle.load(f)
        
    hb_value = model.predict(value)
    with open("scaler/scaling_y.pkl", "rb") as f:
        scaling_y = pickle.load(f)
    print(scaling_y.inverse_transform(hb_value.reshape(-1,1))[0][0])
    # with open("")