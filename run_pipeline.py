from pipelines.training_pipelines import training_pipeline, combine_value_model
import os


if __name__ == "__main__":
    # run the pipeline
    training_pipeline(data_path=r"data/Python Dev Hemoglobin Data - Sheet1.csv")
    combine_value_model(data_path=r"data/Python Dev Hemoglobin Data - Sheet1.csv")