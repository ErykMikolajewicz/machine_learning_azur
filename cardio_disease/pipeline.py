from azure.ai.ml.dsl import pipeline

from preprocessing import med_preprocess
from ada_train import ada_train
from rf_train import rf_train


@pipeline(
    default_compute='medicinal-compute',
    experiment_name='cardio_enasemble'
)
def cardio_pipeline(cardio_data):
    preprocessed_data = med_preprocess(to_preprocess=cardio_data)
    features_path = preprocessed_data.outputs.features_path
    target_path = preprocessed_data.outputs.target_path
    
    ada_train(features_path=features_path, target_path=target_path)
    rf_train(features_path=features_path, target_path=target_path)
