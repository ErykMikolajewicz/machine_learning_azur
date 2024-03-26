from azure.ai.ml.dsl import pipeline

from data_preprocessing import split_and_scale
from knn_train import knn_train


@pipeline(
    default_compute='medicinal-compute',
    experiment_name='kidney_stone_training'
)
def knn_pipeline(kidney_stone_data):
    preprocessed_data = split_and_scale(to_preprocess=kidney_stone_data, features_to_drop='ph;cond')
    features_path = preprocessed_data.outputs.features_path
    target_path = preprocessed_data.outputs.target_path
    
    knn_train(features_path=features_path, target_path=target_path)
