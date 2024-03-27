from azure.ai.ml.dsl import pipeline

from knn_preprocessing import split_and_scale
from knn_train import knn_train

from lr_preprocessing import scale_and_add_polynomial
from lr_train import lr_train


@pipeline(
    default_compute='medicinal-compute',
    experiment_name='kidney_stone_knn'
)
def knn_pipeline(kidney_stone_data):
    preprocessed_data = split_and_scale(to_preprocess=kidney_stone_data, scaler_type='robust', features_to_drop='ph;cond')
    features_path = preprocessed_data.outputs.features_path
    target_path = preprocessed_data.outputs.target_path
    
    knn_train(features_path=features_path, target_path=target_path)


@pipeline(
    default_compute='medicinal-compute',
    experiment_name='kidney_stone_logistic_regresion'
)
def lr_pipeline(kidney_stone_data):
    preprocessed_data = scale_and_add_polynomial(to_preprocess=kidney_stone_data, scaler_type='robust')
    features_path = preprocessed_data.outputs.features_path
    target_path = preprocessed_data.outputs.target_path
    
    lr_train(features_path=features_path, target_path=target_path)
