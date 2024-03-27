import json

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import Input

from pipelines import knn_pipeline, lr_pipeline

with open('../secrets.json') as secrets_file:
    secrets = json.load(secrets_file)

SUBSCRIPTION_ID = secrets['SUBSCRIPTION_ID']
RESOURCE_GROUP = 'medicinal'
WORKSPACE_NAME = 'medicinal_classification'

ml_client = MLClient(credential=DefaultAzureCredential(),
                     subscription_id=SUBSCRIPTION_ID,
                     resource_group_name=RESOURCE_GROUP,
                     workspace_name=WORKSPACE_NAME,
                     enable_telemetry=False)

path = f'azureml://subscriptions/{SUBSCRIPTION_ID}/resourcegroups/medicinal/workspaces/medicinal_classification/datastores/workspaceblobstore/paths/UI/2024-03-23_200730_UTC/kidney_stone.csv'
kidney_stone_data = Input(path=path, type='uri_file')

knn_job = knn_pipeline(kidney_stone_data)
ml_client.jobs.create_or_update(knn_job)

lr_job = lr_pipeline(kidney_stone_data)
ml_client.jobs.create_or_update(lr_job)