import json

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import Input

from pipeline import cardio_pipeline

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

path = f'azureml://subscriptions/{SUBSCRIPTION_ID}/resourcegroups/medicinal/workspaces/medicinal_classification/datastores/medicinal_blobstorage/paths/UI/2024-04-05_194211_UTC/cardio.csv'
cardio_data = Input(path=path, type='uri_file')

cardio_jobs = cardio_pipeline(cardio_data)
ml_client.jobs.create_or_update(cardio_jobs)
