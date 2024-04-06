from pathlib import Path
from datetime import datetime

from mldesigner import command_component, Input


@command_component(
    name='train_lr_model',
    version='1',
    display_name='Train lr model',
    description='Train lr model for kidney stone dataset.',
    environment=dict(
        conda_file=Path(__file__).parent / 'conda.yaml',
        image='mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04'
    )
)
def lr_train(
    features_path: Input(type='uri_file'),
    target_path: Input(type='uri_file')
    ):

    import pandas as pd

    import mlflow
    import mlflow.sklearn

    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, recall_score

    from scipy.stats import loguniform

    features = pd.read_csv(features_path, index_col=False)
    target = pd.read_csv(target_path, index_col=False).squeeze()

    lr = LogisticRegression(random_state=42, solver = 'liblinear')

    randomized_parameters = {'C': loguniform(0.01, 100), 'penalty': ['l1', 'l2']}
    randomized_search = RandomizedSearchCV(lr, randomized_parameters, cv=20, random_state=42, n_iter=30)
    randomized_search.fit(features, target)

    prediction = randomized_search.predict(features)
    precision = precision_score(target, prediction)
    recall = recall_score(target, prediction)
    metrics = {'accuracy': randomized_search.best_score_, 'precision': precision, 'recall': recall}
    params = {'parameters': randomized_search.best_params_}

    run_name = f'kidney_stone_lr_{datetime.now()}'
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(sk_model=lr, registered_model_name='lr_kidney_stone', artifact_path='lr_kidney_stone')