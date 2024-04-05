from pathlib import Path
from datetime import datetime

from mldesigner import command_component, Input


@command_component(
    name='train_rf_model',
    version='1',
    display_name='Train rf model',
    description='Train rf model for cardio dataset.',
    environment=dict(
        conda_file=Path(__file__).parent / 'conda.yaml',
        image='mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04'
    )
)
def rf_train(
    features_path: Input(type='uri_file'),
    target_path: Input(type='uri_file')
    ):

    import pandas as pd

    import mlflow
    import mlflow.sklearn

    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier

    features = pd.read_csv(features_path, index_col=False)
    target = pd.read_csv(target_path, index_col=False).squeeze()

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    grid_parameters = {'criterion': ['gini', 'entropy']}
    grid_search = GridSearchCV(rf, grid_parameters, cv=10)
    grid_search.fit(features, target)

    metrics = {'accuracy': grid_search.best_score_}
    params = {'parameters': grid_search.best_params_}

    run_name = f'cardio_rf_{datetime.now()}'
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(sk_model=rf, registered_model_name='rf_cardio', artifact_path='rf_cardio')
