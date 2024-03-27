from pathlib import Path
from datetime import datetime

from mldesigner import command_component, Input


@command_component(
    name="train_knn_model",
    version="1",
    display_name="Train knn model",
    description="Train knn model for kidney stone dataset.",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
    )
)
def knn_train(
    features_path: Input(type='uri_file'),
    target_path: Input(type='uri_file')
    ):

    import pandas as pd

    import mlflow
    import mlflow.sklearn

    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier

    features = pd.read_csv(features_path, index_col=False)
    target = pd.read_csv(target_path, index_col=False).squeeze()

    knn = KNeighborsClassifier()

    grid_parameters = {'n_neighbors': [i for i in range(1, 17, 2)], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
    grid_search = GridSearchCV(knn, grid_parameters, cv=20)
    grid_search.fit(features, target)

    metrics = {"accuracy": grid_search.best_score_}
    params = {"parameters": grid_search.best_params_}

    run_name = f'kidney_stone_knn_{datetime.now()}'
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(sk_model=knn, registered_model_name='knn_kidney_stone', artifact_path='knn_kidney_stone')
