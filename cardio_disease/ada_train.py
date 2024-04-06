from pathlib import Path
from datetime import datetime

from mldesigner import command_component, Input


@command_component(
    name='train_ada_model',
    version='1',
    display_name='Train ada model',
    description='Train ada model for cardio dataset.',
    environment=dict(
        conda_file=Path(__file__).parent / 'conda.yaml',
        image='mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04'
    )
)
def ada_train(
    features_path: Input(type='uri_file'),
    target_path: Input(type='uri_file')
    ):

    import pandas as pd

    import mlflow
    import mlflow.sklearn

    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    from scipy.stats import uniform, randint

    features = pd.read_csv(features_path, index_col=False)
    target = pd.read_csv(target_path, index_col=False).squeeze()

    ada = AdaBoostClassifier(random_state=42, estimator=DecisionTreeClassifier())

    random_parameters = {'learning_rate': uniform(0, 10), 'estimator__max_depth': randint(1, 11)}
    random_search = RandomizedSearchCV(ada, random_parameters, cv=10, random_state=42, n_jobs=-1)
    random_search.fit(features, target)

    metrics = {'accuracy': random_search.best_score_}
    params = {'parameters': random_search.best_params_}

    run_name = f'cardio_ada_{datetime.now()}'
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(sk_model=ada, registered_model_name='ada_cardio', artifact_path='ada_cardio')
