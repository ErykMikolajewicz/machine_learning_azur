from pathlib import Path

from mldesigner import command_component, Input, Output

@command_component(
    name="drop_features_and_scale",
    version="1",
    display_name="Drop features and scale.",
    description="Drop unrelevant features and scale data by choosen scaler.",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    )
)
def split_and_scale(
    to_preprocess: Input,
    features_path: Output(type='uri_file'),
    target_path: Output(type='uri_file'),
    features_to_drop: str,
    scaler_type: str = 'standard'
):
    from sklearn.preprocessing import RobustScaler, StandardScaler
    import pandas as pd

    data = pd.read_csv(to_preprocess)

    target = data['target']
    target.to_csv(target_path, index=False)

    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError('Invalid scaler type!')
    
    features = data.drop(['target'], axis=1)
    features_to_drop = features_to_drop.split(';')
    features = features.drop(features_to_drop, axis=1)

    features = scaler.fit_transform(features)
    features = pd.DataFrame(features)
    
    features.to_csv(features_path, index=False)
