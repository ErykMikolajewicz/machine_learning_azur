from pathlib import Path

from mldesigner import command_component, Input, Output

@command_component(
    name="interactions_features_and_scale",
    version="1",
    display_name="Interactions features and scale",
    description="create interactions features, by polynomial features and scale by choosen scaler.",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    )
)
def scale_and_add_polynomial(
    to_preprocess: Input,
    features_path: Output(type='uri_file'),
    target_path: Output(type='uri_file'),
    scaler_type: str = 'standard',
    polynomial_degree: int = 2
):
    from sklearn.preprocessing import RobustScaler, StandardScaler, PolynomialFeatures
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
    polynomial_features = PolynomialFeatures(degree=polynomial_degree, include_bias=False, interaction_only=True)
    features = polynomial_features.fit_transform(features)

    features = scaler.fit_transform(features)
    features = pd.DataFrame(features)
    
    features.to_csv(features_path, index=False)
    