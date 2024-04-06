from pathlib import Path

from mldesigner import command_component, Input, Output

@command_component(
    name='medicinal_preprocessing',
    version='1',
    display_name='Medicinal preprocessing',
    description='Medicinal data preprocessing, split gender data, and measure bmi.',
    environment=dict(
        conda_file=Path(__file__).parent / 'conda.yaml',
        image='mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04',
    )
)
def med_preprocess(
    to_preprocess: Input,
    features_path: Output(type='uri_file'),
    target_path: Output(type='uri_file')
):
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder

    data = pd.read_csv(to_preprocess, delimiter=';')

    target = data['cardio']
    target.to_csv(target_path, index=False)
    features = data.drop(['cardio'], axis=1)
    features.drop(['id'], axis=1, inplace=True)

    one_hot_encoder = OneHotEncoder(sparse_output=False)

    encoded_gendere = one_hot_encoder.fit_transform(pd.DataFrame(features['gender']))
    features['female'] = encoded_gendere[0:, 0]
    features['male'] = encoded_gendere[0:, 1]
    features.drop(['gender'], axis=1, inplace=True)

    features['bmi'] = features['weight'] / (features['height']/100)**2
    features.drop(['weight', 'height'], axis=1, inplace=True)
    
    features.to_csv(features_path, index=False, mode='w')
    