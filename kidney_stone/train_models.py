import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import mlflow

from scipy.stats import loguniform


mlflow.start_run()
mlflow.sklearn.autolog()

data = pd.read_csv('kidney_stone.csv')
features = data.drop(['target'], axis=1)
target = data['target']


knn_pipeline = make_pipeline(PCA(n_components=3),
                             StandardScaler(),
                             KNeighborsClassifier(),
                             memory='cache')


grid_parameters = {'kneighborsclassifier__n_neighbors': [i for i in range(1, 17, 2)],
                   'kneighborsclassifier__weights': ['uniform', 'distance']}

grid_search = GridSearchCV(knn_pipeline, grid_parameters)

grid_search.fit(features, target)
print('Best score:', grid_search.best_score_, 'Parameters:', grid_search.best_params_)

mlflow.sklearn.log_model(sk_model=knn_pipeline, registered_model_name='knn_model', artifact_path='artifacts_knn')


svc_pipeline = knn_pipeline
svc_pipeline.steps[-1] = ['svc', SVC(random_state=42, kernel='rbf')]

regularization_range = loguniform(0.001, 10000.0)
gamma_range = loguniform(0.001, 10.0)

random_parameters = {'svc__C': regularization_range, 'svc__gamma':gamma_range}
random_search = RandomizedSearchCV(svc_pipeline, random_parameters, random_state=42)

random_search.fit(features, target)
print('Best score:', random_search.best_score_, 'Parameters:', random_search.best_params_)

mlflow.sklearn.log_model(sk_model=svc_pipeline, registered_model_name='svc_model', artifact_path='artifacts_svc')

mlflow.end_run()
