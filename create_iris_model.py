import os
import mlflow
import mlflow.sklearn
import hydra
import logging


class ExperimentRecorder():
    """MLflow/Hydra simple wrapper for making easy to record experiment details.
    Thanks to https://ymym3412.hatenablog.com/entry/2020/02/09/034644
    """

    def __init__(self, experiment_name, run_name=None,
                 uri='http://0.0.0.0:5000', username='mlflow_user', password='mlflow_pwd'):
        os.environ['MLFLOW_TRACKING_URI']      = uri
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password

        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)

        logging.basicConfig(level=logging.WARN)
    
    def get_things(self):
        org_dir = hydra.utils.get_original_cwd()
        run_dir = os.path.abspath('.')
        return org_dir, run_dir, logging.getLogger(__name__)

    def log_all_params(self, root_param):
        self._explore_recursive('', root_param)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, dict):
            for k, v in element.items():
                if isinstance(v, dict) or isinstance(v, list):
                    self._explore_recursive(f'{parent_name}{k}.', v)
                else:
                    mlflow.log_param(f'{parent_name}{k}', v)
        elif isinstance(element, list):
            for i, v in enumerate(element):
                mlflow.log_param(f'{parent_name}{i}', v)
        else:
            print('ignored to log param:', element)

    # def log_param(self, key, value): # --> simply, `mlflow.log_param(...)`
    #     mlflow.log_param(key, value)

    # def log_metric(self, key, value, step=None):
    #     mlflow.log_metric(key, value, step=step)

    # def log_artifact(self, local_path):
    #     mlflow.log_artifact(local_path)

    def end_run(self):
        mlflow.end_run()


import pandas as pd
import sklearn.datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

def main():
    iris = sklearn.datasets.load_iris()
    iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_label = pd.Series(data=iris.target)
    train_X, test_X, train_y, test_y = train_test_split(iris_data, iris_label)
    clf = SVC()
    model = clf.fit(train_X, train_y)
    pred = clf.predict(test_X)
    ac_score = metrics.accuracy_score(test_y, pred)
    recoder = ExperimentRecorder("sample_exec_iris_SVM", "sample_run_name")
    recoder.log_all_params(clf.get_params())
    mlflow.log_metric("accuracy", ac_score)
    mlflow.sklearn.log_model(model, "model")
    recoder.end_run()


if __name__ == '__main__':
    main()
