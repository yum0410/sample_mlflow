import luigi
from luigi.util import inherits, requires
import pandas as pd
import numpy as np
import sklearn.datasets
import os
import pickle


class LoadRowData(luigi.Task):
    task_namespace = 'iris_tasks'

    # output_path
    output_path = luigi.Parameter()

    def run(self):
        iris = sklearn.datasets.load_iris()
        iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        iris_label = pd.Series(data=iris.target)
        iris_data["Y"] = iris_label
        iris_data.to_pickle(self.output_path)

    def output(self):
        file_anme = self.output_path.split("/")[-1]
        output_dir = self.output_path.split("/")[:-1]
        if not os.path.exists(os.path.join(*output_dir)):
            os.makedirs(os.path.join(*output_dir))
        output = os.path.join(*output_dir + [file_anme])
        return luigi.LocalTarget(output)

@requires(LoadRowData)
class PreprocessAddRandomNum(luigi.Task):
    task_namespace = 'iris_tasks'

    # output_path
    output_path = luigi.Parameter()

    def run(self):
        data = pd.read_pickle(self.input().path)
        data["random_num"] = np.random.rand(data.shape[0])
        data.to_pickle(self.output_path)

    def output(self):
        file_anme = self.output_path.split("/")[-1]
        output_dir = self.output_path.split("/")[:-1]
        if not os.path.exists(os.path.join(*output_dir)):
            os.makedirs(os.path.join(*output_dir))
        output = os.path.join(*output_dir + [file_anme])
        return luigi.LocalTarget(output)

@requires(PreprocessAddRandomNum)
class SplitTrainTest(luigi.Task):
    task_namespace = 'iris_tasks'

    # output_path
    output_path = luigi.Parameter()

    def run(self):
        from sklearn.model_selection import train_test_split
        data = pd.read_pickle(self.input().path)
        X_cols = [c for c in data.columns if c == "Y"]
        train_X, test_X, train_y, test_y = train_test_split(data[X_cols], data["Y"])
        data = {"train_X": train_X, "train_y": train_y, "test_X": test_X, "test_y": test_y}
        with open(self.output_path, "wb") as f:
            pickle.dump(data, f)

    def output(self):
        file_anme = self.output_path.split("/")[-1]
        output_dir = self.output_path.split("/")[:-1]
        if not os.path.exists(os.path.join(*output_dir)):
            os.makedirs(os.path.join(*output_dir))
        output = os.path.join(*output_dir + [file_anme])
        return luigi.LocalTarget(output)

@requires(SplitTrainTest)
class Train(luigi.Task):
    task_namespace = 'iris_tasks'

    # output_path
    output_path = luigi.Parameter()

    def run(self):
        from sklearn.svm import SVC
        data = pd.read_pickle(self.input().path)
        clf = SVC()
        model = clf.fit(data["train_X"], data["train_y"])
        with open(self.output_path, "wb") as f:
            pickle.dump(model, f)

    def output(self):
        file_anme = self.output_path.split("/")[-1]
        output_dir = self.output_path.split("/")[:-1]
        if not os.path.exists(os.path.join(*output_dir)):
            os.makedirs(os.path.join(*output_dir))
        output = os.path.join(*output_dir + [file_anme])
        return luigi.LocalTarget(output)

@inherits(SplitTrainTest)
@inherits(Train)
class Predict(luigi.Task):
    task_namespace = 'iris_tasks'

    # output_path
    output_path = luigi.Parameter()

    def requires(self):
        yield SplitTrainTest()
        yield Train()

    def run(self):
        data = pd.read_pickle(self.input()[0].path)
        model = pd.read_pickle(self.input()[1].path)
        predict = model.predict(data["test_X"])
        with open(self.output_path, "wb") as f:
            pickle.dump(predict, f)

    def output(self):
        file_anme = self.output_path.split("/")[-1]
        output_dir = self.output_path.split("/")[:-1]
        if not os.path.exists(os.path.join(*output_dir)):
            os.makedirs(os.path.join(*output_dir))
        output = os.path.join(*output_dir + [file_anme])
        return luigi.LocalTarget(output)

@inherits(SplitTrainTest)
@inherits(Predict)
class Evaluate(luigi.Task):
    task_namespace = 'iris_tasks'

    # output_path
    output_path = luigi.Parameter()

    def requires(self):
        yield SplitTrainTest()
        yield Predict()

    def run(self):
        from sklearn import metrics
        data = pd.read_pickle(self.input()[0].path)
        predict = pd.read_pickle(self.input()[1].path)
        ac_score = metrics.accuracy_score(data["test_y"], predict)
        evaluate = {"accuracy": ac_score}
        print(evaluate)
        with open(self.output_path, "wb") as f:
            pickle.dump(evaluate, f)

    def output(self):
        file_anme = self.output_path.split("/")[-1]
        output_dir = self.output_path.split("/")[:-1]
        if not os.path.exists(os.path.join(*output_dir)):
            os.makedirs(os.path.join(*output_dir))
        output = os.path.join(*output_dir + [file_anme])
        return luigi.LocalTarget(output)


if __name__ == "__main__":
    luigi.run(["iris_tasks.Train", "--workers", "1"])
