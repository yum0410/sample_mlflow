from datetime import datetime
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
import pandas as pd
import sklearn.datasets


def load_iris_data():
    iris = sklearn.datasets.load_iris()
    iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_label = pd.Series(data=iris.target)
    return {"X": X, "Y": Y}

def split_train_test(X_Y_dict):
    X = X_Y_dict["X"]
    Y = X_Y_dict["Y"]
    print(X)

dag = DAG('01_hello_world', description='Simple tutorial DAG',
          schedule_interval='0 12 * * *',
          start_date=datetime(2019, 6, 16), catchup=False)

load_operator = PythonOperator(task_id='01_load_iris', python_callable=load_iris_data, dag=dag)
split_operator = PythonOperator(task_id='02_train', python_callable=split_train_test, dag=dag, templates_dict={load_operator})

# train_operator = PythonOperator(task_id='02_train', python_callable=load_iris_data, dag=dag, templates_dict={})


load_operator >> split_operator