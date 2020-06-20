# sample_mlflow

### cool code

* https://github.com/ymym3412/mlflow-docker-compose/tree/master/on_premises

```
git clone https://github.com/ymym3412/mlflow-docker-compose.git
cd mlflow-docker-compose/on_premises/
mkdir /tmp/artifacts
docker-compose up --build -d

# localhost:5000
```




### [没]sample airflow

```
cd airflow_home
./create_airflow_server.sh
```

* 一度登録したDAGSを変更した場合は`airflow initdb`を行う必要がある
  * 新規追加も同様

```
export AIRFLOW_HOME=`pwd`/airflow_home
airflow version
airflow initdb
airflow webserver
```