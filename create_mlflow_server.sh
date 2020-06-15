# create mlflow server
git clone https://github.com/ymym3412/mlflow-docker-compose.git
cd mlflow-docker-compose/on_premises/
mkdir /tmp/artifacts
docker-compose up --build -d

# create model
python create_iris_model.py
