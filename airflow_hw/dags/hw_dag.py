import datetime as dt
import os
import sys

from airflow.models import DAG
from airflow.operators.python import PythonOperator



path = os.path.expanduser('~/airflow_hw')
# Добавим путь к коду проекта в переменную окружения, чтобы он был доступен python-процессу
os.environ['PROJECT_PATH'] = path
# Добавим путь к коду проекта в $PATH, чтобы импортировать функции
sys.path.insert(0, path)


# <YOUR_IMPORTS>
from modules.pipeline import pipeline
from modules.predict import predict

args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2024, 3, 10),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}

with DAG(
        dag_id='car_price_prediction',
        schedule_interval="00 15 * * *",
        default_args=args,
) as dag:
    pipeline_create = PythonOperator(
        task_id='pipeline_create',
        python_callable=pipeline,
    )

    predict_car = PythonOperator(
        task_id='predict_car',
        python_callable=predict,
        dag=dag,
    )

    pipeline_create >> predict_car

