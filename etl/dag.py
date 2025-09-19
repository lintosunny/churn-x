from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define DAG
with DAG(
    dag_id="telco_etl_weekly",
    default_args=default_args,
    description="ETL pipeline for Telco churn project",
    schedule_interval="@weekly",  # Runs once every week
    start_date=datetime(2025, 1, 1),  # You can change this
    catchup=False,
    tags=["telco", "etl", "churn"],
) as dag:

    # Task to run your etl/main.py
    run_etl = BashOperator(
        task_id="run_telco_etl",
        bash_command="python /path/to/your/project/etl/main.py"
    )

    run_etl