from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mysql.connector
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# Configuración para la base de datos de datos (no metadatos Airflow)
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "mysql_data"),
    "user": os.getenv("MYSQL_USER", "penguins_user"),
    "password": os.getenv("MYSQL_PASSWORD", "penguins_pass"),
    "database": os.getenv("MYSQL_DB", "penguins"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
}

DATA_PATH = "/opt/airflow/data/penguins_size.csv"
MODEL_PATH = "/opt/airflow/model.pkl"


def clear_database():
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    try:
        cursor.execute("DROP TABLE IF EXISTS penguins_raw;")
        cursor.execute("DROP TABLE IF EXISTS penguins_processed;")
        conn.commit()
    finally:
        conn.close()


def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS penguins_raw (
                culmen_length_mm FLOAT,
                culmen_depth_mm FLOAT,
                flipper_length_mm FLOAT,
                body_mass_g FLOAT,
                island VARCHAR(50),
                sex VARCHAR(10),
                species VARCHAR(50)
            );
        """)
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO penguins_raw 
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, tuple(row))
        conn.commit()
    finally:
        conn.close()


def preprocess_data():
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    df = pd.read_sql("SELECT * FROM penguins_raw", conn)
    df = df.dropna()
    scaler = StandardScaler()
    X = df.drop(["species"], axis=1)
    y = df["species"]
    X_scaled = scaler.fit_transform(X)
    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    df_processed["species"] = y.values
    df_processed.to_sql(
        "penguins_processed",
        conn,
        if_exists="replace",
        index=False
    )
    conn.close()


def train_model():
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    df = pd.read_sql("SELECT * FROM penguins_processed", conn)
    conn.close()
    X = df.drop("species", axis=1)
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)


with DAG(
    dag_id="penguins_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id="clear_database",
        python_callable=clear_database,
    )

    t2 = PythonOperator(
        task_id="load_raw_data",
        python_callable=load_raw_data,
    )

    t3 = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    t4 = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    t1 >> t2 >> t3 >> t4