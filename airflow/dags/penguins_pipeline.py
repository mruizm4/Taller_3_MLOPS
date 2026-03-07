from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mysql.connector
import pandas as pd
import os
import time
from sqlalchemy import create_engine
from src.train import preprocess_data, train_decision_tree, train_knn, train_svm

# Configuración MySQL
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "mysql_data"),
    "user": os.getenv("MYSQL_USER", "penguins_user"),
    "password": os.getenv("MYSQL_PASSWORD", "penguins_pass"),
    "database": os.getenv("MYSQL_DB", "penguins"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
}

DATA_PATH = "/opt/airflow/data/penguins_size.csv"
MODEL_PATH = "/opt/airflow/model.pkl"


def get_engine():
    """Crea el engine de SQLAlchemy (USA ESTO en lugar del string URI)"""
    url = (
        f"mysql+pymysql://{MYSQL_CONFIG['user']}:"
        f"{MYSQL_CONFIG['password']}@"
        f"{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/"
        f"{MYSQL_CONFIG['database']}"
    )
    return create_engine(url)


def wait_for_db(retries=10, sleep=3):
    """Espera a que la DB esté lista"""
    engine = get_engine()
    for i in range(retries):
        try:
            with engine.connect():
                print("✅ DB ready")
                return
        except Exception as e:
            if i == retries - 1:
                raise RuntimeError(f"Database not reachable: {e}")
            print(f"⏳ Waiting for DB... ({i+1}/{retries})")
            time.sleep(sleep)


def clear_database():
    """Limpia las tablas de la base de datos"""
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    try:
        cursor.execute("DROP TABLE IF EXISTS penguins_raw;")
        cursor.execute("DROP TABLE IF EXISTS penguins_processed;")
        conn.commit()
        print("✅ Tables dropped")
    finally:
        cursor.close()
        conn.close()


def load_penguins(csv_path=DATA_PATH):
    """Carga los datos del CSV a MySQL"""
    # Esperar a que la DB esté lista
    wait_for_db()
    
    df = pd.read_csv(csv_path)
    
    expected_cols = [
        "culmen_length_mm", "culmen_depth_mm", "flipper_length_mm",
        "body_mass_g", "island", "sex", "species"
    ]
    
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"El CSV debe contener las columnas: {expected_cols}")
    
    # 🔑 SOLUCIÓN: Usar el engine en lugar del string URI
    engine = get_engine()
    df[expected_cols].to_sql(
        "penguins_raw",
        con=engine,  # ← Usar engine aquí
        if_exists="append",
        index=False,
        method="multi"
    )
    print(f"✅ Loaded {len(df)} rows to penguins_raw")


def preprocess_data_for_training():
    """Preprocesa los datos"""
    engine = get_engine()
    
    # 🔑 SOLUCIÓN: Usar el engine
    df = pd.read_sql("SELECT * FROM penguins_raw", con=engine)
    df = df.dropna()
    
    df_processed, _, _, _ = preprocess_data(df)
    
    # 🔑 SOLUCIÓN: Usar el engine
    df_processed.to_sql(
        "penguins_processed",
        con=engine,
        if_exists="replace",
        index=False,
        method="multi"
    )
    print(f"✅ Processed {len(df_processed)} rows")


def train_model():
    """Entrena los modelos"""
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM penguins_processed", con=engine)
    
    print("🚀 Training SVM...")
    train_svm(df)
    
    print("🚀 Training Decision Tree...")
    train_decision_tree(df)
    
    print("🚀 Training KNN...")
    train_knn(df)
    
    print("✅ All models trained")


# Definición del DAG
with DAG(
    dag_id="penguins_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "penguins"],
) as dag:
    
    t1 = PythonOperator(
        task_id="clear_database",
        python_callable=clear_database,
    )
    
    t2 = PythonOperator(
        task_id="load_raw_data",
        python_callable=load_penguins,
    )
    
    t3 = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data_for_training,
    )
    
    t4 = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )
    
    t1 >> t2 >> t3 >> t4