import os
import time
import pandas as pd
from sqlalchemy import create_engine, text


def get_engine():
    """Create SQLAlchemy engine from env vars."""
    user = os.getenv("MYSQL_USER", "penguins_user")
    password = os.getenv("MYSQL_PASSWORD", "penguins_pass")
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    db = os.getenv("MYSQL_DB", "penguins")

    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)


def wait_for_db(engine, retries=10, sleep=3):
    """Wait until DB is ready (important in Docker)."""
    for i in range(retries):
        try:
            with engine.connect():
                print("✅ DB ready")
                return
        except Exception as e:
            print(f"⏳ waiting for DB... ({i+1}/{retries})")
            time.sleep(sleep)
    raise RuntimeError("Database not reachable")


def load_penguins(csv_path: str):
    engine = get_engine()
    wait_for_db(engine)

    print("📥 Reading CSV...")
    df = pd.read_csv(csv_path)

    print("🗄️ Writing to MySQL (penguins_raw)...")
    df.to_sql(
        "penguins_raw",
        con=engine,
        if_exists="append",  # importante para pipeline real
        index=False,
        chunksize=1000,
    )

    print(f"✅ Loaded {len(df)} rows")


if __name__ == "__main__":
    csv_path = os.getenv("CSV_PATH", "/data/penguins_size.csv")
    load_penguins(csv_path)
