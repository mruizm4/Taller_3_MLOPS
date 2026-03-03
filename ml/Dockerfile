FROM python:3.11-slim

WORKDIR /app

COPY ml/ ./ml/
COPY data/ /data/

RUN pip install --no-cache-dir \
    pandas \
    sqlalchemy \
    pymysql \
    cryptography

CMD ["python", "-m", "ml.load_data"]
