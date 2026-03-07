FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY main.py ./
COPY predict.py ./
COPY models/ ./models/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
