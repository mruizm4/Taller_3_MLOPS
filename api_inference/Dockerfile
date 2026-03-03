FROM python:3.11

WORKDIR /app

# Copiar requirements.txt desde el contexto correcto
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
RUN ls -R


# Copiar solo los archivos necesarios
COPY main.py ./
COPY predict.py ./
COPY models/ ./models/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
