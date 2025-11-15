FROM python:3.12-slim

WORKDIR /app

# 1. Copiamos requirements y los instalamos
COPY requirements.txt /requirements.txt

RUN apt-get update && apt-get install -y curl \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r /requirements.txt

# 2. Copiamos el código (permite usar cache correctamente)
COPY . .

# 3. Exponemos el puerto
EXPOSE 8800

# 4. Comando para arrancar FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8800"]
