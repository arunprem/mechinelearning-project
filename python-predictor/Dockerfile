FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predictor.py .

ENV MLFLOW_URI=http://host.docker.internal:5000

CMD ["uvicorn", "predictor:app", "--host", "0.0.0.0", "--port", "8000"]
