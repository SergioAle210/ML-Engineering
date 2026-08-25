FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml setup.py ./
COPY src/ src/
RUN pip install --no-cache-dir . joblib "fastapi>=0.110" "uvicorn>=0.29"

COPY services/serving/ services/serving/

EXPOSE 8000
CMD ["uvicorn", "services.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
