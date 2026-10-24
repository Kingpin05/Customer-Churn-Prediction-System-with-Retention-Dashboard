# Churn Watch web app. Train first (python -m churn.train) so artifacts/ exists,
# or let the CD workflow build this image after it trains.
FROM python:3.12-slim
WORKDIR /srv
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY churn ./churn
COPY app ./app
COPY wsgi.py config.yaml ./
COPY data ./data
COPY artifacts ./artifacts
RUN mkdir -p var
EXPOSE 8000
# One worker: follow-up notes live in a local SQLite file under var/.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "wsgi:app"]
