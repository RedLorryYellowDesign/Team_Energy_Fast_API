FROM python:3.8.12-slim-buster
COPY app /app
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
