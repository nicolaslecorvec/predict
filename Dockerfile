FROM python:3.10.6
COPY gravitw /gravitw
COPY requirements_fastapi.txt /requirements_fastapi.txt
RUN pip install --upgrade pip
RUN pip install -r requirements_fastapi.txt
CMD uvicorn gravitw.api.fast:app --host 0.0.0.0 --port $PORT
