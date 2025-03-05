FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt requirements.txt
COPY . .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt


CMD python project.py && evidently ui --host 0.0.0.0 --port 8082