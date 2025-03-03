FROM python:3.10-slim
WORKDIR /app
COPY . .
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
ENV FLASK_APP=web.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8081
CMD ["flask", "run", "--host=0.0.0.0", "--port=8081"]