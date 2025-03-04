<<<<<<< HEAD
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Copier le script shell
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Utiliser le script shell comme point d'entrée
CMD ["/app/entrypoint.sh"]

=======
FROM python:3.10-slim
WORKDIR /app
COPY project.py .
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
CMD python project.py && evidently ui --host 0.0.0.0 --port 8082
>>>>>>> ce7455d26707c607cab429c4b8b6566b1f03afbc
