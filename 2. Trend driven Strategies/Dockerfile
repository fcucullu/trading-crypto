FROM python:3.6

ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

ENTRYPOINT '/app/entrypoint.sh'
