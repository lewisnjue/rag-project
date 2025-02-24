FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .

COPY . .
RUN chmod +x entrypoint.sh
CMD [ "entrypoint.sh" ]