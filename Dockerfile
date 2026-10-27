FROM mcr.microsoft.com/playwright/python:v1.49.0-noble

RUN useradd -m -u 1000 user || true
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && playwright install chromium

COPY --chown=1000:1000 autoapply ./autoapply

RUN mkdir -p /tmp/autoapply && chmod -R 777 /tmp/autoapply /app

USER 1000
ENV HOME=/home/user \
    PORT=8000 \
    AUTOAPPLY_DATA_DIR=/tmp/autoapply

EXPOSE 8000

CMD ["python", "-m", "autoapply", "serve", "--host", "0.0.0.0"]
