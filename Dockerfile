FROM mcr.microsoft.com/playwright/python:v1.49.0-noble

# Hugging Face Spaces runs containers as UID 1000
RUN useradd -m -u 1000 user || true
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && playwright install chromium

COPY --chown=1000:1000 autoapply ./autoapply

RUN mkdir -p /tmp/autoapply && chmod -R 777 /tmp/autoapply /app

USER 1000
ENV HOME=/home/user \
    PORT=7860 \
    AUTOAPPLY_DATA_DIR=/tmp/autoapply

EXPOSE 7860

CMD ["python", "-m", "autoapply", "serve", "--host", "0.0.0.0", "--port", "7860"]
