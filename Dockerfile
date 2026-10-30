FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DDDDOCR_HOST=0.0.0.0 \
    DDDDOCR_PORT=8000 \
    DDDDOCR_SHOW_AD=false

ARG VCS_REF
LABEL org.opencontainers.image.source="https://github.com/dnslin/ddddocr" \
      org.opencontainers.image.revision="${VCS_REF}"

COPY . .
RUN pip install --no-cache-dir ".[api]"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['DDDDOCR_PORT'] + '/health', timeout=5).read()"
CMD ["python", "-m", "ddddocr.api"]
