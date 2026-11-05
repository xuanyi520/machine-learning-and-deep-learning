# 推理镜像：只装推理依赖，不装 PyTorch（镜像从 ~6GB 降到 ~300MB）
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 先拷依赖清单再装包：改代码不会让这一层缓存失效
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py best.onnx ./

# 不用 root 跑服务
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
