"""FastAPI 推理服务：接收一张胸部 X 光 PNG，返回肺炎概率。

启动：
    uvicorn app:app --host 0.0.0.0 --port 8000
测试：
    curl -X POST -F "file=@test.png" http://127.0.0.1:8000/predict
交互文档：
    http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

MODEL_PATH = os.getenv("MODEL_PATH", "best.onnx")
CLASS_NAMES = ["normal", "pneumonia"]
_session: ort.InferenceSession | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时加载模型，关闭时释放。加载只做一次。"""
    global _session
    _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    yield
    _session = None


app = FastAPI(title="PneumoniaMNIST 推理服务", version="1.0", lifespan=lifespan)


def preprocess(raw: bytes) -> np.ndarray:
    """PNG 字节 -> (1, 1, 28, 28) float32。

    规则必须和训练时完全一致：灰度、28x28、除以 255。
    """
    try:
        img = Image.open(io.BytesIO(raw)).convert("L").resize((28, 28))
    except Exception as exc:  # 不是图片
        raise HTTPException(status_code=400, detail=f"无法解析图片: {exc}") from exc
    x = np.asarray(img, dtype=np.float32) / 255.0
    return x[None, None, :, :]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if _session is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    if file.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(status_code=415, detail="只支持 PNG/JPEG")

    x = preprocess(await file.read())
    logits = _session.run(None, {"input": x})[0]
    probs = softmax(logits)[0]
    idx = int(probs.argmax())
    return {
        "label": CLASS_NAMES[idx],
        "confidence": round(float(probs[idx]), 4),
        "probabilities": {n: round(float(p), 4) for n, p in zip(CLASS_NAMES, probs)},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
