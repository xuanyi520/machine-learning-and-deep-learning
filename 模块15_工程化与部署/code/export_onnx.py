"""导出 ONNX 并做数值一致性验证。

运行：
    python export_onnx.py
产出：
    best.onnx
    打印 PyTorch 与 ONNXRuntime 的最大绝对误差
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from model import SmallCNN


def main() -> None:
    # 不传参数就取 outputs 下最新一次运行的 best.pt
    if len(sys.argv) > 1:
        ckpt = Path(sys.argv[1])
    else:
        candidates = sorted(Path("outputs").rglob("best.pt"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError("先运行 python train.py 生成 best.pt")
        ckpt = candidates[-1]
    print(f"checkpoint: {ckpt}")

    model = SmallCNN(1, 2)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    # dynamic_axes: 第 0 维是 batch，导出后可以送任意 batch
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy,
        "best.onnx",
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    onnx_model = onnx.load("best.onnx")
    onnx.checker.check_model(onnx_model)
    print(f"ONNX 文件大小: {Path('best.onnx').stat().st_size / 1024:.1f} KB")
    print(f"opset: {[ (o.domain, o.version) for o in onnx_model.opset_import ]}")
    print(f"输入: {[ (i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in onnx_model.graph.input ]}")
    print(f"输出: {[ (o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]) for o in onnx_model.graph.output ]}")

    sess = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

    print("\n=== 一致性验证 ===")
    for bs in (1, 8, 32):
        x = torch.randn(bs, 1, 28, 28)
        with torch.no_grad():
            torch_out = model(x).numpy()
        onnx_out = sess.run(None, {"input": x.numpy()})[0]
        max_abs = float(np.max(np.abs(torch_out - onnx_out)))
        max_rel = float(np.max(np.abs(torch_out - onnx_out) / (np.abs(torch_out) + 1e-8)))
        print(f"batch={bs:2d}  max_abs_err={max_abs:.3e}  max_rel_err={max_rel:.3e}"
              f"  allclose(atol=1e-5)={np.allclose(torch_out, onnx_out, atol=1e-5)}")


if __name__ == "__main__":
    main()
