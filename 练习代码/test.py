import sys
print("解释器:", sys.executable)
import torch
print("torch:", torch.__version__, "| GPU:", torch.cuda.is_available())