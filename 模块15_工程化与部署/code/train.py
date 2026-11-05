"""训练脚本：Hydra 管配置 + TensorBoard 记实验。

命令行覆盖参数：
    python train.py train.lr=0.0003 train.epochs=10
批量跑参数组合：
    python train.py --multirun train.lr=1e-4,1e-3,1e-2 train.batch_size=64,128
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import NpzDataset
from metrics import accuracy
from model import SmallCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += float(criterion(logits, y))
            total_acc += accuracy(logits, y) * len(y)
            n += len(y)
    return total_loss / n, total_acc / n


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(cfg.data.npz_path)
    train_ds = NpzDataset(data["train_images"], data["train_labels"])
    val_ds = NpzDataset(data["val_images"], data["val_labels"])
    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    model = SmallCNN(cfg.model.in_channels, cfg.model.num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # 关键：用 Hydra 给本次运行分配的目录，而不是 os.getcwd()。
    # 否则 multirun 的多个 job 会写进同一个 TensorBoard 目录，曲线互相覆盖。
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    tb_dir = run_dir / "tb" / cfg.log.run_name
    writer = SummaryWriter(log_dir=str(tb_dir))
    # 超参写进 hparams，方便在 TensorBoard 里筛选
    writer.add_text("config", OmegaConf.to_yaml(cfg).replace("\n", "  \n"))

    best_acc = 0.0
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running_loss, seen = 0.0, 0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(y)
            seen += len(y)
            global_step = (epoch - 1) * len(train_loader) + step
            writer.add_scalar("train/loss_step", loss.item(), global_step)

        train_loss = running_loss / seen
        val_loss, val_acc = evaluate(model, val_loader, device)

        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)
        writer.add_scalar("time/epoch_sec", time.perf_counter() - t0, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), run_dir / "best.pt")
        print(
            f"epoch {epoch:02d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

    writer.add_hparams(
        {
            "lr": cfg.train.lr,
            "batch_size": cfg.data.batch_size,
            "epochs": cfg.train.epochs,
            "weight_decay": cfg.train.weight_decay,
        },
        {"hparam/best_val_acc": best_acc},
    )
    writer.close()
    print(f"best_val_acc={best_acc:.4f}")
    print(f"tensorboard logdir: {tb_dir}")


if __name__ == "__main__":
    main()
