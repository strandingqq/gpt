import math
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None
from tqdm import tqdm

try:
    import psutil
except ModuleNotFoundError:
    psutil = None

from gpt import GPT, GPTConfig
from data import build_dataset, clean_text, get_dataloaders


class NullWriter:
    def add_text(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


"""
@dataclass有什么作用？之前我在学习fastapi的时候，知道了它是装饰器，当有http请求时，fastapi会调用这个函数，并传递请求参数。这里它能够自动生成__init__()方法，它还有什么用途？我想你详细的解释一下。
尝试都用这种方法记录参数
"""
@dataclass
class TrainConfig:
    # 模型
    model_name: str = "small"  # small / medium / large
    vocab_size: int = None  # 由数据决定
    n_layer: int = None
    n_head: int = None
    n_embd: int = None
    d_ff: int = None  # 由 cfg_map 覆盖，默认 = n_embd * 4

    # 数据
    data_path: str = "data/hongloumeng.txt"
    block_size: int = 256
    train_split: float = 0.9
    clean_data: bool = True
    batch_size: int = 64

    # 训练
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.1
    warmup_steps: int = 100
    eval_interval: int = 500
    eval_steps: int = None # 评估时的最大验证步数，None表示使用整个验证集
    total_steps: int = None

    # 日志
    output_dir: str = "outputs"


def get_model_output_dir(cfg: TrainConfig) -> Path:
    return Path(cfg.output_dir) / cfg.model_name


def get_best_model_path(cfg: TrainConfig) -> Path:
    return get_model_output_dir(cfg) / "best_model.pt"


def get_log_dir(cfg: TrainConfig) -> Path:
    return get_model_output_dir(cfg) / "logs"


def load_data(cfg: TrainConfig):
    """读取文本，构建数据集。"""
    text = Path(cfg.data_path).read_text(encoding="utf-8")
    raw_chars = len(text)
    if cfg.clean_data:
        text = clean_text(text)
    train_ds, val_ds, stoi, itos, vocab_size = build_dataset(
        text, block_size=cfg.block_size, train_split=cfg.train_split, clean=False
    )
    cfg.vocab_size = vocab_size
    print(f"raw chars: {raw_chars} | used chars: {len(text)} | clean_data: {cfg.clean_data}")
    print(f"random baseline loss ~= log(vocab_size): {math.log(vocab_size):.4f}")
    print(f"词表大小: {vocab_size}")
    print(f"训练样本: {len(train_ds)} | 验证样本: {len(val_ds)}")
    return train_ds, val_ds, stoi, itos, vocab_size


def get_lr(step: int, cfg: TrainConfig):
    """
    warmup机制 在前期小 随着step逐渐增大到lr
    Cosine decay 
    progress是0（warmup结束时) 到1 训练结束时的进度比例
    让学习率 从lr减少到0 
    """
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    total_steps = cfg.total_steps or (cfg.eval_interval * cfg.epochs)
    if total_steps <= cfg.warmup_steps:
        return cfg.lr
    progress = (step - cfg.warmup_steps) / (total_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: nn.Module, val_loader, device: str, max_steps: int = None):
    """在验证集上计算平均 loss 和困惑度。"""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    count = 0
    for step, (x, y) in enumerate(val_loader):
        if max_steps is not None and step >= max_steps:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        """
        .view()：改变形状（类似 reshape）
        -1：自动推断剩余维度
        [B, T, vocab] → [B*T, vocab] 
        [B, T] → [B*T]

        flatten是因为cross_entropy 期望输入logits: [N, C]，target: [N]
        """
        total_loss += loss.item() * y.numel()
        count += y.numel()
    if count == 0:
        raise ValueError("validation loader produced no batches")
    avg_loss = total_loss / count
    perplexity = math.exp(avg_loss)
    if was_training:
        model.train()
    return avg_loss, perplexity


def decode_sample(itos, token_ids: torch.Tensor):
    """将 token id 序列解码为字符串。
    i.item() 将pytorch张量中的单个标量转为int
    """
    return "".join(itos[i.item()] for i in token_ids)


def train(cfg: TrainConfig):
    # 模型规模到超参数的映射
    cfg_map = {
        # "small":  dict(n_layer=4, n_head=4, n_embd=128, batch_size=64, lr=3e-4, warmup_steps=100, epochs=12),
        "medium": dict(n_layer=6, n_head=6, n_embd=192, batch_size=64, lr=2e-4, warmup_steps=300, epochs=15),
        "large":  dict(n_layer=8, n_head=8, n_embd=256, batch_size=64, lr=1e-4, warmup_steps=500, epochs=20),
    }
    override = cfg_map.get(cfg.model_name, cfg_map["medium"])
    for k, v in override.items():
        setattr(cfg, k, v)

    # 加载数据
    train_ds, val_ds, stoi, itos, _ = load_data(cfg)
    train_loader, val_loader = get_dataloaders(train_ds, val_ds, cfg.batch_size)
    cfg.total_steps = len(train_loader) * cfg.epochs

    # 初始化模型
    if cfg.d_ff is None:
        cfg.d_ff = cfg.n_embd * 4

    model = GPT(
        vocab_size=cfg.vocab_size,
        context_length=cfg.block_size,
        d_model=cfg.n_embd,
        num_layers=cfg.n_layer,
        num_heads=cfg.n_head,
        d_ff=cfg.d_ff,
    ).to(cfg.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: get_lr(step, cfg) / cfg.lr
    )
    model_output_dir = get_model_output_dir(cfg)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = get_log_dir(cfg)
    log_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard 记录器
    # TensorBoard 记录器（按模型规模分离）
    writer = (
        SummaryWriter(log_dir=str(log_dir))
        if SummaryWriter is not None
        else NullWriter()
    )

    # 记录超参数
    writer.add_text("model/vocab_size", str(cfg.vocab_size))
    writer.add_text("model/n_layer", str(cfg.n_layer))
    writer.add_text("model/n_head", str(cfg.n_head))
    writer.add_text("model/n_embd", str(cfg.n_embd))
    writer.add_text("model/d_ff", str(cfg.d_ff))
    writer.add_text("model/block_size", str(cfg.block_size))
    writer.add_text("train/batch_size", str(cfg.batch_size))
    writer.add_text("train/lr", str(cfg.lr))
    writer.add_text("train/epochs", str(cfg.epochs))
    writer.add_text("train/weight_decay", str(cfg.weight_decay))
    writer.add_text("train/warmup_steps", str(cfg.warmup_steps))
    writer.add_text("train/eval_steps", str(cfg.eval_steps))
    writer.add_text("train/clean_data", str(cfg.clean_data))
    writer.add_text("paths/output_dir", str(model_output_dir))
    writer.add_scalar("model/num_parameters", num_params)

    # 获取当前进程
    process = psutil.Process() if psutil is not None else None
    best_checkpoint_path = get_best_model_path(cfg)
    best_val_loss = float("inf")

    def save_checkpoint(path: Path, current_step: int, epoch_idx: int, val_loss: float):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": asdict(cfg),
                "stoi": stoi,
                "itos": itos,
                "step": current_step,
                "epoch": epoch_idx + 1,
                "val_loss": val_loss,
            },
            path,
        )

    # 训练循环
    model.train()
    step = 0

    for epoch in range(cfg.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")
        for batch_idx, (x, y) in enumerate(pbar):
            step_start = time.time()

            x, y = x.to(cfg.device), y.to(cfg.device)

            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                y.reshape(-1),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            step_time = time.time() - step_start
            lr_now = scheduler.get_last_lr()[0]
            pbar.set_postfix_str(f"loss={loss.item():.4f}, lr={lr_now:.2e}")

            # 记录训练步指标
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/learning_rate", lr_now, step)
            writer.add_scalar("train/global_step", step, step)
            writer.add_scalar("train/epoch", epoch + batch_idx / len(train_loader), step)
            writer.add_scalar("train/step_time_sec", step_time, step)
            tokens_per_sec = cfg.batch_size * cfg.block_size / step_time
            writer.add_scalar("train/tokens_per_sec", tokens_per_sec, step)

            # 记录内存占用
            if torch.cuda.is_available():
                gpu_allocated = torch.cuda.memory_allocated(cfg.device) / (1024 ** 2)
                writer.add_scalar("memory/gpu_allocated_mb", gpu_allocated, step)
            if process is not None:
                sys_mem = process.memory_info().rss / (1024 ** 2)
                writer.add_scalar("memory/system_rss_mb", sys_mem, step)

            if step % cfg.eval_interval == 0:
                eval_start = time.time()
                val_loss, perplexity = evaluate(
                    model,
                    val_loader,
                    cfg.device,
                    max_steps=cfg.eval_steps,
                )
                eval_time = time.time() - eval_start

                writer.add_scalar("eval/loss", val_loss, step)
                writer.add_scalar("eval/perplexity", perplexity, step)
                writer.add_scalar("eval/duration_sec", eval_time, step)

                pbar.write(f"[Eval] step={step} | train_loss={loss.item():.4f}")
                pbar.write(
                    f"[Eval] step={step} | val_loss={val_loss:.4f} | ppl={perplexity:.2f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(best_checkpoint_path, step, epoch, val_loss)
                    pbar.write(
                        f"[Checkpoint] best saved to {best_checkpoint_path} "
                        f"(val_loss={val_loss:.4f})"
                    )
        pbar.close()

    writer.close()

    print("训练完成！")
    return model, itos, stoi


model_configs = [
    {"name": "small",  "n_layer": 4, "n_head": 4, "n_embd": 128},
    # {"name": "medium", "n_layer": 6, "n_head": 6, "n_embd": 192},
    # {"name": "large",  "n_layer": 8, "n_head": 8, "n_embd": 256},
]

if __name__ == "__main__":
    import argparse

    """
    step1 main创建TrainConfig 用mc["name"]去指定大小，决定模型规模
    step2 train(cfg) 调用train函数，传入cfg 
          train函数override = cfg_map.get(cfg.model_name, cfg_map["medium"])
          cfg_map是完整的模型参数列表， cfg_map.get(key,default) 如果key在cfg_map中，则返回cfg_map[key]，否则返回default
          所以从cfg_map种获取新的模型参数，  for k, v in override.items():   setattr(cfg, k, v)   覆盖TrainConfig
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", choices=["small", "medium", "large"])
    parser.add_argument("--all", action="store_true", help="Train small, medium, and large sequentially.")
    args = parser.parse_args()

    if args.all:
        model_names = ["small", "medium", "large"]
    else:
        model_names = args.model or [mc["name"] for mc in model_configs]

    for model_name in model_names:
        cfg = TrainConfig(model_name=model_name)
        train(cfg)
        print(f"最优模型已保存至 {get_best_model_path(cfg)}")
