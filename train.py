import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from gpt import GPT, GPTConfig
from data import build_dataset, get_dataloaders


@dataclass
class TrainConfig:
    # 数据
    data_path: str = "data/hongloumeng.txt"
    block_size: int = 256 # 每个样本的 token 序列长度（即模型的上下文窗口大小）
    train_split: float = 0.9
    batch_size: int = 64

    # 模型
    vocab_size: int = None  # 由数据决定
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    d_ff: int = None        # 默认 n_embd * 4

    # 训练
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.1
    warmup_steps: int = 100
    eval_interval: int = 100
    eval_steps: int = 50


def load_data(cfg: TrainConfig):
    """读取文本，构建数据集。"""
    text = Path(cfg.data_path).read_text(encoding="utf-8")
    train_ds, val_ds, stoi, itos, vocab_size = build_dataset(
        text, block_size=cfg.block_size, train_split=cfg.train_split
    )
    cfg.vocab_size = vocab_size
    print(f"词表大小: {vocab_size}")
    print(f"训练样本: {len(train_ds)} | 验证样本: {len(val_ds)}")
    return train_ds, val_ds, stoi, itos


def get_lr(step: int, cfg: TrainConfig):
    """
    warmup机制 在前期小 随着step逐渐增大到lr
    Cosine decay 
    progress是0（warmup结束时) 到1 训练结束时的进度比例
    让学习率 从lr减少到0 
    """
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / (cfg.eval_interval * cfg.epochs - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: nn.Module, val_loader, device: str):
    """在验证集上计算平均 loss 和困惑度。"""
    model.eval()
    total_loss = 0.0
    count = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), 
                                            y.view(-1))
        """
        .view()：改变形状（类似 reshape）
        -1：自动推断剩余维度
        [B, T, vocab] → [B*T, vocab] 
        [B, T] → [B*T]

        flatten是因为cross_entropy 期望输入logits: [N, C]，target: [N]
        """
        total_loss += loss.item() * x.size(0)
        count += x.size(0)
    avg_loss = total_loss / count
    perplexity = math.exp(avg_loss)
    model.train()
    return avg_loss, perplexity


def decode_sample(itos, token_ids: torch.Tensor):
    """将 token id 序列解码为字符串。
    i.item() 将pytorch张量中的单个标量转为int
    """
    return "".join(itos[i.item()] for i in token_ids)


def train(cfg: TrainConfig):
    # 加载数据
    train_ds, val_ds, stoi, itos, _ = load_data(cfg)
    train_loader, val_loader = get_dataloaders(train_ds, val_ds, cfg.batch_size)

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
    """
    LambdaLR: 使用一个lambda函数来计算学习率，这个函数接受一个step参数，返回一个学习率
    学习使用不同lr调度
    """
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: get_lr(step, cfg))

    # 训练循环
    model.train()
    step = 0
    for epoch in range(cfg.epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(cfg.device), y.to(cfg.device)

            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, cfg.vocab_size),
                y.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            """
            梯度裁剪 将所有参数的梯度范数裁剪到最大1.0 防止梯度爆炸
            """
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            # 打印训练进度
            if step % cfg.eval_interval == 0:
                lr_now = scheduler.get_last_lr()[0]
                val_loss, perplexity = evaluate(model, val_loader, cfg.device)
                print(
                    f"epoch={epoch+1}/{cfg.epochs} | "
                    f"step={step} | "
                    f"train_loss={loss.item():.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"ppl={perplexity:.2f} | "
                    f"lr={lr_now:.2e}"
                )

    print("训练完成！")
    return model, itos


if __name__ == "__main__":
    cfg = TrainConfig()
    model, itos = train(cfg)

    # 保存模型
    torch.save(model.state_dict(), "gpt_weights.pt")
    print("模型权重已保存至 gpt_weights.pt")

    # 用训练好的模型生成一段文本
    from generate import generate

    prompt = "宝玉"
    prompt_ids = [stoi[ch] for ch in prompt if ch in stoi]
    if len(prompt_ids) == 0:
        prompt_ids = [0]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=cfg.device)
    output = generate(model, idx, max_new_tokens=100, temperature=1.0, top_k=40, top_p=0.9, device=cfg.device)
    generated_text = decode_sample(itos, output[0])
    print("生成结果：")
    print(generated_text)
