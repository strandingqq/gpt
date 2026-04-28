import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    device: str = None,
) -> torch.Tensor:
    """
    自回归文本生成，支持 temperature、top-k、top-p 采样。

    Args:
        model:           GPT 模型实例
        idx:             初始 的token 序列，shape [batch_size, seq_len]
        max_new_tokens:  需要生成的新 token 数量
        temperature:     温度参数（越大越随机，越小越确定性），默认 1.0
        top_k:           若设置，只保留概率最高的 k 个 token
        top_p:           若设置，只保留累计概率不超过 p 的 token（nucleus sampling）
        device:          模型所在设备

    Returns:
        包含生成结果的完整序列，shape [batch_size, seq_len + max_new_tokens]
    """
    if device is None:
        device = idx.device

    model.eval()
    block_size = model.context_length #

    for _ in range(max_new_tokens):
        """
        截断到模型支持的最大长度
        如果idx的序列长度大于block_size，则截断到block_size
        这也就是为什么需要越来越大的上下文窗口
        """
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        """
        前向传播,得到的结果是每个token的logits
        [batch_size, seq_len, vocab_size]
        取最后一个 timestep: [batch_size, vocab_size] 也就是对于新一个词的预测
        """
        logits = model(idx_cond)          # [batch_size, seq_len, vocab_size]
        logits = logits[:, -1, :]          # 取最后一个 timestep: [batch_size, vocab_size]

        # Temperature 采样
        logits = logits / temperature

        # Top-k 采样
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            threshold = v[:, [-1]]         # 第 k 大的值作为阈值
            logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)

        # Top-p (nucleus) 采样
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            # 累计概率超过 p 的位置mask掉
            sorted_masked = sorted_logits.clone()
            sorted_masked[cumsum > top_p] = float("-inf")
            # 恢复到原始顺序
            logits = torch.gather(sorted_masked, -1, sorted_idx.argsort(-1))

        # 转换为概率分布并采样
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]

        # 拼接
        idx = torch.cat([idx, idx_next], dim=1)

    model.train()
    return idx


def decode(itos, token_ids: torch.Tensor) -> str:
    """将 token id 序列解码为字符串（跳过特殊 token）。"""
    return "".join(itos[i.item()] for i in token_ids)
