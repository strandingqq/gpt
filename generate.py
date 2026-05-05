import torch
import torch.nn.functional as F


DEFAULT_PROMPTS = ["宝玉", "黛玉", "凤姐"]
DEFAULT_TEMPERATURES = [0.5, 1.0, 1.5]


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
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    model.eval()
    block_size = model.context_length

    for _ in range(max_new_tokens):
        """
        idx是当前序列的实际长度 初始prompt+已生成token
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
        """
        logits是模型输出的原始分数，未归一化，通过除以temperature来调整分布
        temperature > 1 分布变得更平潭/均匀，更高随机性
        temperature < 1 分布变得更尖锐/确定性，更低随机性
        假设模型对下一个词的打分是 [3.0, 1.0, 0.5]（"猫"最高）
        除以 0.5：[6.0, 2.0, 1.0] → softmax 后"猫"的概率更接近 1.0（更确定）
        除以 2.0：[1.5, 0.5, 0.25] → softmax 后概率更分散（更随机）
        """
        # Temperature 采样
        logits = logits / temperature

        """
        top-k 只从概率最高的k个token中采样
        torch.topk(logits,k) 返回最大的k个值和他们的索引
            min(top_k, logits.size(-1)) 如果词表小于k 就全部取
        v 是 [batch_size, k]
        threshold是取每batch的最小值
        
        logits = torch.where(condition, if_true, if_false)
        condition: logits < threshold 
        内部应该是逐元素修改 full_like dtype、device一致 shape可广播
        """
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            threshold = v[:, [-1]]         # 第 k 大的值作为阈值
            logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)

        """
        选择概率达到阈值p的最小token集合
        torch.cumsum 得到第0项到第i项的概率之和 cumsum = [0.6, 0.85, 1.0]
        累计概率超过 p 的位置mask掉
        torch.gather 按照索引把排序后的logits放回去
        """
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            sorted_indices_to_remove = cumsum > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

        # 转换为概率分布并采样
        probs = F.softmax(logits, dim=-1)
        # 数值安全检查：处理 nan/inf/负数
        probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
        probs = torch.clamp_min(probs, min=0.0)
        probs_sum = probs.sum(dim=-1, keepdim=True)
        zero_rows = probs_sum <= 0
        probs = probs / probs_sum.clamp(min=1e-10)
        if zero_rows.any():
            uniform = torch.full_like(probs, 1.0 / probs.size(-1))
            probs = torch.where(zero_rows, uniform, probs)
        """
        按概率分布 随机采样一个token
        """
        idx_next = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]

        # 拼接
        idx = torch.cat([idx, idx_next], dim=1)

    return idx


def decode(itos, token_ids: torch.Tensor) -> str:
    """将 token id 序列解码为字符串（跳过特殊 token）。"""
    return "".join(itos[i.item()] for i in token_ids)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--checkpoint", help="Optional checkpoint path. Overrides --model default checkpoint lookup.")
    parser.add_argument("--prompt", action="append", help="Prompt text. Can be passed more than once.")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature",
        type=float,
        action="append",
        default=None,
        help="Sampling temperature. Can be passed more than once.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    from train import TrainConfig, decode_sample, get_best_model_path
    from gpt import GPT

    cfg = TrainConfig(model_name=args.model)
    if args.checkpoint:
        load_path = Path(args.checkpoint)
    else:
        load_path = get_best_model_path(cfg)
    if not load_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {load_path}")
    checkpoint = torch.load(load_path, map_location=cfg.device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        saved_cfg = checkpoint.get("config", {})
        for name in ["vocab_size", "block_size", "n_layer", "n_head", "n_embd", "d_ff"]:
            if saved_cfg.get(name) is not None:
                setattr(cfg, name, saved_cfg[name])
        stoi = checkpoint["stoi"]
        itos = checkpoint["itos"]
        state_dict = checkpoint["model_state_dict"]
    else:
        raise ValueError(f"unsupported checkpoint format: {load_path}")
    model = GPT(
        vocab_size=cfg.vocab_size,
        context_length=cfg.block_size,
        d_model=cfg.n_embd,
        num_layers=cfg.n_layer,
        num_heads=cfg.n_head,
        d_ff=cfg.d_ff,
    ).to(cfg.device)

    model.load_state_dict(state_dict)
    model.eval()
    print(f"模型权重已加载 ({args.model})")

    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS
    temperatures = args.temperature if args.temperature is not None else DEFAULT_TEMPERATURES

    for pi, prompt in enumerate(prompts):
        prompt_ids = [stoi[ch] for ch in prompt if ch in stoi]
        if len(prompt_ids) == 0:
            prompt_ids = [0]
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=cfg.device)

        for ti, temp in enumerate(temperatures):
            out = generate(
                model,
                idx,
                max_new_tokens=args.max_new_tokens,
                temperature=temp,
                top_k=args.top_k,
                top_p=args.top_p,
                device=cfg.device,
            )
            text = decode_sample(itos, out[0])
            print(f"\n=== Prompt {pi+1}: '{prompt}' | temp={temp} ===")
            print(text)

