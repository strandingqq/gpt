import torch
from torch.utils.data import Dataset, DataLoader


ALLOWED_PUNCTUATION = set(
    "\n\r\t 　，。！？；：、（）《》【】“”‘’\"'…—-·,.!?;:"
)
ALLOWED_EXTRA_CHARS = set("0123456789一二三四五六七八九十百千万零〇")


def clean_text(text: str) -> str:
    """Normalize the source text and remove rare non-Chinese noise chars."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("`", "’").replace("＂", "”").replace("＇", "’")

    cleaned_chars = []
    previous_newline = False
    for ch in text:
        is_chinese = "\u4e00" <= ch <= "\u9fff"
        keep = is_chinese or ch in ALLOWED_PUNCTUATION or ch in ALLOWED_EXTRA_CHARS
        if not keep:
            continue

        if ch == "\n":
            if previous_newline:
                continue
            previous_newline = True
        elif ch not in {" ", "　", "\t"}:
            previous_newline = False
        cleaned_chars.append(ch)

    return "".join(cleaned_chars).strip()


class TextDataset(Dataset):
    """
    字符级文本数据集。

    每个样本为固定长度的 token 序列：
        x  = data[i : i + block_size]       # 输入
        y  = data[i+1 : i + block_size+1]  # 标签（错位1位，用于预测下一个token）
    """
    def __init__(self, data: list[int], block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        """
        返回可以抽取多少个样本
        # 最后 block_size 个位置无法作为样本起点
        """
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        """
        这里是 自监督学习 我们不需要人工标注 标签就是输入错位一位后的序列
        """
        x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return x, y


def build_dataset(
    text: str,
    block_size: int = 256,
    train_split: float = 0.9,
    clean: bool = True,
):
    """
    将原始文本转换为 token id 列表，划分训练集/验证集，返回 Dataset 对象。

    Args:
        text:        原始文本字符串
        block_size:  最大上下文长度（每个样本的序列长度）
        train_split: 训练集比例（默认 90%）

    Returns:
        train_dataset, val_dataset, stoi, itos, vocab_size

    set(text) 把文本转成集合，自动去重，得到所有不重复的字符
    stoi：strint 2 int 编码用
    itos：int 2 string 解码用

    """
    # 字符级分词：每个汉字作为一个 token
    if clean:
        text = clean_text(text)

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    """
    整个文本转为 id 序列
    stoi[ch] 获取字符 ch 对应的 id
    """
    data = [stoi[ch] for ch in text]

    """
    划分训练集 / 验证集
    train——split默认0.9  前0.9为训练数据 后0.1为val数据
    """
    split = int(train_split * len(data))
    train_data = data[:split]
    val_data   = data[split:]

    train_dataset = TextDataset(train_data, block_size)
    val_dataset   = TextDataset(val_data,   block_size)

    return train_dataset, val_dataset, stoi, itos, vocab_size


def get_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 64,
):
    """
    构建训练集和验证集的 DataLoader。

    Args:
        train_dataset: 训练集 Dataset
        val_dataset:   验证集 Dataset
        batch_size:    批大小

    Returns:
        train_loader, val_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # 快速测试：读取红楼梦，验证数据处理流程
    with open("data/hongloumeng.txt", "r", encoding="utf-8") as f:
        text = f.read()

    train_ds, val_ds, stoi, itos, vocab_size = build_dataset(text, block_size=64)

    print(f"词表大小: {vocab_size}")
    print(f"cleaned chars: {len(clean_text(text))} / raw chars: {len(text)}")
    print(f"训练样本数: {len(train_ds)}")
    print(f"验证样本数: {len(val_ds)}")

    # 查看一个样本
    x, y = train_ds[0]
    print(f"x shape: {x.shape}, y shape: {y.shape}")

    # 解码验证
    decoded = "".join(itos[i.item()] for i in x[:20])
    print(f"解码前20字: {decoded}")

    # DataLoader 测试
    train_loader, val_loader = get_dataloaders(train_ds, val_ds, batch_size=8)
    xb, yb = next(iter(train_loader))
    print(f"Batch x shape: {xb.shape}, Batch y shape: {yb.shape}")
