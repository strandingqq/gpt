from dataclasses import dataclass
import math
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True

def softmax(x: torch.Tensor, dim:int = -1) -> torch.Tensor:
    x_max = torch.max(x,dim = dim, keepdim=True).values
    x_stable = x - x_max

    exp_x = torch.exp(x_stable)
    sum_exp = torch.sum(exp_x, dim=dim,keepdim=True)
    return exp_x / sum_exp

def scaled_dot_product_attention(
    Q:torch.Tensor,
    K:torch.Tensor,
    V:torch.Tensor,
    mask: torch.Tensor = None
)-> torch.Tensor:
    """
    Q: [batch_size , num_heads , seq_len , head_dim]
    torch.einsum 是一种按照字符串规则做张量运算的函数。
    Q: [..., n, k]
    K: [..., m, k]
    V: [..., m, k]
    为什么qn 其他是m
    scores: [..., n, m]

    等价于Q @ K.transpose(-2, -1) 学习这种方法

    .masked_fill(mask = false , float(-inf))
    把mask种为false的位置，在scores种替换为 -inf
    -inf 是为了softmax - 0 
    """
    d_k = Q.size(-1)
    scores = torch.einsum('..nk, ..mk -> ..nm',Q,K) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    probs = softmax(scores, dim = -1)
    output = torch.einsum('..nm , ..mk',probs,V)
    return output

class CasualSelfAttention(nn.Module):
    """
    为什么多头比单头更强？
    多子空间建模 每个head在不同线性子空间工作
    多种注意力分布并行 弹头每个token只能形成一条权重分布 多头可以形成多条
    降低表达冲突
    多头注意力通过在多个低维子空间中并行学习不同的注意力分布，并在输出端进行线性融合，从而显著提升了模型的表示能力、降低了特征干扰，并增强了对复杂依赖关系的建模能力。
    
    不同head之间的结果没有交互，所以需要一个output_proj，实现跨head的混合
    多头注意力中，每个 head 独立建模 token 间关系，但不同 head 的结果在拼接后仍然彼此独立。输出投影层通过线性变换对各个 head 的结果进行加权组合，实现跨 head 的信息融合，并将表示重新映射到统一特征空间。
    """
    def __init__(self, num_heads:int, d_model:int, max_seq_len:int, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须要可以整除num_heads"
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = nn.Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj = nn.Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj = nn.Linear(d_model,d_model,device=device,dtype=dtype)
        self.output_proj = nn.Linear(d_model,d_model,device=device,dtype=dtype)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        q = rearrange(self.q_proj(x),'a b (c d) -> a c b d', c = self.num_heads)
        k = rearrange(self.k_proj(x),'a b (c d) -> a c b d', c=self.num_heads)
        v = rearrange(self.v_proj(x),'a b (c d) -> a c b d', c=self.num_heads)
        # q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len, device=x.device))
        attention_out = scaled_dot_product_attention(q,k,v,mask=mask)
        attn_out = rearrange(attention_out,'a b c d -> a c (b d)')
        return self.output_proj(attn_out)


class FeedForward(nn.Module):
    """
    传统FFN
    attention的作用是不同token之间的信息交互，
    而ffn是为了每个token的多维向量，在内部做特征变换

    FFN是对每个token独立做的mlp
    mlp 多层全连接网络
    前馈网络（FFN）通过先将特征维度扩展至更高维空间（通常为4倍），再通过非线性激活函数进行变换，最后映射回原始维度，从而提升模型的非线性表达能力。传统 FFN 通常采用 GELU 激活，而现代大模型中常使用 SwiGLU 结构，通过引入门控机制（gating）和 Swish 激活函数，实现更灵活的信息选择与特征交互，从而进一步增强模型性能。
    
    常见FFN结构：
    标准ffn是GELU/RELU
    Linear -- GELU -- Linear
    GLU思想 GEGLU SwiGLU 门控机制，选用不同非线性 不同的名字

    """
    def __init__(self,d_model:int, d_ff:int,device=None ,dtype=None):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.w1 = nn.Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = nn.Linear(d_ff, d_model, device=device, dtype=dtype)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = F.gelu(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x

class SwiGLUFeedForward(nn.Module):
    def __init__(self,d_model:int, d_ff:int,device=None, dtype=None):
        super().__init__()
        self.w1 = nn.Linear(d_model,d_ff,device=device, dtype=dtype)
        self.w2 = nn.Linear(d_model,d_ff,device=device, dtype=dtype)
        self.w3 = nn.Linear(d_ff,d_model,device=device, dtype=dtype)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)

        x = x1 * F.silu(x2)
        x = self.w3(x)

        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """
    GPT Transformer Block。

    使用 Pre-LN:
        x = x + Attention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))
    """
    def __init__(self,d_model:int, num_heads:int, d_ff:int, max_seq_len:int,device=None,dtype=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model,device=device,dtype=dtype)
        self.ln2 = nn.LayerNorm(d_model,device=device,dtype=dtype)
        self.ffn = FeedForward(d_model,d_ff,device=device,dtype=dtype)
        self.attn = CasualSelfAttention(num_heads,d_model,max_seq_len,device=device,dtype=dtype)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class LayerNorm(nn.Module):
    """
    Norm的目标是将神经网络的每一层的输出数据强行拉回到规范范围
    “每一层的数值校准器”
    它将每一个Token看作独立个体，在[b,s,D]中，它关注的是D
    所以在一个batch中，执行B*S次归一化计算

    作用：
    如果没有Norm，信息在穿过LLm的数百层block时，每层叠加/叠乘，梯度爆炸
    训练更快，因为自动归一，可以用更大的学习率 每一层输入分布变化更小，后层不用不断适应前层变化

    LayerNorm - RMSNorm

    Pre-Norm 新 x → LayerNorm → Attention → Add
    Post-Norm x → Attention → Add → LayerNorm
    """
    def __init__(self, d_model:int, eps:float=1e-5,device=None,dtype=None):
        super().__init__()
        factory_kwargs = {'device':device,'dtype':dtype}
        # gamma 缩放参数 初始化为全1
        self.gamma = nn.Parameter(torch.ones(d_model,**factory_kwargs))
        # bias/beta  偏移参数 初始化全0
        self.bias = nn.Parameter(torch.zeros(d_model,**factory_kwargs))
        self.eps = eps

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        计算均值 mean
        计算方差 var
        标准化 (x - mean） / sqrt(var^2 + eps)
        可学习缩放+平移
        weight和bias的作用是 让模型可以偏移标准化，而不是被强制标准化
        在完成标准化之后恢复模型的表达能力

        """
        mean = x.mean(dim = -1, keep_dim = True)
        var = x.var(dim = -1, keepdim = True, unbiased=False)
        x_normed = (x - mean) / torch.sqrt(var+self.eps)
        out = self.gamma * x_normed + self.bias
        return out


class GPT(nn.Module):
    def __init__(self,vocab_size:int ,context_length:int, d_model:int, 
                num_layers:int , num_heads:int,
                d_ff:int , device=None,dtype=None):
        super().__init__()
        """
        nn.Embedding(num_embeddings, embedding_dim)
        做的是一个查表的操作
        token_embedding vocab_size 词表大小，表示token种类/数量
        pos——embedding  block_size 最大序列长度
        二者都是词表映射操作，只不过词表不同。
        token——emb的表是一张包含所有token信息的表格，所以vocab_size 是词/token的数量，
        而位置emb的表是一张包含所有token位置信息的位置索引表
        """
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size = vocab_size, embedding_dim = d_model)
        self.pos_embedding = nn.Embedding(vocab_size = context_length, embedding_dim = d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model,num_heads,d_ff,context_length,device=device,dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln1 = LayerNorm(d_model,device=device,dtype=dtype)
        self.lm_head = nn.Linear(d_model, vocab_size,device=device,dtype=dtype)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        b,s = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, s)
        token_embedding = self.token_embedding(x)
        position_embedding = self.pos_embedding(pos)
        x = token_embedding + position_embedding

        for block in self.blocks:
            x = block(x)

        x = self.ln1(x)
        logits = self.lm_head(x)
        return logits
