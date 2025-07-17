import torch
from torch._refs import infer_aten_op
import torch.nn as nn
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # # 使用截断正态分布初始化
        std = math.sqrt(2/(in_features+out_features))  # 输入必须是tensor
        # 初始化一个全0的有形状的矩阵
        weights_tensor = torch.empty(
            (in_features, out_features), device=device, dtype=dtype)
        # 使用torch.nn.init.trunc_normal_初始化
        nn.init.trunc_normal_(weights_tensor, mean=0.0,
                              std=std, a=-3*std, b=3*std)
        self.weight = nn.Parameter(weights_tensor, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用矩阵乘法
        out = torch.matmul(x, self.weight)
        return out


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        num_embeddings : size of the vocabulary
        """
        super().__init__()
        self.num_embed = num_embeddings
        self.embed_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        # 截断正态分布初始化
        weight_tensor = torch.empty(
            (self.num_embed, self.embed_dim), device=device, dtype=dtype)
        nn.init.trunc_normal_(weight_tensor, mean=0, std=1, a=-3, b=3)
        # 嵌入层参数
        self.weight = nn.Parameter(weight_tensor, requires_grad=True)

    def forward(self, input_ids):
        """
        weights: (num_embed,embed_dim)
        input_ids: (batch_size,seq_len)
        weights[0]表示id为0的一个嵌入层行向量 [1,emb_dim]
        """
        # 使用索引来查找向量
        return self.weight[input_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        # RMSNorm的权重一般初始化为1，形状为[d_model]，因为是对每个特征维度进行缩放
        tensor_values = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(tensor_values, requires_grad=True)

    def forward(self, x: torch.Tensor):
        """
        x的形状应该是(batch_size,seq_len,d_model)
        计算一个样本中的带eps的RMS值
        """
        in_type = x.dtype
        # 计算RMS值 x:(batch_size,seq_len,d_model)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        # 归一化
        x = x / rms
        # 应用缩放参数
        x = x * self.weight
        return x.to(in_type)


class SwiGLUFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        # k = (d_model * 8 / 3) // 64
        # self.d_ff = k * 64 # 向下取整的d_ff
        self.d_ff = d_ff
        # 初始化3个w矩阵
        self.w1 = Linear(d_model,d_ff)
        self.w3 = Linear(d_model,d_ff)
        self.w2 = Linear(d_ff,d_model)

    def forward(self, x):
        """
        Args:
            x: (batch_size,seq_len,d_model)
        w1,w3: (d_model,d_ff)
        w2: (d_ff,d_model)
        W2(SiLU(W1x) ⊙ W3x)
        """
        x1 = self.w1(x)

        x1 = torch.sigmoid(x1) * x1  # SiLU

        x2 = self.w3(x)  # GLU

        values = self.w2(x1*x2)  # SwiGLU
        return values


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # 初始化旋转位置编码的参数
        self.theta = theta
        self.d_k = d_k
        self.max_len = max_seq_len
        self.device = device

        # 生成角度值
        position = torch.arange(
            max_seq_len, device=device).unsqueeze(1)  # [S,1]
        div_term = torch.exp(torch.arange(0, d_k, 2) *
                             (- math.log(theta) / d_k))  # [d_k/2]
        self.angel = position * div_term  # [S,d_k/2] 可以按照位置来查对应的隐藏层各个维度的角度

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        """
        x: [batch_size, seq_len, d_k] ,x is query or key
        token_positions: [batch_size, seq_len]

        对x进行旋转特定角度 得到的是RoPE(x) -> x

        """
        # 根据位置索引来查角度值
        angels = self.angel[token_positions]  # [batch_size,n_head,seq_len,d_k/2]
        # 求基数维度和偶数维度的x值
        x_even = x[..., 0::2]  # [ batch_size,n_head,seq_len , d_k/2]
        x_odd = x[..., 1::2]
        # 求旋转后的x值
        rotated_even = x_even * torch.cos(angels) - x_odd * torch.sin(angels)
        rotated_odd = x_even * torch.sin(angels) + x_odd * torch.cos(angels)

        rotated = torch.zeros_like(x, device=self.device)
        rotated[..., 0::2] = rotated_even
        rotated[..., 1::2] = rotated_odd

        return rotated


def softmax(x: torch.Tensor, dim: int = -1):
    # 归一化
    # 在指定维度上求最大值,keepdim=True保持维度不变
    max_val = torch.max(x, dim=dim, keepdim=True)[0]  # [0]是值 [1]是最大值的索引
    x = x - max_val

    # [...,dim,...]->[...,1,...]
    div_term = torch.sum(torch.exp(x), dim=dim, keepdim=True)
    x = torch.exp(x) / div_term

    return x


def scaled_dot_product_attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    缩放点积注意力 把输入的QKV计算成注意力分数
    """
    # Q K内积
    attention = torch.matmul(queries, keys.transpose(-1, -2))
    scale = queries.shape[-1] ** -0.5
    attention:torch.Tensor = attention * scale  # [batch_size,len_q, len_k ]

    # 使用掩码 在seq_len这个维度上面
    if mask is not None:
        # 1.使用广播机制
        # attention = attention + torch.where(mask,False,float('-inf'))
        # 2.错误方法
        # attention[mask == False] += -math.inf 
        # 3.使用masked fill函数
        attention.masked_fill_(~mask , float('-inf')) # mask_fill是返回新张量 masked_fill_是就地操作
    # softmax
    attention = softmax(attention, dim=-1)

    # values shape:[B,len_k,d_k]
    attention_score = torch.matmul(attention, values)

    return attention_score

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = self.d_model // self.n_head

        self.q_proj = Linear(self.d_model, self.d_model)
        self.k_proj = Linear(self.d_model, self.d_model)
        self.v_proj = Linear(self.d_model, self.d_model)
        self.output_proj = Linear(self.d_model, self.d_model)

    def forward(self, in_features: torch.Tensor):
        batch_size = in_features.size(0)

        q = self.q_proj(in_features)
        k = self.k_proj(in_features)
        v = self.v_proj(in_features)
        # q k分头
        q = q.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        # 计算casual mask 后续掩码 保证前面token看不到后面的token
        seq_len = q.size(-2)
        # 下三角序列 可以保证前面看不到后面 mask shape:[len_q,len_k]
        mask = torch.tril(torch.ones((seq_len, seq_len),
                          device=q.device, dtype=torch.bool))

        context = scaled_dot_product_attention(q, k, v, mask)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)

        return self.output_proj(context)

class CausalMultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, n_head: int,theta:float,max_len:int):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = self.d_model // self.n_head
        # 加入rope 这里要注意 rope的d_model在多头自注意力中是head_dim 不能跨头来做角度
        self.rope = RotaryPositionalEmbedding(theta,self.head_dim,max_len)

        self.q_proj = Linear(self.d_model, self.d_model)
        self.k_proj = Linear(self.d_model, self.d_model)
        self.v_proj = Linear(self.d_model, self.d_model)
        self.output_proj = Linear(self.d_model, self.d_model)

    def forward(self, in_features: torch.Tensor,token_positions:torch.Tensor):
        batch_size = in_features.size(0)

        q = self.q_proj(in_features)
        k = self.k_proj(in_features)
        v = self.v_proj(in_features)
        # q k分头
        q = q.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        # 计算q和v的旋转位置编码 旋转位置编码的隐藏层维度应该是head dim的维度 角度旋转依靠的是head_dim
        q = self.rope(q,token_positions)
        k = self.rope(k,token_positions)
        # 计算casual mask 后续掩码 保证前面token看不到后面的token
        seq_len = q.size(-2)
        # 下三角序列 可以保证前面看不到后面 mask shape:[len_q,len_k]
        mask = torch.tril(torch.ones((seq_len, seq_len),
                          device=q.device, dtype=torch.bool))

        context = scaled_dot_product_attention(q, k, v, mask)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)

        return self.output_proj(context)

class TransformerBlock(nn.Module):
    def __init__(self,d_model,n_head,theta,max_len,d_ff):
        super().__init__()
        self.attn = CausalMultiHeadSelfAttentionWithRoPE(d_model,n_head,theta,max_len)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForwardNet(d_model,d_ff)

    def forward(self,in_features):
        # 输入的残差
        residual = in_features
        # 注意力计算的prenorm
        normed_x = self.ln1(in_features)
        # 先计算positions矩阵 [batch_size,seq_len]  默认x是3d张量
        batch_size = in_features.size(0)
        seq_len = in_features.size(1)
        # 生成位置矩阵,从0开始递增到seq_len-1
        positions = torch.arange(seq_len, device=in_features.device)
        # 扩展维度以匹配batch_size [batch_size, seq_len]
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        # 注意力分数
        atten = self.attn(normed_x,positions)
        # 注意力分数+残差x
        x = residual + atten
        
        # ffn层保留残差
        residual = x
        # ffn的prenorm
        normed_x = self.ln2(x)
        # 作用ffn
        x = self.ffn(normed_x)
        # ffn+残差连接
        output = x + residual
        
        return output
        
class Transformer(nn.Module):
    def __init__(self, vocab_size:int,context_length:int,d_model,d_ff,theta,num_layers,num_head):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size,d_model)
        self.layers = nn.ModuleList([
                TransformerBlock(d_model,num_head,theta,context_length,d_ff) 
                for _ in range(num_layers)]
                                    )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model,vocab_size)
        
    def forward(self,in_indicies:torch.Tensor):
        embedding = self.token_embeddings(in_indicies)
        
        x = embedding
        # 经过多层transformer block 
        for layer in self.layers:
            x = layer(x)
        
        # norm一下
        normed_x = self.ln_final(x)
        # 把结果输出位词典的概率分布
        output = self.lm_head(normed_x)
        return output
        

def cross_entropy(inputs:torch.Tensor,targets:torch.Tensor):
    """
    Args:
        inputs: (batch-size,vocab_size)
        targes: (batch_size)
    为了维护数值稳定性 要减去最大值 是减去不是去掉最大值
    """        
    inputs = inputs - inputs.max(dim=-1,keepdim=True).values
    
    # log-softmax 直接先softmax再log可能会数值不稳定 
    # log_sum_exp = torch.log( torch.sum( torch.exp(inputs) ,dim=-1,keepdim=True)) #(dim_1,1)
    # log_probs = inputs - log_sum_exp
    
    log_probs = inputs - torch.logsumexp(inputs,dim=-1,keepdim=True) # 直接调用native接口
    losses = -log_probs[torch.arange(inputs.size(0)) , targets] # 前面选择样本 后面选择目标id的log概率
    # print(losses)
    return losses.mean()

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr<0:
            raise ValueError(f'invalid learning rate: {lr}')
        defaults = {'lr':lr}
        super().__init__(params, defaults)
        
        
    def step(self,closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr'] # 获取学习率
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p] # 获取和p关联的状态
                t = state.get('t',0) # 获取t的迭代次数
                grad = p.grad.data # 获取关于p的损失的梯度
                p.data -= lr / math.sqrt(t+1) * grad # 用梯度 更新 参数
                state['t'] = t + 1
        return loss

if __name__ == '__main__':
    weights = torch.nn.Parameter(2 * torch.randn((10, 10))) 
    opt = SGD([weights], lr=5)
    
    for t in range(100):
        opt.zero_grad()
        loss = ((weights-1) ** 2).mean()
        print(f'loss:{loss.cpu().item()}')
        loss.backward()
        opt.step()
        
        

# if __name__ == "__main__":

#     vocab_size = 1000
#     context_len = 512
#     d_model = 32
#     d_ff = 128
#     theta = 0.1
#     num_layers = 2
#     num_head = 4
    
#     transformer = Transformer(vocab_size,context_len,d_model,d_ff,theta,num_layers,num_head)
    
#     block = TransformerBlock(d_model,num_head,theta,context_len,d_ff)
    
#     self_attn = CausalMultiHeadSelfAttentionWithRoPE(d_model,num_head,theta,context_len)
    
#     ffn = SwiGLUFeedForwardNet(d_model,d_ff)
    
#     print(transformer.state_dict().keys())
    # print(block.state_dict().keys())
    # print(self_attn.state_dict().keys())
    # print(ffn.state_dict().keys())
