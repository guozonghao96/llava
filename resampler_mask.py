import torch
import torch.nn as nn

class CrossAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionModule, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        # query: [seq_len, batch_size, embed_dim]
        # key, value: [seq_len, batch_size, embed_dim]
        # key_padding_mask: [batch_size, seq_len] 如果某些键不应该被注意到，可以设置为True
        # attn_mask: [seq_len, seq_len] 在attention权重上应用的掩码
        attn_output, attn_output_weights = self.multihead_attn(query, key, value,
                                                               key_padding_mask=key_padding_mask,
                                                               attn_mask=attn_mask)
        return attn_output, attn_output_weights

# 参数设置
embed_dim = 256  # 嵌入维度
num_heads = 8    # 注意力头数
seq_len = 10     # 序列长度
batch_size = 3   # 批次大小
seq_len_q = 2     # 序列长度

# 创建模型
cross_attention = CrossAttentionModule(embed_dim, num_heads)

# 随机生成数据
query = torch.rand(seq_len_q, batch_size, embed_dim)
key = torch.rand(seq_len, batch_size, embed_dim)
value = torch.rand(seq_len, batch_size, embed_dim)

# 创建一个attn_mask，这里用一个简单的例子：屏蔽后半部分序列
attn_mask = torch.zeros(batch_size, num_heads, seq_len_q, seq_len)
# attn_mask[:, seq_len//2:] = float('-inf')  # 使用-inf来确保这部分的权重在softmax后为0

token_nums = [
    9, 10, 3
]
for k, num in enumerate(token_nums):
    attn_mask[k][:, :, num:] = float('-inf')
print(attn_mask)
attn_mask = attn_mask.reshape(-1, seq_len_q, seq_len)
# 前向传播
output, weights = cross_attention(query, key, value, attn_mask=attn_mask)
print(output.shape, weights.shape)
# print("Attention Output Shape:", output.shape)
# print("Attention Weights Shape:", weights.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 20))
for i, attn_weight in enumerate(weights):
    plt.subplot(1, len(weights), i + 1)
    print(attn_weight.detach().numpy())
    plt.imshow(attn_weight.detach().numpy())
plt.savefig('attn_weights.png')