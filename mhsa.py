import torch
import torch.nn as nn
from einops import rearrange




class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32):
        super().__init__()
        qkv_dim = dim_head*heads*3 # 3 represents q, k, and v
        # print(f"qkv_dim: {qkv_dim}")
        self.to_qkv = nn.Linear(dim, qkv_dim, bias=False)
        self.W0     = nn.Linear(dim_head*heads, dim, bias=False) # q, k, and v are used for the computation of v * softmax(qk/scale_factor)
        self.scale_factor = dim_head ** -0.5
        self.heads = heads

    def forward(self,x):
        assert x.dim()==3  #[batch, tokens, dim]

        qkv = self.to_qkv(x) #[batch, tokens, qkv_dim]

        # print(f"qkv before {qkv.shape}")
        q, k, v = tuple(rearrange(qkv, "b t (d h k) -> k b h t d", h=self.heads, k=3))
        # print(f"q after  {q.shape}") # [batch, heads, tokens, dim]
        # print(f"k after  {k.shape}") # [batch, heads, tokens, dim]
        # print(f"v after  {v.shape}") # [batch, heads, tokens, dim]

        # resulted shape will be: [batch, heads, tokens, tokens]     
        scaled_dot_product = torch.einsum("b h i d , b h j d -> b h i j", q, k) * self.scale_factor
        # print(f"scaled_dot_product: {scaled_dot_product.shape}")
        attention = torch.softmax(scaled_dot_product, dim=-1)
        # print(f"attention: {attention.shape}")

        out = torch.einsum("b h i j, b h j d -> b h i d", attention, v)
        out = rearrange(out, "b h t d -> b t (h d)")


        return self.W0(out)
        
        


if __name__ == "__main__":
    model = MultiHeadSelfAttention(dim=64)
    x = torch.rand(16,10,64)   # [batch, tokens, dim]
    y = model(x)

    print(f"y: {y.shape}")
