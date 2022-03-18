import torch
from torch import nn
from mhsa import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, dim_linear_block=1024, dropout=0.1, activation=nn.GELU):
        """
        Args:
            dim: token's vector length
            heads: number of heads
            dim_head: if none dim/heads is used
            dim_linear_block: the inner projection dim
            dropout: probability of droppping values
        """

        super().__init__()
        self.mhsa    = MultiHeadSelfAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.norm1   = nn.LayerNorm(dim)
        self.norm2   = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )


    def forward(self,x):
        #y = self.norm1(self.dropout(self.mhsa(x))) + x
        #return self.mlp(self.norm2(y)) + y

        #print('After module x: device {}, shape {}\n'.format(x.device, x.shape))
        y = self.dropout(self.mhsa(self.norm1(x)))+x
        #y = self.dropout(self.mhsa(x))+x

        #print('After module y: device {}, shape {}\n'.format(y.device, y.shape))
        return self.mlp(self.norm2(y)) + y




class TransformerEncoder(nn.Module):
    def __init__(self, dim, blocks=6, heads=8, dim_head=None, dim_linear_block=1024, dropout=0,split_gpus=False):
        super().__init__()
        #self.block_list = [TransformerBlock(dim, heads, dim_head, dim_linear_block, dropout) for _ in range(blocks)]
        #self.encoders = nn.ModuleList(self.block_list)

        self.split_gpus = split_gpus
        #print(f"DEBUG: self.split_gpus: {self.split_gpus}")
        self.block_0 = TransformerBlock(dim, heads, dim_head, dim_linear_block, dropout)
        self.block_1 = TransformerBlock(dim, heads, dim_head, dim_linear_block, dropout)
        self.block_2 = TransformerBlock(dim, heads, dim_head, dim_linear_block, dropout)
        self.block_3 = TransformerBlock(dim, heads, dim_head, dim_linear_block, dropout)
        self.block_4 = TransformerBlock(dim, heads, dim_head, dim_linear_block, dropout)
        self.block_5 = TransformerBlock(dim, heads, dim_head, dim_linear_block, dropout)



        if self.split_gpus:
            self.block_0.cuda(0)
            self.block_1.cuda(1)
            self.block_2.cuda(1)
            self.block_3.cuda(2)
            self.block_4.cuda(2)
            self.block_5.cuda(3)

    def forward(self,x):
        #for enc in self.encoders:
        #    x = enc(x)
        x = self.block_0(x)
        if self.split_gpus:
            x = x.cuda(1)

        x = self.block_1(x)
        x = self.block_2(x)

        if self.split_gpus:
            x = x.cuda(2)
        x = self.block_3(x)
        x = self.block_4(x)

        if self.split_gpus:
            x = x.cuda(3)
        x = self.block_5(x)


        return x


if __name__ == "__main__":
    model = TransformerEncoder(dim=64,dim_head=64)
    x = torch.rand(16,10,64)   # [batch, tokens, dim]
    y = model(x)

    print(f"y: {y.shape}")


