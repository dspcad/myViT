import torch
import torch.nn as nn
from einops import rearrange

from transformer import TransformerEncoder


class ViT(nn.Module):
    def __init__(self,
                 img_h=480, 
                 img_w=640, 
                 in_channels=3,
                 patch_dim=16,
                 dim=768,
                 blocks=6,
                 heads=8,
                 dim_linear_block=3072,
                 dim_head=None,
                 dropout=0,
                 num_classes=41,
                 split_gpus=False):
        """
        Args:
            img_h: the spatial image hight
            img_w: the spatial image width
            in_channels: number of img channels
            patch_dim: desired patch dim
            dim: the linear layer's dim to project the patches for MHSA (i.e. dim of the embeddeding)
            blocks: number of transformer blocks
            heads: number of heads
            dim_linear_block: inner dim of the transformer linear block
            dim_head: dim head in case you want to define it. defaults to dim/heads
            dropout: for pos emb and transformer
            num_classes: classification task classes
        """

        super().__init__()
        assert img_h % patch_dim == 0, f'patch size {patch_dim} not divisible for image height {img_h}'
        assert img_w % patch_dim == 0, f'patch size {patch_dim} not divisible for image width {img_w}'
        self.h = img_h
        self.w = img_w
        self.p = patch_dim
        tokens = (img_h // patch_dim) * (img_w // patch_dim)
        self.token_dim = in_channels * (patch_dim ** 2) # 3 * 16 * 16 = 768
        self.dim = dim
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        self.project_patches = nn.Linear(self.token_dim, dim)
        self.split_gpus = split_gpus
        #self.output_map = nn.Linear(dim*int(img_dim/patch_dim)**2, num_classes*img_dim*img_dim)

        self.emb_dropout = nn.Dropout(dropout)

        #self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_emb1D = nn.Parameter(torch.randn(tokens, dim)).cuda(0) # It is a tensor so we need to specify cuda device here

        self.transformer = TransformerEncoder(dim, blocks=blocks, heads=heads,
                                              dim_head=self.dim_head,
                                              dim_linear_block=dim_linear_block,
                                              dropout=dropout,
                                              split_gpus=self.split_gpus)

        # self.conv = nn.Conv2d(3, 41, 1,padding='same')

        if self.split_gpus:
            self.emb_dropout.cuda(0)
            self.project_patches.cuda(0)




    def forward(self, img, mask=None):
        batch_size = img.shape[0]
        #print(f"img: {img.shape}")
        img_patches = rearrange(
            img, 'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.p, patch_y=self.p)
        # print(f"img_pathes: {img_patches.shape}")
        # project patches with linear layer + add pos emb
        img_patches = self.project_patches(img_patches)

        # print(f"pos_emb1D: {self.pos_emb1D.shape}")
        patch_embeddings = self.emb_dropout(img_patches + self.pos_emb1D)

        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
        y = self.transformer(patch_embeddings)
 
        y = rearrange(y, 'b t d -> b (t d)')
        y = rearrange(y, 'b (c h w) -> b c h w', c=41, h=480, w=640)
        # y = y.unsqueeze(1)
        # print(f"y: {y.shape}")
        # y = rearrange(y, 'b c (f h w) -> b (c f) h w', c=3, h=480, w=640)
        #y = self.conv(y)

        return y




if __name__ == "__main__":
    model = ViT(img_h=480, img_w=640, in_channels=3, patch_dim=16, dim=256, blocks=6, heads=8, dim_linear_block=1024, dim_head=256, dropout=0, num_classes=41)
    x = torch.rand(2, 3, 480, 640)
    y = model(x)

    print(f"y: {y.shape}")
    #print(model)
