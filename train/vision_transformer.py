import math
from functools import partial

import torch
from timm.models.layers import DropPath, Mlp, PatchEmbed, trunc_normal_
from timm.models.helpers import named_apply

from timm.models.vision_transformer import (
    LayerScale,
    init_weights_vit_timm,
    _load_weights,
    get_init_weights_vit,
)
from torch import nn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv3 = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(dim // 3, dim)
        self.proj2 = nn.Linear(dim // 3, dim)
        self.proj3 = nn.Linear(dim // 3, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.gate = nn.Linear(dim, 3, bias=False)

    def forward(self, x, signal):
        B, N, C = x.shape

        prob = self.gate(x)

        prob_soft = torch.nn.functional.softmax(prob, dim=-1)
        prob_hard = torch.nn.functional.one_hot(torch.argmax(prob_soft, dim=-1), 3)

        prob_true = (prob_hard - prob_soft).detach() + prob_soft

        qkv1 = self.qkv1(x).reshape(B, N, 3, self.num_heads // 3, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv2(x).reshape(B, N, 3, self.num_heads // 3, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv3 = self.qkv3(x).reshape(B, N, 3, self.num_heads // 3, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q1, k1, v1 = qkv1.unbind(0)  # make torch script happy (cannot use tensor as tuple)
        q2, k2, v2 = qkv2.unbind(0)
        q3, k3, v3 = qkv3.unbind(0)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)

        attn3 = (q3 @ k3.transpose(-2, -1)) * self.scale
        attn3 = attn3.softmax(dim=-1)
        attn3 = self.attn_drop(attn3)

        x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 3)
        x1 = self.proj1(x1)
        x1 = self.proj_drop(x1)
        # x1 = x1 * prob_true[:,:, 0].view(B, N, 1)

        x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 3)
        x2 = self.proj2(x2)
        x2 = self.proj_drop(x2)
        # x2 = x2 * prob_true[:,:, 1].view(B, N, 1)

        x3 = (attn3 @ v3).transpose(1, 2).reshape(B, N, C // 3)
        x3 = self.proj3(x3)
        x3 = self.proj_drop(x3)
        # x3 = x3 * prob_true[:,:, 2].view(B, N, 1)

        if signal == "ori":
            x = x1 + x2 + x3
            return x
        else:
            prob_d = self.gate(x.detach())
            prob_soft_d = torch.nn.functional.softmax(prob_d, dim=-1)
            prob_hard_d = torch.nn.functional.one_hot(torch.argmax(prob_soft_d, dim=-1), 3)
            prob_true_d = (prob_hard_d - prob_soft_d).detach() + prob_soft_d
            x = (
                x1 * prob_true[:, :, 0].view(B, N, 1)
                + x2 * prob_true[:, :, 1].view(B, N, 1)
                + x3 * prob_true[:, :, 2].view(B, N, 1)
            )
            x_d = (
                x1.detach() * prob_true_d[:, :, 0].view(B, N, 1)
                + x2.detach() * prob_true_d[:, :, 1].view(B, N, 1)
                + x3.detach() * prob_true_d[:, :, 2].view(B, N, 1)
            )
            return x, x_d, torch.argmax(prob_true, dim=-1).view(-1, 1)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp1 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio / 3), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp2 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio / 3), act_layer=act_layer, drop=drop)

        self.mlp3 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio / 3), act_layer=act_layer, drop=drop)

        self.gate = nn.Linear(dim, 3, bias=False)

    def forward(self, x):
        x_ori, x_hard = x[0], x[1]
        x_ori = x_ori + self.drop_path1(self.ls1(self.attn(self.norm1(x_ori), signal="ori")))
        x_ori_d_attn = x_ori.detach()
        x_hard, x_hard_d_attn, prob_attn = self.attn(self.norm1(x_hard), signal="hard")
        x_hard = x[1] + self.drop_path1(self.ls1(x_hard))
        x_hard_d_attn = x[1].detach() + self.drop_path1(self.ls1(x_hard_d_attn))

        B, N, C = x_ori.shape
        x_ori = (
            x_ori
            + self.drop_path2(self.ls2(self.mlp1(self.norm2(x_ori))))
            + self.drop_path2(self.ls2(self.mlp2(self.norm2(x_ori))))
            + self.drop_path2(self.ls2(self.mlp3(self.norm2(x_ori))))
        )
        x_ori_d_mlp = x_ori.detach()

        prob = self.gate(x_hard)

        prob_soft = nn.functional.softmax(prob, dim=-1)
        prob_hard = nn.functional.one_hot(torch.argmax(prob_soft, dim=-1), 3)
        prob_true = (prob_hard - prob_soft).detach() + prob_soft

        x_hard = (
            x_hard
            + self.drop_path2(self.ls2(self.mlp1(self.norm2(x_hard)))) * prob_true[:, :, 0].view(B, N, 1)
            + self.drop_path2(self.ls2(self.mlp2(self.norm2(x_hard)))) * prob_true[:, :, 1].view(B, N, 1)
            + self.drop_path2(self.ls2(self.mlp3(self.norm2(x_hard)))) * prob_true[:, :, 2].view(B, N, 1)
        )
        x_hard_d_mlp = (
            x_hard.detach()
            + self.drop_path2(self.ls2(self.mlp1(self.norm2(x_hard.detach())))) * prob_true[:, :, 0].view(B, N, 1)
            + self.drop_path2(self.ls2(self.mlp2(self.norm2(x_hard.detach())))) * prob_true[:, :, 1].view(B, N, 1)
            + self.drop_path2(self.ls2(self.mlp3(self.norm2(x_hard.detach())))) * prob_true[:, :, 2].view(B, N, 1)
        )

        return (
            [x_ori, x_hard],
            [x_ori_d_attn, x_hard_d_attn],
            [x_ori_d_mlp, x_hard_d_mlp],
            [prob_attn, torch.argmax(prob_true, dim=-1).view(-1, 1)],
        )


class VisionTransformer(nn.Module):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        init_values=None,
        class_token=True,
        fc_norm=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        block_fn=Block,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1 if class_token else 0
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if self.num_tokens > 0 else None
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + self.num_tokens, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    init_values=init_values,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.depth = depth

        if weight_init != "skip":
            self.init_weights(weight_init)

    def init_weights(self, mode=""):
        assert mode in ("jax", "jax_nlhb", "moco", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "token")
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x_ori, x_hard, x_ori_d_attn, x_hard_d_attn, x_ori_d_mlp, x_hard_d_mlp = self.forward_features(x)
        x_ori = self.forward_head(x_ori)
        x_hard = self.forward_head(x_hard)
        if self.training:
            return (self.l2_norm(x_ori), self.l2_norm(x_hard), x_ori_d_attn, x_hard_d_attn, x_ori_d_mlp, x_hard_d_mlp)
        else:
            return self.l2_norm(x_ori)
