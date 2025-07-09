import os
import pickle
from argparse import Namespace
from collections import OrderedDict

from timm.models.deit import default_cfgs

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# from timm.models._hub import download_cached_file
from timm.models.hub import download_cached_file

import torch

from timm.models.layers import trunc_normal_
from torch import nn

from vision_transformer import VisionTransformer


class VisionTransformerDistilled(VisionTransformer):
    """Vision Transformer w/ Distillation Token and Head

    Distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, *args, **kwargs):
        weight_init = kwargs.pop("weight_init", "")
        super().__init__(*args, **kwargs, weight_init="skip")
        assert self.global_pool in ("token",)

        self.num_tokens = 2
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_tokens, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.distilled_training = False  # must set this True to train w/ distillation token

        self.init_weights(weight_init)

    def init_weights(self, mode=""):
        trunc_normal_(self.dist_token, std=0.02)
        super().init_weights(mode=mode)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed|dist_token",
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],  # final norm w/ last block
        )

    @torch.jit.ignore
    def get_classifier(self):
        return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def forward_features(self, x):
        x_ori_d_attn = []
        x_hard_d_attn = []
        x_ori_d_mlp = []
        x_hard_d_mlp = []
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        assert not (self.grad_checkpointing and not torch.jit.is_scripting())
        x = [x, x]
        for i in range(self.depth):
            x, x_d_attn, x_d_mlp, _ = self.blocks[i](x)
            x_ori_d_attn.append(x_d_attn[0])
            x_hard_d_attn.append(x_d_attn[1])
            x_ori_d_mlp.append(x_d_mlp[0])
            x_hard_d_mlp.append(x_d_mlp[1])
        x_ori, x_hard = x[0], x[1]
        x_ori = self.norm(x_ori)
        x_hard = self.norm(x_hard)
        return x_ori, x_hard, x_ori_d_attn, x_hard_d_attn, x_ori_d_mlp, x_hard_d_mlp

    def forward_head(self, x, pre_logits: bool = False):
        if pre_logits:
            return (x[:, 0] + x[:, 1]) / 2
        x, x_dist = self.head(x[:, 0]), self.head_dist(x[:, 1])
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return x, x_dist
        else:
            # during standard train / finetune, inference average the classifier predictions
            return (x + x_dist) / 2


def convert(ckpt_path):
    # load input
    old_ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = old_ckpt["model"]
    new_ckpt = OrderedDict()
    # print(state_dict.keys())
    for k in list(state_dict.keys()):
        if "block" in k:
            if "attn" in k:
                if "qkv" in k:
                    if "weight" in k:
                        value = state_dict[k].view(9, -1, state_dict[k].size()[1])
                        new_ckpt[k[0:-7] + "1.weight"] = torch.cat((value[0], value[3], value[6]), dim=0)
                        new_ckpt[k[0:-7] + "2.weight"] = torch.cat((value[1], value[4], value[7]), dim=0)
                        new_ckpt[k[0:-7] + "3.weight"] = torch.cat((value[2], value[5], value[8]), dim=0)
                    elif "bias" in k:
                        value = state_dict[k].view(9, -1)
                        new_ckpt[k[0:-5] + "1.bias"] = torch.cat((value[0], value[3], value[6]), dim=0)
                        new_ckpt[k[0:-5] + "2.bias"] = torch.cat((value[1], value[4], value[7]), dim=0)
                        new_ckpt[k[0:-5] + "3.bias"] = torch.cat((value[2], value[5], value[8]), dim=0)
                elif "proj" in k:
                    if "weight" in k:
                        value = state_dict[k].view(state_dict[k].size()[0], 3, -1)
                        new_ckpt[k[0:-7] + "1.weight"] = value[:, 0, :]
                        new_ckpt[k[0:-7] + "2.weight"] = value[:, 1, :]
                        new_ckpt[k[0:-7] + "3.weight"] = value[:, 2, :]
                    elif "bias" in k:
                        new_ckpt[k[0:-5] + "1.bias"] = state_dict[k] / 3
                        new_ckpt[k[0:-5] + "2.bias"] = state_dict[k] / 3
                        new_ckpt[k[0:-5] + "3.bias"] = state_dict[k] / 3
            elif "mlp" in k:
                if "fc1" in k:
                    if "weight" in k:
                        value = state_dict[k].view(3, -1, state_dict[k].size()[1])
                        new_ckpt[k[0:-11] + "1.fc1.weight"] = value[0, :, :]
                        new_ckpt[k[0:-11] + "2.fc1.weight"] = value[1, :, :]
                        new_ckpt[k[0:-11] + "3.fc1.weight"] = value[2, :, :]
                    elif "bias" in k:
                        value = state_dict[k].view(3, -1)
                        new_ckpt[k[0:-9] + "1.fc1.bias"] = value[0]
                        new_ckpt[k[0:-9] + "2.fc1.bias"] = value[1]
                        new_ckpt[k[0:-9] + "3.fc1.bias"] = value[2]
                elif "fc2" in k:
                    if "weight" in k:
                        value = state_dict[k].view(state_dict[k].size()[0], 3, -1)
                        new_ckpt[k[0:-11] + "1.fc2.weight"] = value[:, 0, :]
                        new_ckpt[k[0:-11] + "2.fc2.weight"] = value[:, 1, :]
                        new_ckpt[k[0:-11] + "3.fc2.weight"] = value[:, 2, :]
                    elif "bias" in k:
                        new_ckpt[k[0:-9] + "1.fc2.bias"] = state_dict[k] / 3
                        new_ckpt[k[0:-9] + "2.fc2.bias"] = state_dict[k] / 3
                        new_ckpt[k[0:-9] + "3.fc2.bias"] = state_dict[k] / 3
            else:
                new_ckpt[k] = state_dict[k]
        elif "head" in k:
            print(f"ignore: {k}")
        else:
            new_ckpt[k] = state_dict[k]
    return new_ckpt


def build_model(args, pretrained=True):
    if args.backbone == "deit":
        net = VisionTransformerDistilled(patch_size=16, embed_dim=384, depth=12, num_heads=6, num_classes=args.n_bits)
        if pretrained:
            x = args.model_name.split(".")
            # print(default_cfgs[x[0]]['url'])
            # exit()
            # cached_file = download_cached_file(default_cfgs[x[0]]['url'])
            cached_file = '/home/admin00/Downloads/deit_small_distilled_patch16_224-649709d9.pth'
            ckpt = convert(cached_file)
            msg = net.load_state_dict(ckpt, strict=False)
            print(msg)
    else:
        raise NotImplementedError(f"not support: {args.backbone}")
    return net.cuda()


if __name__ == "__main__":
    _args = Namespace(backbone="deit", n_bits=16, model_name="deit_small_distilled_patch16_224.fb_in1k")
    net = build_model(_args)
    _images = torch.rand(10, 3, 224, 224).cuda()
    out = net(_images)
    for _x in out:
        if isinstance(_x, list):
            print(len(_x))
        else:
            print(_x.shape)

    obj = {"images": _images, "out": out}

    with open("C:/Users/QQ/Desktop/1.pkl", "wb") as f:
        pickle.dump(obj, f)

    # with open("C:/Users/QQ/Desktop/1.pkl", "rb") as f:
    #     obj = pickle.load(f)
    # _images = obj["images"]
    # out = obj["out"]
