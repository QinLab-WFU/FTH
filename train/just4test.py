import pickle
from collections import OrderedDict

import timm
import torch
from timm.models.hub import download_cached_file


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
        else:
            new_ckpt[k] = state_dict[k]
    return new_ckpt


if __name__ == "__main__":
    net = timm.create_model("deit_small_distilled_patch16_224", pretrained=False, num_classes=0).cuda()
    cached_file = download_cached_file(
        "https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth"
    )
    print(cached_file)
    ckpt = convert(cached_file)
    msg = net.load_state_dict(ckpt, strict=False)
    print(msg)

    with open("C:/Users/QQ/Desktop/1.pkl", "rb") as f:
        obj = pickle.load(f)
    _images = obj["images"]
    out = obj["out"]

    out2 = net(_images)

    for i in range(len(out)):
        if isinstance(out[i], torch.Tensor):
            x1 = out[i]
            x2 = out2[i]
        else:
            x1 = torch.cat(out[i])
            x2 = torch.cat(out2[i])

        if (x1 == x2).all():
            print(i, "equals")
        else:
            print(i, "equals")
