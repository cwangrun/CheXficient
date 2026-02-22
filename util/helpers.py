import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F


@torch.no_grad()
def sinkhorn_knopp(out, n_iterations=3, epsilon=0.05, use_gumbel=False):
    L = torch.exp(out / epsilon).t()  # shape: [K, B,]
    K, B = L.shape

    # make the matrix sums to 1
    sum_L = torch.sum(L)
    L /= sum_L

    for _ in range(n_iterations):
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K

        L /= torch.sum(L, dim=0, keepdim=True)
        L /= B

    L *= B
    L = L.t()

    indices = torch.argmax(L, dim=1)
    if use_gumbel:
        L = F.gumbel_softmax(L, tau=0.5, hard=True)
    else:
        L = F.one_hot(indices, num_classes=K).to(dtype=torch.float32)

    return L, indices


def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)


def list_of_similarities_2d(X, Y):
    return - torch.sum((X.unsqueeze(dim=3) - Y.unsqueeze(dim=2)) ** 2, dim=1)


def make_one_hot(target, target_one_hot):
    target = target.view(-1,1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)


def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def print_and_write(str, file):
    print(str)
    file.write(str + '\n')


def interpolate_pos_embed(model, new_img_size=512, patch_size_config=16):
    pos_embed = model.image_encoder.model.pos_embed.data  # [197, hidden_dim]
    cls_pos_embed = pos_embed[:, 0:1, :]  # [1, hidden_dim]
    patch_pos_embed = pos_embed[:, 1:, :]  # [196, hidden_dim]

    # raw patch grid size
    orig_size = int(patch_pos_embed.shape[1] ** 0.5)  # 37
    new_size = new_img_size // patch_size_config  # 512 // 16 = 32

    patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)  # [1, dim, 37, 37]
    patch_pos_embed = F.interpolate(patch_pos_embed, size=(new_size, new_size), mode='bicubic', align_corners=False)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)

    new_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)  # [1, 1+new_size*new_size, dim]

    model.image_encoder.model.pos_embed = torch.nn.Parameter(new_pos_embed, requires_grad=True)

    return model