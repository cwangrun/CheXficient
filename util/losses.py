# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from util import misc


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        output = [torch.empty_like(tensor) for _ in range(misc.get_world_size())]
        dist.all_gather(output, tensor)
        ctx.rank = misc.get_rank()
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[
                ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)    
            ],
            None,
        )


# class CiTCLIPLossGrad(nn.Module):
#     def forward(self, image_embeds, text_embeds, logit_scale):
#         # normalized features
#         # image_embeds = F.normalize(image_embeds, dim=-1, p=2)
#         # text_embeds = F.normalize(text_embeds, dim=-1, p=2)
#         if misc.get_world_size() > 1:
#             # gather features from all GPUs
#             image_embeds = AllGather.apply(image_embeds)
#             text_embeds = AllGather.apply(text_embeds)
#
#         # cosine similarity as logits
#         logits_per_image = logit_scale * image_embeds @ text_embeds.t()
#         labels = torch.arange(logits_per_image.size(0), device=image_embeds.device)
#         loss = F.cross_entropy(logits_per_image, labels)
#         return loss


class CLIPLossGrad(nn.Module):
    def forward(self, image_embeds, text_embeds, logit_scale):
        # image_embeds = F.normalize(image_embeds, dim=-1, p=2)
        # text_embeds = F.normalize(text_embeds, dim=-1, p=2)
        if misc.get_world_size() > 1:
            # gather features from all GPUs
            image_embeds = AllGather.apply(image_embeds)
            text_embeds = AllGather.apply(text_embeds)

        # cosine similarity as logits
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logit_scale * text_embeds @ image_embeds.t()
        labels = torch.arange(logits_per_image.size(0), device=image_embeds.device)

        loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2.
        return loss, logits_per_image, logits_per_text


class Prototype_Based_Loss(nn.Module):
    def forward(self, f_img, f_txt, proto_img, proto_txt, logit_scale):
        # # CLIP InfoNCE loss
        # f_img = F.normalize(f_img, dim=1)
        # f_txt = F.normalize(f_txt, dim=1)
        # sim_matrix = logit_scale * torch.matmul(f_img, f_txt.T)
        # labels = torch.arange(f_img.size(0), device=f_img.device)
        # loss_clip = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)

        if misc.get_world_size() > 1:
            # gather features from all GPUs
            f_img = AllGather.apply(f_img)
            f_txt = AllGather.apply(f_txt)

        # Prototype losses
        proto_img = F.normalize(proto_img, dim=1)  # [K, D]
        proto_txt = F.normalize(proto_txt, dim=1)  # [K, D]

        sim_img_proto = logit_scale * torch.matmul(f_img, proto_img.T)   # [B, K]
        sim_txt_proto = logit_scale * torch.matmul(f_txt, proto_txt.T)   # [B, K]

        loss_proto_img = F.cross_entropy(sim_img_proto, sim_img_proto.argmax(dim=1))
        loss_proto_txt = F.cross_entropy(sim_txt_proto, sim_txt_proto.argmax(dim=1))

        return (loss_proto_img + loss_proto_txt) / 2.