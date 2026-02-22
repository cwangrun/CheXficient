# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import logging
import numpy as np
import torch
from torch import nn
import os
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModel
from torchvision.transforms import transforms
from dinov2.models.vision_transformer import vit_base

from util.projection import load_projection_head
from util.helpers import sinkhorn_knopp

log = logging.getLogger(__name__)


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        output = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(output, tensor)
        ctx.rank = dist.get_rank()
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


URL_DICT = {
    "dinov2_vits14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
    "dinov2_vitb14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
    "dinov2_vitl14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
}


def load_tokenizer(source, pretrained_model_name_or_path, cache_dir='huggingface/tokenizers', **kwargs):
    if source == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=os.path.exists(os.path.join(cache_dir, f'models--{pretrained_model_name_or_path.replace("/", "--")}')),
            **kwargs,
        )
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token_id = tokenizer.cls_token_id
    else:
        raise KeyError(f"Not supported tokenizer source: {source}")

    return tokenizer


class TextEncoder(nn.Module):
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT'):
        super().__init__()
        # self.model = AutoModel.from_pretrained(model_name, ignore_mismatched_sizes=False, cache_dir='./huggingface',)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./huggingface/tokenizers')
        self.model = AutoModel.from_pretrained(model_name, use_safetensors=True, ignore_mismatched_sizes=False, )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, )
        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token_id = self.tokenizer.cls_token_id
        self.out_dim = self.model.config.hidden_size

    def forward(self, inputs):
        outputs = self.model(**inputs)
        return outputs["last_hidden_state"]  # (batch, seq_len, hidden_size)


class ImageEncoder(nn.Module):
    def __init__(self, model_name='dinov2_vitb14', image_size=224):
        super().__init__()
        self.model = vit_base(patch_size=14, img_size=image_size, init_values=1.0, block_chunks=0)
        stact_dict = torch.hub.load_state_dict_from_url(URL_DICT[model_name], map_location="cpu")
        ##########################################################
        if self.model.pos_embed.shape[1] != stact_dict['pos_embed'].shape[1]:
            cls_pos_embed = stact_dict['pos_embed'][:, 0:1, :]  # [1, hidden_dim]
            patch_pos_embed = stact_dict['pos_embed'][:, 1:, :]  # [1369, hidden_dim]
            # raw patch grid size
            orig_size = int(patch_pos_embed.shape[1] ** 0.5)  # 37
            new_size = image_size // self.model.patch_size  # 512 // 16 = 32
            patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)  # [1, dim, 37, 37]
            patch_pos_embed = F.interpolate(patch_pos_embed, size=(new_size, new_size), mode='bicubic', align_corners=False)
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
            stact_dict['pos_embed'] = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)  # [1, 1+new_size*new_size, dim]
        ##########################################################
        res = self.model.load_state_dict(stact_dict, strict=False)
        print('load dinov2 pretrained model:', res)
        self.out_dim = self.model.embed_dim

    def forward(self, x):
        feats = self.model(x)  # Shape: (b, d)
        return feats


class CheXficient(nn.Module):
    def __init__(self, visual_name='dinov2_vitb14', text_name='emilyalsentzer/Bio_ClinicalBERT', image_size=224, proj_dim=512, temperature=0.01, num_prototypes=6, ema_decay=0.9):
        super().__init__()
        self.image_encoder = ImageEncoder(model_name=visual_name, image_size=image_size)
        self.text_encoder = TextEncoder(model_name=text_name)

        self.text_pooling = 'eos'

        self.projection = True

        if self.projection:
            self.image_projection = load_projection_head(
                embedding_dim=self.image_encoder.out_dim, config_projection_head={'name': 'linear', 'dropout': 0.1, 'proj_dim': proj_dim}
            )
            # self.image_projection = nn.Identity()
            self.text_projection = load_projection_head(
                embedding_dim=self.text_encoder.out_dim, config_projection_head={'name': 'linear', 'dropout': 0.1, 'proj_dim': proj_dim}
            )
        else:
            assert (
                self.image_encoder.out_dim == self.text_encoder.out_dim
            ), "Without 'projection_head', embedding_dim of the image and text encoder must be the same."

        self.temperature = temperature
        if self.temperature:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))
        else:
            self.logit_scale = torch.tensor(1, dtype=torch.float32)
            log.warning("missing temperature scaling factor")

        self.feature_dim = proj_dim
        self.num_prototypes = num_prototypes

        self.use_ema_update = True
        self.ema_start_step = 0
        self.ema_decay = ema_decay

        self.prototypes = nn.Parameter(torch.rand(self.num_prototypes, self.feature_dim * 2))
        self.prototypes.data.copy_(F.normalize(self.prototypes.data, p=2, dim=1))

        self.register_buffer('prototype_usage', torch.zeros(self.num_prototypes))
        self.register_buffer('step_count', torch.zeros(1, dtype=torch.long))

        self.register_buffer('batch_prototype_updates', torch.zeros(self.num_prototypes, self.feature_dim * 2))
        self.register_buffer('batch_update_weight_stats', torch.zeros(1, 2))

        self.register_buffer('pending_ema_features', torch.zeros(0, self.feature_dim * 2))
        self.register_buffer('pending_ema_similarities', torch.zeros(0, self.num_prototypes))
        self._has_pending_ema_update = False

    def encode_image(self, image):
        image_features = self.image_encoder(image)
        image_embeddings = self.image_projection(image_features) if self.projection else image_features
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)

        return image_embeddings

    def encode_text(self, text_tokens):
        text_features = self.text_encoder(text_tokens)

        if self.text_pooling == "eos":
            # take features from the eot embedding (eos_token is the highest number in each sequence)
            eos_token_indices = text_tokens["attention_mask"].sum(dim=-1) - 1
            text_features = text_features[torch.arange(text_features.shape[0]), eos_token_indices]
        elif self.text_pooling == "bos":   # [CLS] token
            text_features = text_features[:, 0]
        elif self.text_pooling == "mean":
            input_mask_expanded = text_tokens["attention_mask"].unsqueeze(axis=-1).expand(text_features.size()).float()
            text_features = torch.sum(text_features * input_mask_expanded, axis=1) / torch.clamp(input_mask_expanded.sum(axis=1), min=1e-9)
        else:
            raise NotImplementedError("Not supported pooling method : %s", self.text_pooling)

        text_embeddings = self.text_projection(text_features) if self.projection else text_features

        text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        return text_embeddings

    def forward_vanilla(self, images, text_tokens):

        # get image and text features
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(text_tokens)

        # normalize features
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        return image_embeddings, text_embeddings, logit_scale

    def forward(self, images, text_tokens, image_ids=None, do_curation=False, return_loss=True):

        # image and text features
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(text_tokens)

        # normalize features
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        if dist.is_initialized() and dist.get_world_size() > 1:
            all_image_embeds = AllGather.apply(image_embeddings)
            all_text_embeds = AllGather.apply(text_embeddings)
            all_image_ids = AllGather.apply(image_ids) if image_ids is not None else None
        else:
            all_image_embeds = image_embeddings
            all_text_embeds = text_embeddings
            all_image_ids = image_ids

        if not return_loss:
            return all_image_embeds, all_text_embeds, all_image_ids

        loss_dict = self._compute_losses(all_image_embeds, all_text_embeds, all_image_ids, do_curation)

        return all_image_embeds, all_text_embeds, all_image_ids, loss_dict

    def _compute_losses(self, all_image_embeds, all_text_embeds, all_image_ids, do_curation):

        device = all_image_embeds.device
        batch_size_all = all_image_embeds.size(0)

        # normalize prototypes
        normalized_prototypes = F.normalize(self.prototypes, dim=-1, p=2)
        concat_features = torch.cat([all_image_embeds, all_text_embeds], dim=-1)
        concat_features = F.normalize(concat_features, dim=-1, p=2)

        if do_curation:
            # Prototype-driven data selection
            if dist.is_initialized() and dist.get_world_size() > 1:
                if dist.get_rank() == 0:
                    mask_outlier, keep_mask_support, keep_mask_central = self._select_samples(concat_features, outlier_r=0.95, support_r=0.85)
                else:
                    mask_outlier = torch.zeros(batch_size_all, dtype=torch.bool, device=device)
                    keep_mask_support = torch.zeros(batch_size_all, dtype=torch.bool, device=device)
                    keep_mask_central = torch.zeros(batch_size_all, dtype=torch.bool, device=device)
                torch.distributed.broadcast(mask_outlier, src=0)
                torch.distributed.broadcast(keep_mask_support, src=0)
                torch.distributed.broadcast(keep_mask_central, src=0)
            else:
                mask_outlier, keep_mask_support, keep_mask_central = self._select_samples(concat_features, outlier_r=0.95, support_r=0.85)
            keep_mask = (keep_mask_support | keep_mask_central)
        else:
            keep_mask = torch.ones(batch_size_all, device=device, dtype=torch.bool)
            mask_outlier = torch.zeros(batch_size_all, device=device, dtype=torch.bool)
            keep_mask_support = torch.zeros(batch_size_all, device=device, dtype=torch.bool)
            keep_mask_central = torch.zeros(batch_size_all, device=device, dtype=torch.bool)

        if keep_mask.sum() > 0:
            contrastive_loss = self.compute_contrastive_loss(all_image_embeds, all_text_embeds, keep_mask)
            selected_features, prototype_similarities = self.compute_prototype_similarities(normalized_prototypes, concat_features, keep_mask)
        else:
            # using all samples if no samples selected for a super-batch data
            contrastive_loss =  self.compute_contrastive_loss(all_image_embeds, all_text_embeds, None)
            selected_features, prototype_similarities = self.compute_prototype_similarities(normalized_prototypes, concat_features, None)
            keep_mask = torch.ones_like(keep_mask, device=device, dtype=torch.bool)

        if self.use_ema_update and self.step_count >= self.ema_start_step and do_curation:
            self._store_ema_update_data(selected_features.detach().clone(), prototype_similarities.detach().clone())

        self.step_count += 1

        return {
            'contrastive_loss': contrastive_loss,
            'keep mask': keep_mask,
            'outlier mask': mask_outlier,
            'support mask': keep_mask_support,
            'central mask': keep_mask_central,
        }

    def compute_prototype_similarities(self, normalized_prototypes, concat_features, mask=None):
        if mask is not None:
            concat_features = concat_features[mask]
        prototype_similarities = torch.mm(concat_features, normalized_prototypes.t())
        return concat_features, prototype_similarities

    def compute_contrastive_loss(self, image_embeds, text_embeds, mask=None):
        if mask is not None:
            image_embeds = image_embeds[mask]
            text_embeds = text_embeds[mask]

        # cosine similarity as logits
        logits_per_image = self.logit_scale.exp() * image_embeds @ text_embeds.t()
        logits_per_text  = self.logit_scale.exp() * text_embeds @ image_embeds.t()
        labels = torch.arange(logits_per_image.size(0), device=image_embeds.device)

        loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2.
        return loss


    @torch.no_grad()
    def _select_samples(self, batch_feat, outlier_r=0.95, support_r=0.80, max_per_cluster=10, method='farthest'):
        """
        Assign each feature to the nearest prototype, and filter based on top-R closeness.
        :param batch_feat: torch.Tensor [B, dim]
        :return: cluster_ids [B], keep_mask [B]
        """
        dists = torch.cdist(batch_feat, self.prototypes, p=2)  # [B, K]
        min_dist, cluster_id = dists.min(dim=1)  # [B], [B]
        keep_mask_outlier = torch.zeros_like(min_dist, dtype=torch.bool, device=batch_feat.device)
        keep_mask_support = torch.zeros_like(min_dist, dtype=torch.bool, device=batch_feat.device)
        keep_mask_central = torch.zeros_like(min_dist, dtype=torch.bool, device=batch_feat.device)

        for k in range(self.num_prototypes):
            mask = (cluster_id == k)  # 每个cluster保留r比例的样本
            if mask.sum() == 0: continue
            threshold_outlier = torch.quantile(min_dist[mask], outlier_r)
            threshold_support = torch.quantile(min_dist[mask], support_r)
            keep_mask_outlier[mask] = (min_dist[mask] > threshold_outlier)
            keep_mask_support[mask] = (min_dist[mask] > threshold_support) & (min_dist[mask] <= threshold_outlier)
            keep_mask_central[mask] = (min_dist[mask] <= threshold_support)

        # --- Redundancy removal ---
        if max_per_cluster is not None:
            new_keep_mask = torch.zeros_like(keep_mask_central, dtype=torch.bool, device=batch_feat.device)
            for k in range(self.num_prototypes):
                mask = (cluster_id == k) & keep_mask_central
                idx = mask.nonzero(as_tuple=True)[0]  # sample indices in cluster k
                if idx.numel() == 0:
                    continue
                if idx.numel() <= max_per_cluster:
                    new_keep_mask[idx] = True
                else:
                    subset = batch_feat[idx]  # [M, D]
                    if method == 'density':  # local density sampling
                        chosen_idx = self.local_density_sampling(subset, max_per_cluster, k=5)
                    elif method == 'farthest':  # Farthest-Point Sampling (FPS)
                        chosen_idx = self.farthest_point_sampling(subset, max_per_cluster)
                    else:  # random sampling
                        chosen_idx = torch.randperm(idx.numel(), device=batch_feat.device)[:max_per_cluster]
                    chosen = idx[chosen_idx]
                    new_keep_mask[chosen] = True
            keep_mask_central = new_keep_mask
        return keep_mask_outlier, keep_mask_support, keep_mask_central
    
    @torch.no_grad()
    def local_density_sampling(self, x, n_samples, k=5):
        dists = torch.cdist(x, x)
        dists.fill_diagonal_(float('inf'))
        knn_dists, _ = dists.topk(k, largest=False, dim=1)
        score = knn_dists.mean(dim=1)  # bigger = more sparse
        topk = torch.topk(score, k=n_samples).indices
        return topk

    @torch.no_grad()
    def farthest_point_sampling(self, x, n_samples):
        """
        x: [N, D] feature tensor
        returns: [n_samples] indices
        """
        N, D = x.shape
        selected = [torch.randint(0, N, (1,), device=x.device).item()]
        dists = torch.full((N,), float('inf'), device=x.device)

        for _ in range(1, n_samples):
            current = x[selected[-1]].unsqueeze(0)  # [1, D]
            dist = torch.norm(x - current, dim=1)  # [N]
            dists = torch.minimum(dists, dist)  # update nearest distance to selected
            next_idx = torch.argmax(dists).item()
            selected.append(next_idx)

        return torch.tensor(selected, device=x.device)


    @torch.no_grad()
    def get_prototype_diversity_stats(self):
        """prototype statistics"""
        with torch.no_grad():
            normalized_prototypes = F.normalize(self.prototypes, dim=-1, p=2)
            dist_matrix = torch.cdist(normalized_prototypes, normalized_prototypes)

            mask = torch.eye(self.num_prototypes, device=dist_matrix.device)
            off_diagonal = dist_matrix * (1 - mask)

            stats = {
                'max_distance': off_diagonal.max().item(),
                'min_distance': off_diagonal.min().item(),
                'avg_distance': off_diagonal.sum().item() / (self.num_prototypes * (self.num_prototypes - 1)),
                'prototype_usage': self.prototype_usage,
                'step_count': self.step_count.item()
            }

            return stats

    @torch.no_grad()
    def _store_ema_update_data(self, selected_features, prototype_similarities):
        """
        Args:
            selected_features: [N_selected, feature_dim*2]
            prototype_similarities: [N_selected, num_prototypes]
        """
        if not self.use_ema_update:
            return

        self.pending_ema_features = selected_features
        self.pending_ema_similarities = prototype_similarities
        self._has_pending_ema_update = True

    @torch.no_grad()
    def apply_pending_ema_update(self):
        """
        apply ema update of prototypes
        """
        if not self.use_ema_update or not self._has_pending_ema_update:
            return

        if self.pending_ema_features.size(0) == 0:
            self._has_pending_ema_update = False
            return

        self._ema_update_prototypes(self.pending_ema_features, self.pending_ema_similarities)

        self._has_pending_ema_update = False

    @torch.no_grad()
    def _ema_update_prototypes(self, selected_features, prototype_similarities, tau=0.01, eps=1e-8, use_sinkhorn=True):
        """
        Args:
            selected_features: [N_selected, feature_dim*2]
            prototype_similarities: [N_selected, num_prototypes]
        """
        if not self.use_ema_update:
            return

        self.batch_prototype_updates.zero_()       # (num_prototypes, feature_dim * 2)
        self.batch_update_weight_stats.zero_()     # (*, 2)

        if use_sinkhorn:
            weights, indices = sinkhorn_knopp(prototype_similarities.detach(), epsilon=0.5, use_gumbel=False)
        else:
            weights = F.softmax(prototype_similarities.detach() / tau, dim=1)  # temperature-controlled softmax

        self.batch_update_weight_stats = torch.stack([weights.min(dim=1)[0], weights.max(dim=1)[0]], dim=1)
        for k in range(self.num_prototypes):
            w = weights[:, k].unsqueeze(1)
            avg_feature = torch.sum(w * selected_features, dim=0) / (w.sum() + eps)
            self.prototypes[k].data.copy_(self.ema_decay * self.prototypes[k].data + (1 - self.ema_decay) * avg_feature)
        self.prototypes.data.copy_(self.prototypes.data / self.prototypes.data.norm(dim=1, keepdim=True))
        self.prototype_usage = weights.sum(0)
        

    def get_prototype_update_stats(self):
        if not self.use_ema_update:
            return {}

        with torch.no_grad():
            stats = {
                'mim_probs': self.batch_update_weight_stats[:, 0],
                'max_probs': self.batch_update_weight_stats[:, 1],
            }
            return stats

    def has_pending_ema_update(self):
        return getattr(self, '_has_pending_ema_update', False)


class CheXficient_linear_probing_classification(nn.Module):
    def __init__(self, visual_name='dinov2_vitb14', num_classes=14, image_size=224, proj_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(model_name=visual_name, image_size=image_size)

        self.text_pooling = 'eos'

        self.projection = True

        if self.projection:
            self.image_projection = load_projection_head(
                embedding_dim=self.image_encoder.out_dim, config_projection_head={'name': 'linear', 'dropout': 0.1, 'proj_dim': 512}
            )

        self.classification_layer = nn.Linear(self.image_encoder.out_dim, num_classes)
        # self.classification_layer = nn.Linear(proj_dim, num_classes)

    def encode_image(self, image):
        image_features = self.image_encoder(image)
        # image_features = self.image_projection(image_features) if self.projection else image_features
        image_embeddings = image_features / image_features.norm(dim=1, keepdim=True)

        return image_embeddings

    def forward(self, images):

        # get image feature
        image_embeddings = self.encode_image(images)
        logits = self.classification_layer(image_embeddings)

        return logits


class CheXficient_unet_segmentation(nn.Module):
    def __init__(self, visual_name='dinov2_vitb14', num_classes=2, image_size=224, n_last_blocks=5, decoder_type='unet'):
        super().__init__()
        self.image_encoder = ImageEncoder(model_name=visual_name, image_size=image_size)

        self.text_pooling = 'eos'

        self.projection = True

        self.decoder_type = decoder_type

        if self.projection:
            self.image_projection = load_projection_head(
                embedding_dim=self.image_encoder.out_dim, config_projection_head={'name': 'linear', 'dropout': 0.1, 'proj_dim': 512}
            )

        self.n_last_blocks = n_last_blocks

        if decoder_type == "linear":
            self.decoder = LinearDecoder(in_channels=self.image_encoder.out_dim, num_classes=num_classes, image_size=image_size, patch_size=14)
        elif decoder_type == "unet":
            self.decoder = UNetDecoder(in_channels=self.image_encoder.out_dim, out_channels=num_classes, image_size=image_size, patch_size=14, resize_image=True)
        else:
            raise NotImplementedError

    def forward(self, images):

        # # get image feature
        if self.decoder_type == "linear" or self.decoder_type == "vitdet":
            features = self.image_encoder.model.forward_features(images)['x_norm_patchtokens']
        else:
            features = self.image_encoder.model.get_intermediate_layers(images, self.n_last_blocks, return_class_token=False)

        logits = self.decoder(features)

        return logits


class UNetDecoder(nn.Module):
    """Unet decoder head"""
    DECODER_TYPE = "unet"

    def __init__(self, in_channels, out_channels, image_size=224, resize_image=False, patch_size=14):
        super(UNetDecoder, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = in_channels
        self.image_size = image_size
        self.resize_image = resize_image

        self.up1 = UNetDecoderUpBlock(in_channels=in_channels, out_channels=in_channels // 2, embed_dim=self.embed_dim)  # number of params: 9.69 M
        self.up2 = UNetDecoderUpBlock(in_channels=in_channels // 2, out_channels=in_channels // 4, embed_dim=self.embed_dim)
        self.up3 = UNetDecoderUpBlock(in_channels=in_channels // 4, out_channels=in_channels // 8, embed_dim=self.embed_dim)
        self.up4 = UNetDecoderUpBlock(in_channels=in_channels // 8, out_channels=out_channels, embed_dim=self.embed_dim)

        self.apply(init_weights)

    def forward(self, x):
        h = w = self.image_size // self.patch_size

        skip1 = x[3].reshape(-1, h, w, self.embed_dim).permute(0, 3, 1, 2)
        skip2 = x[2].reshape(-1, h, w, self.embed_dim).permute(0, 3, 1, 2)
        skip3 = x[1].reshape(-1, h, w, self.embed_dim).permute(0, 3, 1, 2)
        skip4 = x[0].reshape(-1, h, w, self.embed_dim).permute(0, 3, 1, 2)
        x1 = x[4].reshape(-1, h, w, self.embed_dim).permute(0, 3, 1, 2)

        x2 = self.up1(x1, skip1)
        x3 = self.up2(x2, skip2)
        x4 = self.up3(x3, skip3)
        x5 = self.up4(x4, skip4)

        if self.resize_image:
            x5 = transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BILINEAR)(x5)  # BICUBIC
        return x5


class UNetDecoderUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=1024, dropout=0.0) -> None:
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.skip_conv = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x2 = self.skip_conv(x2)

        x1 = self.dropout(x1)
        x2 = self.dropout(x2)

        scale_factor = (x1.size()[2] / x2.size()[2])
        x2 = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)(x2)
        x = torch.concat([x1, x2], dim=1)
        return self.conv(x)


class LinearDecoder(torch.nn.Module):
    """Linear decoder head"""
    DECODER_TYPE = "linear"

    def __init__(self, in_channels, num_classes, image_size=224, patch_size=14):
        super().__init__()
        print(patch_size)
        self.image_size = image_size
        self.in_channels = in_channels
        self.width = self.height = image_size // patch_size
        self.decoder = torch.nn.Conv2d(in_channels, num_classes, (1, 1))
        self.decoder.weight.data.normal_(mean=0.0, std=0.1)
        self.decoder.bias.data.zero_()

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels).permute(0, 3, 1, 2)
        output = self.decoder(embeddings)
        # Upsample (interpolate) output/logit map.
        output = F.interpolate(output, size=self.image_size, mode="bilinear", align_corners=False)
        return output


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

