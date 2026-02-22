# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import json
import os
import numpy as np
from sklearn import metrics
from prompt import prompts
from collections import Counter
import sklearn
from sklearn.metrics import roc_auc_score
import torch.distributed as dist


def load_metadata(metadir="clipeval"):
    with open(os.path.join(metadir, 'dataset_catalog.json')) as f:
        dataset = json.load(f)

    with open(os.path.join(metadir, 'labels.json')) as f:
        all_labels = json.load(f)
        
    return dataset, all_labels


def evaluate(args, d, val_loader, labels, model, tokenizer, max_bert_length):

    if args.rank == 0:
        print('Evaluating: {}'.format(d), ', Number of samples: {}'.format(len(val_loader.dataset)))

    if d == 'chexpert_test':
        acc_or_outputs = validate_zeroshot(args, model, val_loader, labels, tokenizer, max_bert_length)
    elif d == 'mimic':
        acc_or_outputs = validate_zeroshot(args, model, val_loader, labels, tokenizer, max_bert_length)
    elif d == 'nihchestxray14':
        acc_or_outputs = validate_zeroshot(args, model, val_loader, labels, tokenizer, max_bert_length)
    elif d == 'pneumonia_Xray2017':
        acc_or_outputs = validate_zeroshot(args, model, val_loader, labels, tokenizer, max_bert_length)
    elif d == 'covid19_pneumonia_normal':
        acc_or_outputs = validate_zeroshot(args, model, val_loader, labels, tokenizer, max_bert_length)
    elif d == 'simm_pneumothorax':
        acc_or_outputs = validate_zeroshot(args, model, val_loader, labels, tokenizer, max_bert_length)
    elif d == 'tbx11k':
        acc_or_outputs = validate_zeroshot(args, model, val_loader, labels, tokenizer, max_bert_length)
    elif d == 'vindr_cxr':
        acc_or_outputs = validate_zeroshot(args, model, val_loader, labels, tokenizer, max_bert_length)
    elif d == 'vindr_pcxr':
        acc_or_outputs = validate_zeroshot(args, model, val_loader, labels, tokenizer, max_bert_length)
    elif d == 'chexpert_5x200_retrieve':
        acc_or_outputs = retrieval_image_text(args, model, val_loader, tokenizer, max_bert_length)
    elif d == 'mimic_retrieve':
        acc_or_outputs = retrieval_image_text(args, model, val_loader, tokenizer, max_bert_length)
    else:
        raise Exception('Unknown evaluation dataset')

    metric = acc_or_outputs
    
    return metric

def flatten_and_concat(nested_array_list, axis=0):
    flat_list = [arr for part in nested_array_list for arr in part]
    return np.concatenate(flat_list, axis=axis)

@torch.no_grad()
def validate_zeroshot(args, model, dataloader, cxr_labels, tokenizer, max_bert_length=256):
    # model.eval()

    cxr_templates = ['{}', 'no {}']

    with torch.no_grad():
        zeroshot_weights = []
        # compute embedding through model for each class
        for classname in cxr_labels:
            texts = [template.format(classname) for template in cxr_templates]  # format with class

            if classname == "No Finding" or classname == "No finding" or classname == "no Finding" or classname == "no finding":
                texts[1] = "there is Finding"
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%% use my modified prompt for no findings ')

            texts = tokenizer(texts, return_tensors='pt', padding="longest", truncation=True, max_length=max_bert_length)
            for key in texts:
                texts[key] = texts[key].to(next(model.parameters()).device, non_blocking=True)

            class_embeddings = model.encode_text(texts)  # embed with text encoder

            # normalize class_embeddings
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # average over templates
            # class_embedding = class_embeddings.mean(dim=0)
            # norm over new averaged templates
            # class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embeddings)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1)  # [2, 14, 512]

    local_predictions = []
    local_targets = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images = data[0]
            target = data[1]

            images = images.to(next(model.parameters()).device, non_blocking=True)

            image_features = model.encode_image(images)

            image_features /= image_features.norm(dim=-1, keepdim=True)  # (1, 768)

            # obtain logits
            logits = torch.stack([image_features @ (zeroshot_weights[0].permute(1, 0)), image_features @ (zeroshot_weights[1].permute(1, 0))], dim=2)  # (batch, num_classes, 2)

            # logits = logits * model.logit_scale.exp()

            probs = logits.softmax(dim=2)  # we can take the softmax to get the label probabilities

            # if softmax_eval is False:
            #     norm_logits = (logits - logits.mean()) / (logits.std())
            #     logits = sigmoid(norm_logits)

            local_predictions.append(probs.cpu().numpy())
            local_targets.append(target.cpu().numpy())

    if dist.is_initialized() and dist.get_world_size() > 1:
        predictions = [None for _ in range(args.world_size)]
        dist.all_gather_object(predictions, local_predictions)
        targets = [None for _ in range(args.world_size)]
        dist.all_gather_object(targets, local_targets)

        predictions = flatten_and_concat(predictions, axis=0)
        targets = flatten_and_concat(targets, axis=0)
    else:
        predictions = np.concatenate(local_predictions, axis=0)
        targets = np.concatenate(local_targets, axis=0)

    all_auc = []
    for c in range(len(cxr_labels)):
        mask = targets[:, c] != -1   # -1: uncertainty in mimic
        y_true = targets[mask, c]
        y_score = predictions[mask, c, 0]
        if len(np.unique(y_true)) > 1:   #
            auc = roc_auc_score(y_true, y_score)
        else:
            auc = np.nan
        all_auc.append(auc)
    all_auc = np.asarray([auc for auc in all_auc if not np.isnan(auc)])   # check classes that have 0 positive samples
    mean_auc = all_auc.mean()

    # model.train()
    return {"mean": mean_auc, "individual": all_auc.tolist()}

@torch.no_grad()
def retrieval_image_text(args, model, dataloader, tokenizer, max_bert_length=256):
    # model.eval()

    local_image_embeddings = []
    local_text_embeddings = []
    local_text_list = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images = data[0]
            # text_tokens = data[1]
            text = data[1]

            text_tokens = tokenizer(text, return_tensors='pt', padding="longest", truncation=True, max_length=max_bert_length)

            images = images.to(next(model.parameters()).device, non_blocking=True)
            for key in text_tokens:
                text_tokens[key] = text_tokens[key].to(next(model.parameters()).device, non_blocking=True)

            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)  # (1, 768)
            text_features /= text_features.norm(dim=-1, keepdim=True)  # (1, 768)

            local_image_embeddings.append(image_features.cpu().numpy())
            local_text_embeddings.append(text_features.cpu().numpy())
            local_text_list.extend(text)

    if dist.is_initialized() and dist.get_world_size() > 1:
        image_embeddings = [None for _ in range(args.world_size)]
        dist.all_gather_object(image_embeddings, local_image_embeddings)
        text_embeddings = [None for _ in range(args.world_size)]
        dist.all_gather_object(text_embeddings, local_text_embeddings)
        text_list = [None for _ in range(args.world_size)]
        dist.all_gather_object(text_list, local_text_list)

        image_embeddings = flatten_and_concat(image_embeddings, axis=0)
        text_embeddings = flatten_and_concat(text_embeddings, axis=0)
        text_list = [text for part in text_list for text in part]
    else:
        image_embeddings = np.concatenate(local_image_embeddings, axis=0)
        text_embeddings = np.concatenate(local_text_embeddings, axis=0)
        text_list = [item for sublist in local_text_list for item in sublist]


    identical_text_set = []
    idx2label = {}
    identical_indexes = []
    for i, text in enumerate(text_list):
        if text not in identical_text_set:
            identical_text_set.append(text)
            identical_indexes.append(i)
            idx2label[i] = len(identical_text_set) - 1
        else:
            idx2label[i] = identical_text_set.index(text)
    identical_text_embedding = text_embeddings[np.array(identical_indexes)]

    num_samples = image_embeddings.shape[0]
    n_text = len(identical_text_set)
    
    # print('debug shape image-text retrieval:', args.rank, len(dataloader), image_embeddings.shape, text_embeddings.shape, len(text_list), n_text, idx2label)

    similarities = metrics.pairwise.cosine_similarity(image_embeddings, identical_text_embedding)  # n x m
    recall_dict = {1: 0, 5: 0, 10: 0}
    mean_rank = 0
    for idx in range(num_samples):
        label = idx2label[idx]
        similarity = similarities[idx]
        similarity_args = similarity.argsort()

        # rank of the paired text
        rank = n_text - np.argwhere(similarity_args == label).ravel()[0]
        mean_rank += rank

        for k in recall_dict:
            if rank <= k:
                recall_dict[k] += 1
    # results
    # print(
        # '\n',
        # "\n".join([f"Recall@{k}: {v / num_samples:.3f}" for k, v in recall_dict.items()]) + f"\nmean rank: {mean_rank / num_samples:.3f}"
    # )
    result = {}
    result.update({f"Recall@{k}": v / num_samples for k, v in recall_dict.items()})
    result.update({"MeanRank": mean_rank / num_samples})

    # model.train()
    return {"mean": result['Recall@1'], "individual": list(result.values())}

@torch.no_grad()
def retrieval_text_image(args, model, dataloader, tokenizer, max_bert_length=256):
    # model.eval()

    local_image_embeddings = []
    local_text_embeddings = []
    local_text_list = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images = data[0]
            # text_tokens = data[1]
            text = data[1]

            text_tokens = tokenizer(text, return_tensors='pt', padding="longest", truncation=True, max_length=max_bert_length)

            images = images.to(next(model.parameters()).device, non_blocking=True)
            for key in text_tokens:
                text_tokens[key] = text_tokens[key].to(next(model.parameters()).device, non_blocking=True)

            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)  # (1, 768)
            text_features /= text_features.norm(dim=-1, keepdim=True)  # (1, 768)

            local_image_embeddings.append(image_features.cpu().numpy())
            local_text_embeddings.append(text_features.cpu().numpy())
            local_text_list.append(text)

    if dist.is_initialized() and dist.get_world_size() > 1:
        image_embeddings = [None for _ in range(args.world_size)]
        dist.all_gather_object(image_embeddings, local_image_embeddings)
        text_embeddings = [None for _ in range(args.world_size)]
        dist.all_gather_object(text_embeddings, local_text_embeddings)
        text_list = [None for _ in range(args.world_size)]
        dist.all_gather_object(text_list, local_text_list)

        image_embeddings = flatten_and_concat(image_embeddings, axis=0)
        text_embeddings = flatten_and_concat(text_embeddings, axis=0)
        text_list = [text for part in text_list for text in part]
    else:
        image_embeddings = np.concatenate(local_image_embeddings, axis=0)
        text_embeddings = np.concatenate(local_text_embeddings, axis=0)
        text_list = [item for sublist in local_text_list for item in sublist]

    identical_text_set = []
    idx2label = {}
    identical_indexes = []
    for i, text in enumerate(text_list):
        if text not in identical_text_set:
            identical_text_set.append(text)
            identical_indexes.append(i)
            idx2label[i] = len(identical_text_set) - 1
        else:
            idx2label[i] = identical_text_set.index(text)
    identical_text_embedding = text_embeddings[np.array(identical_indexes)]

    num_samples = len(text_list)
    n_text = len(identical_text_set)

    print('number of unique texts: {}'.format(n_text))

    similarities = metrics.pairwise.cosine_similarity(identical_text_embedding, image_embeddings)  # n x m
    recall_dict = {1: 0, 5: 0, 10: 0}
    mean_rank = 0
    for idx in range(num_samples):
        label = idx2label[idx]
        similarity = similarities[label]
        similarity_args = similarity.argsort()

        # rank of the paired text
        rank = num_samples - np.argwhere(similarity_args == idx).ravel()[0]
        mean_rank += rank

        for k in recall_dict:
            if rank <= k:
                recall_dict[k] += 1
    # results
    # print(
    #     '\n',
    #     "\n".join([f"Recall@{k}: {v / num_samples:.3f}" for k, v in recall_dict.items()]) + f"\nmean rank: {mean_rank / num_samples:.3f}"
    # )
    result = {}
    result.update({f"Recall@{k}": v / num_samples for k, v in recall_dict.items()})
    result.update({"MeanRank": mean_rank / num_samples})

    # model.train()
    return {"mean": result['Recall@1'], "individual": list(result.values())}


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mean_per_class(outputs, targets):
    pred = outputs.argmax(1)
    confusion_matrix = metrics.confusion_matrix(targets, pred)
    per_classes = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)

    return 100 * per_classes.mean()


def roc_auc(outputs, targets):
    pos_score = outputs[:, 1] - outputs[:, 0]
    metric = metrics.roc_auc_score(targets, pos_score)

    return 100 * metric


if __name__ == '__main__':
    evaluate(args, 'chexpert_test', None, ["Cardiomegaly", "Lung Lesion"], model, None, 256)