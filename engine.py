# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math
import sys
import json
import os
import numpy as np
import torch
import time
import util.misc as misc
import util.lr_scheduler as lr_scheduler

from typing import Iterable
import wandb
from prompt import prompts
import torch.nn.functional as F
import torch.distributed as dist

from clipeval import eval_zeroshot


def to_device(samples, device, args):
    inputs = {}
    for key in samples:
        if key in ["pixel_values", 'image_ids']:
            samples[key] = samples[key].to(device, non_blocking=True)
        if key in ["text_tokens"]:
            for k in samples[key]:
                samples[key][k] = samples[key][k].to(device, non_blocking=True)
            # if key == "pixel_values" and inputs[key].dtype == torch.uint8:
            #     from main import get_mean_std
            #     # inmem data. normalize it.
            #     inputs[key] = inputs[key].to(torch.float32).div_(255.)  # b, 3, 224, 224
            #     mean, std = get_mean_std(args)
            #     mean = torch.as_tensor(mean, device=inputs[key].device)[None, :, None, None]
            #     std = torch.as_tensor(std, device=inputs[key].device)[None, :, None, None]
            #     inputs[key] = inputs[key].sub_(mean).div_(std)
    return samples


@torch.no_grad()
def evaluate(args, model, tokenizer):

    model.eval()

    dataset, all_labels = eval_zeroshot.load_metadata("clipeval")

    metrics = {}

    start_time = time.time()

    for d in dataset:
        val_dataset = args.val_dataset[d]
        labels = all_labels[d]

        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.eval_batch_size,
                                                 sampler=val_sampler,
                                                 shuffle=False,
                                                 num_workers=3,   # args.num_workers
                                                 pin_memory=False,
                                                 drop_last=False,
                                                 )
        metric = eval_zeroshot.evaluate(args, d, val_loader, labels, model, tokenizer, args.max_bert_length)
        metrics[d] = metric
        if args.eval:
            json_str = json.dumps({"task": d, "acc": metric})
            misc.print_json(args.output_dir, json_str)

    eval_time = time.time() - start_time
    print(f"evaluation time: {eval_time:.2f}s")

    model.train()

    return metrics


def train_one_epoch(model: torch.nn.Module, model_without_ddp, tokenizer, data_loader,
                    best_acc, optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, step, loss_scaler, eff_batch_size, max_norm: float = 0,
                    global_example_ids: set = (),
                    log_writer=None,
                    args=None):
    model.train(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter
    
    # metric = evaluate(args, model_without_ddp, tokenizer)
    # print(metric)
    # print(debug)

    optimizer.zero_grad()

    do_curation = epoch < args.curation_epochs

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header, args.max_update)):
        if step[0] > args.max_update:
            break

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_scheduler.adjust_step_learning_rate(optimizer, step[0], args.lr, args.min_lr, args.warmup_steps, args.epochs * len(data_loader))  # args.max_update

        inputs = to_device(samples, device, args)

        with torch.cuda.amp.autocast(enabled=args.fp16):
            image_embeds, text_embeds, image_ids, loss_dict = model(inputs['pixel_values'], inputs['text_tokens'], inputs['image_ids'], do_curation)

            loss_total = loss_dict['contrastive_loss']

            if hasattr(model, 'module'):
                proto_stats = model.module.get_prototype_diversity_stats()
            else:
                proto_stats = model.get_prototype_diversity_stats()

        loss_value = loss_total.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_total /= accum_iter
        update_grad = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(loss_total, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False, update_grad=update_grad)

        logits_per_image = image_embeds.detach() @ text_embeds.detach().t()
        logits_per_text = logits_per_image.t()

        cosine_score = (logits_per_image.diagonal()).mean()
        non_cosine_score = ((logits_per_image[~torch.eye(logits_per_image.size(0), dtype=torch.bool, device=logits_per_image.device)]).mean() + \
                           (logits_per_text[~torch.eye(logits_per_text.size(0), dtype=torch.bool, device=logits_per_text.device)]).mean()) / 2

        if do_curation:
            global_example_ids.update(image_ids[loss_dict['keep mask'].cpu()].cpu().numpy().tolist())
        args.subset_ratio = len(global_example_ids) / len(data_loader.dataset.all_filenames)

        if update_grad:
            step[0] += 1
            optimizer.zero_grad()

        if hasattr(model, 'module'):
            has_pending_ema = model.module.has_pending_ema_update()
        else:
            has_pending_ema = model.has_pending_ema_update()

        # EMA update
        ema_stats = None
        if has_pending_ema and do_curation:
            if hasattr(model, 'module'):
                model.module.apply_pending_ema_update()
                ema_stats = model.module.get_prototype_update_stats()
            else:
                model.apply_pending_ema_update()
                ema_stats = model.get_prototype_update_stats()

        torch.cuda.synchronize()

        if data_iter_step % 3 == 0 and args.rank == 0:
            print("[CTR LOSS]: " + str(data_iter_step) + "   [Cosine Score]: " + str(cosine_score.item()) + "   [Non cosine Score]: " + str(non_cosine_score.item()))
            wandb.log({
                'Epoch': epoch,
                'Global step': step[0],
                'Local step': data_iter_step,
                "Optimizer LR": optimizer.param_groups[0]['lr'],
                "Temperature": model.module.logit_scale.data.item() if hasattr(model, 'module') else model.logit_scale.data.item(),
                "Actual batch": loss_dict['keep mask'].sum().item(),
                "Proto_dist": proto_stats['avg_distance'],
                "prob_min_stats": ema_stats['mim_probs'].mean().item() if ema_stats is not None else 0.0,
                "prob_max_stats": ema_stats['max_probs'].mean().item() if ema_stats is not None else 1.0,
                'Loss_ctr': loss_dict['contrastive_loss'].item(),
                'cosine_score': cosine_score.item(),
                'non cosine_score': non_cosine_score.item(),
                'subset ratio': args.subset_ratio,
                'batch ratio': loss_dict['keep mask'].sum() / len(loss_dict['keep mask']),
            })
            wandb.log({
                f"assignment_{ttt}_stats": proto_stats['prototype_usage'][ttt] for ttt in range(len(proto_stats['prototype_usage']))
            })

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            log_writer.add_scalar('lr', max_lr, step[0])
            log_writer.add_scalar('loss', loss_value_reduce, step[0])

        if step[0] and step[0] % args.eval_steps == 0 and step[0] >= 500: # 50, 2000
            metric = evaluate(args, model_without_ddp, tokenizer)
            if args.rank == 0:
                json_str = json.dumps({"@@@ step": step[0], "acc": metric, "seen": eff_batch_size * step[0]})
                misc.print_json(args.output_dir, json_str)
                wandb.log(
                    {f"Eval/{k}": v['mean'] for k, v in metric.items()}
                )
                if log_writer is not None:
                    log_writer.add_scalar('acc', metric, step[0])

                if isinstance(data_loader, list) or (hasattr(data_loader, "dataset") and isinstance(data_loader.dataset, torch.utils.data.IterableDataset)):
                    misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, global_example_ids=global_example_ids,
                                    loss_scaler=loss_scaler, epoch=0, epoch_name="last", best_acc=best_acc[0], step=step[0])
                # if metric > best_acc[0]:
                if (np.array([v['mean'] for v in metric.values()]).mean()) > best_acc[0]:
                    best_acc[0] = (np.array([v['mean'] for v in metric.values()]).mean())
                    misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, global_example_ids=global_example_ids,
                                    loss_scaler=loss_scaler, epoch=step[0], epoch_name="best", best_acc=best_acc[0], step=step[0])
            model.train(True)
        
        if step[0] and step[0] % 1000 == 0 and step[0] > 2000:
            if args.rank == 0:
                torch.save({'args': args, 'epoch': epoch, 'step': step[0], 'model': model_without_ddp.state_dict(), }, os.path.join(args.output_dir, 'regular_step_' + str(step[0])))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def while_condition(example_ids, step, args):
    return len(example_ids) < (args.prototype_warmup_steps * args.batch_size)


@torch.no_grad()
def warmup_prototypes(step, gpu, producer_iter, model, args):
    model.eval()

    if hasattr(args, "sublist") and args.sublist:
        example_ids = []
    else:
        example_ids = set()

    total_example = 0
    image_embeds_all = []
    text_embeds_all = []
    while while_condition(example_ids, step[0], args):
        samples = next(producer_iter)
        image_ids = samples["image_ids"]
        total_example += len(image_ids)

        inputs = to_device(samples, torch.device(f'cuda:{gpu}'), args)
        with torch.cuda.amp.autocast():
            image_embeds, text_embeds, _ = model(inputs['pixel_values'], inputs['text_tokens'], return_loss=False)
        image_embeds = image_embeds.to(torch.float32)
        text_embeds = text_embeds.to(torch.float32)
        image_embeds_all.append(image_embeds.cpu().numpy())
        text_embeds_all.append(text_embeds.cpu().numpy())
        example_ids.update(image_ids.numpy().tolist())

    image_embeds_all = np.concatenate(image_embeds_all, axis=0)
    text_embeds_all = np.concatenate(text_embeds_all, axis=0)

    if dist.is_initialized() and dist.get_world_size() > 1:
        centroids = run_faiss_kmeans_on_gpu(np.concatenate([image_embeds_all, text_embeds_all], axis=1),
                                            K=model.module.num_prototypes,
                                            D=model.module.feature_dim * 2,
                                            gpu_id=gpu
                                            )
        model.module.prototypes.data.copy_(centroids)
    else:
        centroids = run_faiss_kmeans_on_gpu(np.concatenate([image_embeds_all, text_embeds_all], axis=1),
                                            K=model.num_prototypes,
                                            D=model.feature_dim * 2,
                                            gpu_id=gpu
                                            )
        model.prototypes.data.copy_(centroids)

    model.train()


def run_faiss_kmeans_on_gpu(features_np, K, D, gpu_id, seed=42, niter=20):
    import faiss

    kmeans = faiss.Kmeans(d=D, k=K, niter=niter, seed=seed, verbose=True, gpu=True)
    kmeans.train(features_np)

    cen_kmeans = torch.tensor(kmeans.centroids).cuda(gpu_id)
    cen_kmeans = cen_kmeans / cen_kmeans.norm(dim=1, keepdim=True)

    return cen_kmeans
