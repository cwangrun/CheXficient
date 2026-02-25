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

import argparse
import datetime
import os
import time
import json
from pathlib import Path

import util.misc as misc

from util.misc import NativeScalerWithGradNormCount as NativeScaler
import torch.nn as nn
from models_clip import CheXficient
from engine import train_one_epoch, evaluate, warmup_prototypes
import torchvision.transforms as transforms
from PIL import Image
import wandb
import shutil
import torch.multiprocessing as mp
from util.funs import *


def get_mean_std(args):
    if "augreg" in args.vision_backbone:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    return mean, std


def get_val_transform(args):
    """moved from SLIP's eval_zeroshot.py"""
    import torchvision.transforms as transforms
    mean, std = get_mean_std(args)
    print(args.vision_backbone, "val_normalizer", mean, std)
    return transforms.Compose([
            # transforms.Resize(256),
            transforms.Resize(args.image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(args.image_size),
            # lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
    ])


def _transform(args):
    mean, std = get_mean_std(args)
    return transforms.Compose([
        transforms.Resize(args.image_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(args.image_size),
        # lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_dataset(args, tokenizer):
    from clipeval import datasets
    from clipeval import eval_zeroshot

    train_dataset = datasets.MultimodalPretrainingDataset(split='train', transform=_transform(args), max_bert_length=args.max_bert_length, tokenizer=tokenizer)
    dataset, all_labels = eval_zeroshot.load_metadata("clipeval")

    val_dataset = {}
    for d in dataset:    # 'chexpert_test, ...'
        val_dataset[d] = datasets.get_downstream_dataset(dataset, d, is_train=False, transform=get_val_transform(args))
    args.val_dataset = val_dataset

    return train_dataset


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    source_files = [
        __file__, 'configs.py', 'constants.py', 'engine.py', 'main.py', 'models_clip.py',
        'util/funs.py', 'util/helpers.py', 'util/projection.py', 'util/lr_scheduler.py',
        'clipeval/datasets.py', 'clipeval/eval_zeroshot.py',
        'clipeval/dataset_catalog.json', 'clipeval/labels.json',
    ]
    for src_file in source_files:
        if os.path.exists(src_file):
            shutil.copy(src_file, args.output_root)

    ngpus_per_node = torch.cuda.device_count()

    if ngpus_per_node > 1:
        args.distributed = True
        args.world_size = ngpus_per_node * args.nodes
        args.dist_url = f'tcp://127.0.0.1:' + str(12000 + np.random.randint(0, 1000))
        print(f"Starting distributed training on {ngpus_per_node} GPUs")
        print(f"Distribution URL: {args.dist_url}")
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
    else:
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        print("Starting single GPU training")
        main_worker(0, args)


def main_worker(gpu, args):

    args.gpu = gpu
    args.rank = gpu if not hasattr(args, 'rank') else args.rank

    torch.cuda.set_device(gpu)

    if args.distributed:
        dist.init_process_group(
            backend='nccl',
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )
        dist.barrier()
        print(f"Process {args.rank} initialized.")

    # fix seed for reproducibility
    seed_everything(args.seed)

    if args.rank == 0:
        # WandB – Initialize a new run
        wandb.init(project='CheXficient', mode='disabled')  # mode='disabled'
        wandb.run.name = 'Dino-' + wandb.run.id

    model = CheXficient(image_size=args.image_size, temperature=args.temperature, num_prototypes=args.num_prototypes)
    model.to(torch.device(f'cuda:{gpu}'))
    tokenizer = model.text_encoder.tokenizer
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    print('image size: %d' % (args.image_size))
    print('max_bert_length: %d' % (args.max_bert_length))

    eff_batch_size = args.batch_size * args.accum_iter * args.world_size

    log_writer = None

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    dataset_train = build_dataset(args, tokenizer=tokenizer)

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    if not isinstance(dataset_train, torch.utils.data.IterableDataset):
        print("len(dataset)", len(dataset_train))
    else:
        print("cannot estimate len of torch.utils.data.IterableDataset.")

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # https://github.com/rwightman/pytorch-image-models/blob/fd360ac951a179474917f4b2d21db8669bf87f68/timm/models/vision_transformer.py#L407
    no_weight_decay_list = {'pos_embed', 'cls_token', 'dist_token'}  # THIS DOESN'T MATTER YET as we frozen all.
    head_weight_decay_list = {"visual_projection", "text_projection"}

    p_wd, p_no_wd = [], []
    p_head_wd = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim == 1 or n in no_weight_decay_list or 'prototypes' in n or 'centroids' in n:
            p_no_wd.append(p)
        elif hasattr(args, "no_wd_emb") and isinstance(p, torch.nn.Embedding):
            p_no_wd.append(p)
        elif hasattr(args, "no_wd_ln") and isinstance(p, torch.nn.LayerNorm):
            p_no_wd.append(p)
        elif hasattr(args, "head_weight_decay") and [True for _part in head_weight_decay_list if _part in n]:
            p_head_wd.append(p)
        else:
            p_wd.append(p)   # prototypes in this group

    param_groups = [{"params": p_wd, "weight_decay": args.weight_decay},
                    {"params": p_no_wd, "weight_decay": 0.},
                    ]

    if p_head_wd:
        param_groups.append({"params": p_head_wd, "weight_decay": args.head_weight_decay})

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, eps=1e-8)

    loss_scaler = NativeScaler(args.fp16)


    # checkpoint = torch.load("./to_hf/checkpoint.pth", map_location='cpu')
    # subset_used = list(checkpoint['subset'])
    # res = model.load_state_dict(checkpoint['model'], strict=False)
    # print(res, 'training epoch: %d' % checkpoint['epoch'], 'training step: %d' % checkpoint['step'])


    start_epoch, best_acc, step = 0, [0.], [0]
    if args.resume:
        if args.resume.endswith(".pth"):  # a pytorch checkpoint for resuming training.
            if args.resume.startswith("checkpoint"):
                args.resume = os.path.join(args.output_dir, args.resume)
            start_epoch, _, best_acc, step = misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
            best_acc, step = [best_acc], [step if step is not None else 0]

            if isinstance(dataset_train, torch.utils.data.IterableDataset):
                # random from step to avoid dupped train.
                dataset_train.start_shard_id = step[0] % dataset_train.num_shards
            print("resuming", args.resume, "from step", step[0], "with best_acc", best_acc[0])
        else:
            print("assuming a huggingface transformer pretrained model (no optimizer states).")
            from models_clip import TextEncoder
            metric = evaluate(args, model, tokenizer)
            model = TextEncoder.from_pretrained(args.resume)

    if args.eval:
        metric = evaluate(args, model, tokenizer)
        json_str = json.dumps({"step": step[0], "acc": metric, "seen": eff_batch_size * step[0]})
        print(json_str)
        exit(0)


    if args.prototype_warmup_steps is not None and args.prototype_warmup_steps > 1:

        warmup_batch_size = args.batch_size * 2

        warmup_data_loader_producer = torch.utils.data.DataLoader(
            dataset_train,
            sampler=None,
            batch_size=warmup_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            collate_fn=getattr(dataset_train, "collate_fn", None),
            persistent_workers=True
        )

        def producer_fn(epoch):
            while True:
                for batch in warmup_data_loader_producer:
                    yield batch
                epoch += 1

        producer_iter = iter(producer_fn(start_epoch))

        if dist.is_initialized() and dist.get_world_size() > 1:
            warmup_prototypes(step, gpu, producer_iter, model, args)
            dist.barrier()
            if hasattr(model, 'module'):
                dist.broadcast(model.module.prototypes, src=0)
            else:
                dist.broadcast(model.prototypes, src=0)
        else:
            warmup_prototypes(step, gpu, producer_iter, model, args)

    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True
        )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        collate_fn=getattr(dataset_train, "collate_fn", None),
        pin_memory=args.pin_mem,
        sampler=train_sampler,
        drop_last=True
    )

    start_time = time.time()
    global_example_ids = set()
    for epoch in range(start_epoch, args.epochs):

        if step[0] >= args.max_update:
            if args.rank == 0:
                print(f"Reach max steps ({args.max_update}), terminating training")
            break

        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, model_without_ddp, tokenizer, data_loader_train, best_acc, optimizer, torch.device(f'cuda:{gpu}'),
            epoch, step, loss_scaler, eff_batch_size, args.clip_grad, global_example_ids, log_writer=log_writer,
            args=args
        )

        if epoch < args.curation_epochs:
            dataset_train.set_subset(global_example_ids)
            train_sampler = None
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset_train,
                    num_replicas=args.world_size,
                    rank=args.rank,
                    shuffle=True
                )
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                num_workers=args.num_workers,
                collate_fn=getattr(dataset_train, "collate_fn", None),
                pin_memory=args.pin_mem,
                sampler=train_sampler,
                drop_last=True
            )

        if args.rank == 0:
            if not isinstance(dataset_train, torch.utils.data.IterableDataset):
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, global_example_ids=global_example_ids,
                    loss_scaler=loss_scaler, epoch=epoch, epoch_name="last", best_acc=best_acc[0], step=step[0])
            else:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, global_example_ids=global_example_ids,
                    loss_scaler=loss_scaler, epoch=0, epoch_name="last", best_acc=best_acc[0], step=step[0])

    metric = evaluate(args, model_without_ddp, tokenizer)


    if args.rank == 0:

        with open(os.path.join(args.output_dir, 'subset.json'), 'w') as f_subset:
            for item in global_example_ids:
                f_subset.write(json.dumps(item) + '\n')
        print('final subset ratio:', len(global_example_ids) / len(dataset_train.all_filenames))

        json_str = json.dumps({"step": step[0], "acc": metric, "seen": eff_batch_size * step[0]})
        print(json_str)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        cleanup_distributed()


def parse_args():
    '''see configs.py or sweep.py (we only allow pre-defined config).'''
    parser = argparse.ArgumentParser(description='CheXficient', add_help=False)

    parser.add_argument('--config_name', default='chexficient', help='see configs.py')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', default=False)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--eval', default=None, action='store_true')

    # parser.add_argument('--config_name', default='chexficient', help='see configs.py')
    # parser.add_argument('--world_size', default=1, type=int)
    # parser.add_argument('--local_rank', default=-1, type=int)
    # parser.add_argument('--dist_on_itp', action='store_true')
    # parser.add_argument('--dist_url', default='env://')
    # parser.add_argument('--resume', default=None, type=str)
    # parser.add_argument('--eval', default=None, action='store_true')

    cmd_args = parser.parse_args()

    import run_configs
    config = getattr(run_configs, cmd_args.config_name)().add_cmd_args(cmd_args)
    return config


if __name__ == '__main__':
    args = parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
