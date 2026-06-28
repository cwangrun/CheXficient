# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from configs import Config

########### running ###########
# torchrun --nproc_per_node=8 main.py <config>


def chexficient():
    return Config(
        min_ratio=0.003,
        extra_prompt=True,
        aug_tag=True,
        nodes=1,
        ngpus=8,
    )
