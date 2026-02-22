import torch
import random, os
import numpy as np
import torch.distributed as dist


def setup_for_distributed(rank):
    if rank == 0:
        return
    import builtins
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force or rank == 0:
            builtin_print(*args, **kwargs)

    builtins.print = print


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, step, loss, save_path, is_distributed=False):
    model_to_save = model.module if is_distributed else model
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, f"{save_path}/step_{step}.pt")
    print(f"Checkpoint saved at step {step}")


def load_checkpoint(model, optimizer, checkpoint_path):
    try:
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint.get('epoch', 0), checkpoint.get('step', 0)
        else:
            print("No checkpoint found, starting from scratch")
            return 0, 0
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0, 0


def convert_single_to_ddp_state_dict(single_state_dict):
    ddp_state_dict = {}
    for key, value in single_state_dict.items():
        ddp_state_dict[f'module.{key}'] = value
    return ddp_state_dict


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor, world_size):
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return tensor