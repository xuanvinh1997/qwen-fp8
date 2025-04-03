

import torch, os
import torch.distributed as dist



def setup_distributed(backend="nccl"):
    """Initialize distributed training environment"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        return False, 0, 1

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
    dist.barrier()
    return True, local_rank, world_size