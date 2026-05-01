import os
import math
import torch
from torch.utils.data import Sampler

class ShardAwareSampler(Sampler):
    def __init__(self, dataset, batches_per_shard, shuffle=True, seed=42):
        self.dataset = dataset
        self.batches_per_shard = batches_per_shard
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # PyTorch Lightning sets these environment variables automatically in DDP
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("GLOBAL_RANK", 0))

        self.total_batches = len(dataset)
        self.total_shards = math.ceil(self.total_batches / self.batches_per_shard)

        # Calculate exact length of batches for this specific GPU rank
        shard_indices = list(range(self.total_shards))
        shards_for_this_rank = shard_indices[self.rank :: self.world_size]
        
        self._len = 0
        for shard_idx in shards_for_this_rank:
            start = shard_idx * self.batches_per_shard
            end = min(start + self.batches_per_shard, self.total_batches)
            self._len += (end - start)

    def __iter__(self):
        if self.shuffle:
            # Shuffle the order of the shards globally
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            shard_indices = torch.randperm(self.total_shards, generator=g).tolist()
        else:
            shard_indices = list(range(self.total_shards))

        # Distribute whole shards to this GPU (Block distribution)
        shards_for_this_rank = shard_indices[self.rank :: self.world_size]

        batch_indices = []
        for shard_idx in shards_for_this_rank:
            start_batch = shard_idx * self.batches_per_shard
            end_batch = min(start_batch + self.batches_per_shard, self.total_batches)
            
            batches_in_shard = list(range(start_batch, end_batch))
            
            if self.shuffle:
                # Randomize batch order within the shard for extra stochasticity
                # while keeping reads localized to this specific shard file.
                g_inner = torch.Generator()
                g_inner.manual_seed(self.seed + self.epoch + shard_idx)
                shuffled_idx = torch.randperm(len(batches_in_shard), generator=g_inner).tolist()
                batches_in_shard = [batches_in_shard[i] for i in shuffled_idx]
                
            batch_indices.extend(batches_in_shard)

        return iter(batch_indices)

    def __len__(self):
        return self._len

    def set_epoch(self, epoch):
        self.epoch = epoch
