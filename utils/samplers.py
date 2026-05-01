import math
from collections import Counter

import torch
from torch.utils.data import Sampler


def sample_group_key(sample, group_by="source_label"):
    source_group = str(sample.get("source_group", "default"))
    label = int(sample.get("class_label", -1))
    if group_by == "source":
        return source_group
    if group_by == "label":
        return f"label={label}"
    if group_by == "source_label":
        return f"{source_group}|label={label}"
    raise ValueError(f"Unsupported balanced sampler group_by: {group_by}")


class BalancedDistributedSampler(Sampler):
    """Replacement sampler that balances source/label groups across DDP ranks."""

    def __init__(
        self,
        dataset,
        group_by="source_label",
        replacement=True,
        samples_per_epoch=None,
        seed=0,
        num_replicas=1,
        rank=0,
    ):
        if len(dataset) <= 0:
            raise ValueError("BalancedDistributedSampler requires a non-empty dataset.")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got: {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got: {rank}")

        self.dataset = dataset
        self.group_by = group_by
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0
        self.group_keys = [sample_group_key(sample, group_by=group_by) for sample in dataset.samples]
        self.group_counts = Counter(self.group_keys)
        self.weights = torch.as_tensor(
            [1.0 / float(self.group_counts[key]) for key in self.group_keys],
            dtype=torch.double,
        )

        total_samples = len(dataset) if samples_per_epoch is None else int(samples_per_epoch)
        if total_samples <= 0:
            raise ValueError(f"samples_per_epoch must be positive when set, got: {samples_per_epoch}")
        self.num_samples = int(math.ceil(total_samples / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.replacement:
            indices = torch.multinomial(
                self.weights,
                self.total_size,
                replacement=True,
                generator=generator,
            ).tolist()
        else:
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
            repeats = int(math.ceil(self.total_size / len(indices)))
            indices = (indices * repeats)[: self.total_size]
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def summary(self):
        groups = ", ".join(f"{key}:{count}" for key, count in sorted(self.group_counts.items()))
        return (
            f"BalancedDistributedSampler(group_by={self.group_by}, replacement={self.replacement}, "
            f"rank={self.rank}/{self.num_replicas}, samples_per_rank={self.num_samples}, groups={groups})"
        )


def build_balanced_train_sampler(dataset, sampler_config, distributed=False, seed=0):
    if not sampler_config.get("enabled", False):
        return None

    if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        num_replicas = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        num_replicas = 1
        rank = 0

    return BalancedDistributedSampler(
        dataset=dataset,
        group_by=sampler_config.get("group_by", "source_label"),
        replacement=sampler_config.get("replacement", True),
        samples_per_epoch=sampler_config.get("samples_per_epoch"),
        seed=seed,
        num_replicas=num_replicas,
        rank=rank,
    )
