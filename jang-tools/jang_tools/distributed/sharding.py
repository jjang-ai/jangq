"""Sharding plans for MiMo-V2.5-Pro (and any DSV-style MoE).

Three orthogonal axes:

1. **Expert-Parallel (EP)** — split `n_routed_experts` across nodes.
   MiMo-V2.5-Pro upstream already ships expert-pipeline shards
   `model_pp0_ep{0..31}_shard*.safetensors`; map ep-shard → node directly.

2. **Pipeline-Parallel (PP)** — split `num_hidden_layers` across nodes.
   Replicate dense layer-0 (16384 inter) on whichever node owns it.

3. **Tensor-Parallel (TP)** — split attention projections across heads.
   Only used when one head-group is too big for a node; mostly off for
   MiMo because GQA already keeps KV small.

The default plan for the two-node rig (Studio 256 + M4 Max 128):
    EP  : Studio = experts 0–255 (≈83 GB FP8), MacBook = 256–383 (≈42 GB)
    PP  : both nodes own all 70 layers; experts gated by EP
    TP  : off (each node runs full attention locally)

Routed gather: each step the gate computes top-8 expert ids; tokens whose
top expert lives on a remote node get sent there (compact, tokens are
6144 * bf16 = 12 KB each). Returns are summed via all_sum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class ShardPlan:
    rank: int
    world_size: int
    expert_owner: list[int]  # length n_routed_experts: which rank owns each expert
    layer_owner: list[int]   # length num_hidden_layers; -1 = replicated
    tp_size: int = 1
    tp_rank: int = 0

    @property
    def my_experts(self) -> list[int]:
        return [e for e, r in enumerate(self.expert_owner) if r == self.rank]

    @property
    def my_layers(self) -> list[int]:
        return [l for l, r in enumerate(self.layer_owner)
                if r == self.rank or r == -1]


def even_expert_split(n_experts: int, world_size: int,
                      ram_weights: Sequence[float] | None = None) -> list[int]:
    """Assign each expert id to a rank.

    `ram_weights[r]` ∈ (0,1] scales rank r's share. For Studio(256) +
    MacBook(128), pass [1.0, 0.5] to give Studio 2/3 of experts.
    """
    if ram_weights is None:
        ram_weights = [1.0] * world_size
    total = sum(ram_weights)
    cuts = [0]
    acc = 0.0
    for w in ram_weights[:-1]:
        acc += w
        cuts.append(int(round(acc / total * n_experts)))
    cuts.append(n_experts)
    owner = [0] * n_experts
    for r, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        for e in range(a, b):
            owner[e] = r
    return owner


def replicate_dense_layer0(num_layers: int) -> list[int]:
    """Default PP plan: replicate every layer (-1) so attn runs locally;
    EP handles the heavy MoE FFN. Override if pipelining attention too."""
    return [-1] * num_layers


def default_mimo_v2_plan(rank: int, world_size: int,
                         n_experts: int = 384,
                         num_layers: int = 70,
                         ram_weights: Sequence[float] | None = None) -> ShardPlan:
    return ShardPlan(
        rank=rank,
        world_size=world_size,
        expert_owner=even_expert_split(n_experts, world_size, ram_weights),
        layer_owner=replicate_dense_layer0(num_layers),
    )
