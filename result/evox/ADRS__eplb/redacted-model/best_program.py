# SPDX-License-Identifier: Apache-2.0
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.

The rearrangement algorithm is adapted from
[DeepSeek EPLB](https://github.com/deepseek-ai/eplb).

Please find at [#12](https://github.com/deepseek-ai/EPLB/issues/12) an example
on how the EPLB algorithm works.
"""

# EVOLVE-BLOCK-START

import torch


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    device = weight.device
    sorted_vals, sorted_indices = weight.sort(-1, descending=True)
    
    # Initialize pack tracking tensors
    pack_weights = torch.zeros(num_layers, num_packs, device=device)
    pack_items = torch.zeros(num_layers, num_packs, dtype=torch.int64, device=device)
    
    # Create output tensors with sorted indices shape
    pack_index_sorted = torch.empty(num_layers, num_groups, dtype=torch.int64, device=device)
    rank_in_pack_sorted = torch.empty(num_layers, num_groups, dtype=torch.int64, device=device)
    
    # Precompute layer indices for efficient indexing
    layer_indices = torch.arange(num_layers, device=device)
    
    # Process each item position in sorted order
    for pos in range(num_groups):
        # Get current item indices and weights
        current_indices = sorted_indices[:, pos]
        current_weights = weight[layer_indices, current_indices]
        
        # Find pack with smallest weight that still has space
        mask = pack_items < groups_per_pack
        # Replace weights of full packs with infinity so they're not selected
        candidate_weights = torch.where(mask, pack_weights, torch.inf)
        selected_packs = candidate_weights.argmin(-1)
        
        # Update tracking tensors
        pack_index_sorted[:, pos] = selected_packs
        rank_in_pack_sorted[:, pos] = pack_items[layer_indices, selected_packs]
        pack_weights[layer_indices, selected_packs] += current_weights
        pack_items[layer_indices, selected_packs] += 1
    
    # Map back to original indices using advanced indexing
    inverse_indices = torch.argsort(sorted_indices, dim=-1)
    pack_index = pack_index_sorted.gather(1, inverse_indices)
    rank_in_pack = rank_in_pack_sorted.gather(1, inverse_indices)
    
    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum
    load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    
    # Initialize with base experts
    phy2log = torch.arange(num_log, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_log, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    
    if num_redundant <= 0:
        return phy2log, rank, logcnt
    
    # Preallocate remaining physical experts for better memory efficiency
    remaining_phy = torch.empty(n, num_redundant, dtype=torch.int64, device=device)
    remaining_rank = torch.empty(n, num_redundant, dtype=torch.int64, device=device)
    
    # Precompute layer indices and weight float conversion
    layer_indices = torch.arange(n, device=device)
    weight_float = weight.float()
    
    # Vectorized replication selection with precomputed load per replica
    load_per_replica = weight_float / logcnt.float()
    
    # Batch process replication using argmax with top-k for multiple selections
    for i in range(num_redundant):
        # Find expert with highest load per replica
        selected_experts = load_per_replica.argmax(-1)
        
        remaining_phy[:, i] = selected_experts
        remaining_rank[:, i] = logcnt[layer_indices, selected_experts]
        
        # Update counters and load per replica in-place
        logcnt[layer_indices, selected_experts] += 1
        load_per_replica[layer_indices, selected_experts] = weight_float[layer_indices, selected_experts] / logcnt[layer_indices, selected_experts].float()
    
    # Combine base and replicated experts
    phy2log = torch.cat([phy2log, remaining_phy], dim=1)
    rank = torch.cat([rank, remaining_rank], dim=1)
    
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
        (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64,
                         device=perm.device).expand(perm.shape),
        )
        return inv

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    
    # Vectorized logical to mapped logical transformation
    node_base_indices = torch.arange(0, num_logical_experts, 
                                    num_logical_experts // num_nodes, 
                                    device=weight.device).view(1, -1, 1)
    group_positions = (group_pack_index * groups_per_node + group_rank_in_pack).unsqueeze(-1)
    log2mlog = (group_positions * group_size + 
                torch.arange(group_size, device=weight.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs
    tokens_per_phy = (tokens_per_mlog / mlogcnt.float()).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    # Vectorized physical to logical mapping
    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + node_base_indices).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all
            logical experts
        num_replicas: number of physical experts, must be a multiple of
            `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
            (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of
            each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica
            indices for each expert
        expert_count: [layers, num_logical_experts], number of physical
            replicas for each logical expert
    """
    num_layers, num_logical_experts = weight.shape
    device = weight.device
    
    # Keep processing on GPU for performance
    weight = weight.float()
    
    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # use global load-balance policy with optimized grouping
        optimal_groups = num_nodes if num_logical_experts % num_nodes == 0 else 1
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, optimal_groups, num_nodes if optimal_groups !=1 else 1, num_gpus)
    
    # Optimize logical to physical mapping creation with precomputed max replicas
    max_replicas = logcnt.max().item()
    log2phy = torch.full(
        (num_layers, num_logical_experts, max_replicas),
        -1,
        dtype=torch.int64,
        device=device,
    )
    
    # Vectorized scatter operation with precomputed indices
    flat_indices = phy2log * max_replicas + phyrank
    flat_values = torch.arange(num_replicas, dtype=torch.int64,
                               device=device).expand(num_layers, -1)
    log2phy.view(num_layers, -1).scatter_(1, flat_indices, flat_values)
    
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]