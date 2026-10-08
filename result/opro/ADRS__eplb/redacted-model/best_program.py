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

    # Vectorized sorting and packing for improved efficiency
    device = weight.device
    sorted_vals, sorted_indices = weight.sort(-1, descending=True)
    
    # Initialize pack tracking tensors on device
    pack_weights = torch.zeros(num_layers, num_packs, dtype=weight.dtype, device=device)
    pack_items = torch.zeros(num_layers, num_packs, dtype=torch.int64, device=device)
    
    pack_index = torch.full_like(weight, -1, dtype=torch.int64, device=device)
    rank_in_pack = torch.full_like(weight, -1, dtype=torch.int64, device=device)
    
    # Process items in batches using vectorized operations
    for idx in range(num_groups):
        current_items = sorted_indices[:, idx]
        
        # Find packs with available slots
        available_packs = (pack_items < groups_per_pack)
        
        # Get current weights only for available packs
        masked_weights = torch.where(available_packs, pack_weights, torch.inf)
        
        # Select pack with minimum weight for each layer
        selected_pack = masked_weights.argmin(dim=-1)
        
        # Update pack assignments
        pack_index[torch.arange(num_layers), current_items] = selected_pack
        rank_in_pack[torch.arange(num_layers), current_items] = pack_items[torch.arange(num_layers), selected_pack]
        
        # Update pack tracking
        pack_weights[torch.arange(num_layers), selected_pack] += weight[torch.arange(num_layers), current_items]
        pack_items[torch.arange(num_layers), selected_pack] += 1
    
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
    
    # Early exit if no replication needed
    if num_redundant == 0:
        return phy2log, rank, logcnt
    
    # Preallocate remaining slots
    remaining_phy = torch.empty(n, num_redundant, dtype=torch.int64, device=device)
    remaining_rank = torch.empty(n, num_redundant, dtype=torch.int64, device=device)
    
    # Vectorized replication selection
    for i in range(num_redundant):
        # Calculate load per replica ratio
        load_ratio = weight / logcnt.float()
        redundant_indices = load_ratio.max(dim=-1).indices
        
        remaining_phy[:, i] = redundant_indices
        remaining_rank[:, i] = logcnt[torch.arange(n), redundant_indices]
        logcnt[torch.arange(n), redundant_indices] += 1
    
    # Combine base and replicated experts
    phy2log = torch.cat([phy2log, remaining_phy], dim=-1)
    rank = torch.cat([rank, remaining_rank], dim=-1)
    
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

    # Step 1: pack groups to nodes - keep device consistency
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    
    # Vectorized logical to mapped logical transformation
    group_base = (group_pack_index * groups_per_node + group_rank_in_pack) * group_size
    log2mlog = group_base.unsqueeze(-1) + torch.arange(group_size, dtype=torch.int64, device=group_base.device)
    log2mlog = log2mlog.flatten(-2)
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
    node_offset = torch.arange(0, num_logical_experts, 
                              num_logical_experts // num_nodes, 
                              device=pphy2mlog.device).view(1, -1, 1)
    pphy2mlog = pphy2mlog.view(num_layers, num_nodes, -1) + node_offset
    pphy2mlog = pphy2mlog.flatten(-2)
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
    
    # Keep tensor on original device instead of moving to CPU
    weight = weight.float()
    
    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)
    
    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    
    # Precompute indices for scatter operation
    scatter_indices = phy2log * maxlogcnt + phyrank
    expand_indices = torch.arange(num_replicas, dtype=torch.int64, device=device).expand(num_layers, -1)
    
    # Initialize and populate logical to physical map
    log2phy = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        scatter_indices,
        expand_indices,
    )
    
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]