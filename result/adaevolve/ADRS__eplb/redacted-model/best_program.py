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
    # Sort weights descending and get indices, keep on device
    _, indices = weight.sort(-1, descending=True)
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64)
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    
    # Initialize tracking tensors on device
    pack_weights = torch.zeros((num_layers, num_packs), dtype=weight.dtype, device=device)
    pack_items = torch.zeros((num_layers, num_packs), dtype=torch.int64, device=device)

    # Precompute batch indices for repeated use
    batch_indices = torch.arange(num_layers, device=device)
    
    # Process all layers in parallel using vectorized operations
    for group_idx in range(num_groups):
        # Get current group indices for all layers
        current_groups = indices[:, group_idx]
        
        # Create mask for available packs (those not yet full)
        available_mask = pack_items < groups_per_pack
        
        # Calculate minimum weight among available packs for each layer
        # Set unavailable pack weights to infinity so they're not selected
        masked_weights = torch.where(available_mask, pack_weights, torch.full_like(pack_weights, float('inf')))
        selected_packs = torch.argmin(masked_weights, dim=1)
        
        # Update pack assignments
        pack_index[batch_indices, current_groups] = selected_packs
        rank_in_pack[batch_indices, current_groups] = pack_items[batch_indices, selected_packs]
        
        # Update pack weights and item counts using in-place operations for efficiency
        pack_weights.scatter_add_(1, selected_packs.unsqueeze(1), weight[batch_indices, current_groups].unsqueeze(1))
        pack_items.scatter_add_(1, selected_packs.unsqueeze(1), torch.ones_like(selected_packs, dtype=torch.int64).unsqueeze(1))
    
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
    
    # Early exit if no replication needed
    if num_redundant == 0:
        phy2log = torch.arange(num_log, dtype=torch.int64, device=device).repeat(n, 1)
        rank = torch.zeros(n, num_log, dtype=torch.int64, device=device)
        logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
        return phy2log, rank, logcnt
    
    # Initialize with base logical experts
    phy2log = torch.arange(num_log, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_log, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    
    # Precompute arange for indexing
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    
    # Pre-allocate tensors for new replicas to avoid repeated cat operations
    new_phy2log = torch.empty((n, num_redundant), dtype=torch.int64, device=device)
    new_rank = torch.empty((n, num_redundant), dtype=torch.int64, device=device)
    
    # Vectorized replication loop with incremental load updates for efficiency
    load_per_replica = weight / logcnt.float()
    
    for i in range(num_redundant):
        # Find logical expert with maximum load per replica for each layer
        redundant_indices = torch.argmax(load_per_replica, dim=-1)
        
        # Store new replica mappings
        new_phy2log[:, i] = redundant_indices
        # Get current count for selected experts and use as rank
        current_rank = logcnt[arangen, redundant_indices]
        new_rank[:, i] = current_rank
        # Increment replica count using in-place operation
        logcnt[arangen, redundant_indices] += 1
        
        # Update load per replica incrementally instead of recalculating entire tensor
        load_per_replica[arangen, redundant_indices] = weight[arangen, redundant_indices] / logcnt[arangen, redundant_indices]
    
    # Concatenate all replicas once
    phy2log = torch.cat([phy2log, new_phy2log], dim=1)
    rank = torch.cat([rank, new_rank], dim=1)
    
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
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt.float()).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy)  # [num_layers * num_nodes, num_log_per_nodes]
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + torch.arange(
        0,
        num_logical_experts,
        num_logical_experts // num_nodes,
        device=group_pack_index.device,
    ).view(1, -1, 1)).flatten(-2)
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
    # Keep tensor on original device instead of moving to CPU
    weight = weight.float()
    
    # Early exit if no replication needed
    if num_replicas == num_logical_experts:
        phy2log = torch.arange(num_logical_experts, dtype=torch.int64, device=weight.device).repeat(num_layers, 1)
        phyrank = torch.zeros(num_layers, num_logical_experts, dtype=torch.int64, device=weight.device)
        logcnt = torch.ones(num_layers, num_logical_experts, dtype=torch.int64, device=weight.device)
        log2phy = torch.zeros(num_layers, num_logical_experts, 1, dtype=torch.int64, device=weight.device)
        log2phy[:, :, 0] = torch.arange(num_logical_experts, device=weight.device).expand(num_layers, -1)
        return phy2log, log2phy, logcnt
    
    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)
    
    # Calculate maximum required replica count per logical expert to avoid over-allocation
    maxlogcnt = logcnt.max().item()
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64,
                     device=log2phy.device).expand(num_layers, -1),
    )
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]