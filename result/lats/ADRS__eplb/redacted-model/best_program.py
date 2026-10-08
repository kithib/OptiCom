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

    # Sort weights descending and get indices
    _, indices = weight.sort(-1, descending=True)
    device = weight.device

    # Initialize pack tracking tensors on device
    pack_weights = torch.zeros(num_layers, num_packs, device=device, dtype=torch.float32)
    pack_items = torch.zeros(num_layers, num_packs, device=device, dtype=torch.int64)
    
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64)
    rank_in_pack = torch.full_like(weight, fill_value=-1, dtype=torch.int64)

    # Process each position in sorted order
    for pos in range(num_groups):
        # Get current group for each layer
        current_groups = indices[:, pos]
        
        # Find pack with smallest weight among those not full
        mask = pack_items < groups_per_pack
        # Add large value to full packs to exclude them
        pack_weights_with_penalty = torch.where(mask, pack_weights, torch.finfo(torch.float32).max)
        selected_packs = pack_weights_with_penalty.argmin(dim=1)
        
        # Update pack assignments
        batch_indices = torch.arange(num_layers, device=device)
        pack_index[batch_indices, current_groups] = selected_packs
        rank_in_pack[batch_indices, current_groups] = pack_items[batch_indices, selected_packs]
        
        # Update pack tracking
        pack_weights[batch_indices, selected_packs] += weight[batch_indices, current_groups]
        pack_items[batch_indices, selected_packs] += 1

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
    num_layers, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    
    # Initialize with original experts
    phy2log = torch.arange(num_log, dtype=torch.int64, device=device).repeat(num_layers, 1)
    rank = torch.zeros(num_layers, num_log, dtype=torch.int64, device=device)
    logcnt = torch.ones(num_layers, num_log, dtype=torch.int64, device=device)
    
    if num_redundant <= 0:
        return phy2log, rank, logcnt
    
    # Preallocate tensors for redundant experts
    redundant_phy2log = torch.empty(num_layers, num_redundant, dtype=torch.int64, device=device)
    redundant_rank = torch.empty(num_layers, num_redundant, dtype=torch.int64, device=device)
    
    batch_indices = torch.arange(num_layers, device=device)
    
    # Vectorized calculation for all replicas at once
    for i in range(num_redundant):
        # Calculate load per replica
        load_per_replica = weight / logcnt.float()
        
        # Find expert with maximum load per replica
        selected_experts = load_per_replica.argmax(dim=1)
        
        # Store results
        redundant_phy2log[:, i] = selected_experts
        redundant_rank[:, i] = logcnt[batch_indices, selected_experts]
        
        # Update replica count
        logcnt[batch_indices, selected_experts] += 1
    
    # Combine original and redundant experts
    phy2log = torch.cat([phy2log, redundant_phy2log], dim=1)
    rank = torch.cat([rank, redundant_rank], dim=1)
    
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
        rank: [num_moe_layers, num_physical_experts]
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
    device = weight.device

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm, device=device)
        arange = torch.arange(perm.size(1), dtype=torch.int64, device=device).expand(perm.shape)
        inv.scatter_(1, perm, arange)
        return inv

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    
    # Create logical to mapped logical indices with efficient broadcasting
    base_indices = (group_pack_index * groups_per_node + group_rank_in_pack) * group_size
    offset_indices = torch.arange(group_size, dtype=torch.int64, device=device).view(1, 1, -1)
    log2mlog = (base_indices.unsqueeze(-1) + offset_indices).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical experts to GPUs
    tokens_per_phy = (tokens_per_mlog / mlogcnt.float()).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
    
    # Create physical to physical packed indices
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    # Map packed physical to mapped logical
    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    node_offset = torch.arange(0, num_logical_experts, 
                              num_logical_experts // num_nodes, 
                              device=device).view(1, -1, 1)
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + node_offset).flatten(-2)
    
    # Map packed physical to original logical
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
    
    # Keep tensor on device for all operations
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
    
    # Create logical to physical mapping
    log2phy = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=device,
    )
    
    # Calculate scatter indices and update mapping efficiently
    scatter_indices = phy2log * maxlogcnt + phyrank
    phy_indices = torch.arange(num_replicas, dtype=torch.int64, device=device).expand(num_layers, -1)
    log2phy.view(num_layers, -1).scatter_(
        -1,
        scatter_indices,
        phy_indices,
    )
    
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]