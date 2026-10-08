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

    # Vectorized greedy packing with sorted weights
    device = weight.device
    sorted_vals, indices = weight.sort(dim=-1, descending=True)
    
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device=device)
    rank_in_pack = torch.full_like(pack_index, fill_value=-1, dtype=torch.int64, device=device)
    
    pack_weights = torch.zeros((num_layers, num_packs), dtype=weight.dtype, device=device)
    pack_items = torch.zeros((num_layers, num_packs), dtype=torch.int64, device=device)
    
    # Precompute layer indices for repeated use
    layer_indices = torch.arange(num_layers, device=device)
    
    for idx in range(num_groups):
        current_groups = indices[:, idx]
        
        # Find available packs and select those with minimum weight
        available_packs = pack_items < groups_per_pack
        masked_weights = torch.where(available_packs, pack_weights, torch.finfo(weight.dtype).max)
        selected_packs = masked_weights.argmin(dim=-1)
        
        # Update assignments
        pack_index[layer_indices, current_groups] = selected_packs
        rank_in_pack[layer_indices, current_groups] = pack_items[layer_indices, selected_packs]
        
        # Update pack tracking
        pack_weights[layer_indices, selected_packs] += sorted_vals[:, idx]
        pack_items[layer_indices, selected_packs] += 1
    
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
    
    # Initialize base mappings
    phy2log = torch.arange(num_log, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_log, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    
    if num_redundant <= 0:
        return phy2log, rank, logcnt
    
    # Preallocate extended tensors for better performance
    ext_phy2log = torch.empty(n, num_redundant, dtype=torch.int64, device=device)
    ext_rank = torch.empty(n, num_redundant, dtype=torch.int64, device=device)
    
    # Vectorized replication loop with precomputed load per replica
    load_per_replica = weight / logcnt.float()
    layer_indices = torch.arange(n, device=device)
    
    for i in range(num_redundant):
        redundant_indices = load_per_replica.argmax(dim=-1)
        
        ext_phy2log[:, i] = redundant_indices
        ext_rank[:, i] = logcnt[layer_indices, redundant_indices]
        
        # Update counts and load per replica in-place
        logcnt[layer_indices, redundant_indices] += 1
        load_per_replica[layer_indices, redundant_indices] = weight[layer_indices, redundant_indices] / logcnt[layer_indices, redundant_indices]
    
    # Combine base and extended mappings
    phy2log = torch.cat([phy2log, ext_phy2log], dim=-1)
    rank = torch.cat([rank, ext_rank], dim=-1)
    
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

    # Step 1: pack groups to nodes using vectorized balanced packing
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes with optimized replication
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical experts to GPUs with accurate load calculation
    tokens_per_phy = (tokens_per_mlog / mlogcnt.float()).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy)
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
    device = weight.device
    weight = weight.float()
    
    # Optimize group compatibility for hierarchical balancing when possible
    original_num_groups = num_groups
    if num_groups % num_nodes != 0:
        # Find largest compatible group count <= original num_groups
        new_num_groups = num_nodes * (num_groups // num_nodes)
        if new_num_groups == 0:
            new_num_groups = num_nodes
        # Ensure divisibility by logical experts while maintaining compatibility
        while new_num_groups > 0 and num_logical_experts % new_num_groups != 0:
            new_num_groups -= num_nodes
            if new_num_groups < num_nodes:
                new_num_groups = num_nodes
                break
        num_groups = new_num_groups
    
    # Fallback to global policy with optimal grouping if needed
    if num_groups % num_nodes == 0 and num_logical_experts % num_groups == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # Choose optimal global grouping for best balance
        optimal_global_groups = num_gpus if num_logical_experts % num_gpus == 0 else 1
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, optimal_global_groups, 1, num_gpus)
    
    # Calculate actual maximum replicas to minimize memory usage
    max_replicas = logcnt.max().item()
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, max_replicas),
        -1,
        dtype=torch.int64,
        device=device,
    )
    
    # Efficiently build logical to physical mapping with precomputed indices
    scatter_indices = phy2log * max_replicas + phyrank
    log2phy_view = log2phy.view(num_layers, -1)
    log2phy_view.scatter_(
        -1,
        scatter_indices,
        torch.arange(num_replicas, dtype=torch.int64,
                     device=device).expand(num_layers, -1),
    )
    
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]