import networkx as nx
import os
import pandas as pd
from typing import Dict, List, Optional


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Precompute shared paths to reduce redundant transfers
    shared_paths = {}
    for dst in dsts:
        try:
            path = nx.dijkstra_path(h, src, dst, weight="cost")
            shared_paths[dst] = path
        except nx.NetworkXNoPath:
            continue

    # Build a shared multicast tree for initial segments
    if dsts:
        # Find common subpaths across destinations
        common_prefix = _find_common_prefix(list(shared_paths.values()))
        if len(common_prefix) > 1:
            # Add common path to all partitions
            for i in range(len(common_prefix) - 1):
                s, t = common_prefix[i], common_prefix[i + 1]
                edge_data = G[s][t].copy()
                for part in range(num_partitions):
                    bc_topology.append_dst_partition_path(dst, part, [s, t, edge_data])

    # Add destination-specific paths
    for dst, path in shared_paths.items():
        # Skip common prefix if already added
        start_idx = len(common_prefix) - 1 if 'common_prefix' in locals() else 0
        for i in range(start_idx, len(path) - 1):
            s, t = path[i], path[i + 1]
            edge_data = G[s][t].copy()
            for part in range(num_partitions):
                bc_topology.append_dst_partition_path(dst, part, [s, t, edge_data])

    # Balance load across partitions by distributing paths
    _balance_partition_load(bc_topology, G)

    return bc_topology


def _find_common_prefix(paths: List[List[str]]) -> List[str]:
    """Find the longest common prefix across multiple paths"""
    if not paths:
        return []
    
    prefix = paths[0].copy()
    for path in paths[1:]:
        min_len = min(len(prefix), len(path))
        while min_len > 0 and prefix[:min_len] != path[:min_len]:
            min_len -= 1
        prefix = prefix[:min_len]
        if not prefix:
            break
    return prefix


def _balance_partition_load(bc_topology: 'BroadCastTopology', G: nx.DiGraph):
    """Balance network load across partitions by adjusting path assignments"""
    edge_load = {}
    # Calculate current load per edge
    for dst in bc_topology.dsts:
        for part_id, paths in bc_topology.paths[dst].items():
            if paths is None:
                continue
            for edge in paths:
                s, t = edge[0], edge[1]
                key = (s, t)
                edge_load[key] = edge_load.get(key, 0) + 1

    # Redistribute paths to balance load
    for dst in bc_topology.dsts:
        for part_id, paths in bc_topology.paths[dst].items():
            if paths is None or len(paths) == 0:
                continue
            
            # Find alternative paths for high-load edges
            new_paths = []
            for edge in paths:
                s, t = edge[0], edge[1]
                key = (s, t)
                if edge_load.get(key, 0) > bc_topology.num_partitions * 1.5:
                    # Find alternative edge with lower cost and sufficient capacity
                    try:
                        alt_path = nx.shortest_path(G, s, t, weight=lambda u, v, d: d['cost'] + (1/d['throughput'] if d['throughput'] > 0 else 1e6))
                        if len(alt_path) == 2:
                            alt_s, alt_t = alt_path
                            alt_edge_data = G[alt_s][alt_t].copy()
                            new_paths.append([alt_s, alt_t, alt_edge_data])
                            edge_load[(alt_s, alt_t)] = edge_load.get((alt_s, alt_t), 0) + 1
                            edge_load[key] -= 1
                        else:
                            new_paths.append(edge)
                    except nx.NetworkXNoPath:
                        new_paths.append(edge)
                else:
                    new_paths.append(edge)
            
            bc_topology.set_dst_partition_paths(dst, int(part_id), new_paths)


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Optional[Dict[str, SingleDstPath]] = None):
        self.src = src  # single str
        self.dsts = dsts  # list of strs
        self.num_partitions = num_partitions

        # dict(dst) --> dict(partition) --> list(nx.edges)
        # example: {dst1: {partition1: [src->node1, node1->dst1], partition 2: [src->dst1]}}
        if paths is not None:
            self.paths = paths
        else:
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions
        # Update paths structure if increasing partitions
        current_parts = len(next(iter(self.paths.values()))) if self.paths else 0
        if num_partitions > current_parts and self.paths:
            for dst in self.paths:
                for i in range(current_parts, num_partitions):
                    self.paths[dst][str(i)] = None

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        """
        Set paths for partition = partition to reach dst
        """
        partition = str(partition)
        if dst in self.paths:
            self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        """
        Append path for partition = partition to reach dst
        """
        partition = str(partition)
        if dst not in self.paths:
            self.paths[dst] = {str(i): None for i in range(self.num_partitions)}
        
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)
    
    def set_graph(self):
        """Placeholder for backward compatibility"""
        pass


def make_nx_graph(cost_path=None, throughput_path=None, num_vms=1):
    """
    Default graph with capacity constraints and cost info
    nodes: regions, edges: links
    per edge:
        throughput: max tput achievable (gbps)
        cost: $/GB
        flow: actual flow (gbps), must be < throughput, default = 0
    """
    # Use relative path from this file's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if cost_path is None:
        cost = pd.read_csv(os.path.join(current_dir, "profiles/cost.csv"))
    else:
        cost = pd.read_csv(cost_path)

    if throughput_path is None:
        throughput = pd.read_csv(os.path.join(current_dir, "profiles/throughput.csv"))
    else:
        throughput = pd.read_csv(throughput_path)

    G = nx.DiGraph()
    for _, row in throughput.iterrows():
        if row["src_region"] == row["dst_region"]:
            continue
        G.add_edge(row["src_region"], row["dst_region"], 
                   cost=None, 
                   throughput=num_vms * row["throughput_sent"] / 1e9,
                   flow=0.0)

    for _, row in cost.iterrows():
        if row["src"] in G and row["dest"] in G[row["src"]]:
            G[row["src"]][row["dest"]]["cost"] = row["cost"]

    # some pairs not in the cost grid
    no_cost_pairs = []
    for edge in G.edges.data():
        src, dst = edge[0], edge[1]
        if edge[-1]["cost"] is None:
            no_cost_pairs.append((src, dst))
    if no_cost_pairs:
        print("Unable to get costs for: ", no_cost_pairs)

    return G


# Helper functions that won't be evolved
def create_broadcast_topology(src: str, dsts: List[str], num_partitions: int = 4):
    """Create a broadcast topology instance"""
    return BroadCastTopology(src, dsts, num_partitions)

def run_search_algorithm(src: str, dsts: List[str], G, num_partitions: int):
    """Run the search algorithm and return the topology"""
    return search_algorithm(src, dsts, G, num_partitions)