import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Precompute shared paths to avoid redundant calculations
    shared_paths = {}
    for dst in dsts:
        try:
            path = nx.dijkstra_path(h, src, dst, weight="cost")
            shared_paths[dst] = path
        except nx.NetworkXNoPath:
            # Fallback to direct edge if available, else skip
            if src in h and dst in h[src]:
                shared_paths[dst] = [src, dst]
            else:
                continue

    # Distribute partitions across paths to balance load
    for dst in dsts:
        if dst not in shared_paths:
            continue
        path = shared_paths[dst]
        path_edges = []
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            path_edges.append([s, t, G[s][t].copy()])
        
        # Assign full path to each partition initially
        for j in range(bc_topology.num_partitions):
            bc_topology.set_dst_partition_paths(dst, j, path_edges.copy())
        
        # Optimize: Try to find alternative paths to balance load across edges
        alternative_paths = []
        try:
            # Get k-shortest paths to find lower-cost or less congested alternatives
            for alt_path in nx.shortest_simple_paths(h, src, dst, weight="cost"):
                alternative_paths.append(alt_path)
                if len(alternative_paths) >= min(num_partitions, 3):
                    break
        except nx.NetworkXNoPath:
            pass
        
        # Distribute partitions across multiple paths to balance load
        if len(alternative_paths) > 1:
            paths_to_use = alternative_paths[:min(num_partitions, len(alternative_paths))]
            partitions_per_path = num_partitions // len(paths_to_use)
            remaining_partitions = num_partitions % len(paths_to_use)
            
            assigned_partitions = 0
            for path_idx, alt_path in enumerate(paths_to_use):
                alt_path_edges = []
                for i in range(len(alt_path) - 1):
                    s, t = alt_path[i], alt_path[i + 1]
                    alt_path_edges.append([s, t, G[s][t].copy()])
                
                # Calculate how many partitions to assign to this path
                assign_count = partitions_per_path + (1 if path_idx < remaining_partitions else 0)
                
                # Assign partitions to this path
                for j in range(assigned_partitions, assigned_partitions + assign_count):
                    if j < bc_topology.num_partitions:
                        bc_topology.set_dst_partition_paths(dst, j, alt_path_edges.copy())
                assigned_partitions += assign_count

    # Optimize by finding shared subpaths between destinations to reduce redundant transfers
    if len(dsts) > 1 and all(dst in shared_paths for dst in dsts):
        # Find longest common prefix across all destination paths
        paths_list = [shared_paths[dst] for dst in dsts]
        min_path_length = min(len(p) for p in paths_list)
        common_prefix_length = 0
        
        while common_prefix_length < min_path_length:
            current_node = paths_list[0][common_prefix_length]
            if all(p[common_prefix_length] == current_node for p in paths_list):
                common_prefix_length += 1
            else:
                break
        
        # If there's a meaningful shared prefix (longer than just source)
        if common_prefix_length > 1:
            shared_edges = []
            for i in range(common_prefix_length - 1):
                s = paths_list[0][i]
                t = paths_list[0][i + 1]
                shared_edges.append([s, t, G[s][t].copy()])
            
            # Update each destination's paths to use shared prefix + unique suffix
            for dst in dsts:
                path = shared_paths[dst]
                unique_edges = []
                for i in range(common_prefix_length - 1, len(path) - 1):
                    s = path[i]
                    t = path[i + 1]
                    unique_edges.append([s, t, G[s][t].copy()])
                
                # Validate the combined path reaches the destination
                combined_path_nodes = [src]
                for edge in shared_edges + unique_edges:
                    combined_path_nodes.append(edge[1])
                if combined_path_nodes[-1] == dst:
                    for j in range(bc_topology.num_partitions):
                        bc_topology.set_dst_partition_paths(dst, j, shared_edges + unique_edges)

    # Add edge load balancing across partitions to avoid congestion
    for dst in dsts:
        if dst not in shared_paths:
            continue
        all_paths = []
        for j in range(bc_topology.num_partitions):
            partition_path = bc_topology.paths[dst][str(j)]
            if partition_path:
                all_paths.append(partition_path)
        
        if not all_paths:
            continue
        
        # Calculate edge usage counts
        edge_usage = {}
        for path in all_paths:
            for edge in path:
                edge_key = (edge[0], edge[1])
                edge_usage[edge_key] = edge_usage.get(edge_key, 0) + 1
        
        # Find most congested edges
        max_usage = max(edge_usage.values()) if edge_usage else 0
        congested_edges = [k for k, v in edge_usage.items() if v == max_usage]
        
        # For congested edges, try to find alternative paths for some partitions
        for edge in congested_edges:
            src_edge, dst_edge = edge
            # Find partitions using this edge
            partitions_to_reassign = []
            for j in range(bc_topology.num_partitions):
                partition_path = bc_topology.paths[dst][str(j)]
                if any((e[0] == src_edge and e[1] == dst_edge) for e in partition_path):
                    partitions_to_reassign.append(j)
            
            if not partitions_to_reassign:
                continue
            
            # Try to find an alternative path that avoids this congested edge
            try:
                # Create a temporary graph without the congested edge
                temp_h = h.copy()
                temp_h.remove_edge(src_edge, dst_edge)
                alt_path = nx.dijkstra_path(temp_h, src, dst, weight="cost")
                alt_path_edges = []
                for i in range(len(alt_path) - 1):
                    s, t = alt_path[i], alt_path[i + 1]
                    alt_path_edges.append([s, t, G[s][t].copy()])
                
                # Reassign half of the partitions to the alternative path
                reassign_count = len(partitions_to_reassign) // 2
                for j in partitions_to_reassign[:reassign_count]:
                    bc_topology.set_dst_partition_paths(dst, j, alt_path_edges.copy())
            except nx.NetworkXNoPath:
                continue

    return bc_topology


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, SingleDstPath] = None):
        self.src = src  # single str
        self.dsts = dsts  # list of strs
        self.num_partitions = num_partitions

        # dict(dst) --> dict(partition) --> list(nx.edges)
        # example: {dst1: {partition1: [src->node1, node1->dst1], partition 2: [src->dst1]}}
        if paths is not None:
            self.paths = paths
            self.set_graph()
        else:
            self.paths = {dst: {str(i): [] for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        print(f"now the set path is: {self.paths}")
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        """
        Set paths for partition = partition to reach dst
        """
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        """
        Append path for partition = partition to reach dst
        """
        partition = str(partition)
        self.paths[dst][partition].append(path)

    def set_graph(self):
        """Placeholder for graph setup if needed"""
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
        G.add_edge(
            row["src_region"], 
            row["dst_region"], 
            cost=None, 
            throughput=num_vms * row["throughput_sent"] / 1e9,
            flow=0.0
        )

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