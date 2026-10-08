import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Cache paths to avoid redundant Dijkstra computations
    path_cache = {}
    for dst in dsts:
        if dst not in path_cache:
            try:
                path_cache[dst] = nx.dijkstra_path(h, src, dst, weight="cost")
            except nx.NetworkXNoPath:
                raise ValueError(f"No path found from source {src} to destination {dst}")
        
        path = path_cache[dst]
        path_edges = []
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            edge_data = h[s][t].copy()
            path_edges.append([s, t, edge_data])
        
        # Distribute partitions across alternative paths if available to balance load
        primary_path = path_edges
        alternative_paths = []
        
        # Try to find alternative paths with slightly higher cost but different edges
        try:
            alt_paths = nx.shortest_simple_paths(h, src, dst, weight="cost")
            # Skip first path (primary), take next two if available
            next(alt_paths)
            for _ in range(2):
                alt_path = next(alt_paths)
                alt_edges = []
                for j in range(len(alt_path) - 1):
                    s_alt, t_alt = alt_path[j], alt_path[j + 1]
                    alt_edges.append([s_alt, t_alt, h[s_alt][t_alt].copy()])
                alternative_paths.append(alt_edges)
        except (nx.NetworkXNoPath, StopIteration):
            pass
        
        # Assign partitions to paths to balance load across edges
        paths = [primary_path] + alternative_paths
        partitions_per_path = num_partitions // len(paths)
        remaining_partitions = num_partitions % len(paths)
        
        current_partition = 0
        for idx, assigned_path in enumerate(paths):
            count = partitions_per_path + (1 if idx < remaining_partitions else 0)
            for _ in range(count):
                if current_partition >= num_partitions:
                    break
                bc_topology.set_dst_partition_paths(dst, current_partition, assigned_path.copy())
                current_partition += 1
        
        # If still have partitions left (shouldn't happen), assign to primary path
        while current_partition < num_partitions:
            bc_topology.set_dst_partition_paths(dst, current_partition, primary_path.copy())
            current_partition += 1
    
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
        # Update paths structure when changing partitions
        for dst in self.paths:
            current_partitions = list(self.paths[dst].keys())
            # Add new partitions if increasing
            for i in range(num_partitions):
                part_str = str(i)
                if part_str not in self.paths[dst]:
                    self.paths[dst][part_str] = []
            # Remove excess partitions if reducing
            for part_str in current_partitions:
                if int(part_str) >= num_partitions:
                    del self.paths[dst][part_str]

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
        if dst in self.paths and partition in self.paths[dst]:
            self.paths[dst][partition].append(path)
    
    def set_graph(self):
        """Placeholder for graph setup (matches original API)"""
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
    
    # Handle missing files gracefully
    if cost_path is None:
        cost_path = os.path.join(current_dir, "profiles/cost.csv")
    if throughput_path is None:
        throughput_path = os.path.join(current_dir, "profiles/throughput.csv")
    
    if not os.path.exists(cost_path):
        raise FileNotFoundError(f"Cost file not found at {cost_path}")
    if not os.path.exists(throughput_path):
        raise FileNotFoundError(f"Throughput file not found at {throughput_path}")
    
    cost = pd.read_csv(cost_path)
    throughput = pd.read_csv(throughput_path)

    G = nx.DiGraph()
    for _, row in throughput.iterrows():
        src_region = row["src_region"]
        dst_region = row["dst_region"]
        if src_region == dst_region:
            continue
        tput = num_vms * row["throughput_sent"] / 1e9
        G.add_edge(src_region, dst_region, cost=None, throughput=tput, flow=0.0)

    for _, row in cost.iterrows():
        src = row["src"]
        dest = row["dest"]
        if src in G and dest in G[src]:
            G[src][dest]["cost"] = row["cost"]

    # Identify missing cost pairs
    no_cost_pairs = []
    for edge in G.edges.data():
        src_edge, dst_edge, data = edge
        if data["cost"] is None:
            no_cost_pairs.append((src_edge, dst_edge))
    
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