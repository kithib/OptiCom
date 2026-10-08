import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Precompute shortest paths once per destination
    shared_paths = {}
    for dst in dsts:
        try:
            path = nx.dijkstra_path(h, src, dst, weight="cost")
            shared_paths[dst] = path
        except nx.NetworkXNoPath:
            continue

    # Identify shared path segments to minimize redundant transfers
    segment_usage = {}
    for dst, path in shared_paths.items():
        for i in range(len(path) - 1):
            segment = (path[i], path[i+1])
            if segment not in segment_usage:
                segment_usage[segment] = []
            segment_usage[segment].append(dst)

    # Distribute partitions to balance load across segments
    for dst in dsts:
        if dst not in shared_paths:
            continue
            
        path = shared_paths[dst]
        segments = [[path[i], path[i+1], G[path[i]][path[i+1]]] for i in range(len(path)-1)]
        
        # Basic distribution: assign full path to all partitions for consistency
        for partition in range(num_partitions):
            bc_topology.set_dst_partition_paths(dst, partition, segments.copy())
        
        # Optimize: Use alternative paths for some partitions if available and beneficial
        if num_partitions > 1:
            try:
                # Use A* for faster alternative path search
                alt_path = nx.astar_path(h, src, dst, weight="cost")
                if alt_path != path:
                    alt_segments = [[alt_path[i], alt_path[i+1], G[alt_path[i]][alt_path[i+1]]] 
                                   for i in range(len(alt_path)-1)]
                    # Assign alternative path to half of the partitions
                    for partition in range(num_partitions // 2):
                        bc_topology.set_dst_partition_paths(dst, partition, alt_segments.copy())
            except nx.NetworkXNoPath:
                pass

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
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions
        # Update paths to include new partitions if needed
        for dst in self.paths:
            for i in range(num_partitions):
                if str(i) not in self.paths[dst]:
                    self.paths[dst][str(i)] = []

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
        """Build graph from paths"""
        self.graph = nx.DiGraph()
        for dst in self.paths:
            for partition in self.paths[dst]:
                edges = self.paths[dst][partition]
                if edges:
                    for s, t, data in edges:
                        if not self.graph.has_edge(s, t):
                            self.graph.add_edge(s, t, **data)


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
    cost_path = cost_path or os.path.join(current_dir, "profiles/cost.csv")
    throughput_path = throughput_path or os.path.join(current_dir, "profiles/throughput.csv")
    
    if not os.path.exists(cost_path):
        print(f"Warning: Cost file not found at {cost_path}, using empty cost data")
        cost = pd.DataFrame(columns=["src", "dest", "cost"])
    else:
        cost = pd.read_csv(cost_path)

    if not os.path.exists(throughput_path):
        print(f"Warning: Throughput file not found at {throughput_path}, using default throughput")
        throughput = pd.DataFrame(columns=["src_region", "dst_region", "throughput_sent"])
    else:
        throughput = pd.read_csv(throughput_path)

    G = nx.DiGraph()
    for _, row in throughput.iterrows():
        if pd.isna(row["src_region"]) or pd.isna(row["dst_region"]):
            continue
        if row["src_region"] == row["dst_region"]:
            continue
        try:
            tput = num_vms * float(row["throughput_sent"]) / 1e9 if not pd.isna(row["throughput_sent"]) else 1.0
            G.add_edge(row["src_region"], row["dst_region"], cost=None, throughput=tput)
        except (ValueError, TypeError):
            continue

    for _, row in cost.iterrows():
        if pd.isna(row["src"]) or pd.isna(row["dest"]):
            continue
        if row["src"] in G and row["dest"] in G[row["src"]]:
            try:
                G[row["src"]][row["dest"]]["cost"] = float(row["cost"]) if not pd.isna(row["cost"]) else 0.01
            except (ValueError, TypeError):
                G[row["src"]][row["dest"]]["cost"] = 0.01

    # some pairs not in the cost grid
    no_cost_pairs = []
    for edge in G.edges.data():
        src, dst = edge[0], edge[1]
        if edge[-1]["cost"] is None:
            no_cost_pairs.append((src, dst))
            G[src][dst]["cost"] = 0.01  # Set default cost for missing pairs
    if no_cost_pairs:
        print("Unable to get costs for: ", no_cost_pairs, " using default cost of 0.01")

    return G


# EVOLVE-BLOCK-END

# Helper functions that won't be evolved
def create_broadcast_topology(src: str, dsts: List[str], num_partitions: int = 4):
    """Create a broadcast topology instance"""
    return BroadCastTopology(src, dsts, num_partitions)

def run_search_algorithm(src: str, dsts: List[str], G, num_partitions: int):
    """Run the search algorithm and return the topology"""
    return search_algorithm(src, dsts, G, num_partitions)