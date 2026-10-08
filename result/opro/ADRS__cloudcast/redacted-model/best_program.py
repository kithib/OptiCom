# EVOLVE-BLOCK-START
import networkx as nx
import os
import pandas as pd
from typing import Dict, List, Optional


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Precompute shortest paths with fallback to ensure valid routes
    all_paths = {}
    for dst in dsts:
        try:
            # Limit to max 3 paths to avoid excessive computation
            paths = list(nx.all_shortest_paths(h, src, dst, weight="cost"))[:3]
            if not paths:
                raise nx.NetworkXNoPath
            all_paths[dst] = paths
        except nx.NetworkXNoPath:
            # Fallback to single dijkstra path as reliable default
            path = nx.dijkstra_path(h, src, dst, weight="cost")
            all_paths[dst] = [path]

    # Distribute partitions evenly across paths with strict validation
    for dst in dsts:
        paths = all_paths[dst]
        path_count = len(paths)
        
        # Ensure every partition gets a valid path
        for partition_idx in range(num_partitions):
            # Cycle through paths to distribute load
            path = paths[partition_idx % path_count]
            edges = []
            prev_node = src
            
            # Build valid edge sequence with continuity checks
            for i in range(len(path) - 1):
                s, t = path[i], path[i + 1]
                if s != prev_node:
                    raise ValueError(f"Path discontinuity at partition {partition_idx} for {dst}: expected {prev_node}, got {s}")
                edges.append([s, t, G[s][t].copy()])
                prev_node = t
            
            # Verify path reaches destination
            if prev_node != dst:
                raise ValueError(f"Path for {dst} partition {partition_idx} ends at {prev_node}, expected {dst}")
            
            bc_topology.set_dst_partition_paths(dst, partition_idx, edges)

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
            self.validate_topology()
        else:
            self.paths = {dst: {str(i): [] for i in range(num_partitions)} for dst in dsts}

    def validate_topology(self):
        """Validate all paths are complete and continuous"""
        for dst, partitions in self.paths.items():
            for part_id, edges in partitions.items():
                if not edges:
                    raise ValueError(f"Empty partition {part_id} for destination {dst}")
                
                current_node = self.src
                for edge in edges:
                    s, t, _ = edge
                    if s != current_node:
                        raise ValueError(f"Path discontinuity in {dst}:{part_id} - expected {current_node}, got {s}")
                    current_node = t
                
                if current_node != dst:
                    raise ValueError(f"Path in {dst}:{part_id} ends at {current_node}, expected {dst}")

    def get_paths(self):
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions
        # Initialize new partitions if needed
        for dst in self.dsts:
            for i in range(self.num_partitions):
                if str(i) not in self.paths[dst]:
                    self.paths[dst][str(i)] = []

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
        G.add_edge(row["src_region"], row["dst_region"], cost=None, throughput=num_vms * row["throughput_sent"] / 1e9)

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


# EVOLVE-BLOCK-END

# Helper functions that won't be evolved
def create_broadcast_topology(src: str, dsts: List[str], num_partitions: int = 4):
    """Create a broadcast topology instance"""
    return BroadCastTopology(src, dsts, num_partitions)

def run_search_algorithm(src: str, dsts: List[str], G, num_partitions: int):
    """Run the search algorithm and return the topology"""
    return search_algorithm(src, dsts, G, num_partitions)