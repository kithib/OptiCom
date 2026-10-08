import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Cache paths to avoid redundant Dijkstra computations for shared destinations
    path_cache = {}
    for dst in dsts:
        if dst not in path_cache:
            try:
                path_cache[dst] = nx.dijkstra_path(h, src, dst, weight="cost")
            except nx.NetworkXNoPath:
                raise ValueError(f"No path found from {src} to {dst}")
        
        path = path_cache[dst]
        edge_list = []
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            edge_data = G[s][t].copy()
            edge_list.append([s, t, edge_data])
        
        # Limit alternative paths to min(num_partitions-1, 1) to strictly control computation time
        alternative_paths = []
        try:
            max_alternatives = min(num_partitions - 1, 1)
            alt_path_count = 0
            # Skip the first path which is already our main path
            path_iter = nx.shortest_simple_paths(h, src, dst, weight="cost")
            next(path_iter, None)  # Consume the first path
            
            for alt_path in path_iter:
                if alt_path_count >= max_alternatives:
                    break
                alt_edge_list = []
                for j in range(len(alt_path) - 1):
                    s_alt, t_alt = alt_path[j], alt_path[j + 1]
                    alt_edge_data = G[s_alt][t_alt].copy()
                    alt_edge_list.append([s_alt, t_alt, alt_edge_data])
                alternative_paths.append(alt_edge_list)
                alt_path_count += 1
        except nx.NetworkXNoPath:
            pass
        
        # Add direct path only if it's lower cost than current shortest path
        if src in G and dst in G[src] and G[src][dst]["cost"] is not None:
            direct_cost = G[src][dst]["cost"]
            path_cost = sum(G[path[i]][path[i+1]]["cost"] for i in range(len(path)-1))
            if direct_cost < path_cost:
                direct_edge_data = G[src][dst].copy()
                direct_path = [[src, dst, direct_edge_data]]
                # Insert at front to prioritize lower cost path
                alternative_paths.insert(0, direct_path)
        
        # Distribute partitions across main path and alternative paths to balance load
        paths_to_use = [edge_list] + alternative_paths
        paths_count = len(paths_to_use)
        
        for j in range(bc_topology.num_partitions):
            selected_path = paths_to_use[j % paths_count]
            bc_topology.set_dst_partition_paths(dst, j, selected_path.copy())

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
            self._validate_paths()
            self.set_graph()
        else:
            self.paths = {dst: {str(i): [] for i in range(num_partitions)} for dst in dsts}

    def _validate_paths(self):
        """Validate that all partitions have valid non-empty paths"""
        for dst in self.dsts:
            if dst not in self.paths:
                raise ValueError(f"Missing paths for destination {dst}")
            for partition in range(self.num_partitions):
                partition_key = str(partition)
                if partition_key not in self.paths[dst]:
                    raise ValueError(f"Missing partition {partition} for destination {dst}")
                if not self.paths[dst][partition_key]:
                    raise ValueError(f"Empty path for destination {dst}, partition {partition}")

    def get_paths(self):
        print(f"now the set path is: {self.paths}")
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions
        # Update paths to include new partitions if needed
        for dst in self.dsts:
            current_partitions = len(self.paths[dst])
            for i in range(current_partitions, num_partitions):
                # Reuse existing paths for new partitions to maintain consistency
                self.paths[dst][str(i)] = self.paths[dst][str(i % current_partitions)].copy()

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        """
        Set paths for partition = partition to reach dst
        """
        partition = str(partition)
        if not paths:
            raise ValueError(f"Paths cannot be empty for destination {dst}, partition {partition}")
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        """
        Append path for partition = partition to reach dst
        """
        partition = str(partition)
        self.paths[dst][partition].append(path)

    def set_graph(self):
        """Initialize graph structure from paths - implemented to avoid attribute errors"""
        self.graph = nx.DiGraph()
        for dst in self.paths:
            for partition in self.paths[dst]:
                for edge in self.paths[dst][partition]:
                    if len(edge) >= 2:
                        src_node, dst_node = edge[0], edge[1]
                        edge_data = edge[2] if len(edge) > 2 else {}
                        self.graph.add_edge(src_node, dst_node, **edge_data)


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