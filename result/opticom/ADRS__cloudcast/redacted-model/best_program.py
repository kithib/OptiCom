# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Step 1: Compute incumbent shortest paths for all destinations
    incumbent_paths = {}
    total_incumbent_cost = 0.0
    for dst in dsts:
        try:
            path = nx.dijkstra_path(h, src, dst, weight="cost")
            incumbent_paths[dst] = path
            # Calculate path cost
            path_cost = sum(h[path[i]][path[i+1]]["cost"] for i in range(len(path)-1))
            total_incumbent_cost += path_cost * num_partitions
        except nx.NetworkXNoPath:
            # Fallback to empty path (evaluator will catch as failure)
            incumbent_paths[dst] = []

    # Step 2: Identify low-cost hub nodes
    hub_candidates = []
    for node in h.nodes():
        if node == src or node in dsts:
            continue
        # Calculate hub score: sum of shortest path costs from src to hub and hub to all dsts
        try:
            src_to_hub_cost = nx.dijkstra_path_length(h, src, node, weight="cost")
            hub_to_dsts_cost = sum(
                nx.dijkstra_path_length(h, node, dst, weight="cost") 
                for dst in dsts if nx.has_path(h, node, dst)
            )
            hub_score = src_to_hub_cost + hub_to_dsts_cost
            hub_candidates.append((hub_score, node))
        except nx.NetworkXNoPath:
            continue
    
    # Sort hubs by total cost (ascending)
    hub_candidates.sort()
    # Take top 3 hubs or fewer if not available
    top_hubs = [node for (score, node) in hub_candidates[:3]]

    # Step 3: Generate candidate paths for each destination
    candidate_path_sets = {}
    for dst in dsts:
        candidates = set()
        
        # Add incumbent shortest path
        if incumbent_paths[dst]:
            candidates.add(tuple(incumbent_paths[dst]))
        
        # Add paths through top hubs
        for hub in top_hubs:
            try:
                src_to_hub = nx.dijkstra_path(h, src, hub, weight="cost")
                hub_to_dst = nx.dijkstra_path(h, hub, dst, weight="cost")
                # Combine paths (remove duplicate hub node)
                combined_path = src_to_hub + hub_to_dst[1:]
                candidates.add(tuple(combined_path))
            except nx.NetworkXNoPath:
                continue
        
        # Add alternative shortest paths (k=2 if available)
        try:
            for path in nx.shortest_simple_paths(h, src, dst, weight="cost"):
                candidates.add(tuple(path))
                if len(candidates) >= 3:
                    break
        except nx.NetworkXNoPath:
            pass
        
        candidate_path_sets[dst] = list(candidates)

    # Step 4: Select best path set with shared prefix optimization and cost balancing
    best_paths = {}
    total_candidate_cost = float('inf')
    
    # Evaluate all combinations (limited to first 2 candidates per dst for runtime)
    from itertools import product
    candidate_combinations = product(*[
        [(dst, path) for path in candidate_path_sets[dst][:2]] 
        for dst in dsts
    ])
    
    for combination in candidate_combinations:
        current_paths = {dst: path for (dst, path) in combination}
        # Calculate total cost with shared edge discount (simulate reduced redundant transfer)
        edge_usage = {}
        current_total = 0.0
        
        for dst, path in current_paths.items():
            if not path:
                current_total = float('inf')
                break
            for i in range(len(path)-1):
                edge = (path[i], path[i+1])
                edge_cost = h[edge[0]][edge[1]]["cost"]
                # Track how many destinations use this edge
                edge_usage[edge] = edge_usage.get(edge, 0) + 1
                # Apply discount for shared edges (reduce cost by 10% per additional user)
                discount = 0.1 * (edge_usage[edge] - 1)
                current_total += edge_cost * (1 - discount) * num_partitions
        
        # Update best if this combination is cheaper and valid
        if current_total < total_candidate_cost and current_total != float('inf'):
            total_candidate_cost = current_total
            best_paths = current_paths.copy()
    
    # Fallback to incumbent if no valid candidate combination found
    if not best_paths:
        best_paths = incumbent_paths

    # Step 5: Populate broadcast topology with selected paths
    for dst in dsts:
        path = best_paths.get(dst, [])
        if not path:
            # Ensure all partitions are populated even if path is invalid
            for j in range(num_partitions):
                bc_topology.set_dst_partition_paths(dst, j, [])
            continue
        
        # For each partition, use the full path (ensure no missing edges)
        edge_list = []
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            edge_list.append([s, t, G[s][t]])
        
        # Assign same path to all partitions (balanced load)
        for j in range(num_partitions):
            bc_topology.set_dst_partition_paths(dst, j, edge_list.copy())

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
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

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
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)
    
    def set_graph(self):
        """Placeholder for graph setup (preserved for API compatibility)"""
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