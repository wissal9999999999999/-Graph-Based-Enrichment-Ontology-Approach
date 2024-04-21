import json
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split

def read_graph_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    graph = nx.DiGraph()
    for edge in data['graphs']['edges']:
        graph.add_edge(edge['sub'], edge['obj'], pred=edge['pred'])
    return graph, data['graphs']['nodes'], data['graphs']['edges']

def remove_edges(edges, remove_ratio=0.3):
    num_edges_to_remove = int(len(edges) * remove_ratio)
    np.random.shuffle(edges)
    removed_edges = edges[:num_edges_to_remove]
    remaining_edges = edges[num_edges_to_remove:]
    return remaining_edges, removed_edges

# Load the graph
graph, nodes, edges = read_graph_from_json('out.json')

# Remove a small percentage of edges from the graph
remaining_edges, removed_edges = remove_edges(edges, remove_ratio=0.01)
# Save the removed edges to a file
with open('removed-edges.json', 'w') as f:
    json.dump(removed_edges, f, indent=4)

# Generate a new graph without the removed edges
new_graph = {
    "graphs": {
        "nodes": nodes,
        "edges": remaining_edges
    }
}

# Save the new graph to a file
with open('out-for-validation.json', 'w') as f:
    json.dump(new_graph, f, indent=4)