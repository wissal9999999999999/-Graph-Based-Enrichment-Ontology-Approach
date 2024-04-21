import os
import sys
import json
import networkx as nx
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/src")

from transe_model import TransE, RotatE
from models import PreferentialAttachment
from utils import read_graph_from_json

def get_missing_links(preds, adj, node, node_ids):
    adj_array = nx.to_numpy_array(adj)

    # Calculate redundant and missing   
    redundant = preds * adj_array[node, :]
    missing = preds - redundant

    sorted_missing = np.argsort(-missing)

    missing_links = []
    for i in range(num_recommendations):
        score = missing[sorted_missing[i]]
        print("{} - {}, Score: {}".format(node_ids[node], node_ids[sorted_missing[i]], missing[sorted_missing[i]]))
        if score >= 0.5 and score < 1:
            missing_link = {
                "sub": node_ids[node],
                "pred": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "obj": node_ids[sorted_missing[i]],
                "score": score
            }

            missing_links.append(missing_link)
    return missing_links

if __name__ == '__main__':
    num_recommendations = 20

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to go.json relative to the script directory
    data_path = os.path.join(script_dir, "../..", "out.json")

    # Read the graph from the JSON file
    graph, node_ids, edge_ids = read_graph_from_json(data_path)
    adj = nx.Graph(graph)
    
    # Print the number of edges in your graph
    print("Number of edges in the graph: ", adj.number_of_edges())
    
    # Generate positive edges
    pos_edges = [(u, v) for u, v in adj.edges()]

    # Generate negative edges
    neg_edges = [(u, v) for u in adj.nodes() for v in adj.nodes() if not adj.has_edge(u, v)]

    # Create an instance of each model
    model1 = TransE()
    model2 = RotatE()
    model3 = PreferentialAttachment()

    # Train each model
    model1.train(adj, pos_edges, neg_edges, [], [], None, (0, 0))
    model2.train(adj, pos_edges, neg_edges, [], [], None, (0, 0))
    model3.train(adj, pos_edges, neg_edges, [], [], None, (0, 0))

    # Initialize a list to store all the missing links
    all_missing_links = []

    # Iterate over all nodes in the graph
    for node in node_ids:
        
        # Use each model to predict the missing links and add them to all_missing_links
        preds1 = model1.predict([(node, i) for i in range(adj.number_of_nodes())])
        preds2 = model2.predict([(node, i) for i in range(adj.number_of_nodes())])
        preds3 = model3.predict([(node, i) for i in range(adj.number_of_nodes())])

        missing_links1 = get_missing_links(preds1, adj, node, node_ids)
        missing_links2 = get_missing_links(preds2, adj, node, node_ids)
        missing_links3 = get_missing_links(preds3, adj, node, node_ids)

        all_missing_links.extend(missing_links1)
        all_missing_links.extend(missing_links2)
        all_missing_links.extend(missing_links3)

    # Load the existing data
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Add the missing links to the edges
    for link in all_missing_links:
        data['graphs']['edges'].append({
            "sub": link['sub'],
            "pred": link['pred'],
            "obj": link['obj']
        })

    # Write the updated data to a new file
    with open('out-with-missing-links-final.json', 'w') as f:
        json.dump(data, f, indent=3)