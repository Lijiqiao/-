import os
import pickle
import numpy as np
import networkx as nx
import torch
from joblib import Parallel, delayed

def compute_node_features(G, device):
    degrees = torch.tensor([degree for _, degree in G.degree()], dtype=torch.float32, device=device)
    clustering_coeffs = torch.tensor([nx.clustering(G, node) for node in G.nodes()], dtype=torch.float32, device=device)
    betweenness_centralities = torch.tensor([nx.betweenness_centrality(G)[node] for node in G.nodes()], dtype=torch.float32, device=device)
    node_features = torch.stack([degrees, clustering_coeffs, betweenness_centralities], dim=1)
    print(f"Computed node features for graph with {G.number_of_nodes()} nodes")
    return node_features.cpu().numpy()

def compute_edge_features(edges, G, device):
    degrees = {node: degree for node, degree in G.degree()}
    clustering = nx.clustering(G)
    edge_features = []
    for u, v in edges:
        deg_u, deg_v = degrees[u], degrees[v]
        clust_u, clust_v = clustering[u], clustering[v]
        avg_deg = (deg_u + deg_v) / 2
        avg_clust = (clust_u + clust_v) / 2
        edge_features.append([1, avg_deg * avg_clust])
    edge_features = torch.tensor(edge_features, dtype=torch.float32, device=device)
    print(f"Computed edge features for {len(edges)} edges")
    return edge_features.cpu().numpy()

def process_graph(graph, idx, device):
    print(f"Processing graph {idx + 1}...")
    num_nodes = graph.number_of_nodes()
    edges = np.array(list(graph.edges), dtype=np.int64)

    edge_weights = np.array([d['weight'] if 'weight' in d else 1.0 for u, v, d in graph.edges(data=True)], dtype=np.float32)

    # Compute node and edge features
    node_features = compute_node_features(graph, device)
    edge_features = compute_edge_features(edges, graph, device)

    # Convert edge list to bidirectional
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    edge_weights = np.tile(edge_weights, 2)
    edge_features = np.tile(edge_features, (2, 1))

    print(f"Processed graph {idx + 1} with {num_nodes} nodes and {edges.shape[0]} edges")
    print(f"  Node features shape: {node_features.shape}")
    print(f"  Edge features shape: {edge_features.shape}")
    return num_nodes, edges.T, edge_weights, node_features, edge_features

def save_graph_labels(data_file, label_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(data_file, "rb") as f:
        graphs = pickle.load(f)

    print(f'Starting to process {len(graphs)} graphs.')

    results = Parallel(n_jobs=-1)(delayed(process_graph)(graph, idx, device) for idx, graph in enumerate(graphs))

    with open(label_file, "w") as f:
        for idx, result in enumerate(results):
            num_nodes, edges, edge_weights, node_features, edge_features = result
            # Save num_nodes, edge_index, edge_weights, node_features, and edge_features
            num_nodes_str = str(num_nodes)
            edge_index_str = edges.flatten().tolist()
            edge_weights_str = edge_weights.tolist()
            node_features_str = node_features.flatten().tolist()
            edge_features_str = edge_features.flatten().tolist()

            f.write(f"{num_nodes_str}\n")
            f.write(" ".join(map(str, edge_index_str)) + "\n")
            f.write(" ".join(map(str, edge_weights_str)) + "\n")
            f.write(" ".join(map(str, node_features_str)) + "\n")
            f.write(" ".join(map(str, edge_features_str)) + "\n")

            print(f'Processed and saved graph {idx + 1}/{len(graphs)}')

    print(f'Labels saved to {label_file}')

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    data_file = "data/list_0313.pkl"  # Modify with your actual data file path
    label_file = data_file.replace(".pkl", ".txt")
    
    save_graph_labels(data_file, label_file)
    print(f"Labels saved to {label_file}")