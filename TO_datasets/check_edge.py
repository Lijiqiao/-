import pickle
import numpy as np

def load_graphs(data_file):
    with open(data_file, "rb") as f:
        graphs = pickle.load(f)
    return graphs

def load_labels(label_file):
    with open(label_file, "r") as f:
        labels = f.readlines()
    return labels

def parse_labels(labels):
    idx = 0
    num_nodes_list = []
    edges_list = []
    edge_weights_list = []
    node_features_list = []
    edge_features_list = []

    while idx < len(labels):
        num_nodes = int(labels[idx].strip())
        num_nodes_list.append(num_nodes)
        idx += 1

        edges = list(map(int, labels[idx].strip().split()))
        edges = np.array(edges).reshape(2, -1)
        edges_list.append(edges)
        idx += 1

        edge_weights = list(map(float, labels[idx].strip().split()))
        edge_weights_list.append(np.array(edge_weights, dtype=np.float32))
        idx += 1

        node_features = list(map(float, labels[idx].strip().split()))
        node_features = np.array(node_features).reshape(num_nodes, -1)
        node_features_list.append(node_features)
        idx += 1

        edge_features = list(map(float, labels[idx].strip().split()))
        edge_features = np.array(edge_features).reshape(-1, 2)
        edge_features_list.append(edge_features)
        idx += 1

    return num_nodes_list, edges_list, edge_weights_list, node_features_list, edge_features_list

def check_edge_features(data_file, label_file):
    graphs = load_graphs(data_file)
    labels = load_labels(label_file)
    num_nodes_list, edges_list, edge_weights_list, node_features_list, edge_features_list = parse_labels(labels)

    all_correct = True
    for idx, (edges, edge_features) in enumerate(zip(edges_list, edge_features_list)):
        expected_num_edges = edges.shape[1]
        actual_num_edge_features = edge_features.shape[0]
        if actual_num_edge_features != 2 * expected_num_edges:
            print(f"Graph {idx + 1} has mismatched edge features. Expected {2 * expected_num_edges}, got {actual_num_edge_features}")
            all_correct = False
        else:
            print(f"Graph {idx + 1} edge features are correct.")

    if all_correct:
        print("All graphs have correct edge features.")
    else:
        print("Some graphs have mismatched edge features.")

if __name__ == "__main__":
    data_file = "data/list_0313.pkl"  # Modify with your actual data file path
    label_file = data_file.replace(".pkl", ".txt")
    
    check_edge_features(data_file, label_file)