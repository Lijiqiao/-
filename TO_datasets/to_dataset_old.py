import torch
import numpy as np
from torch_geometric.data import Data as GraphData
import pickle
import os

# 设置 CUDA_VISIBLE_DEVICES 环境变量，只使用卡号为3的显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class TopologyOptimizationDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, label_file=None):
        self.data_file = data_file
        self.label_file = label_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(data_file, "rb") as f:
            self.graphs = pickle.load(f)

        self.labels = None
        if label_file is not None:
            with open(label_file, "r") as f:
                self.labels = f.readlines()

        print(f'Loaded "{data_file}" with {len(self.graphs)} examples')

        # Read node and edge features from label file
        self.node_features = []
        self.edge_features = []
        self.num_nodes_list = []
        self.edges_list = []
        self.edge_weights_list = []

        self.read_labels()

        if len(self.graphs) > 0:
            # Use the first graph as standard
            self.standard_num_nodes = self.num_nodes_list[0]
            self.standard_num_edges = self.edges_list[0].shape[1]

            # Filter graphs
            self.filter_graphs()

        print(f'Loaded {len(self.node_features)} node_features')
        print(f'Loaded {len(self.edge_features)} edge_features')

    def read_labels(self):
        idx = 0
        while idx < len(self.labels):
            num_nodes = int(self.labels[idx].strip())
            self.num_nodes_list.append(num_nodes)
            idx += 1
            
            edges = list(map(int, self.labels[idx].strip().split()))
            edges = np.array(edges).reshape(2, -1)
            self.edges_list.append(edges)
            idx += 1
            
            edge_weights = list(map(float, self.labels[idx].strip().split()))
            self.edge_weights_list.append(np.array(edge_weights, dtype=np.float32))
            idx += 1
            
            node_features = list(map(float, self.labels[idx].strip().split()))
            node_features = np.array(node_features).reshape(num_nodes, -1)
            self.node_features.append(node_features)
            idx += 1
            
            edge_features = list(map(float, self.labels[idx].strip().split()))
            edge_features = np.array(edge_features).reshape(-1, 2)
            self.edge_features.append(edge_features)
            idx += 1

    def filter_graphs(self):
        print("Filtering graphs...")
        filtered_indices = []
        for i in range(len(self.edges_list)):
            num_nodes = self.num_nodes_list[i]
            num_edges = self.edges_list[i].shape[1]
            if num_nodes == self.standard_num_nodes and num_edges == self.standard_num_edges:
                filtered_indices.append(i)

        self.num_nodes_list = [self.num_nodes_list[i] for i in filtered_indices]
        self.edges_list = [self.edges_list[i] for i in filtered_indices]
        self.edge_weights_list = [self.edge_weights_list[i] for i in filtered_indices]
        self.node_features = [self.node_features[i] for i in filtered_indices]
        self.edge_features = [self.edge_features[i] for i in filtered_indices]

    def __len__(self):
        return len(self.node_features)

    def get_example(self, idx):
        num_nodes = self.num_nodes_list[idx]
        edges = self.edges_list[idx]
        edge_weights = self.edge_weights_list[idx]

        # Get node and edge features
        node_features = self.node_features[idx]
        edge_features = self.edge_features[idx]  # 直接使用原始的边特征

        return num_nodes, edges, edge_weights, node_features, edge_features
    
    def __getitem__(self, idx):
        num_nodes, edge_index, edge_weights, node_features, edge_features = self.get_example(idx)
        graph_data = GraphData(
            x=torch.from_numpy(node_features).float(),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_features).float(),
            num_nodes=num_nodes
        )

        point_indicator = np.array([num_nodes], dtype=np.int64)
        return (
            torch.LongTensor(np.array([idx], dtype=np.int64)),
            graph_data,
            torch.from_numpy(point_indicator).long(),
        )

# Example usage:
if __name__ == "__main__":
    data_file = "data/list_0313.pkl"  # Modify with your actual data file path
    label_file = "data/list_0313.txt"  # Modify with your actual label file path
    
    dataset = TopologyOptimizationDataset(data_file, label_file)
    print(f"Dataset length: {len(dataset)}")
    for I in range(min(3, len(dataset))):  # Print first 3 examples for verification
        idx, graph_data, point_indicator = dataset[I]
        print(f"Graph {I}:")
        print(f"  Num nodes: {graph_data.num_nodes}")
        print(f"  Edge index shape: {graph_data.edge_index.shape}")
        print(f"  Node features shape: {graph_data.x.shape}")
        print(f"  Edge features shape: {graph_data.edge_attr.shape}")
        print(f"  Point indicator: {point_indicator}")