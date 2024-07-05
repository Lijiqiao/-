import numpy as np
import torch
from torch_geometric.data import Data as GraphData
import pickle
import networkx as nx

class TopologyOptimizationDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, sparse_factor=-1):
        self.data_file = data_file
        self.sparse_factor = sparse_factor
        
        with open(data_file, "rb") as f:
            self.graphs = pickle.load(f)
        
        print(f'Loaded "{data_file}" with {len(self.graphs)} examples')

        # 打印第一个图的内容以调试
        if len(self.graphs) > 0:
            print("First graph data:", self.graphs[0])

    def __len__(self):
        return len(self.graphs)

    def get_example(self, idx):
        # 选择样本
        graph = self.graphs[idx]

        # 确保这是一个 networkx 图
        assert isinstance(graph, nx.Graph), "Graph is not a networkx object"

        # 提取节点特征和边
        nodes = list(graph.nodes(data=True))
        edges = list(graph.edges(data=True))
        
        # # 打印每个节点的结构以调试
        # for i, data in graph.nodes(data=True):
        #     print(f"Node {i} data: {data}")

        # 提取坐标
        loc = np.array([data['loc'] for _, data in graph.nodes(data=True)])
        node_features = np.array([data['loc'] for _, data in graph.nodes(data=True)])
        edge_indices = [(edge[0], edge[1]) for edge in edges]

        return node_features, edge_indices, loc

    def __getitem__(self, idx):
        nodes, edges, loc = self.get_example(idx)
        if self.sparse_factor <= 0:
            # 返回密集连接图
            adj_matrix = np.zeros((len(nodes), len(nodes)))
            for edge in edges:
                adj_matrix[edge[0], edge[1]] = 1
                adj_matrix[edge[1], edge[0]] = 1  # 无向图
            return (
                torch.LongTensor(np.array([idx], dtype=np.int64)),
                torch.from_numpy(nodes).float(),
                torch.from_numpy(adj_matrix).float(),
                torch.from_numpy(np.array(edges)).long(),
                torch.from_numpy(loc).float()
            )
        else:
            # 返回稀疏图，使用节点的'neighbor'属性
            edge_index_0 = []
            edge_index_1 = []
            
            for node, data in enumerate(nodes):
                neighbors = data[1]['neighbor']
                for neighbor in neighbors:
                    edge_index_0.append(node)
                    edge_index_1.append(neighbor)
            
            edge_index_0 = torch.tensor(edge_index_0, dtype=torch.long)
            edge_index_1 = torch.tensor(edge_index_1, dtype=torch.long)
            edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

            # 初始化边属性
            edge_attr = torch.zeros(edge_index.shape[1], dtype=torch.float32)

            # 将 edges 转换为 set 以便快速查找
            edge_set = set((edge[0], edge[1]) for edge in edges)
            
            # 设置边属性
            for i in range(edge_index.shape[1]):
                if (edge_index[0, i].item(), edge_index[1, i].item()) in edge_set or (edge_index[1, i].item(), edge_index[0, i].item()) in edge_set:
                    edge_attr[i] = 1.0

            graph_data = GraphData(x=torch.from_numpy(nodes).float(),
                                   edge_index=edge_index,
                                   edge_attr=edge_attr)

            point_indicator = np.array([len(nodes)], dtype=np.int64)
            edge_indicator = np.array([edge_index.shape[1]], dtype=np.int64)
            return (
                torch.LongTensor(np.array([idx], dtype=np.int64)),
                graph_data,
                torch.from_numpy(point_indicator).long(),
                torch.from_numpy(edge_indicator).long(),
                torch.from_numpy(np.array(edges)).long(),
                torch.from_numpy(loc).float()
            )