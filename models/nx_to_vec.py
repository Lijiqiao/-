import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, DataLoader

# 从gnn_encoder.py中导入所需的类和函数
from gnn_encoder import GNNEncoder, PositionEmbeddingSine, ScalarEmbeddingSine, ScalarEmbeddingSine1D, GNNLayer


# 定义用于将NetworkX图转换为PyG图数据的辅助函数
def nx_to_pyg_data(graph):
    data = from_networkx(graph)
    data.x = torch.eye(data.num_nodes)  # 简单的one-hot编码作为示例
    return data


# 自定义Dataset类
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        data = nx_to_pyg_data(graph)
        return data


# 自定义DataLoader类
def create_data_loader(graphs, batch_size=1):
    dataset = GraphDataset(graphs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 加载图数据
graph_list = [nx.gnm_random_graph(200, 400) for _ in range(10)]  # 生成10个随机图作为示例

# 创建DataLoader
data_loader = create_data_loader(graph_list, batch_size=1)

# 定义GNNEncoder模型
n_layers = 4
hidden_dim = 128
out_channels = 1

model = GNNEncoder(
    n_layers=n_layers,
    hidden_dim=hidden_dim,
    out_channels=out_channels,
    aggregation="sum",
    norm="layer",
    learn_norm=True,
    track_norm=False,
    gated=True,
    sparse=False,
    use_activation_checkpoint=False,
    node_feature_only=False
)

# 将模型移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 进行特征编码
for batch in data_loader:
    data = batch.to(device)
    node_features = data.x
    edge_index = data.edge_index
    adjacency_matrix = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.float32)
    adjacency_matrix[edge_index[0], edge_index[1]] = 1.0

    # 假设时间步长和图数据一起提供，这里我们用随机数作为示例
    timesteps = torch.randint(0, 100, (data.num_nodes,)).to(device)

    # 执行前向传播
    encoded_features = model(node_features, timesteps, graph=adjacency_matrix, edge_index=edge_index)

    # 这里你可以根据你的需求对编码后的特征进行进一步处理或优化
    print(encoded_features)
