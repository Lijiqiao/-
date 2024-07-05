import os
import warnings
from multiprocessing import Pool

import numpy as np
import scipy.sparse
import scipy.spatial
import torch
import networkx as nx
from .cython_merge import cython_merge
from .cython_merge.cython_merge import merge_cython

def calculate_robustness(G, node):
    sum_largest_cc = 0
    sorted_degree_descending = sorted(dict(G.degree()).items(), key=lambda item: item[1], reverse=True)
    
    while len(G.edges()) > 0:
        G.remove_node(sorted_degree_descending[0][0])
        largest_cc = max(nx.connected_components(G), key=len)
        sum_largest_cc += len(largest_cc)
        sorted_degree_descending = sorted(dict(G.degree()).items(), key=lambda item: item[1], reverse=True)

    sum_largest_cc += len(G.nodes())
    R = float((sum_largest_cc - 1)) / float((node * (node - 1)))
    return R

def batched_two_opt_robustness(points, adj_matrix, max_iterations=1000, device="cpu"):
    iterator = 0
    adj_matrix = adj_matrix.copy()
    with torch.inference_mode():
        cuda_points = torch.from_numpy(points).to(device)
        cuda_adj_matrix = torch.from_numpy(adj_matrix).to(device)
        batch_size = cuda_adj_matrix.shape[0]
        min_change = -1.0
        while min_change < 0.0:
            dists = torch.cdist(cuda_points, cuda_points, p=2)
            i, j = torch.triu_indices(dists.shape[0], dists.shape[1], offset=1)

            current_dists = dists[adj_matrix[i, j].bool()]
            new_dists = dists[adj_matrix[i, j].bool() == False]

            change = new_dists.sum() - current_dists.sum()
            valid_change = torch.triu(change, diagonal=2)

            min_change = torch.min(valid_change)
            flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
            min_i = torch.div(flatten_argmin_index, len(points), rounding_mode='floor')
            min_j = torch.remainder(flatten_argmin_index, len(points))

            if min_change < -1e-6:
                for i in range(batch_size):
                    cuda_adj_matrix[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(cuda_adj_matrix[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
                iterator += 1
            else:
                break

            if iterator >= max_iterations:
                break
        adj_matrix = cuda_adj_matrix.cpu().numpy()
    return adj_matrix, iterator

def numpy_merge_robustness(points, adj_mat, total_edges):
    dists = np.linalg.norm(points[:, None] - points, axis=-1)
    G = nx.from_numpy_matrix(adj_mat)
    components = np.zeros((adj_mat.shape[0], 2)).astype(int)
    components[:] = np.arange(adj_mat.shape[0])[..., None]
    real_adj_mat = np.zeros_like(adj_mat)
    merge_iterations = 0
    for edge in (-adj_mat / dists).flatten().argsort():
        merge_iterations += 1
        a, b = edge // adj_mat.shape[0], edge % adj_mat.shape[0]
        if not (a in components and b in components):
            continue
        ca = np.nonzero((components == a).sum(1))[0][0]
        cb = np.nonzero((components == b).sum(1))[0][0]
        if ca == cb:
            continue
        cca = sorted(components[ca], key=lambda x: x == a)
        ccb = sorted(components[cb], key=lambda x: x == b)
        newc = np.array([[cca[0], ccb[0]]])
        m, M = min(ca, cb), max(ca, cb)
        real_adj_mat[a, b] = 1
        components = np.concatenate([components[:m], components[m + 1:M], components[M + 1:], newc], 0)
        if len(components) == 1:
            break
    real_adj_mat[components[0, 1], components[0, 0]] = 1
    real_adj_mat += real_adj_mat.T
    
    # 保证边的数量不变
    while np.sum(real_adj_mat) // 2 > total_edges:
        edges = np.array(np.nonzero(real_adj_mat))
        edge_to_remove = edges[:, np.random.randint(edges.shape[1])]
        real_adj_mat[edge_to_remove[0], edge_to_remove[1]] = 0
        real_adj_mat[edge_to_remove[1], edge_to_remove[0]] = 0
    
    return real_adj_mat, merge_iterations

def cython_merge_robustness(points, adj_mat, total_edges):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        real_adj_mat, merge_iterations = merge_cython(points.astype("double"), adj_mat.astype("double"))
        real_adj_mat = np.asarray(real_adj_mat)
    
    # 保证边的数量不变
    while np.sum(real_adj_mat) // 2 > total_edges:
        edges = np.array(np.nonzero(real_adj_mat))
        edge_to_remove = edges[:, np.random.randint(edges.shape[1])]
        real_adj_mat[edge_to_remove[0], edge_to_remove[1]] = 0
        real_adj_mat[edge_to_remove[1], edge_to_remove[0]] = 0
    
    return real_adj_mat, merge_iterations

def merge_robustness(adj_mat, np_points, edge_index_np, sparse_graph=False, parallel_sampling=1):
    """
    Merge edges to form a robust graph with maximum robustness R.
    """
    splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)
    total_edges = np.sum(splitted_adj_mat[0]) // 2

    if not sparse_graph:
        splitted_adj_mat = [
            adj_mat[0] + adj_mat[0].T for adj_mat in splitted_adj_mat
        ]
    else:
        splitted_adj_mat = [
            scipy.sparse.coo_matrix(
                (adj_mat, (edge_index_np[0], edge_index_np[1])),
            ).toarray() + scipy.sparse.coo_matrix(
                (adj_mat, (edge_index_np[1], edge_index_np[0])),
            ).toarray() for adj_mat in splitted_adj_mat
        ]

    splitted_points = [
        np_points for _ in range(parallel_sampling)
    ]

    if np_points.shape[0] > 1000 and parallel_sampling > 1:
        with Pool(parallel_sampling) as p:
            results = p.starmap(
                cython_merge_robustness,
                zip(splitted_points, splitted_adj_mat, [total_edges] * parallel_sampling),
            )
    else:
        results = [
            cython_merge_robustness(_np_points, _adj_mat, total_edges) for _np_points, _adj_mat in zip(splitted_points, splitted_adj_mat)
        ]

    splitted_real_adj_mat, splitted_merge_iterations = zip(*results)

    robustness_values = []
    total_lengths = []
    for i in range(parallel_sampling):
        G = nx.from_numpy_matrix(splitted_real_adj_mat[i])
        R = calculate_robustness(G, np_points.shape[0])
        robustness_values.append(R)
        total_length = sum([nx.linalg.graphmatrix.adjacency_matrix(G).toarray()[edge[0], edge[1]] for edge in G.edges()])
        total_lengths.append(total_length)

    merge_iterations = np.mean(splitted_merge_iterations)
    
    # 优先选择 R 值最大化的图，在 R 值相同的情况下选择边长最小的图
    best_index = np.argmax(robustness_values)
    best_adj_mat = splitted_real_adj_mat[best_index]
    
    for i in range(parallel_sampling):
        if robustness_values[i] == robustness_values[best_index] and total_lengths[i] < total_lengths[best_index]:
            best_index = i
            best_adj_mat = splitted_real_adj_mat[best_index]
    
    return best_adj_mat, merge_iterations, max(robustness_values)

class TOEvaluator(object):
    def __init__(self, points):
        self.dist_mat = scipy.spatial.distance_matrix(points, points)

    def evaluate(self, adj_mat):
        G = nx.from_numpy_matrix(adj_mat)
        total_length = np.sum([self.dist_mat[edge[0], edge[1]] for edge in G.edges()])
        return total_length