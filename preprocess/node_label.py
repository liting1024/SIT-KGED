import math
from typing import List, Tuple
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch_sparse
from torch import Tensor
from torch_sparse import SparseTensor, matmul

import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data

from torch_geometric.utils import k_hop_subgraph as pyg_k_hop_subgraph, to_edge_index

# import torchhd

MINIMUM_SIGNATURE_DIM=64

class HoCN(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x: Tensor, edges: Tensor, adj_t: SparseTensor, node_weight: Tensor=None, adj2: SparseTensor=None):
        adj_t, new_edges, subset_nodes = subgraph(edges, adj_t, 2)
        x = x[subset_nodes]
        node_weight = node_weight[subset_nodes] if node_weight is not None else None
        subset = new_edges.view(-1) # flatten the target nodes [row, col]

        # remove values from adj_t
        # adj_t = adj_t.set_value(None)
        
        degree_one_hop = adj_t.sum(dim=1)
        degree_u = degree_one_hop[new_edges[0]]
        degree_v = degree_one_hop[new_edges[1]]


        adj_t = normalize_adj(adj_t,subset_nodes)
        subset_unique, inverse_indices = torch.unique(subset, return_inverse=True)
        one_hop_x_subgraph_nodes = matmul(adj_t, x)
        one_hop_x = one_hop_x_subgraph_nodes[subset]
        two_hop_x = matmul(adj_t[subset_unique], one_hop_x_subgraph_nodes)[inverse_indices]

        one_hop_x = one_hop_x.view(2, new_edges.size(1), -1)
        two_hop_x = two_hop_x.view(2, new_edges.size(1), -1)

        count_1_1 = dot_product(one_hop_x[0,:,:], one_hop_x[1,:,:])
        count_1_2 = dot_product(one_hop_x[0,:,:], two_hop_x[1,:,:])
        count_2_1 = dot_product(two_hop_x[0,:,:] , one_hop_x[1,:,:])
        count_2_2 = dot_product((two_hop_x[0,:,:]-degree_one_hop[new_edges[0]].view(-1,1)*x[new_edges[0]]) , (two_hop_x[1,:,:]-degree_one_hop[new_edges[1]].view(-1,1)*x[new_edges[1]]))

        count_self_1_2 = dot_product(one_hop_x[0,:,:] , two_hop_x[0,:,:])
        count_self_2_1 = dot_product(one_hop_x[1,:,:] , two_hop_x[1,:,:])

        if adj2 is None:
            return count_1_1, count_1_2, count_2_1, count_2_2, count_self_1_2, count_self_2_1, degree_u, degree_v
        else:
            raise NotImplementedError()


def normalize_adj(adj: SparseTensor, subset_nodes,eps=1e-12) -> SparseTensor:
    """
    Symmetric normalization: D^{-1/2} A D^{-1/2}
    """
    from torch_geometric.utils import normalize_edge_index

    # edge_index: [2, num_edges]
    row, col, _ = adj.coo()
    edge_index = torch.stack([row, col], dim=0)
    edges, value = normalize_edge_index(edge_index, num_nodes=subset_nodes.shape[0])

    return SparseTensor(row=edges[0], col=edges[1], value=value, sparse_sizes=adj.sizes()).coalesce()

def get_high_order_adj(adj_t: SparseTensor, hop: int=3):
    adjs = [adj_t]
    for _ in range(1, hop):
        adjs.append(matmul(adjs[-1], adj_t))
    return adjs

def subgraph(edges: Tensor, adj_t: SparseTensor, k: int=2):
    row,col = edges
    nodes = torch.cat((row,col),dim=-1)
    edge_index,_ = to_edge_index(adj_t)
    subset, new_edge_index, inv, edge_mask = pyg_k_hop_subgraph(nodes, k, edge_index=edge_index, 
                                                                num_nodes=adj_t.size(0), relabel_nodes=True)
    # subset[inv] = nodes. The new node id is based on `subset`'s order.
    # inv means the new idx (in subset) of the old nodes in `nodes`
    new_adj_t = SparseTensor(row=new_edge_index[0], col=new_edge_index[1], 
                                sparse_sizes=(subset.size(0), subset.size(0)))
    new_edges = inv.view(2,-1)
    return new_adj_t, new_edges, subset

def get_two_hop_adj(adj_t):
    # adj_t = adj_t.fill_value_(1.0) # no need to fill value because of subgraph op
    one_and_two_hop_adj = adj_t @ adj_t
    adj_t_with_self_loop = adj_t.fill_diag(1)
    two_hop_adj = spmdiff_(one_and_two_hop_adj, adj_t_with_self_loop)
    return adj_t, two_hop_adj

def dotproduct_naive(tensor1, tensor2):
    return (tensor1 * tensor2).sum(dim=-1)

def dotproduct_bmm(tensor1, tensor2):
    return torch.bmm(tensor1.unsqueeze(1), tensor2.unsqueeze(2)).view(-1)

def dotproduct_dim(tensor1, tensor2):
    return (tensor1 * tensor2)

dot_product = dotproduct_dim

def sparsesample(adj: SparseTensor, deg: int) -> SparseTensor:
    '''
    sampling elements from a adjacency matrix
    '''
    rowptr, col, _ = adj.csr()
    rowcount = adj.storage.rowcount()
    mask = rowcount > 0
    rowcount = rowcount[mask]
    rowptr = rowptr[:-1][mask]

    rand = torch.rand((rowcount.size(0), deg), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).reshape(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.reshape(-1, 1))

    samplecol = col[rand]

    samplerow = torch.arange(adj.size(0), device=adj.device())[mask]

    ret = SparseTensor(row=samplerow.reshape(-1, 1).expand(-1, deg).flatten(),
                       col=samplecol.flatten(),
                       sparse_sizes=adj.sparse_sizes()).to_device(
                           adj.device()).coalesce().fill_value_(1.0)
    #print(ret.storage.value())
    return ret


def sparsesample2(adj: SparseTensor, deg: int) -> SparseTensor:
    '''
    another implementation for sampling elements from a adjacency matrix
    '''
    rowptr, col, _ = adj.csr()
    rowcount = adj.storage.rowcount()
    mask = rowcount > deg

    rowcount = rowcount[mask]
    rowptr = rowptr[:-1][mask]

    rand = torch.rand((rowcount.size(0), deg), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).reshape(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.reshape(-1, 1))

    samplecol = col[rand].flatten()

    samplerow = torch.arange(adj.size(0), device=adj.device())[mask].reshape(
        -1, 1).expand(-1, deg).flatten()

    mask = torch.logical_not(mask)
    nosamplerow, nosamplecol = adj[mask].coo()[:2]
    nosamplerow = torch.arange(adj.size(0),
                               device=adj.device())[mask][nosamplerow]

    ret = SparseTensor(
        row=torch.cat((samplerow, nosamplerow)),
        col=torch.cat((samplecol, nosamplecol)),
        sparse_sizes=adj.sparse_sizes()).to_device(
            adj.device()).fill_value_(1.0).coalesce()  #.fill_value_(1)
    #assert (ret.sum(dim=-1) == torch.clip(adj.sum(dim=-1), 0, deg)).all()
    return ret


def sparsesample_reweight(adj: SparseTensor, deg: int) -> SparseTensor:
    '''
    another implementation for sampling elements from a adjacency matrix. It will also scale the sampled elements.
    
    '''
    rowptr, col, _ = adj.csr()
    rowcount = adj.storage.rowcount()
    mask = rowcount > deg

    rowcount = rowcount[mask]
    rowptr = rowptr[:-1][mask]

    rand = torch.rand((rowcount.size(0), deg), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).reshape(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.reshape(-1, 1))

    samplecol = col[rand].flatten()

    samplerow = torch.arange(adj.size(0), device=adj.device())[mask].reshape(
        -1, 1).expand(-1, deg).flatten()
    samplevalue = (rowcount * (1/deg)).reshape(-1, 1).expand(-1, deg).flatten()

    mask = torch.logical_not(mask)
    nosamplerow, nosamplecol = adj[mask].coo()[:2]
    nosamplerow = torch.arange(adj.size(0),
                               device=adj.device())[mask][nosamplerow]

    ret = SparseTensor(row=torch.cat((samplerow, nosamplerow)),
                       col=torch.cat((samplecol, nosamplecol)),
                       value=torch.cat((samplevalue,
                                        torch.ones_like(nosamplerow))),
                       sparse_sizes=adj.sparse_sizes()).to_device(
                           adj.device()).coalesce()  #.fill_value_(1)
    #assert (ret.sum(dim=-1) == torch.clip(adj.sum(dim=-1), 0, deg)).all()
    return ret


def elem2spm(element: Tensor, sizes: List[int], val: Tensor=None) -> SparseTensor:
    # Convert adjacency matrix to a 1-d vector
    col = torch.bitwise_and(element, 0xffffffff)
    row = torch.bitwise_right_shift(element, 32)
    if val is None:
        sp_tensor =  SparseTensor(row=row, col=col, sparse_sizes=sizes).to_device(
            element.device).fill_value_(1.0)
    else:
        sp_tensor =  SparseTensor(row=row, col=col, value=val, sparse_sizes=sizes).to_device(
            element.device)
    return sp_tensor


def spm2elem(spm: SparseTensor) -> Tensor:
    # Convert 1-d vector to an adjacency matrix
    sizes = spm.sizes()
    elem = torch.bitwise_left_shift(spm.storage.row(),
                                    32).add_(spm.storage.col())
    val = spm.storage.value()
    #elem = spm.storage.row()*sizes[-1] + spm.storage.col()
    #assert torch.all(torch.diff(elem) > 0)
    return elem, val


def spmoverlap_(adj1: SparseTensor, adj2: SparseTensor) -> SparseTensor:
    '''
    Compute the overlap of neighbors (rows in adj). The returned matrix is similar to the hadamard product of adj1 and adj2
    '''
    assert adj1.sizes() == adj2.sizes()
    element1, val1 = spm2elem(adj1)
    element2, val2 = spm2elem(adj2)

    if element2.shape[0] > element1.shape[0]:
        element1, element2 = element2, element1

    idx = torch.searchsorted(element1[:-1], element2)
    mask = (element1[idx] == element2)
    retelem = element2[mask]
    '''
    nnz1 = adj1.nnz()
    element = torch.cat((adj1.storage.row(), adj2.storage.row()), dim=-1)
    element.bitwise_left_shift_(32)
    element[:nnz1] += adj1.storage.col()
    element[nnz1:] += adj2.storage.col()
    
    element = torch.sort(element, dim=-1)[0]
    mask = (element[1:] == element[:-1])
    retelem = element[:-1][mask]
    '''

    return elem2spm(retelem, adj1.sizes())


def spmnotoverlap_(adj1: SparseTensor,
                   adj2: SparseTensor) -> Tuple[SparseTensor, SparseTensor]:
    '''
    return elements in adj1 but not in adj2 and in adj2 but not adj1
    '''
    # assert adj1.sizes() == adj2.sizes()

    
    element1, val1 = spm2elem(adj1)
    element2, val2 = spm2elem(adj2)

    if element1.shape[0] == 0:
        retelem1 = element1
        retelem2 = element2
    else:
        idx = torch.searchsorted(element1[:-1], element2)
        matchedmask = (element1[idx] == element2)

        maskelem1 = torch.ones_like(element1, dtype=torch.bool)
        maskelem1[idx[matchedmask]] = 0
        retelem1 = element1[maskelem1]

        retelem2 = element2[torch.logical_not(matchedmask)]
    return elem2spm(retelem1, adj1.sizes()), elem2spm(retelem2, adj2.sizes())

def spmdiff_(adj1: SparseTensor,
                   adj2: SparseTensor, keep_val=False) -> Tuple[SparseTensor, SparseTensor]:
    '''
    return elements in adj1 but not in adj2 and in adj2 but not adj1
    '''
    # assert adj1.sizes() == adj2.sizes()

    
    element1, val1 = spm2elem(adj1)
    element2, val2 = spm2elem(adj2)

    if element1.shape[0] == 0:
        retelem1 = element1
        retelem2 = element2
    else:
        idx = torch.searchsorted(element1[:-1], element2)
        matchedmask = (element1[idx] == element2)

        maskelem1 = torch.ones_like(element1, dtype=torch.bool)
        maskelem1[idx[matchedmask]] = 0
        retelem1 = element1[maskelem1]
    
    if keep_val and val1 is not None:
        retval1 = val1[maskelem1]
        return elem2spm(retelem1, adj1.sizes(), retval1)
    else:
        return elem2spm(retelem1, adj1.sizes())


def spmoverlap_notoverlap_(
        adj1: SparseTensor,
        adj2: SparseTensor) -> Tuple[SparseTensor, SparseTensor, SparseTensor]:
    '''
    return elements in adj1 but not in adj2 and in adj2 but not adj1
    '''
    # assert adj1.sizes() == adj2.sizes()
    element1, val1 = spm2elem(adj1)
    element2, val2 = spm2elem(adj2)

    if element1.shape[0] == 0:
        retoverlap = element1
        retelem1 = element1
        retelem2 = element2
    else:
        idx = torch.searchsorted(element1[:-1], element2)
        matchedmask = (element1[idx] == element2)

        maskelem1 = torch.ones_like(element1, dtype=torch.bool)
        maskelem1[idx[matchedmask]] = 0
        retelem1 = element1[maskelem1]

        retoverlap = element2[matchedmask]
        retelem2 = element2[torch.logical_not(matchedmask)]
    sizes = adj1.sizes()
    return elem2spm(retoverlap,
                    sizes), elem2spm(retelem1,
                                     sizes), elem2spm(retelem2, sizes)


def adjoverlap(adj1: SparseTensor,
               adj2: SparseTensor,
               calresadj: bool = False,
               cnsampledeg: int = -1,
               ressampledeg: int = -1):
    """
        returned sparse matrix shaped as [tarei.size(0), num_nodes]
        where each row represent the corresponding target edge,
        and each column represent whether that target edge has such a neighbor.
    """
    # a wrapper for functions above.
    if calresadj:
        adjoverlap, adjres1, adjres2 = spmoverlap_notoverlap_(adj1, adj2)
        if cnsampledeg > 0:
            adjoverlap = sparsesample_reweight(adjoverlap, cnsampledeg)
        if ressampledeg > 0:
            adjres1 = sparsesample_reweight(adjres1, ressampledeg)
            adjres2 = sparsesample_reweight(adjres2, ressampledeg)
        return adjoverlap, adjres1, adjres2
    else:
        adjoverlap = spmoverlap_(adj1, adj2)
        if cnsampledeg > 0:
            adjoverlap = sparsesample_reweight(adjoverlap, cnsampledeg)
    return adjoverlap


def isSymmetric(mat):
    """detect whether a sparse matrix is symmetric"""
    N = mat.shape[0]
    for i in range(N):
        for j in range(N):
            if (mat[i][j] != mat[j][i]):
                return False
    return True

def check_all(pred, real):
    pred = pred.to_dense().numpy()
    real = real.to_dense().numpy()
    assert (pred == real).all()


def neighbors(fringe, A, outgoing=True):
    # Find all 1-hop neighbors of nodes in fringe from graph A, 
    # where A is a scipy csr adjacency matrix.
    # If outgoing=True, find neighbors with outgoing edges;
    # otherwise, find neighbors with incoming edges (you should
    # provide a csc matrix in this case).
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res

def k_hop_subgraph(src, dst, num_hops, A):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A. 
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, num_hops+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes]

    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    return nodes, subgraph, dists






