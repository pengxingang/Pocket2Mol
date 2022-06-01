import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import knn, knn_graph
# from torch_geometric.nn.pool import knn_graph
import numpy as np

def split_tensor_by_batch(x, batch, num_graphs=None):
    """
    Args:
        x:      (N, ...)
        batch:  (B, )
    Returns:
        [(N_1, ), (N_2, ) ..., (N_B, ))]
    """
    if num_graphs is None:
        num_graphs = batch.max().item() + 1
    x_split = []
    for i in range (num_graphs):
        mask = batch == i
        x_split.append(x[mask])
    return x_split


def concat_tensors_to_batch(x_split):
    x = torch.cat(x_split, dim=0)
    batch = torch.repeat_interleave(
        torch.arange(len(x_split)), 
        repeats=torch.LongTensor([s.size(0) for s in x_split])
    ).to(device=x.device)
    return x, batch


def split_tensor_to_segments(x, segsize):
    num_segs = math.ceil(x.size(0) / segsize)
    segs = []
    for i in range(num_segs):
        segs.append(x[i*segsize : (i+1)*segsize])
    return segs


def split_tensor_by_lengths(x, lengths):
    segs = []
    for l in lengths:
        segs.append(x[:l])
        x = x[l:]
    return segs


def batch_intersection_mask(batch, batch_filter):
    batch_filter = batch_filter.unique()
    mask = (batch.view(-1, 1) == batch_filter.view(1, -1)).any(dim=1)
    return mask


def get_batch_edge(ligand_context_bond_index, ligand_context_bond_type):
    return ligand_context_bond_index, ligand_context_bond_type

class MeanReadout(nn.Module):
    """Mean readout operator over graphs with variadic sizes."""

    def forward(self, input, batch, num_graphs):
        """
        Perform readout over the graph(s).
        Parameters:
            data (torch_geometric.data.Data): batched graph
            input (Tensor): node representations
        Returns:
            Tensor: graph representations
        """
        output = scatter_mean(input, batch, dim=0, dim_size=num_graphs)
        return output


class SumReadout(nn.Module):
    """Sum readout operator over graphs with variadic sizes."""

    def forward(self, input, batch, num_graphs):
        """
        Perform readout over the graph(s).
        Parameters:
            data (torch_geometric.data.Data): batched graph
            input (Tensor): node representations
        Returns:
            Tensor: graph representations
        """
        output = scatter_add(input, batch, dim=0, dim_size=num_graphs)
        return output


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no activation or dropout in the last layer.
    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

    def forward(self, input):
        """"""
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


class EdgeExpansion(nn.Module):
    def __init__(self, edge_channels):
        super().__init__()
        self.nn = nn.Linear(in_features=1, out_features=edge_channels, bias=False)
    
    def forward(self, edge_vector):
        edge_vector = edge_vector / (torch.norm(edge_vector, p=2, dim=1, keepdim=True)+1e-7)
        expansion = self.nn(edge_vector.unsqueeze(-1)).transpose(1, -1)
        return expansion


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        self.stop = stop
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.clamp_max(self.stop)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class GaussianSmearingVN(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=64):
        super().__init__()
        assert num_gaussians % 8 == 0
        num_per_direction = num_gaussians // 8
        delta = (stop - start) / num_per_direction
        offset = torch.linspace(start+delta/2, stop-delta/2, num_per_direction)
        unit_vector = self.get_unit_vector()
        kernel_vectors = unit_vector.unsqueeze(1) * offset.reshape([1, -1, 1])
        self.kernel_vectors = kernel_vectors.reshape([-1, 3])
        self.coeff = -0.5 / delta.item()**2
        self.register_buffer('offset', offset)
    
    def get_unit_vector(self,):
        vec = torch.tensor([-1., 1.])
        vec = torch.stack([a.reshape(-1) for a in torch.meshgrid(vec, vec, vec, indexing=None)], dim=-1)
        vec = vec / np.sqrt(3)
        return vec

    def forward(self, dist):
        dist = dist.view(-1, 1, 3) - self.kernel_vectors.view(1, -1, 3)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        # self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.leaky_relu(x)
        # return F.softplus(x) - self.shift


def compose_context(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
    sort_idx = batch_ctx.argsort()

    is_mol_atom = torch.cat([
        torch.zeros([batch_protein.size(0)], device=batch_protein.device).bool(),
        torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
    ], dim=0)[sort_idx]

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]       # (N_protein+N_ligand, H)
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx] # (N_protein+N_ligand, 3)

    return h_ctx, pos_ctx, batch_ctx, is_mol_atom


def embed_compose(compose_feature, compose_pos, idx_ligand, idx_protein,
                                      ligand_atom_emb, protein_atom_emb,
                                      emb_dim):

    h_ligand = ligand_atom_emb(compose_feature[idx_ligand], compose_pos[idx_ligand])
    h_protein = protein_atom_emb(compose_feature[idx_protein], compose_pos[idx_protein])
    
    h_sca = torch.zeros([len(compose_pos), emb_dim[0]],).to(h_ligand[0])
    h_vec = torch.zeros([len(compose_pos), emb_dim[1], 3],).to(h_ligand[1])
    h_sca[idx_ligand], h_sca[idx_protein] = h_ligand[0], h_protein[0]
    h_vec[idx_ligand], h_vec[idx_protein] = h_ligand[1], h_protein[1]
    return [h_sca, h_vec]


def compose_context_vn(h_ligand, h_protein, pos_ligand, pos_protein, batch_ligand, batch_protein):
    batch_ctx = torch.cat([batch_ligand, batch_protein], dim=0)
    A = batch_ctx[:, None] == torch.arange(batch_protein.max()+1, device=batch_ctx.device)
    sort_idx = torch.nonzero(A.T)[:, -1]
    # sort_idx2 = batch_ctx.argsort()
    batch_ctx = batch_ctx[sort_idx]

    sca_ctx = torch.cat([h_ligand[0], h_protein[0]], dim=0)[sort_idx]       # (N_protein+N_ligand, H)
    vec_ctx = torch.cat([h_ligand[1], h_protein[1]], dim=0)[sort_idx]       # (N_protein+N_ligand, H)
    pos_ctx = torch.cat([pos_ligand, pos_protein], dim=0)[sort_idx] # (N_protein+N_ligand, 3)

    is_mol_atom = torch.cat([
        torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
        torch.zeros([batch_protein.size(0)], device=batch_protein.device).bool(),
    ], dim=0)[sort_idx]

    return (sca_ctx, vec_ctx), pos_ctx, batch_ctx, is_mol_atom


def get_compose_knn_graph(
        pos_compose,
        knn,
        ligand_context_bond_index,
        ligand_context_bond_type,
        is_mol_atom,
        batch_compose
    ):
    compose_knn_edge_index = knn_graph(pos_compose, knn, flow='target_to_source', batch=batch_compose)
    # init edge features
    compose_knn_edge_feature = torch.cat([
        torch.ones([len(compose_knn_edge_index[0]), 1], dtype=torch.float32),
        torch.zeros([len(compose_knn_edge_index[0]), 3], dtype=torch.float32),
    ], dim=-1).to(pos_compose)
    # get bond index in compose
    idx_ligand_ctx_in_compose = torch.nonzero(is_mol_atom).squeeze(-1)
    compose_bond_index = idx_ligand_ctx_in_compose[ligand_context_bond_index]
    compose_bond_type = ligand_context_bond_type
    # find the bond in all edges
    len_compose = len(batch_compose)
    id_compose_edge = compose_knn_edge_index[0] * len_compose + compose_knn_edge_index[1]
    id_compose_bond = compose_bond_index[0] * len_compose + compose_bond_index[1]
    idx_bond = [torch.nonzero(id_compose_edge == id_) for id_ in id_compose_bond]
    idx_bond = torch.tensor([a.squeeze() if len(a) >0 else torch.tensor(-1) for a in idx_bond], dtype=torch.long)
    compose_knn_edge_feature[idx_bond[idx_bond>=0]] = F.one_hot(compose_bond_type[idx_bond>=0], num_classes=4).to(torch.float32)    # 0 (1,2,3)-onehot
    return compose_knn_edge_index, compose_knn_edge_feature


def get_query_compose_knn_edge(
        pos_query,
        pos_compose,
        k,
        batch_query,
        batch_compose,
    ):
    query_compose_knn_edge_index = knn(
        x=pos_compose,
        y=pos_query, 
        k=k,
        batch_x=batch_compose,
        batch_y=batch_query
    )
    return query_compose_knn_edge_index


def get_edge_atten_input(edge_index_query, n_query, context_bond_index, context_bond_type):
    if (len(edge_index_query) != 0) and (edge_index_query.size(1) > 0):
        device = edge_index_query.device
        row, col = edge_index_query
        acc_num_edges = 0
        index_real_cps_edge_i_list, index_real_cps_edge_j_list = [], []  # index of real-ctx edge (for attention)
        for node in torch.arange(n_query):
            num_edges = (row == node).sum()
            index_edge_i = torch.arange(num_edges, dtype=torch.long, device=device) + acc_num_edges
            index_edge_i, index_edge_j = torch.meshgrid(index_edge_i, index_edge_i, indexing=None)
            index_edge_i, index_edge_j = index_edge_i.flatten(), index_edge_j.flatten()
            index_real_cps_edge_i_list.append(index_edge_i)
            index_real_cps_edge_j_list.append(index_edge_j)
            acc_num_edges += num_edges
        index_real_cps_edge_i = torch.cat(index_real_cps_edge_i_list, dim=0)  # add len(real_compose_edge_index) in the dataloader for batch
        index_real_cps_edge_j = torch.cat(index_real_cps_edge_j_list, dim=0)

        node_a_cps_tri_edge = col[index_real_cps_edge_i]  # the node of tirangle edge for the edge attention (in the compose)
        node_b_cps_tri_edge = col[index_real_cps_edge_j]

        if context_bond_index.size(1)  > 0:
            n_context = 1 + torch.maximum(context_bond_index.flatten().max(), col.max())  # NOTE:for only one batch
            adj_mat = torch.zeros([n_context, n_context], dtype=torch.long, device=device) - torch.eye(n_context, dtype=torch.long, device=device)
            adj_mat[context_bond_index[0], context_bond_index[1]] = context_bond_type
            tri_edge_type = adj_mat[node_a_cps_tri_edge, node_b_cps_tri_edge]
            tri_edge_feat = (tri_edge_type.view([-1, 1]) == torch.tensor([[-1, 0, 1, 2, 3]], device=device)).long()
        else:
            n_context = 1 + col.max()
            adj_mat = torch.zeros([n_context, n_context], dtype=torch.long) - torch.eye(n_context, dtype=torch.long)
            tri_edge_type = adj_mat[node_a_cps_tri_edge, node_b_cps_tri_edge]
            tri_edge_feat = (tri_edge_type.view([-1, 1]) == torch.tensor([[-1, 0, 1, 2, 3]])).long().to(device)

        index_real_cps_edge_for_atten = torch.stack([
            index_real_cps_edge_i, index_real_cps_edge_j  # plus len(real_compose_edge_index_0) for dataloader batch
        ], dim=0)
        tri_edge_index = torch.stack([
            node_a_cps_tri_edge, node_b_cps_tri_edge  # plus len(compose_pos) for dataloader batch
        ], dim=0)
        tri_edge_feat = tri_edge_feat
        return index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat
    else:
        return [], [], []


def get_complete_graph(batch):
    """
    Args:
        batch:  Batch index.
    Returns:
        edge_index: (2, N_1 + N_2 + ... + N_{B-1}), where N_i is the number of nodes of the i-th graph.
        neighbors:  (B, ), number of edges per graph.
    """
    natoms = scatter_add(torch.ones_like(batch), index=batch, dim=0)

    natoms_sqr = (natoms ** 2).long()
    num_atom_pairs = torch.sum(natoms_sqr)
    natoms_expand = torch.repeat_interleave(natoms, natoms_sqr)

    index_offset = torch.cumsum(natoms, dim=0) - natoms
    index_offset_expand = torch.repeat_interleave(index_offset, natoms_sqr)

    index_sqr_offset = torch.cumsum(natoms_sqr, dim=0) - natoms_sqr
    index_sqr_offset = torch.repeat_interleave(index_sqr_offset, natoms_sqr)

    atom_count_sqr = torch.arange(num_atom_pairs, device=num_atom_pairs.device) - index_sqr_offset

    index1 = (atom_count_sqr // natoms_expand).long() + index_offset_expand
    index2 = (atom_count_sqr % natoms_expand).long() + index_offset_expand
    edge_index = torch.cat([index1.view(1, -1), index2.view(1, -1)])
    mask = torch.logical_not(index1 == index2)
    edge_index = edge_index[:, mask]

    num_edges = natoms_sqr - natoms # Number of edges per graph

    return edge_index, num_edges


def compose_context_stable(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    num_graphs = batch_ligand.max().item() + 1

    batch_ctx = []
    h_ctx = []
    pos_ctx = []
    mask_protein = []

    for i in range(num_graphs):
        mask_p, mask_l = (batch_protein == i), (batch_ligand == i)
        batch_p, batch_l = batch_protein[mask_p], batch_ligand[mask_l]

        batch_ctx += [batch_p, batch_l]
        h_ctx += [h_protein[mask_p], h_ligand[mask_l]]
        pos_ctx += [pos_protein[mask_p], pos_ligand[mask_l]]
        mask_protein += [
            torch.ones([batch_p.size(0)], device=batch_p.device, dtype=torch.bool),
            torch.zeros([batch_l.size(0)], device=batch_l.device, dtype=torch.bool),
        ]

    batch_ctx = torch.cat(batch_ctx, dim=0)
    h_ctx = torch.cat(h_ctx, dim=0)
    pos_ctx = torch.cat(pos_ctx, dim=0)
    mask_protein = torch.cat(mask_protein, dim=0)

    return h_ctx, pos_ctx, batch_ctx, mask_protein


    