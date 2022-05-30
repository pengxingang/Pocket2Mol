import torch
from torch.nn import Module, Sequential, LayerNorm
from torch_scatter import scatter_add, scatter_softmax, scatter_sum

from math import pi as PI

from ..common import GaussianSmearing, EdgeExpansion
from ..invariant import GVLinear, GVPerceptronVN, MessageModule

class SpatialClassifierVN(Module):

    def __init__(self, num_classes, num_bond_types, in_sca, in_vec, num_filters, edge_channels, num_heads, k=32, cutoff=10.0):
        super().__init__()
        self.num_bond_types = num_bond_types
        self.message_module = MessageModule(in_sca, in_vec, edge_channels, edge_channels, num_filters[0], num_filters[1], cutoff)

        self.nn_edge_ij = Sequential(
            GVPerceptronVN(edge_channels, edge_channels, num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1])
        )
        
        self.classifier = Sequential(
            GVPerceptronVN(num_filters[0], num_filters[1], num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_classes, 1)
        )

        self.edge_feat = Sequential(
            GVPerceptronVN(num_filters[0] * 2 + in_sca, num_filters[1] * 2 + in_vec, num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1])
        )
        self.edge_atten = AttentionEdges(num_filters, num_filters, num_heads, num_bond_types)
        self.edge_pred = GVLinear(num_filters[0], num_filters[1], num_bond_types + 1, 1)
        
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.distance_expansion_3A = GaussianSmearing(stop=3., num_gaussians=edge_channels)
        self.vector_expansion = EdgeExpansion(edge_channels)  # Linear(in_features=1, out_features=edge_channels, bias=False)
        self.k = k
        self.cutoff = cutoff

    def forward(self, pos_query, edge_index_query, pos_compose, node_attr_compose, edge_index_q_cps_knn,
                         index_real_cps_edge_for_atten=[], tri_edge_index=[], tri_edge_feat=[]):
        # (self, pos_query, edge_index_query, pos_ctx, node_attr_ctx, is_mol_atom, batch_query, batch_edge, batch_ctx):
        """
        Args:
            pos_query:   (N_query, 3)
            edge_index_query: (2, N_q_c, )
            pos_ctx:     (N_ctx, 3)
            node_attr_ctx:  (N_ctx, H)
            is_mol_atom: (N_ctx, )
            batch_query: (N_query, )
            batch_ctx:   (N_ctx, )
        Returns
            (N_query, num_classes)
        """

        # Pairwise distances and contextual node features
        vec_ij = pos_query[edge_index_q_cps_knn[0]] - pos_compose[edge_index_q_cps_knn[1]]
        dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)  # (A, 1)
        edge_ij = self.distance_expansion(dist_ij), self.vector_expansion(vec_ij)
        
        # node_attr_ctx_j = [node_attr_ctx_[edge_index_q_cps_knn[1]] for node_attr_ctx_ in node_attr_ctx]  # (A, H)
        h = self.message_module(node_attr_compose, edge_ij, edge_index_q_cps_knn[1], dist_ij, annealing=True)

        # Aggregate messages
        y = [scatter_add(h[0], index=edge_index_q_cps_knn[0], dim=0, dim_size=pos_query.size(0)), # (N_query, F)
                scatter_add(h[1], index=edge_index_q_cps_knn[0], dim=0, dim_size=pos_query.size(0))]

        # element prediction
        y_cls, _ = self.classifier(y)  # (N_query, num_classes)


        # edge prediction
        if (len(edge_index_query) != 0) and (edge_index_query.size(1) > 0):
            # print(edge_index_query.shape)
            idx_node_i = edge_index_query[0]
            node_mol_i = [
                y[0][idx_node_i],
                y[1][idx_node_i]
            ]
            idx_node_j = edge_index_query[1]
            node_mol_j = [
                node_attr_compose[0][idx_node_j],
                node_attr_compose[1][idx_node_j]
            ]
            vec_ij = pos_query[idx_node_i] - pos_compose[idx_node_j]
            dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)  # (E, 1)

            edge_ij = self.distance_expansion_3A(dist_ij), self.vector_expansion(vec_ij) 
            edge_feat = self.nn_edge_ij(edge_ij)  # (E, F)

            edge_attr = (torch.cat([node_mol_i[0], node_mol_j[0], edge_feat[0]], dim=-1),  # (E, F)
                                                torch.cat([node_mol_i[1], node_mol_j[1], edge_feat[1]], dim=1))
            edge_attr = self.edge_feat(edge_attr)  # (E, N_edgetype)
            edge_attr = self.edge_atten(edge_attr, edge_index_query, pos_compose, index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat) #
            edge_pred, _ = self.edge_pred(edge_attr)

        else:
            edge_pred = torch.empty([0, self.num_bond_types+1], device=pos_query.device)

        return y_cls, edge_pred


class AttentionEdges(Module):

    def __init__(self, hidden_channels, key_channels, num_heads=1, num_bond_types=3):
        super().__init__()
        
        assert (hidden_channels[0] % num_heads == 0) and (hidden_channels[1] % num_heads == 0)
        assert (key_channels[0] % num_heads == 0) and (key_channels[1] % num_heads == 0)

        self.hidden_channels = hidden_channels
        self.key_channels = key_channels
        self.num_heads = num_heads

        # linear transformation for attention 
        self.q_lin = GVLinear(hidden_channels[0], hidden_channels[1], key_channels[0], key_channels[1])
        self.k_lin = GVLinear(hidden_channels[0], hidden_channels[1], key_channels[0], key_channels[1])
        self.v_lin = GVLinear(hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1])

        self.atten_bias_lin = AttentionBias(self.num_heads, hidden_channels, num_bond_types=num_bond_types)
        self.layernorm_sca = LayerNorm([hidden_channels[0]])
        self.layernorm_vec = LayerNorm([hidden_channels[1], 3])

    def forward(self, edge_attr, edge_index, pos_compose, 
                          index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat,):
        """
        Args:
            x:  edge features: scalar features (N, feat), vector features(N, feat, 3)
            edge_attr:  (E, H)
            edge_index: (2, E). the row can be seen as batch_edge
        """
        scalar, vector = edge_attr
        N = scalar.size(0)
        row, col = edge_index   # (N,) 

        # Project to multiple key, query and value spaces
        h_queries = self.q_lin(edge_attr)
        h_queries = (h_queries[0].view(N, self.num_heads, -1),  # (N, heads, K_per_head)
                    h_queries[1].view(N, self.num_heads, -1, 3))  # (N, heads, K_per_head, 3)
        h_keys = self.k_lin(edge_attr)
        h_keys = (h_keys[0].view(N, self.num_heads, -1),  # (N, heads, K_per_head)
                    h_keys[1].view(N, self.num_heads, -1, 3))  # (N, heads, K_per_head, 3)
        h_values = self.v_lin(edge_attr)
        h_values = (h_values[0].view(N, self.num_heads, -1),  # (N, heads, K_per_head)
                    h_values[1].view(N, self.num_heads, -1, 3))  # (N, heads, K_per_head, 3)

        # assert (index_edge_i_list == index_real_cps_edge_for_atten[0]).all()
        # assert (index_edge_j_list == index_real_cps_edge_for_atten[1]).all()
        index_edge_i_list, index_edge_j_list = index_real_cps_edge_for_atten

        # # get nodes of triangle edges

        atten_bias = self.atten_bias_lin(
            tri_edge_index,
            tri_edge_feat,
            pos_compose,
        )


        # query * key
        queries_i = [h_queries[0][index_edge_i_list], h_queries[1][index_edge_i_list]]
        keys_j = [h_keys[0][index_edge_j_list], h_keys[1][index_edge_j_list]]

        qk_ij = [
            (queries_i[0] * keys_j[0]).sum(-1),  # (N', heads)
            (queries_i[1] * keys_j[1]).sum(-1).sum(-1)  # (N', heads)
        ]

        alpha = [
            atten_bias[0] + qk_ij[0],
            atten_bias[1] + qk_ij[1]
        ]
        alpha = [
            scatter_softmax(alpha[0], index_edge_i_list, dim=0),  # (N', heads)
            scatter_softmax(alpha[1], index_edge_i_list, dim=0)  # (N', heads)
        ] 

        values_j = [h_values[0][index_edge_j_list], h_values[1][index_edge_j_list]]
        num_attens = len(index_edge_j_list)
        output =[
            scatter_sum((alpha[0].unsqueeze(-1) * values_j[0]).view(num_attens, -1), index_edge_i_list, dim=0, dim_size=N),   # (N, H, 3)
            scatter_sum((alpha[1].unsqueeze(-1).unsqueeze(-1) * values_j[1]).view(num_attens, -1, 3), index_edge_i_list, dim=0, dim_size=N)   # (N, H, 3)
        ]

        # output 
        output = [edge_attr[0] + output[0], edge_attr[1] + output[1]]
        output = [self.layernorm_sca(output[0]), self.layernorm_vec(output[1])]

        return output



class AttentionBias(Module):

    def __init__(self, num_heads, hidden_channels, cutoff=10., num_bond_types=3): #TODO: change the cutoff
        super().__init__()
        num_edge_types = num_bond_types + 1
        self.num_bond_types = num_bond_types
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=hidden_channels[0] - num_edge_types-1)  # minus 1 for self edges (e.g. edge 0-0)
        self.vector_expansion = EdgeExpansion(hidden_channels[1])  # Linear(in_features=1, out_features=hidden_channels[1], bias=False)
        self.gvlinear = GVLinear(hidden_channels[0], hidden_channels[1], num_heads, num_heads)

    def forward(self,  tri_edge_index, tri_edge_feat, pos_compose):
        node_a, node_b = tri_edge_index
        pos_a = pos_compose[node_a]
        pos_b = pos_compose[node_b]
        vector = pos_a - pos_b
        dist = torch.norm(vector, p=2, dim=-1)
        
        dist_feat = self.distance_expansion(dist)
        sca_feat = torch.cat([
            dist_feat,
            tri_edge_feat,
        ], dim=-1)
        vec_feat = self.vector_expansion(vector)
        output_sca, output_vec = self.gvlinear([sca_feat, vec_feat])
        output_vec = (output_vec * output_vec).sum(-1)
        return output_sca, output_vec
