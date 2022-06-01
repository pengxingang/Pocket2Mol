import copy
# from multiprocessing import context
import os
import sys
sys.path.append('.')
import random
import time
import uuid
from itertools import compress
# from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.pool import knn_graph
from torch_geometric.transforms import Compose
from torch_geometric.utils.subgraph import subgraph
from torch_geometric.nn import knn, radius
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
# import multiprocessing as multi
# from torch_geometric.data import DataLoader
try:
    from .data import ProteinLigandData
    from .datasets import *
    from .misc import *
    from .train import inf_iterator
    from .protein_ligand import ATOM_FAMILIES
except:
    from utils.data import ProteinLigandData
    from utils.datasets import *
    from utils.misc import *
    from utils.train import inf_iterator
    from utils.protein_ligand import ATOM_FAMILIES
import argparse
import logging

class RefineData(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        # delete H atom of pocket
        protein_element = data.protein_element
        is_H_protein = (protein_element == 1)
        if torch.sum(is_H_protein) > 0:
            not_H_protein = ~is_H_protein
            data.protein_atom_name = list(compress(data.protein_atom_name, not_H_protein)) 
            data.protein_atom_to_aa_type = data.protein_atom_to_aa_type[not_H_protein]
            data.protein_element = data.protein_element[not_H_protein]
            data.protein_is_backbone = data.protein_is_backbone[not_H_protein]
            data.protein_pos = data.protein_pos[not_H_protein]
        # delete H atom of ligand
        ligand_element = data.ligand_element
        is_H_ligand = (ligand_element == 1)
        if torch.sum(is_H_ligand) > 0:
            not_H_ligand = ~is_H_ligand
            data.ligand_atom_feature = data.ligand_atom_feature[not_H_ligand]
            data.ligand_element = data.ligand_element[not_H_ligand]
            data.ligand_pos = data.ligand_pos[not_H_ligand]
            # nbh
            index_atom_H = torch.nonzero(is_H_ligand)[:, 0]
            index_changer = -np.ones(len(not_H_ligand), dtype=np.int64)
            index_changer[not_H_ligand] = np.arange(torch.sum(not_H_ligand))
            new_nbh_list = [value for ind_this, value in zip(not_H_ligand, data.ligand_nbh_list.values()) if ind_this]
            data.ligand_nbh_list = {i:[index_changer[node] for node in neigh if node not in index_atom_H] for i, neigh in enumerate(new_nbh_list)}
            # bond
            ind_bond_with_H = np.array([(bond_i in index_atom_H) | (bond_j in index_atom_H) for bond_i, bond_j in zip(*data.ligand_bond_index)])
            ind_bond_without_H = ~ind_bond_with_H
            old_ligand_bond_index = data.ligand_bond_index[:, ind_bond_without_H]
            data.ligand_bond_index = torch.tensor(index_changer)[old_ligand_bond_index]
            data.ligand_bond_type = data.ligand_bond_type[ind_bond_without_H]

        return data


class FeaturizeProteinAtom(object):

    def __init__(self):
        super().__init__()
        # self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])    # H, C, N, O, S, Se
        self.atomic_numbers = torch.LongTensor([6, 7, 8, 16, 34])    # H, C, N, O, S, Se
        self.max_num_aa = 20

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1 + 1

    def __call__(self, data:ProteinLigandData):
        element = data.protein_element.view(-1, 1) == self.atomic_numbers.view(1, -1)   # (N_atoms, N_elements)
        amino_acid = F.one_hot(data.protein_atom_to_aa_type, num_classes=self.max_num_aa)
        is_backbone = data.protein_is_backbone.view(-1, 1).long()
        is_mol_atom = torch.zeros_like(is_backbone, dtype=torch.long)
        # x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        x = torch.cat([element, amino_acid, is_backbone, is_mol_atom], dim=-1)
        data.protein_atom_feature = x
        # data.compose_index = torch.arange(len(element), dtype=torch.long)
        return data


class FeaturizeLigandAtom(object):
    
    def __init__(self):
        super().__init__()
        # self.atomic_numbers = torch.LongTensor([1,6,7,8,9,15,16,17])  # H C N O F P S Cl
        self.atomic_numbers = torch.LongTensor([6,7,8,9,15,16,17])  # C N O F P S Cl
        assert len(self.atomic_numbers) == 7, NotImplementedError('fix the staticmethod: chagne_bond')

    # @property
    # def num_properties(self):
        # return len(ATOM_FAMILIES)

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + (1 + 1 + 1) + 3

    def __call__(self, data:ProteinLigandData):
        element = data.ligand_element.view(-1, 1) == self.atomic_numbers.view(1, -1)   # (N_atoms, N_elements)
        # chem_feature = data.ligand_atom_feature
        is_mol_atom = torch.ones([len(element), 1], dtype=torch.long)
        n_neigh = data.ligand_num_neighbors.view(-1, 1)
        n_valence = data.ligand_atom_valence.view(-1, 1)
        ligand_atom_num_bonds = data.ligand_atom_num_bonds
        # x = torch.cat([element, chem_feature, ], dim=-1)
        x = torch.cat([element, is_mol_atom, n_neigh, n_valence, ligand_atom_num_bonds], dim=-1)
        data.ligand_atom_feature_full = x
        return data

    @staticmethod
    def change_features_of_neigh(ligand_feature_full, new_num_neigh, new_num_valence, ligand_atom_num_bonds):
        idx_n_neigh = 7 + 1
        idx_n_valence = idx_n_neigh + 1
        idx_n_bonds = idx_n_valence + 1
        ligand_feature_full[:, idx_n_neigh] = new_num_neigh.long()
        ligand_feature_full[:, idx_n_valence] = new_num_valence.long()
        ligand_feature_full[:, idx_n_bonds:idx_n_bonds+3] = ligand_atom_num_bonds.long()
        return ligand_feature_full



class FeaturizeLigandBond(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data:ProteinLigandData):
        data.ligand_bond_feature = F.one_hot(data.ligand_bond_type - 1 , num_classes=3)    # (1,2,3) to (0,1,2)-onehot
        return data


class LigandCountNeighbors(object):

    @staticmethod
    def count_neighbors(edge_index, symmetry, valence=None, num_nodes=None):
        assert symmetry == True, 'Only support symmetrical edges.'

        if num_nodes is None:
            num_nodes = maybe_num_nodes(edge_index)

        if valence is None:
            valence = torch.ones([edge_index.size(1)], device=edge_index.device)
        valence = valence.view(edge_index.size(1))

        return scatter_add(valence, index=edge_index[0], dim=0, dim_size=num_nodes).long()

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.ligand_num_neighbors = self.count_neighbors(
            data.ligand_bond_index, 
            symmetry=True,
            num_nodes=data.ligand_element.size(0),
        )
        data.ligand_atom_valence = self.count_neighbors(
            data.ligand_bond_index, 
            symmetry=True, 
            valence=data.ligand_bond_type,
            num_nodes=data.ligand_element.size(0),
        )
        data.ligand_atom_num_bonds = torch.stack([
            self.count_neighbors(
                data.ligand_bond_index, 
                symmetry=True, 
                valence=(data.ligand_bond_type == i).long(),
                num_nodes=data.ligand_element.size(0),
            ) for i in [1, 2, 3]
        ], dim = -1)
        return data


class LigandRandomMask(object):

    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_num_masked = min_num_masked
        self.min_num_unmasked = min_num_unmasked

    def __call__(self, data:ProteinLigandData):
        ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
        num_atoms = data.ligand_element.size(0)
        num_masked = int(num_atoms * ratio)

        if num_masked < self.min_num_masked:
            num_masked = self.min_num_masked
        if (num_atoms - num_masked) < self.min_num_unmasked:
            num_masked = num_atoms - self.min_num_unmasked

        idx = np.arange(num_atoms)
        np.random.shuffle(idx)
        idx = torch.LongTensor(idx)
        masked_idx = idx[:num_masked]
        context_idx = idx[num_masked:]

        data.context_idx = context_idx  # for change bond index
        data.masked_idx = masked_idx

        # masked ligand atom element/feature/pos.
        data.ligand_masked_element = data.ligand_element[masked_idx]
        # data.ligand_masked_feature = data.ligand_atom_feature[masked_idx]   # For Prediction. these features are chem properties
        data.ligand_masked_pos = data.ligand_pos[masked_idx]

        # context ligand atom elment/full features/pos. Note: num_neigh and num_valence features should be changed
        data.ligand_context_element = data.ligand_element[context_idx]
        data.ligand_context_feature_full = data.ligand_atom_feature_full[context_idx]   # For Input
        data.ligand_context_pos = data.ligand_pos[context_idx]

        # new bond with ligand context atoms
        if data.ligand_bond_index.size(1) != 0:
            data.ligand_context_bond_index, data.ligand_context_bond_type = subgraph(
                context_idx,
                data.ligand_bond_index,
                edge_attr = data.ligand_bond_type,
                relabel_nodes = True,
            )
        else:
            data.ligand_context_bond_index = torch.empty([2, 0], dtype=torch.long)
            data.ligand_context_bond_type = torch.empty([0], dtype=torch.long)
        # change context atom features that relate to bonds
        data.ligand_context_num_neighbors = LigandCountNeighbors.count_neighbors(
            data.ligand_context_bond_index,
            symmetry=True,
            num_nodes = context_idx.size(0),
        )
        data.ligand_context_valence = LigandCountNeighbors.count_neighbors(
            data.ligand_context_bond_index,
            symmetry=True,
            valence=data.ligand_context_bond_type,
            num_nodes=context_idx.size(0)
        )
        data.ligand_context_num_bonds = torch.stack([
            LigandCountNeighbors.count_neighbors(
                data.ligand_context_bond_index, 
                symmetry=True, 
                valence=(data.ligand_context_bond_type == i).long(),
                num_nodes=context_idx.size(0),
            ) for i in [1, 2, 3]
        ], dim = -1)
        # re-calculate ligand_context_featrure_full
        data.ligand_context_feature_full = FeaturizeLigandAtom.change_features_of_neigh(
            data.ligand_context_feature_full,
            data.ligand_context_num_neighbors,
            data.ligand_context_valence,
            data.ligand_context_num_bonds
        )

        data.ligand_frontier = data.ligand_context_num_neighbors < data.ligand_num_neighbors[context_idx]

        data._mask = 'random'

        return data


class LigandBFSMask(object):
    
    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0, inverse=False):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_num_masked = min_num_masked
        self.min_num_unmasked = min_num_unmasked
        self.inverse = inverse

    @staticmethod
    def get_bfs_perm(nbh_list):
        num_nodes = len(nbh_list)
        num_neighbors = torch.LongTensor([len(nbh_list[i]) for i in range(num_nodes)])

        bfs_queue = [random.randint(0, num_nodes-1)]
        bfs_perm = []
        num_remains = [num_neighbors.clone()]
        bfs_next_list = {}
        visited = {bfs_queue[0]}   

        num_nbh_remain = num_neighbors.clone()
        
        while len(bfs_queue) > 0:
            current = bfs_queue.pop(0)
            for nbh in nbh_list[current]:
                num_nbh_remain[nbh] -= 1
            bfs_perm.append(current)
            num_remains.append(num_nbh_remain.clone())
            next_candid = []
            for nxt in nbh_list[current]:
                if nxt in visited: continue
                next_candid.append(nxt)
                visited.add(nxt)
                
            random.shuffle(next_candid)
            bfs_queue += next_candid
            bfs_next_list[current] = copy.copy(bfs_queue)

        return torch.LongTensor(bfs_perm), bfs_next_list, num_remains

    def __call__(self, data):
        bfs_perm, bfs_next_list, num_remaining_nbs = self.get_bfs_perm(data.ligand_nbh_list)

        ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
        num_atoms = data.ligand_element.size(0)
        num_masked = int(num_atoms * ratio)
        if num_masked < self.min_num_masked:
            num_masked = self.min_num_masked
        if (num_atoms - num_masked) < self.min_num_unmasked:
            num_masked = num_atoms - self.min_num_unmasked

        if self.inverse:
            masked_idx = bfs_perm[:num_masked]
            context_idx = bfs_perm[num_masked:]
        else:
            masked_idx = bfs_perm[-num_masked:]
            context_idx = bfs_perm[:-num_masked]

        data.context_idx = context_idx  # for change bond index
        data.masked_idx = masked_idx

        # masked ligand atom element/feature/pos.
        data.ligand_masked_element = data.ligand_element[masked_idx]
        # data.ligand_masked_feature = data.ligand_atom_feature[masked_idx]   # For Prediction. these features are chem properties
        data.ligand_masked_pos = data.ligand_pos[masked_idx]
        
        # context ligand atom elment/full features/pos. Note: num_neigh and num_valence features should be changed
        data.ligand_context_element = data.ligand_element[context_idx]
        data.ligand_context_feature_full = data.ligand_atom_feature_full[context_idx]   # For Input
        data.ligand_context_pos = data.ligand_pos[context_idx]

        # new bond with ligand context atoms
        if data.ligand_bond_index.size(1) != 0:
            data.ligand_context_bond_index, data.ligand_context_bond_type = subgraph(
                context_idx,
                data.ligand_bond_index,
                edge_attr = data.ligand_bond_type,
                relabel_nodes = True,
            )
        else:
            data.ligand_context_bond_index = torch.empty([2, 0], dtype=torch.long)
            data.ligand_context_bond_type = torch.empty([0], dtype=torch.long)
        # re-calculate atom features that relate to bond
        data.ligand_context_num_neighbors = LigandCountNeighbors.count_neighbors(
            data.ligand_context_bond_index,
            symmetry=True,
            num_nodes = context_idx.size(0),
        )
        data.ligand_context_valence = LigandCountNeighbors.count_neighbors(
            data.ligand_context_bond_index,
            symmetry=True,
            valence=data.ligand_context_bond_type,
            num_nodes=context_idx.size(0)
        )
        data.ligand_context_num_bonds = torch.stack([
            LigandCountNeighbors.count_neighbors(
                data.ligand_context_bond_index, 
                symmetry=True, 
                valence=data.ligand_context_bond_type == i,
                num_nodes=context_idx.size(0),
            ) for i in [1, 2, 3]
        ], dim = -1)
        # re-calculate ligand_context_featrure_full
        data.ligand_context_feature_full = FeaturizeLigandAtom.change_features_of_neigh(
            data.ligand_context_feature_full,
            data.ligand_context_num_neighbors,
            data.ligand_context_valence,
            data.ligand_context_num_bonds
        )

        data.ligand_frontier = data.ligand_context_num_neighbors < data.ligand_num_neighbors[context_idx]

        data._mask = 'invbfs' if self.inverse else 'bfs'

        return data


# class LigandMaskLinker(object):
#     def __init__(self, linker_data):
#         super().__init__()
#         self.linker_data = linker_data
        
    
#     def __call__(self, data:ProteinLigandData):
#         # data_name = os.path.basename(data.ligand_filename)
#         # indicator = np.where(self.sdf_names == data_name)[0][0]
#         linker_data = self.linker_data
#         context_idx = torch.LongTensor(linker_data['frag_idx'])
#         masked_idx = torch.LongTensor(linker_data['linker_idx'])
#         data.context_idx = context_idx
#         data.masked_idx = masked_idx

#         # masked ligand atom element/feature/pos.
#         data.ligand_masked_element = data.ligand_element[masked_idx]
#         # data.ligand_masked_feature = data.ligand_atom_feature[masked_idx]   # For Prediction. these features are chem properties
#         data.ligand_masked_pos = data.ligand_pos[masked_idx]

#         # context ligand atom elment/full features/pos. Note: num_neigh and num_valence features should be changed
#         data.ligand_context_element = data.ligand_element[context_idx]
#         data.ligand_context_feature_full = data.ligand_atom_feature_full[context_idx]   # For Input
#         data.ligand_context_pos = data.ligand_pos[context_idx]

#         # new bond with ligand context atoms
#         data.ligand_context_bond_index, data.ligand_context_bond_type = subgraph(
#             context_idx,
#             data.ligand_bond_index,
#             edge_attr = data.ligand_bond_type,
#             relabel_nodes = True,
#         )
#         # change context atom features that relate to bonds
#         data.ligand_context_num_neighbors = LigandCountNeighbors.count_neighbors(
#             data.ligand_context_bond_index,
#             symmetry=True,
#             num_nodes = context_idx.size(0),
#         )
#         data.ligand_context_valence = LigandCountNeighbors.count_neighbors(
#             data.ligand_context_bond_index,
#             symmetry=True,
#             valence=data.ligand_context_bond_type,
#             num_nodes=context_idx.size(0)
#         )
#         data.ligand_context_num_bonds = torch.stack([
#             LigandCountNeighbors.count_neighbors(
#                 data.ligand_context_bond_index, 
#                 symmetry=True, 
#                 valence=(data.ligand_context_bond_type == i).long(),
#                 num_nodes=context_idx.size(0),
#             ) for i in [1, 2, 3]
#         ], dim = -1)
#         # re-calculate ligand_context_featrure_full
#         data.ligand_context_feature_full = FeaturizeLigandAtom.change_features_of_neigh(
#             data.ligand_context_feature_full,
#             data.ligand_context_num_neighbors,
#             data.ligand_context_valence,
#             data.ligand_context_num_bonds
#         )

#         data.ligand_frontier = data.ligand_context_num_neighbors < data.ligand_num_neighbors[context_idx]

#         data._mask = 'linker'
#         return data


class LigandMaskAll(LigandRandomMask):

    def __init__(self):
        super().__init__(min_ratio=1.0)


class LigandMixedMask(object):

    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0, p_random=0.5, p_bfs=0.25, p_invbfs=0.25):
        super().__init__()

        self.t = [
            LigandRandomMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked),
            LigandBFSMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked, inverse=False),
            LigandBFSMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked, inverse=True),
        ]
        self.p = [p_random, p_bfs, p_invbfs]

    def __call__(self, data):
        f = random.choices(self.t, k=1, weights=self.p)[0]
        return f(data)


def get_mask(cfg):
    if cfg.type == 'bfs':
        return LigandBFSMask(
            min_ratio=cfg.min_ratio, 
            max_ratio=cfg.max_ratio, 
            min_num_masked=cfg.min_num_masked,
            min_num_unmasked=cfg.min_num_unmasked,
        )
    elif cfg.type == 'random':
        return LigandRandomMask(
            min_ratio=cfg.min_ratio, 
            max_ratio=cfg.max_ratio, 
            min_num_masked=cfg.min_num_masked,
            min_num_unmasked=cfg.min_num_unmasked,
        )
    elif cfg.type == 'mixed':
        return LigandMixedMask(
            min_ratio=cfg.min_ratio, 
            max_ratio=cfg.max_ratio, 
            min_num_masked=cfg.min_num_masked,
            min_num_unmasked=cfg.min_num_unmasked,
            p_random = cfg.p_random,
            p_bfs = cfg.p_bfs,
            p_invbfs = cfg.p_invbfs,
        )
    elif cfg.type == 'all':
        return LigandMaskAll()
    else:
        raise NotImplementedError('Unknown mask: %s' % cfg.type)


class ContrastiveSample(object):
    def __init__(self, num_real=50, num_fake=50, pos_real_std=0.05, pos_fake_std=2.0, knn=32, elements=None):
    # def __init__(self, knn=32, elements=None):
        super().__init__()
        self.num_real = num_real
        self.num_fake = num_fake
        self.pos_real_std = pos_real_std
        self.pos_fake_std = pos_fake_std
        self.knn = knn
        if elements is None:
            elements = [6,7,8,9,15,16,17]
        self.elements = torch.LongTensor(elements)

    @property
    def num_elements(self):
        return self.elements.size(0)

    def __call__(self, data:ProteinLigandData):
        # Positive samples
        pos_real_mode = data.ligand_masked_pos
        element_real = data.ligand_masked_element
        # ind_real = data.ligand_masked_feature
        cls_real = data.ligand_masked_element.view(-1, 1) == self.elements.view(1, -1)
        assert (cls_real.sum(-1) > 0).all(), 'Unexpected elements.'
        p = np.zeros(len(pos_real_mode), dtype=np.float32)
        p[data.idx_generated_in_ligand_masked] = 1.
        real_sample_idx = np.random.choice(np.arange(pos_real_mode.size(0)), size=self.num_real, p=p/p.sum())

        data.pos_real = pos_real_mode[real_sample_idx]
        data.pos_real += torch.randn_like(data.pos_real) * self.pos_real_std
        data.element_real = element_real[real_sample_idx]
        data.cls_real = cls_real[real_sample_idx]
        # data.ind_real = ind_real[real_sample_idx]
        # data.num_neighbors_real = data.ligand_masked_num_neighbors[real_sample_idx]

        mask_ctx_edge_index_0 = data.mask_ctx_edge_index_0
        mask_ctx_edge_index_1 = data.mask_ctx_edge_index_1
        mask_ctx_edge_type = data.mask_ctx_edge_type
        real_ctx_edge_idx_0_list, real_ctx_edge_idx_1_list, real_ctx_edge_type_list = [], [], []
        for new_idx, real_node in enumerate(real_sample_idx):
            idx_edge = (mask_ctx_edge_index_0 == real_node)
            # real_ctx_edge_idx_0 = mask_ctx_edge_index_0[idx_edge]  # get edges related to this node
            real_ctx_edge_idx_1 = mask_ctx_edge_index_1[idx_edge]  # get edges related to this node
            real_ctx_edge_type = mask_ctx_edge_type[idx_edge]
            real_ctx_edge_idx_0 = new_idx * torch.ones(idx_edge.sum(), dtype=torch.long)  # change to new node index
            real_ctx_edge_idx_0_list.append(real_ctx_edge_idx_0)
            real_ctx_edge_idx_1_list.append(real_ctx_edge_idx_1)
            real_ctx_edge_type_list.append(real_ctx_edge_type)

        data.real_ctx_edge_index_0 = torch.cat(real_ctx_edge_idx_0_list, dim=-1)
        data.real_ctx_edge_index_1 = torch.cat(real_ctx_edge_idx_1_list, dim=-1)
        data.real_ctx_edge_type = torch.cat(real_ctx_edge_type_list, dim=-1)
        data.real_compose_edge_index_0 = data.real_ctx_edge_index_0
        data.real_compose_edge_index_1 = data.idx_ligand_ctx_in_compose[data.real_ctx_edge_index_1]  # actually are the same
        data.real_compose_edge_type = data.real_ctx_edge_type

        # the triangle edge of the mask-compose edge
        row, col = data.real_compose_edge_index_0, data.real_compose_edge_index_1
        acc_num_edges = 0
        index_real_cps_edge_i_list, index_real_cps_edge_j_list = [], []  # index of real-ctx edge (for attention)
        for node in torch.arange(data.pos_real.size(0)):
            num_edges = (row == node).sum()
            index_edge_i = torch.arange(num_edges, dtype=torch.long, ) + acc_num_edges
            index_edge_i, index_edge_j = torch.meshgrid(index_edge_i, index_edge_i, indexing=None)
            index_edge_i, index_edge_j = index_edge_i.flatten(), index_edge_j.flatten()
            index_real_cps_edge_i_list.append(index_edge_i)
            index_real_cps_edge_j_list.append(index_edge_j)
            acc_num_edges += num_edges
        index_real_cps_edge_i = torch.cat(index_real_cps_edge_i_list, dim=0)  # add len(real_compose_edge_index) in the dataloader for batch
        index_real_cps_edge_j = torch.cat(index_real_cps_edge_j_list, dim=0)

        node_a_cps_tri_edge = col[index_real_cps_edge_i]  # the node of tirangle edge for the edge attention (in the compose)
        node_b_cps_tri_edge = col[index_real_cps_edge_j]
        n_context = len(data.ligand_context_pos)
        adj_mat = torch.zeros([n_context, n_context], dtype=torch.long) - torch.eye(n_context, dtype=torch.long)
        adj_mat[data.ligand_context_bond_index[0], data.ligand_context_bond_index[1]] = data.ligand_context_bond_type
        tri_edge_type = adj_mat[node_a_cps_tri_edge, node_b_cps_tri_edge]
        tri_edge_feat = (tri_edge_type.view([-1, 1]) == torch.tensor([[-1, 0, 1, 2, 3]])).long()

        data.index_real_cps_edge_for_atten = torch.stack([
            index_real_cps_edge_i, index_real_cps_edge_j  # plus len(real_compose_edge_index_0) for dataloader batch
        ], dim=0)
        data.tri_edge_index = torch.stack([
            node_a_cps_tri_edge, node_b_cps_tri_edge  # plus len(compose_pos) for dataloader batch
        ], dim=0)
        data.tri_edge_feat = tri_edge_feat


        # Negative samples
        if len(data.ligand_context_pos) != 0: # all mask
            pos_fake_mode = data.ligand_context_pos[data.ligand_frontier]
        else:
            pos_fake_mode = data.protein_pos[data.y_protein_frontier]
        fake_sample_idx = np.random.choice(np.arange(pos_fake_mode.size(0)), size=self.num_fake)
        pos_fake = pos_fake_mode[fake_sample_idx]
        data.pos_fake = pos_fake + torch.randn_like(pos_fake) * self.pos_fake_std / 2.

        # knn of query nodes
        real_compose_knn_edge_index = knn(x=data.compose_pos, y=data.pos_real, k=self.knn, num_workers=16)
        data.real_compose_knn_edge_index_0, data.real_compose_knn_edge_index_1 = real_compose_knn_edge_index
        fake_compose_knn_edge_index = knn(x=data.compose_pos, y=data.pos_fake, k=self.knn, num_workers=16)
        data.fake_compose_knn_edge_index_0, data.fake_compose_knn_edge_index_1 =fake_compose_knn_edge_index

        return data


# def get_contrastive_sampler(cfg):
#     return ContrastiveSample(
#         num_real = cfg.num_real,
#         num_fake = cfg.num_fake,
#         pos_real_std = cfg.pos_real_std,
#         pos_fake_std = cfg.pos_fake_std,
#     )



class AtomComposer(object):

    def  __init__(self, protein_dim, ligand_dim, knn):
        super().__init__()
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.knn = knn  # knn of compose atoms
    
    def __call__(self, data:ProteinLigandData):
        # fetch ligand context and protein from data
        ligand_context_pos = data.ligand_context_pos
        ligand_context_feature_full = data.ligand_context_feature_full
        protein_pos = data.protein_pos
        protein_atom_feature = data.protein_atom_feature
        len_ligand_ctx = len(ligand_context_pos)
        len_protein = len(protein_pos)

        # compose ligand context and protein. save idx of them in compose
        data.compose_pos = torch.cat([ligand_context_pos, protein_pos], dim=0)
        len_compose = len_ligand_ctx + len_protein
        ligand_context_feature_full_expand = torch.cat([
            ligand_context_feature_full, torch.zeros([len_ligand_ctx, self.protein_dim - self.ligand_dim], dtype=torch.long)
        ], dim=1)
        data.compose_feature = torch.cat([ligand_context_feature_full_expand, protein_atom_feature], dim=0)
        data.idx_ligand_ctx_in_compose = torch.arange(len_ligand_ctx, dtype=torch.long)  # can be delete
        data.idx_protein_in_compose = torch.arange(len_protein, dtype=torch.long) + len_ligand_ctx  # can be delete

        # build knn graph and bond type
        data = self.get_knn_graph(data, self.knn, len_ligand_ctx, len_compose, num_workers=16)
        return data

    @staticmethod
    def get_knn_graph(data:ProteinLigandData, knn, len_ligand_ctx, len_compose, num_workers=1, ):
        data.compose_knn_edge_index = knn_graph(data.compose_pos, knn, flow='target_to_source', num_workers=num_workers)

        id_compose_edge = data.compose_knn_edge_index[0, :len_ligand_ctx*knn] * len_compose + data.compose_knn_edge_index[1, :len_ligand_ctx*knn]
        id_ligand_ctx_edge = data.ligand_context_bond_index[0] * len_compose + data.ligand_context_bond_index[1]
        idx_edge = [torch.nonzero(id_compose_edge == id_) for id_ in id_ligand_ctx_edge]
        idx_edge = torch.tensor([a.squeeze() if len(a) > 0 else torch.tensor(-1) for a in idx_edge], dtype=torch.long)
        data.compose_knn_edge_type = torch.zeros(len(data.compose_knn_edge_index[0]), dtype=torch.long)  # for encoder edge embedding
        data.compose_knn_edge_type[idx_edge[idx_edge>=0]] = data.ligand_context_bond_type[idx_edge>=0]
        data.compose_knn_edge_feature = torch.cat([
            torch.ones([len(data.compose_knn_edge_index[0]), 1], dtype=torch.long),
            torch.zeros([len(data.compose_knn_edge_index[0]), 3], dtype=torch.long),
        ], dim=-1) 
        data.compose_knn_edge_feature[idx_edge[idx_edge>=0]] = F.one_hot(data.ligand_context_bond_type[idx_edge>=0], num_classes=4)    # 0 (1,2,3)-onehot
        return data


class FocalBuilder(object):
    def __init__(self, close_threshold=0.8, max_bond_length=2.4):
        self.close_threshold = close_threshold
        self.max_bond_length = max_bond_length
        super().__init__()

    def __call__(self, data:ProteinLigandData):
        # ligand_context_pos = data.ligand_context_pos
        # ligand_pos = data.ligand_pos
        ligand_masked_pos = data.ligand_masked_pos
        protein_pos = data.protein_pos
        context_idx = data.context_idx
        masked_idx = data.masked_idx
        old_bond_index = data.ligand_bond_index
        # old_bond_types = data.ligand_bond_type  # type: 0, 1, 2
        has_unmask_atoms = context_idx.nelement() > 0
        if has_unmask_atoms:
            # # get bridge bond index (mask-context bond)
            ind_edge_index_candidate = [
                (context_node in context_idx) and (mask_node in masked_idx)
                for mask_node, context_node in zip(*old_bond_index)
            ]  # the mask-context order is right
            bridge_bond_index = old_bond_index[:, ind_edge_index_candidate]
            # candidate_bond_types = old_bond_types[idx_edge_index_candidate]
            idx_generated_in_whole_ligand = bridge_bond_index[0]
            idx_focal_in_whole_ligand = bridge_bond_index[1]
            
            index_changer_masked = torch.zeros(masked_idx.max()+1, dtype=torch.int64)
            index_changer_masked[masked_idx] = torch.arange(len(masked_idx))
            idx_generated_in_ligand_masked = index_changer_masked[idx_generated_in_whole_ligand]
            pos_generate = ligand_masked_pos[idx_generated_in_ligand_masked]

            data.idx_generated_in_ligand_masked = idx_generated_in_ligand_masked
            data.pos_generate = pos_generate

            index_changer_context = torch.zeros(context_idx.max()+1, dtype=torch.int64)
            index_changer_context[context_idx] = torch.arange(len(context_idx))
            idx_focal_in_ligand_context = index_changer_context[idx_focal_in_whole_ligand]
            idx_focal_in_compose = idx_focal_in_ligand_context  # if ligand_context was not before protein in the compose, this was not correct
            data.idx_focal_in_compose = idx_focal_in_compose

            data.idx_protein_all_mask = torch.empty(0, dtype=torch.long)  # no use if has context
            data.y_protein_frontier = torch.empty(0, dtype=torch.bool)  # no use if has context
            
        else:  # # the initial atom. surface atoms between ligand and protein
            assign_index = radius(x=ligand_masked_pos, y=protein_pos, r=4., num_workers=16)
            if assign_index.size(1) == 0:
                dist = torch.norm(data.protein_pos.unsqueeze(1) - data.ligand_masked_pos.unsqueeze(0), p=2, dim=-1)
                assign_index = torch.nonzero(dist <= torch.min(dist)+1e-5)[0:1].transpose(0, 1)
            idx_focal_in_protein = assign_index[0]
            data.idx_focal_in_compose = idx_focal_in_protein  # no ligand context, so all composes are protein atoms
            data.pos_generate = ligand_masked_pos[assign_index[1]]
            data.idx_generated_in_ligand_masked = torch.unique(assign_index[1])  # for real of the contractive transform

            data.idx_protein_all_mask = data.idx_protein_in_compose  # for input of initial frontier prediction
            y_protein_frontier = torch.zeros_like(data.idx_protein_all_mask, dtype=torch.bool)  # for label of initial frontier prediction
            y_protein_frontier[torch.unique(idx_focal_in_protein)] = True
            data.y_protein_frontier = y_protein_frontier
            
        
        # generate not positions: around pos_focal ( with `max_bond_length` distance) but not close to true generated within `close_threshold` 
        # pos_focal = ligand_context_pos[idx_focal_in_ligand_context]
        # pos_notgenerate = pos_focal + torch.randn_like(pos_focal) * self.max_bond_length  / 2.4
        # dist = torch.norm(pos_generate - pos_notgenerate, p=2, dim=-1)
        # ind_close = (dist < self.close_threshold)
        # while ind_close.any():
        #     new_pos_notgenerate = pos_focal[ind_close] + torch.randn_like(pos_focal[ind_close]) * self.max_bond_length  / 2.3
        #     dist[ind_close] = torch.norm(pos_generate[ind_close] - new_pos_notgenerate, p=2, dim=-1)
        #     pos_notgenerate[ind_close] = new_pos_notgenerate
        #     ind_close = (dist < self.close_threshold)
        # data.pos_notgenerate = pos_notgenerate

        return data



class EdgeSample(object):

    def __init__(self, cfg, num_bond_types=3):
        super().__init__()
        # self.neg_pos_ratio = cfg.neg_pos_ratio
        self.k = cfg.k
        # self.r = cfg.r
        self.num_bond_types = num_bond_types


    def __call__(self, data:ProteinLigandData):
        
        ligand_context_pos = data.ligand_context_pos
        ligand_masked_pos = data.ligand_masked_pos
        context_idx = data.context_idx
        masked_idx = data.masked_idx
        old_bond_index = data.ligand_bond_index
        old_bond_types = data.ligand_bond_type  # type: 0, 1, 2
        
        # candidate edge: mask-contex edge
        idx_edge_index_candidate = [
            (context_node in context_idx) and (mask_node in masked_idx)
            for mask_node, context_node in zip(*old_bond_index)
        ]  # the mask-context order is right
        candidate_bond_index = old_bond_index[:, idx_edge_index_candidate]
        candidate_bond_types = old_bond_types[idx_edge_index_candidate]
        
        # index changer
        index_changer_masked = torch.zeros(masked_idx.max()+1, dtype=torch.int64)
        index_changer_masked[masked_idx] = torch.arange(len(masked_idx))

        has_unmask_atoms = context_idx.nelement() > 0
        if has_unmask_atoms:
            index_changer_context = torch.zeros(context_idx.max()+1, dtype=torch.int64)
            index_changer_context[context_idx] = torch.arange(len(context_idx))

            # new edge index (positive)
            new_edge_index_0 = index_changer_masked[candidate_bond_index[0]]
            new_edge_index_1 = index_changer_context[candidate_bond_index[1]]
            new_edge_index = torch.stack([new_edge_index_0, new_edge_index_1])
            new_edge_type = candidate_bond_types

            neg_version = 0
            if neg_version == 1:  # radiu + tri_edge
                # negative edge index (types = 0)
                id_edge_pos = new_edge_index[0] * len(context_idx) + new_edge_index[1]
                # 1. radius all edges
                edge_index_radius = radius(ligand_context_pos, ligand_masked_pos, r=self.r, num_workers=16)  # r = 3
                id_edge_radius = edge_index_radius[0] * len(context_idx) + edge_index_radius[1]
                not_pos_in_radius = torch.tensor([id_ not in id_edge_pos for id_ in id_edge_radius])
                # 2. pick true neg edges and random choice
                if not_pos_in_radius.size(0) > 0:
                    edge_index_neg = edge_index_radius[:, not_pos_in_radius]
                    dist = torch.norm(ligand_masked_pos[edge_index_neg[0]] - ligand_context_pos[edge_index_neg[1]], p=2, dim=-1)
                    probs = torch.clip(0.8 * (dist ** 2) - 4.8 * dist + 7.3 + 0.4, min=0.5, max=0.95)
                    values = torch.rand(len(dist))
                    choice = values < probs
                    edge_index_neg = edge_index_neg[:, choice]
                else: 
                    edge_index_neg = torch.empty([2, 0], dtype=torch.long)
                # 3. edges form ring should be choicen
                bond_index_ctx = data.ligand_context_bond_index
                edge_index_ring_candidate = [[], []]
                for node_i, node_j in zip(*new_edge_index):
                    node_k_all = bond_index_ctx[1, bond_index_ctx[0] == node_j]
                    edge_index_ring_candidate[0].append( torch.ones_like(node_k_all) * node_i)
                    edge_index_ring_candidate[1].append(node_k_all)
                edge_index_ring_candidate[0] = torch.cat(edge_index_ring_candidate[0], dim=0)
                edge_index_ring_candidate[1] = torch.cat(edge_index_ring_candidate[1], dim=0)
                id_ring_candidate = edge_index_ring_candidate[0] * len(context_idx) + edge_index_ring_candidate[1]
                edge_index_ring_candidate = torch.stack(edge_index_ring_candidate, dim=0)
                not_pos_in_ring = torch.tensor([id_ not in id_edge_pos for id_ in id_ring_candidate])
                if not_pos_in_ring.size(0) > 0:
                    edge_index_ring = edge_index_ring_candidate[:, not_pos_in_ring]
                    dist = torch.norm(ligand_masked_pos[edge_index_ring[0]] - ligand_context_pos[edge_index_ring[1]], p=2, dim=-1)
                    edge_index_ring = edge_index_ring[:, dist < 4.0]
                else:
                    edge_index_ring = torch.empty([2, 0], dtype=torch.long)
                # 4.cat neg and ring
                false_edge_index = torch.cat([
                    edge_index_neg, edge_index_ring
                ], dim=-1)
                false_edge_types = torch.zeros(len(false_edge_index[0]), dtype=torch.int64)
            elif neg_version == 0:  # knn edge
                edge_index_knn = knn(ligand_context_pos, ligand_masked_pos, k=self.k, num_workers=16)
                dist = torch.norm(ligand_masked_pos[edge_index_knn[0]] - ligand_context_pos[edge_index_knn[1]], p=2, dim=-1)
                idx_sort = torch.argsort(dist)  #  choose negative edges as short as possible
                num_neg_edges = min(len(ligand_masked_pos) * (self.k // 2) + len(new_edge_index[0]), len(idx_sort))
                idx_sort = torch.unique(
                    torch.cat([
                        idx_sort[:num_neg_edges],
                        torch.linspace(0, len(idx_sort), len(ligand_masked_pos)+1, dtype=torch.long)[:-1]  # each mask pos at least has one negative edge
                    ], dim=0)
                )
                edge_index_knn = edge_index_knn[:, idx_sort]
                id_edge_knn = edge_index_knn[0] * len(context_idx) + edge_index_knn[1]  # delete false negative edges
                id_edge_new = new_edge_index[0] * len(context_idx) + new_edge_index[1]
                idx_real_edge_index = torch.tensor([id_ in id_edge_new for id_ in id_edge_knn])
                false_edge_index = edge_index_knn[:, ~idx_real_edge_index]
                false_edge_types = torch.zeros(len(false_edge_index[0]), dtype=torch.int64)


            # cat 
            # print('Num of pos : neg edge:', len(new_edge_type), len(false_edge_types), len(new_edge_type) / len(false_edge_types))
            new_edge_index = torch.cat([new_edge_index, false_edge_index], dim=-1)
            new_edge_type = torch.cat([new_edge_type, false_edge_types], dim=0)


            data.mask_ctx_edge_index_0 = new_edge_index[0]
            data.mask_ctx_edge_index_1 = new_edge_index[1]
            data.mask_ctx_edge_type = new_edge_type
            data.mask_compose_edge_index_0 = data.mask_ctx_edge_index_0
            data.mask_compose_edge_index_1 = data.idx_ligand_ctx_in_compose[data.mask_ctx_edge_index_1]  # actually are the same
            data.mask_compose_edge_type = new_edge_type

            
        else:
            data.mask_ctx_edge_index_0 = torch.empty([0], dtype=torch.int64)
            data.mask_ctx_edge_index_1 = torch.empty([0], dtype=torch.int64)
            data.mask_ctx_edge_type = torch.empty([0], dtype=torch.int64)
            data.mask_compose_edge_index_0 = torch.empty([0], dtype=torch.int64)
            data.mask_compose_edge_index_1 = torch.empty([0], dtype=torch.int64)
            data.mask_compose_edge_type = torch.empty([0], dtype=torch.int64)

        return data
