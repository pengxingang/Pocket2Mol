import copy
import torch
import numpy as np
# from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

FOLLOW_BATCH = [] #['protein_element', 'ligand_context_element', 'pos_real', 'pos_fake']


class ProteinLigandData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item

        instance['ligand_nbh_list'] = {i.item():[j.item() for k, j in enumerate(instance.ligand_bond_index[1]) if instance.ligand_bond_index[0, k].item() == i] for i in instance.ligand_bond_index[0]}
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ligand_bond_index':
            return self['ligand_element'].size(0)
        elif key == 'ligand_context_bond_index':
            return self['ligand_context_element'].size(0)

        elif key == 'mask_ctx_edge_index_0':
            return self['ligand_masked_element'].size(0)
        elif key == 'mask_ctx_edge_index_1':
            return self['ligand_context_element'].size(0)
        elif key == 'mask_compose_edge_index_0':
            return self['ligand_masked_element'].size(0)
        elif key == 'mask_compose_edge_index_1':
            return self['compose_pos'].size(0)

        elif key == 'compose_knn_edge_index':  # edges for message passing of encoder 
            return self['compose_pos'].size(0)

        elif key == 'real_ctx_edge_index_0':
            return self['pos_real'].size(0)
        elif key == 'real_ctx_edge_index_1':
            return self['ligand_context_element'].size(0)
        elif key == 'real_compose_edge_index_0':  # edges for edge type prediction
            return self['pos_real'].size(0)
        elif key == 'real_compose_edge_index_1':
            return self['compose_pos'].size(0)

        elif key == 'real_compose_knn_edge_index_0':  # edges for message passing of  field
            return self['pos_real'].size(0)
        elif key == 'fake_compose_knn_edge_index_0':
            return self['pos_fake'].size(0)
        elif (key == 'real_compose_knn_edge_index_1') or (key == 'fake_compose_knn_edge_index_1'):
            return self['compose_pos'].size(0)

        elif (key == 'idx_protein_in_compose') or (key == 'idx_ligand_ctx_in_compose'):
            return self['compose_pos'].size(0)
            
        elif key == 'index_real_cps_edge_for_atten':
            return self['real_compose_edge_index_0'].size(0)
        elif key == 'tri_edge_index':
            return self['compose_pos'].size(0)

        elif key == 'idx_generated_in_ligand_masked':
            return self['ligand_masked_element'].size(0)
        elif key == 'idx_focal_in_compose':
            return self['compose_pos'].size(0)
        elif key == 'idx_protein_all_mask':
            return self['compose_pos'].size(0)
        else:
            return super().__inc__(key, value)


class ProteinLigandDataLoader(DataLoader):

    def __init__(
        self, 
        dataset, 
        batch_size = 1, 
        shuffle = False, 
        follow_batch = ['ligand_element', 'protein_element'], 
        **kwargs
    ):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, follow_batch=follow_batch, **kwargs)


def batch_from_data_list(data_list):
    return Batch.from_data_list(data_list, follow_batch=['ligand_element', 'protein_element'])


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output

    