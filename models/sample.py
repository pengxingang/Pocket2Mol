import random
import torch
import numpy as np
from torch.nn import functional as F


def add_ligand_atom_to_data(data, pos, element, bond_index, bond_type, type_map=[6,7,8,9,15,16,17]):
    """
    """
    data = data.clone()

    data.ligand_context_pos = torch.cat([
        data.ligand_context_pos, pos.view(1, 3).to(data.ligand_context_pos)
    ], dim=0)

    data.ligand_context_feature_full = torch.cat([
        data.ligand_context_feature_full,
        torch.cat([
            F.one_hot(element.view(1), len(type_map)).to(data.ligand_context_feature_full), # (1, num_elements)
            torch.tensor([[1, 0, 0]]).to(data.ligand_context_feature_full),  # is_mol_atom, num_neigh (placeholder), valence (placeholder)
            torch.tensor([[0, 0, 0]]).to(data.ligand_context_feature_full)  # num_of_bonds 1, 2, 3(placeholder)
        ], dim=1)
    ], dim=0)
    idx_num_neigh = 7 + 1
    idx_valence = idx_num_neigh + 1
    idx_num_of_bonds = idx_valence + 1

    element = torch.LongTensor([type_map[element.item()]])
    data.ligand_context_element = torch.cat([
        data.ligand_context_element, element.view(1).to(data.ligand_context_element)
    ])

    if len(bond_type) != 0:
        bond_index[0, :] = len(data.ligand_context_pos) - 1
        bond_vec = data.ligand_context_pos[bond_index[0]] - data.ligand_context_pos[bond_index[1]]
        bond_lengths = torch.norm(bond_vec, dim=-1, p=2)
        if (bond_lengths > 3).any():
            print(bond_lengths)

        bond_index_all = torch.cat([bond_index, torch.stack([bond_index[1, :], bond_index[0, :]], dim=0)], dim=1)
        bond_type_all = torch.cat([bond_type, bond_type], dim=0)
        
        data.ligand_context_bond_index = torch.cat([
            data.ligand_context_bond_index, bond_index_all.to(data.ligand_context_bond_index)
        ], dim=1)

        data.ligand_context_bond_type = torch.cat([
            data.ligand_context_bond_type,
            bond_type_all
        ])
        # modify atom features related to bonds
        # previous atom
        data.ligand_context_feature_full[bond_index[1, :], idx_num_neigh] += 1 # num of neigh of previous nodes
        data.ligand_context_feature_full[bond_index[1, :], idx_valence] += bond_type # valence of previous nodes
        data.ligand_context_feature_full[bond_index[1, :], idx_num_of_bonds + bond_type - 1] += 1  # num of bonds of 
        # the new atom
        data.ligand_context_feature_full[-1, idx_num_neigh] += len(bond_index[1]) # num of neigh of last node
        data.ligand_context_feature_full[-1, idx_valence] += torch.sum(bond_type) # valence of last node
        for bond in [1, 2, 3]:
            data.ligand_context_feature_full[-1, idx_num_of_bonds + bond - 1] += (bond_type == bond).sum()  # num of bonds of last node
            
    return data
    


def get_next_step(
        parent_sample,
        p_focal,
        pos_generated,
        pdf_pos,
        element_pred,
        element_prob,
        has_atom_prob,
        bond_index,
        bond_type,
        bond_prob,
        transform,
        type_map=[6,7,8,9,15,16,17],
        threshold=None,
    ):
    results = []
    for i in range(len(pos_generated)):
        # # add new atom information
        index_bond_i = (bond_index[0, :] == i)
        data_new = add_ligand_atom_to_data(
            parent_sample,
            pos = pos_generated[i],
            element = element_pred[i],
            bond_index = bond_index[:, index_bond_i],
            bond_type = bond_type[index_bond_i],
            type_map = type_map
        )
        # # make new compose
        data_new = transform(data_new)
        # # add generation probabilities
        logp_focal = np.log(p_focal[i].item()+1e-16)
        logpdf_pos = np.log(pdf_pos[i].item()+1e-16)
        logp_element = np.log(element_prob[i].item()+1e-16)
        logp_hasatom = np.log(has_atom_prob[i].item()+1e-16)
        logp_bond = np.mean(np.log(bond_prob[index_bond_i].cpu().detach().numpy()))
        is_high_prob = ((logp_focal >= np.log(threshold.focal_threshold)) and
                        (logpdf_pos >= np.log(threshold.pos_threshold)) and
                        (logp_element >= np.log(  threshold.element_threshold)) and
                        (logp_hasatom >= np.log(  threshold.hasatom_threshold)))
        if not np.isnan(logp_bond):
            is_high_prob = is_high_prob and (logp_bond >= np.log( threshold.bond_threshold))
        data_new.is_high_prob = is_high_prob
        
        if 'logp_focal' not in data_new:
                data_new.logp_focal = [logp_focal]
                data_new.logpdf_pos = [logpdf_pos]
                data_new.logp_element = [logp_element]
                data_new.logp_hasatom = [logp_hasatom]
        else:
            data_new.logp_focal.append(logp_focal)
            data_new.logpdf_pos.append(logpdf_pos)
            data_new.logp_element.append(logp_element)
            data_new.logp_hasatom.append(logp_hasatom)

        if not np.isnan(logp_bond):
            if ('logp_bond' not in data_new):
                data_new.logp_bond = [logp_bond]
            else:
                data_new.logp_bond.append(logp_bond)
            data_new.average_logp = np.array([np.mean(logps) for logps in [data_new.logp_focal, data_new.logpdf_pos, data_new.logp_element, data_new.logp_hasatom, data_new.logp_bond]])
        else:
            data_new.average_logp = np.array([np.mean(logps) for logps in [data_new.logp_focal, data_new.logpdf_pos, data_new.logp_element, data_new.logp_hasatom]])

        results.append(data_new)

    random.shuffle(results)
    return results
