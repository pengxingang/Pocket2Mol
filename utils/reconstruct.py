import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit import Geometry


class MolReconsError(Exception):
    pass

def add_context(data):
    data.ligand_context_pos = data.ligand_pos
    data.ligand_context_element = data.ligand_element
    data.ligand_context_bond_index = data.ligand_bond_index
    data.ligand_context_bond_type = data.ligand_bond_type
    return data

def reconstruct_from_generated_with_edges(data, raise_error=True, sanitize=True):
    xyz = data.ligand_context_pos.clone().cpu().tolist()
    atomic_nums = data.ligand_context_element.clone().cpu().tolist()
    # indicators = data.ligand_context_feature_full[:, -len(ATOM_FAMILIES_ID):].clone().cpu().bool().tolist()
    bond_index = data.ligand_context_bond_index.clone().cpu().tolist()
    bond_type = data.ligand_context_bond_type.clone().cpu().tolist()
    n_atoms = len(atomic_nums)

    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)
    
    # add atoms and coordinates
    for i, atom in enumerate(atomic_nums):
        rd_atom = Chem.Atom(atom)
        rd_mol.AddAtom(rd_atom)
        rd_coords = Geometry.Point3D(*xyz[i])
        rd_conf.SetAtomPosition(i, rd_coords)
    rd_mol.AddConformer(rd_conf)
    
    # add bonds
    for i, type_this in enumerate(bond_type):
        node_i, node_j = bond_index[0][i], bond_index[1][i]
        if node_i < node_j:
            if type_this == 1:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.SINGLE)
            elif type_this == 2:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.DOUBLE)
            elif type_this == 3:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.TRIPLE)
            elif type_this == 12:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.AROMATIC)
            else:
                raise Exception('unknown bond order {}'.format(type_this))
    
    # modify
    try:
        rd_mol = modify_submol(rd_mol)
    except:
        if raise_error:
            raise MolReconsError()
        else:
            print('MolReconsError')
    # check valid
    rd_mol_check = Chem.MolFromSmiles(Chem.MolToSmiles(rd_mol))
    if rd_mol_check is None:
        if raise_error:
            raise MolReconsError()
        else:
            print('MolReconsError')
    
    rd_mol = rd_mol.GetMol()
    if 12 in bond_type:  # mol may directlu come from ture mols and contains aromatic bonds
        Chem.Kekulize(rd_mol, clearAromaticFlags=True)
    if sanitize:
        Chem.SanitizeMol(rd_mol, Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE^Chem.SANITIZE_SETAROMATICITY)
    return rd_mol


def modify_submol(mol):  # modify mols containing C=N(C)O
    submol = Chem.MolFromSmiles('C=N(C)O', sanitize=False)
    sub_fragments = mol.GetSubstructMatches(submol)
    for fragment in sub_fragments:
        atomic_nums = np.array([mol.GetAtomWithIdx(atom).GetAtomicNum() for atom in fragment])
        idx_atom_N = fragment[np.where(atomic_nums == 7)[0][0]]
        idx_atom_O = fragment[np.where(atomic_nums == 8)[0][0]]
        mol.GetAtomWithIdx(idx_atom_N).SetFormalCharge(1)  # set N to N+
        mol.GetAtomWithIdx(idx_atom_O).SetFormalCharge(-1)  # set O to O-
    return mol

