import os
from time import sleep
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit.Chem.QED import qed
from easydict import EasyDict
from utils.reconstruct import reconstruct_from_generated_with_edges
from sascorer import compute_sa_score
from docking import QVinaDockingTask
from utils.datasets import get_dataset
from rdkit.Chem.FilterCatalog import *



def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    rule_4 = (logp:=Crippen.MolLogP(mol)>=-2) & (logp<=5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])
    

def get_basic(mol):
    n_atoms = len(mol.GetAtoms())
    n_bonds = len(mol.GetBonds())
    n_rings = len(Chem.GetSymmSSSR(mol))
    weight = Descriptors.ExactMolWt(mol)
    return n_atoms, n_bonds, n_rings, weight


def get_rdkit_rmsd(mol, n_conf=20, random_seed=42):
    """
    Calculate the alignment of generated mol and rdkit predicted mol
    Return the rmsd (max, min, median) of the `n_conf` rdkit conformers
    """
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    mol3d = Chem.AddHs(mol)
    rmsd_list = []
    # predict 3d
    confIds = AllChem.EmbedMultipleConfs(mol3d, n_conf, randomSeed=random_seed)
    for confId in confIds:
        AllChem.UFFOptimizeMolecule(mol3d, confId=confId)
        rmsd = Chem.rdMolAlign.GetBestRMS(mol, mol3d, refId=confId)
        rmsd_list.append(rmsd)
    # mol3d = Chem.RemoveHs(mol3d)
    rmsd_list = np.array(rmsd_list)
    return [np.max(rmsd_list), np.min(rmsd_list), np.median(rmsd_list)]


def get_logp(mol):
    return Crippen.MolLogP(mol)


def get_chem(mol):
    qed_score = qed(mol)
    sa_score = compute_sa_score(mol)
    logp_score = Crippen.MolLogP(mol)
    hacc_score = Lipinski.NumHAcceptors(mol)
    hdon_score = Lipinski.NumHDonors(mol)
    return qed_score, sa_score, logp_score, hacc_score, hdon_score


class SimilarityWithTrain:
    def __init__(self) -> None:
        self.cfg_dataset = EasyDict({
            'name': 'pl',
            'path': './data/crossdocked_pocket10', 
            'split': './data/split_by_name.pt', 
            'fingerprint': './data/crossdocked_pocket10_fingerprint.pt',
            'smiles': './data/crossdocked_pocket10_smiles.pt', 
        })
        self.train_smiles = None
        self.train_fingers = None
        
    def _get_train_mols(self):
        file_not_exists = (not os.path.exists(self.cfg_dataset.fingerprint)) or (not os.path.exists(self.cfg_dataset.smiles))
        if file_not_exists:
            _, subsets = get_dataset(config = self.cfg_dataset)
            train_set = subsets['train']
            self.train_smiles = []
            self.train_fingers = []
            for data in tqdm(train_set):  # calculate fingerprint and smiles of train data
                data.ligand_context_pos = data.ligand_pos
                data.ligand_context_element = data.ligand_element
                data.ligand_context_bond_index = data.ligand_bond_index
                data.ligand_context_bond_type = data.ligand_bond_type
                mol = reconstruct_from_generated_with_edges(data, sanitize=True)
                mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))  # automately sanitize 
                smiles = Chem.MolToSmiles(mol)
                fg = Chem.RDKFingerprint(mol)
                self.train_fingers.append(fg)
                self.train_smiles.append(smiles)
            self.train_smiles = np.array(self.train_smiles)
            # self.train_fingers = np.array(self.train_fingers)
            torch.save(self.train_smiles, self.cfg_dataset.smiles)
            torch.save(self.train_fingers, self.cfg_dataset.fingerprint)
        else:
            self.train_smiles = torch.load(self.cfg_dataset.smiles)
            self.train_fingers = torch.load(self.cfg_dataset.fingerprint)
            self.train_smiles = np.array(self.train_smiles)
            # self.train_fingers = np.array(self.train_fingers)

    def _get_uni_mols(self):
        self.train_uni_smiles, self.index_in_train = np.unique(self.train_smiles, return_index=True)
        self.train_uni_fingers = [self.train_fingers[idx] for idx in self.index_in_train]

    def get_similarity(self, mol):
        if self.train_fingers is None:
            self._get_train_mols()
            self._get_uni_mols()
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))  # automately sanitize 
        fp_mol = Chem.RDKFingerprint(mol)
        sims = [DataStructs.TanimotoSimilarity(fp, fp_mol) for fp in self.train_uni_fingers]
        return np.array(sims)


    def get_top_sims(self, mol, top=3):
        similarities = self.get_similarity(mol)
        idx_sort = np.argsort(similarities)[::-1]
        top_scores = similarities[idx_sort[:top]]
        top_smiles = self.train_uni_smiles[idx_sort[:top]]
        return top_scores, top_smiles


