import pickle
import torch
import os
import numpy as np
from torch.utils.data import Subset
from .pl import PocketLigandPairDataset


def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)
    
    if 'split' in config:
        split_by_name = torch.load(config.split)
        split = {
            k: [dataset.name2id[n] for n in names if n in dataset.name2id]
            for k, names in split_by_name.items()
        }
        subsets = {k:Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset

def get_data_new_mol(config, pdb_id):
    mol_dir = config.data_dir
    # get path
    files = os.listdir(mol_dir)
    is_pdb_files = np.array([pdb_id in fi for fi in files])  # sift this pdb
    is_process_files = np.array(['processed' in fi for fi in files])  # get the processed pk file
    file = np.array(files)[(is_pdb_files & is_process_files)][0]
    path = os.path.join(mol_dir, file)
    # get data
    with open(path, 'rb') as f:
        data = pickle.loads(f.read())
    return data

def transform_data(data, transform):
    assert data.protein_pos.size(0) > 0
    if transform is not None:
        data = transform(data)
    return data