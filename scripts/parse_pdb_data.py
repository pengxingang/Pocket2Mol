import pandas as pd
import os
import shutil
import argparse
import pickle
import sys
sys.path.append('.')
from utils.data import ProteinLigandData
from utils.protein_ligand import *
from utils.visualize import *
from utils.data import torchify_dict
from utils.misc import load_config
from utils.align_structures import align_protein_ligand_pairs


def load_item(item, path):
    pdb_path = os.path.join(path, item[0])
    sdf_path = os.path.join(path, item[1])
    with open(pdb_path, 'r') as f:
        pdb_block = f.read()
    with open(sdf_path, 'r') as f:
        sdf_block = f.read()
    return pdb_block, sdf_block


def get_save_pocket(source,  protein_name, ligand_name, radius):
    # load protein and ligand
    pdb_path = os.path.join(source, protein_name + '.pdb')
    with open(pdb_path, 'r') as f:
        pdb_block = f.read()
    protein = PDBProtein(pdb_block)
    sdf_path = os.path.join(source, ligand_name + '.sdf')
    ligand = parse_sdf_file(sdf_path)
    # get pocket
    pdb_block_pocket = protein.residues_to_pdb_block(
        protein.query_residues_ligand(ligand, radius)
    )
    # pocket = PDBProtein(pdb_block_pocket)
        
    # save pocket pdb
    os.makedirs(os.path.join(source, 'processed_pocket'), exist_ok=True)
    save_dir = os.path.join(source, 'processed_pocket')
    pocket_file = protein_name + '_' + ligand_name + ('_pocket%d.pdb' % radius)
    pocket_path = os.path.join(save_dir, pocket_file)
    with open(pocket_path, 'w') as f:
        f.write(pdb_block_pocket)

    # copy molecule sdf
    ligand_path = os.path.join(save_dir, ligand_name + '.sdf')
    shutil.copyfile(
        src = sdf_path,
        dst = ligand_path
    )
    return pocket_file[:-4]

def align_pocket(refer_pocket_name, refer_ligand_name, sample_pocket_names, sample_ligand_names, files_dir):
    # align pocket
    refer_pocket_path = os.path.join(files_dir, refer_pocket_name + '.pdb')
    refer_ligand_path = os.path.join(files_dir, refer_ligand_name + '.sdf')
    for sample_pocket_name, sample_ligand_name in zip(sample_pocket_names, sample_ligand_names):
        sample_pocket_path = os.path.join(files_dir, sample_pocket_name + '.pdb')
        sample_ligand_path = os.path.join(files_dir, sample_ligand_name + '.sdf')
        align_protein_ligand_pairs(sample_ligand_path, sample_pocket_path, refer_ligand_path, refer_pocket_path)
    

def save_pocket_to_pkl(source, dest, pocket_name, ligand_name):
    # load pocket
    src_pocket_path = os.path.join(source, pocket_name + '.pdb')
    with open(src_pocket_path, 'r') as f:
        pocket_block = f.read()
    pocket = PDBProtein(pocket_block)
    # load ligand
    ligand_path = os.path.join(source, ligand_name + '.sdf')
    ligand = parse_sdf_file(ligand_path)
    # get pocket data
    pocket_dict = pocket.to_dict_atom()
    ligand_dict = ligand
    data = ProteinLigandData.from_protein_ligand_dicts(
            protein_dict=torchify_dict(pocket_dict),
            ligand_dict=torchify_dict(ligand_dict),
    )
    data.protein_filename = pocket_name + '.pdb'
    data.ligand_filename = ligand_name + '.sdf'
    # save
    processed_dest = os.path.join(dest, pocket_name + '_processed.pk')
    with open(processed_dest, 'wb+') as f:
            f.write(pickle.dumps(data))


def process_to_pocket(args):
    try:
        # get pocket 
        pdb_path = os.path.join(args.source, args.protein)
        with open(pdb_path, 'r') as f:
            pdb_block = f.read()
        protein = PDBProtein(pdb_block)
        sdf_path = os.path.join(args.source, args.ligand)
        ligand = parse_sdf_file(sdf_path)

        pdb_block_pocket = protein.residues_to_pdb_block(
            protein.query_residues_ligand(ligand, args.radius)
        )
        pocket = PDBProtein(pdb_block_pocket)
        
        # save pocket pdb
        protein_fn = args.protein
        ligand_fn = args.ligand
        pocket_fn = protein_fn[:-4] + '_' + ligand_fn[:-4] + ('_pocket%d.pdb' % args.radius)
        pocket_dest = os.path.join(args.dest, pocket_fn)
        with open(pocket_dest, 'w') as f:
            f.write(pdb_block_pocket)
        
        # copy molecule sdf
        shutil.copyfile(
            src = os.path.join(args.source, ligand_fn),
            dst = os.path.join(args.dest, ligand_fn)
        )

        # make ProteinLigandData
        pocket_dict = pocket.to_dict_atom()
        ligand_dict = ligand
        data = ProteinLigandData.from_protein_ligand_dicts(
                protein_dict=torchify_dict(pocket_dict),
                ligand_dict=torchify_dict(ligand_dict),
        )
        data.protein_filename = pocket_fn
        data.ligand_filename = ligand_fn

        # save ProteinLigandData
        processed_fn = pocket_fn[:-4] + '_processed.pk'
        processed_dest = os.path.join(args.dest, processed_fn)
        with open(processed_dest, 'wb+') as f:
            f.write(pickle.dumps(data))
        return data, pdb_block_pocket
    except Exception:
        print('Exception occured.')
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    cfg = load_config(args.config)
    # process_to_pocket(cfg)
    
    df_summay = pd.read_csv(cfg.file, index_col=0)
    # df_summay['receptor'].values
    # # get and save pocket
    pocket_names_list = []
    for index, line in df_summay.iterrows():
        pocket_name = get_save_pocket(cfg.source,  line['receptor'], line['ligand'], cfg.radius)
        pocket_names_list.append(pocket_name)
    
    # # align pocket to the wild type reference 
    align_pocket(
        refer_pocket_name = pocket_names_list[0],
        refer_ligand_name = df_summay.loc[0, 'ligand'],
        sample_pocket_names = pocket_names_list[1:],
        sample_ligand_names = df_summay.loc[1:, 'ligand'].values,
        files_dir = os.path.join(cfg.source, 'processed_pocket'),
    )

    # # save pocket to dest as pkl
    for index, line in df_summay.iterrows():
        if index == 0:
            pocket_name = pocket_names_list[index]
        else:
            pocket_name = pocket_names_list[index] + '_aligned'
        save_pocket_to_pkl(os.path.join(cfg.source, 'processed_pocket'), cfg.dest, pocket_name, line['ligand'])

    print('Done protein-ligand pairs.')
    