from copy import deepcopy
import os
# import sys
# sys.path.append('.')
import shutil
import argparse
# import random
import torch
import numpy as np
from torch_geometric.data import Batch
from easydict import EasyDict
from tqdm.auto import tqdm
from rdkit import Chem

from models.maskfill import MaskFillModelVN
from models.sample import *
from utils.transforms import *
from utils.datasets import get_dataset
from utils.misc import *
from utils.data import FOLLOW_BATCH
from utils.reconstruct import *
# from utils.chem import *
STATUS_RUNNING = 'running'
STATUS_FINISHED = 'finished'
STATUS_FAILED = 'failed'


def logp_to_rank_prob(logp, weight=1.0):

    logp_sum = np.array([np.sum(l) for l in logp])
    prob = np.exp(logp_sum) + 1
    prob = prob * np.array(weight)
    return prob / prob.sum()


@torch.no_grad()  # for a protein-ligand
def get_init(data, model, transform, threshold):
    batch = Batch.from_data_list([data], follow_batch=FOLLOW_BATCH) #batch only contains one data

    ### Predict next atoms
    model.eval()
    predicitions = model.sample_init(
        compose_feature = batch.compose_feature.float(),
        compose_pos = batch.compose_pos,
        # idx_ligand = batch.idx_ligand_ctx_in_compose,
        idx_protein = batch.idx_protein_in_compose,
        compose_knn_edge_index = batch.compose_knn_edge_index,
        compose_knn_edge_feature = batch.compose_knn_edge_feature,
        n_samples_pos=-1,
        n_samples_atom=5,
    )
    data = data.to('cpu')
    # no frontier
    if not predicitions[0]:
        data.status = STATUS_FINISHED
        return [data]
    # has frontiers
    data.status = STATUS_RUNNING
    (has_frontier, idx_frontier, p_frontier,
    idx_focal_in_compose, p_focal,
    pos_generated, pdf_pos, abs_pos_mu, pos_sigma, pos_pi,
    element_pred, element_prob, has_atom_prob) = [p.cpu() for p in predicitions]

    while True:
        data_next_list = get_next_step(
            data,
            p_focal = p_focal,
            pos_generated = pos_generated,
            pdf_pos = pdf_pos,
            element_pred = element_pred,
            element_prob = element_prob,
            has_atom_prob = has_atom_prob,
            # ind_pred = ind_pred,
            # ind_prob = ind_prob,
            bond_index = torch.empty([2, 0]),
            bond_type = torch.empty([0]),
            bond_prob = torch.empty([0]),
            transform = transform,
            threshold=threshold
        )
        data_next_list = [data for data in data_next_list if data.is_high_prob]
        if len(data_next_list) == 0:
            if torch.all(pdf_pos < threshold.pos_threshold):
                threshold.pos_threshold = threshold.pos_threshold / 2
                print('Positional probability threshold is too high. Change to %f' % threshold.pos_threshold)
            elif torch.all(p_focal < threshold.focal_threshold):
                threshold.focal_threshold = threshold.focal_threshold / 2
                print('Focal probability threshold is too high. Change to %f' % threshold.focal_threshold)
            elif torch.all(element_prob < threshold.element_threshold):
                threshold.element_threshold = threshold.element_threshold / 2
                print('Element probability threshold is too high. Change to %f' % threshold.element_threshold)
            else:
                print('Initialization failed.')
        else:
            break

    return data_next_list


@torch.no_grad()  # for a protein-ligand
def get_next(data, model, transform, threshold):
    batch = Batch.from_data_list([data], follow_batch=FOLLOW_BATCH) #batch only contains one data

    ### Predict next atoms
    model.eval()
    predicitions = model.sample(
        compose_feature = batch.compose_feature.float(),
        compose_pos = batch.compose_pos,
        idx_ligand = batch.idx_ligand_ctx_in_compose,
        idx_protein = batch.idx_protein_in_compose,
        compose_knn_edge_index = batch.compose_knn_edge_index,
        compose_knn_edge_feature = batch.compose_knn_edge_feature,
        ligand_context_bond_index = batch.ligand_context_bond_index,
        ligand_context_bond_type = batch.ligand_context_bond_type,
        n_samples_pos=-1,
        n_samples_atom=5
    )
    data = data.to('cpu')
    # no frontier
    if not predicitions[0]:
        data.status = STATUS_FINISHED
        return [data]
    # has frontiers
    (has_frontier, idx_frontier, p_frontier,
    idx_focal_in_compose, p_focal,
    pos_generated, pdf_pos, abs_pos_mu, pos_sigma, pos_pi,
    element_pred, element_prob, has_atom_prob,
    bond_index, bond_type, bond_prob) = [p.cpu() for p in predicitions]

    data_next_list = get_next_step(
        data,
        p_focal = p_focal,
        pos_generated = pos_generated,
        pdf_pos = pdf_pos,
        element_pred = element_pred,
        element_prob = element_prob,
        has_atom_prob = has_atom_prob,
        bond_index = bond_index,
        bond_type = bond_type,
        bond_prob = bond_prob,
        transform = transform,
        threshold = threshold
    )
    data_next_list = [data for data in data_next_list if data.is_high_prob]

    return data_next_list


def print_pool_status(pool, logger):
    logger.info('[Pool] Queue %d | Finished %d | Failed %d' % (
        len(pool.queue), len(pool.finished), len(pool.failed)
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sample.yml')
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('-i', '--data_id', type=str, default='0')
    args = parser.parse_args()

    # check existing in output dir
    os.makedirs(args.outdir, exist_ok=True)
    all_files = os.listdir(args.outdir)
    file_name = os.path.basename(args.config)[:-4] + '_' + args.data_id + '_'
    files_target = [f for f in all_files if f.startswith(file_name)]
    if len(files_target) > 0:  # # has been sampled before
        for file_name in files_target:
            file_dir = os.path.join(args.outdir, file_name)
            if os.path.exists(os.path.join(file_dir, 'samples_all.pt')):  # # finished
                print('Already finished! data_id: %s' % args.data_id)
                exit(0)
            else:
                print('Has been terminated data_id: %s' % args.data_id)
                shutil.rmtree(file_dir)

    # # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.sample.seed)

    # # Get pdb id or data_idx
    data_id = int(args.data_id)

    # # Logging
    log_dir = get_new_log_dir(args.outdir, prefix='%s_%s' % (config_name, data_id))
    logger = get_logger('sample', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))    

    # # Transform
    logger.info('Loading data...')
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    contrastive_sampler = ContrastiveSample()
    transform = Compose([
        RefineData(),
        LigandCountNeighbors(),
        protein_featurizer,
        ligand_featurizer,
    ])
    # # Data
    if config.data.data_name == 'test':
        dataset, subsets = get_dataset(
            config = config.data.dataset,
            transform = transform,
        )
        testset = subsets['test']
        base_data = testset[data_id]

    # # Model (Main)
    logger.info('Loading main model...')
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    model = MaskFillModelVN(
        ckpt['config'].model, 
        num_classes = contrastive_sampler.num_elements,
        protein_atom_feature_dim = protein_featurizer.feature_dim,
        ligand_atom_feature_dim = ligand_featurizer.feature_dim,
        num_bond_types = 3,
    ).to(args.device)
    model.load_state_dict(ckpt['model'])

    pool = EasyDict({
        'queue': [],
        'failed': [],
        'finished': [],
        'duplicate': [],
        'smiles': set(),
    })
    # # Sample the first atoms
    logger.info('Initialization')
    pbar = tqdm(total=config.sample.beam_size, desc='InitSample')
    mask = LigandMaskAll()
    atom_composer = AtomComposer(protein_featurizer.feature_dim, ligand_featurizer.feature_dim, model.config.encoder.knn)
    if config.sample.mask_init:
        masking = Compose([
            mask,
            atom_composer
        ]) 
        data = transform_data(deepcopy(base_data), masking)
        init_data_list = get_init(data.to(args.device),   # sample the initial atoms
                model = model,
                transform=atom_composer,
                threshold=config.sample.threshold
        )
        pool.queue = init_data_list
        if len(pool.queue) > config.sample.beam_size:
            pool.queue = init_data_list[:config.sample.beam_size]
            pbar.update(config.sample.beam_size)
        else:
            pbar.update(len(pool.queue))
    else:
        masking = Compose([
            LigandBFSMask(min_ratio=1., min_num_unmasked=1),
            atom_composer
        ]) 
        while len(pool.queue) < config.sample.num_samples // 10:
            queue_size_before = len(pool.queue)
            data = transform_data(deepcopy(base_data), masking)
            data.status = STATUS_RUNNING
            pool.queue += [data]
            if len(pool.queue) > config.sample.num_samples:
                pool.queue = pool.queue[:config.sample.num_samples]
            pbar.update(len(pool.queue) - queue_size_before)
    pbar.close()

    print_pool_status(pool, logger)
    logger.info('Saving samples...')
    # torch.save(pool, os.path.join(log_dir, 'samples_init.pt'))

    # # Sampling loop
    logger.info('Start sampling')
    global_step = 0

    while len(pool.finished) < config.sample.num_samples:
        global_step += 1
        if global_step > config.sample.max_steps:
            break
        queue_size = len(pool.queue)
        # # sample candidate new mols from each parent mol
        queue_tmp = []
        for data in tqdm(pool.queue):
            nexts = []
            data_next_list = get_next(
                data.to(args.device), 
                model = model,
                transform = atom_composer,
                threshold = config.sample.threshold
            )

            for data_next in data_next_list:
                if data_next.status == STATUS_FINISHED:
                    try:
                        rdmol = reconstruct_from_generated_with_edges(data_next)
                        data_next.rdmol = rdmol
                        mol = Chem.MolFromSmiles(Chem.MolToSmiles(rdmol))
                        smiles = Chem.MolToSmiles(mol)
                        data_next.smiles = smiles
                        if smiles in pool.smiles:
                            logger.warning('Duplicate molecule: %s' % smiles)
                            pool.duplicate.append(data_next)
                        elif '.' in smiles:
                            logger.warning('Failed molecule: %s' % smiles)
                            pool.failed.append(data_next)
                        else:   # Pass checks
                            logger.info('Success: %s' % smiles)
                            pool.finished.append(data_next)
                            pool.smiles.add(smiles)
                    except MolReconsError:
                        logger.warning('Reconstruction error encountered.')
                        pool.failed.append(data_next)
                elif data_next.status == STATUS_RUNNING:
                    nexts.append(data_next)

            queue_tmp += nexts
        # # random choose mols from candidates
        prob = logp_to_rank_prob(np.array([p.average_logp[2:] for p in queue_tmp]),)  # (logp_focal, logpdf_pos), logp_element, logp_hasatom, logp_bond
        n_tmp = len(queue_tmp)
        next_idx = np.random.choice(np.arange(n_tmp), p=prob, size=min(config.sample.beam_size, n_tmp), replace=False)
        pool.queue = [queue_tmp[idx] for idx in next_idx]

        print_pool_status(pool, logger)
        # torch.save(pool, os.path.join(log_dir, 'samples_%d.pt' % global_step))

    # # Save sdf mols
    sdf_dir = os.path.join(log_dir, 'SDF')
    os.makedirs(sdf_dir)
    with open(os.path.join(log_dir, 'SMILES.txt'), 'a') as smiles_f:
        for i, data_finished in enumerate(pool['finished']):
            smiles_f.write(data_finished.smiles + '\n')
            rdmol = data_finished.rdmol
            Chem.MolToMolFile(rdmol, os.path.join(sdf_dir, '%d.sdf' % i))

    torch.save(pool, os.path.join(log_dir, 'samples_all.pt'))
