import os
import argparse
import warnings
from easydict import EasyDict
from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Selection import unfold_entities
from rdkit import Chem

from utils.protein_ligand import PDBProtein
from sample import *    # Import everything from `sample.py`


def pdb_to_pocket_data(pdb_path, center, bbox_size):
    center = torch.FloatTensor(center)
    warnings.simplefilter('ignore', BiopythonWarning)
    ptable = Chem.GetPeriodicTable()
    parser = PDBParser()
    model = parser.get_structure(None, pdb_path)[0]

    protein_dict = EasyDict({
        'element': [],
        'pos': [],
        'is_backbone': [],
        'atom_to_aa_type': [],
    })
    for atom in unfold_entities(model, 'A'):
        res = atom.get_parent()
        resname = res.get_resname()
        if resname == 'MSE': resname = 'MET'
        if resname not in PDBProtein.AA_NAME_NUMBER: continue   # Ignore water, heteros, and non-standard residues.

        element_symb = atom.element.capitalize()
        if element_symb == 'H': continue
        x, y, z = atom.get_coord()
        pos = torch.FloatTensor([x, y, z])
        if (pos - center).abs().max() > (bbox_size / 2): 
            continue

        protein_dict['element'].append( ptable.GetAtomicNumber(element_symb))
        protein_dict['pos'].append(pos)
        protein_dict['is_backbone'].append(atom.get_name() in ['N', 'CA', 'C', 'O'])
        protein_dict['atom_to_aa_type'].append(PDBProtein.AA_NAME_NUMBER[resname])
        
    if len(protein_dict['element']) == 0:
        raise ValueError('No atoms found in the bounding box (center=%r, size=%f).' % (center, bbox_size))

    protein_dict['element'] = torch.LongTensor(protein_dict['element'])
    protein_dict['pos'] = torch.stack(protein_dict['pos'], dim=0)
    protein_dict['is_backbone'] = torch.BoolTensor(protein_dict['is_backbone'])
    protein_dict['atom_to_aa_type'] = torch.LongTensor(protein_dict['atom_to_aa_type'])

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict = protein_dict,
        ligand_dict = {
            'element': torch.empty([0,], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0,], dtype=torch.long),
        }
    )
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str,
                        default='./example/4yhj.pdb')
    parser.add_argument('--center', type=lambda s: list(map(float, s.split(','))),
                        default=[32.0, 28.0, 36.0], 
                        help='Center of the pocket bounding box, in format x,y,z')
    parser.add_argument('--bbox_size', type=float, default=23.0, 
                        help='Pocket bounding box size')
    parser.add_argument('--config', type=str, default='./configs/sample_for_pdb.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--outdir', type=str, default='./outputs')
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.sample.seed)

    # Logging
    log_dir = get_new_log_dir(args.outdir, prefix='%s_%s' % (
        config_name, 
        os.path.basename(args.pdb_path),
    ))
    logger = get_logger('sample', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))    
    shutil.copyfile(args.pdb_path, os.path.join(log_dir, os.path.basename(args.pdb_path)))    

    # # Transform
    logger.info('Loading data...')
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    contrastive_sampler = ContrastiveSample(num_real=0, num_fake=0)
    masking = LigandMaskAll()
    transform = Compose([
        RefineData(),
        LigandCountNeighbors(),
        protein_featurizer,
        ligand_featurizer,
        masking,
    ])
    # # Data
    data = pdb_to_pocket_data(args.pdb_path, args.center, args.bbox_size)
    data = transform(data)

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

    # Sampling
    # The algorithm is the same as the one `sample.py`.

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
    atom_composer = AtomComposer(protein_featurizer.feature_dim, ligand_featurizer.feature_dim, model.config.encoder.knn)
    data = transform_data(data, atom_composer)
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
    pbar.close()

    print_pool_status(pool, logger)
    logger.info('Saving samples...')
    torch.save(pool, os.path.join(log_dir, 'samples_init.pt'))

    # # Sampling loop
    logger.info('Start sampling')
    global_step = 0

    try:
        while len(pool.finished) < config.sample.num_samples:
            global_step += 1
            if global_step > config.sample.max_steps:
                break
            queue_size = len(pool.queue)
            # # sample candidate new mols from each parent mol
            queue_tmp = []
            queue_weight = []
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
                            logger.warning('Ignoring, because reconstruction error encountered.')
                            pool.failed.append(data_next)
                    elif data_next.status == STATUS_RUNNING:
                        nexts.append(data_next)

                queue_tmp += nexts
                if len(nexts) > 0:
                    queue_weight += [1. / len(nexts)] * len(nexts)
            # # random choose mols from candidates
            prob = logp_to_rank_prob(np.array([p.average_logp[2:] for p in queue_tmp]), queue_weight)  # (logp_focal, logpdf_pos), logp_element, logp_hasatom, logp_bond
            n_tmp = len(queue_tmp)
            next_idx = np.random.choice(np.arange(n_tmp), p=prob, size=min(config.sample.beam_size, n_tmp), replace=False)
            pool.queue = [queue_tmp[idx] for idx in next_idx]

            print_pool_status(pool, logger)
            torch.save(pool, os.path.join(log_dir, 'samples_%d.pt' % global_step))
    except KeyboardInterrupt:
        logger.info('Terminated. Generated molecules will be saved.')


    # # Save sdf mols
    sdf_dir = os.path.join(log_dir, 'SDF')
    os.makedirs(sdf_dir)
    with open(os.path.join(log_dir, 'SMILES.txt'), 'a') as smiles_f:
        for i, data_finished in enumerate(pool['finished']):
            smiles_f.write(data_finished.smiles + '\n')
            rdmol = data_finished.rdmol
            Chem.MolToMolFile(rdmol, os.path.join(sdf_dir, '%d.sdf' % i))
            
    torch.save(pool, os.path.join(log_dir, 'samples_all.pt'))
