# import sys
# sys.path.append('.')
import os
import shutil
import argparse
from tqdm.auto import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
# import torch_geometric
# assert not torch_geometric.__version__.startswith('2'), 'Please use torch_geometric lower than version 2.0.0'
from torch_geometric.loader import DataLoader

from models.maskfill import MaskFillModelVN
from utils.datasets import *
from utils.transforms import *
from utils.misc import *
from utils.train import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)
    if config.train.use_apex:
        from apex import amp

    # Logging
    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))

    # Transforms
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    masking = get_mask(config.train.transform.mask)
    composer = AtomComposer(protein_featurizer.feature_dim, ligand_featurizer.feature_dim, config.model.encoder.knn)
    
    edge_sampler = EdgeSample(config.train.transform.edgesampler)
    cfg_ctr = config.train.transform.contrastive
    contrastive_sampler = ContrastiveSample(cfg_ctr.num_real, cfg_ctr.num_fake, cfg_ctr.pos_real_std, cfg_ctr.pos_fake_std, config.model.field.knn)
    transform = Compose([
        RefineData(),
        LigandCountNeighbors(),
        protein_featurizer,
        ligand_featurizer,
        masking,
        composer,

        FocalBuilder(),
        edge_sampler,
        contrastive_sampler,
    ])

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config = config.dataset,
        transform = transform,
    )
    train_set, val_set = subsets['train'], subsets['test']
    follow_batch = []
    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = inf_iterator(DataLoader(
        train_set, 
        batch_size = config.train.batch_size, 
        shuffle = True,
        num_workers = config.train.num_workers,
        pin_memory = config.train.pin_memory,
        follow_batch = follow_batch,
        exclude_keys = collate_exclude_keys,
    ))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False, follow_batch=follow_batch, exclude_keys = collate_exclude_keys,)

    # Model
    logger.info('Building model...')
    if config.model.vn == 'vn':
        model = MaskFillModelVN(
            config.model, 
            num_classes = contrastive_sampler.num_elements,
            num_bond_types = edge_sampler.num_bond_types,
            protein_atom_feature_dim = protein_featurizer.feature_dim,
            ligand_atom_feature_dim = ligand_featurizer.feature_dim,
        ).to(args.device)
    print('Num of parameters is', np.sum([p.numel() for p in model.parameters()]))

    # Optimizer and scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    if config.train.use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    def train(it):
        # model.train()  has been moved to the end of validation function
        optimizer.zero_grad()
        batch = next(train_iterator).to(args.device)

        compose_noise = torch.randn_like(batch.compose_pos) * config.train.pos_noise_std
        loss, loss_frontier, loss_pos, loss_cls, loss_edge, loss_real, loss_fake, loss_surf = model.get_loss(

            pos_real = batch.pos_real,
            y_real = batch.cls_real.long(),
            # p_real = batch.ind_real.float(),    # Binary indicators: float
            pos_fake = batch.pos_fake,

            edge_index_real = torch.stack([batch.real_compose_edge_index_0, batch.real_compose_edge_index_1], dim=0),
            edge_label = batch.real_compose_edge_type,
            
            index_real_cps_edge_for_atten = batch.index_real_cps_edge_for_atten,
            tri_edge_index = batch.tri_edge_index,
            tri_edge_feat = batch.tri_edge_feat,

            compose_feature = batch.compose_feature.float(),
            compose_pos = batch.compose_pos + compose_noise,
            idx_ligand = batch.idx_ligand_ctx_in_compose,
            idx_protein = batch.idx_protein_in_compose,

            y_frontier = batch.ligand_frontier,
            idx_focal = batch.idx_focal_in_compose,
            pos_generate=batch.pos_generate,
            idx_protein_all_mask = batch.idx_protein_all_mask,
            y_protein_frontier = batch.y_protein_frontier,

            compose_knn_edge_index = batch.compose_knn_edge_index,
            compose_knn_edge_feature = batch.compose_knn_edge_feature,
            real_compose_knn_edge_index = torch.stack([batch.real_compose_knn_edge_index_0, batch.real_compose_knn_edge_index_1], dim=0),
            fake_compose_knn_edge_index = torch.stack([batch.fake_compose_knn_edge_index_0, batch.fake_compose_knn_edge_index_1], dim=0),
        )
        if config.train.use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm, error_if_nonfinite=True)  # 5% running time
        optimizer.step()

        logger.info('[Train] Iter %d | Loss %.6f | Loss(Fron) %.6f | Loss(Pos) %.6f | Loss(Cls) %.6f | Loss(Edge) %.6f | Loss(Real) %.6f | Loss(Fake) %.6f | Loss(Surf) %.6f  ' % (
            it, loss.item(), loss_frontier.item(), loss_pos.item(), loss_cls.item(), loss_edge.item(), loss_real.item(), loss_fake.item(), loss_surf.item()
        ))
        writer.add_scalar('train/loss', loss, it)
        writer.add_scalar('train/loss_fron', loss_frontier, it)
        writer.add_scalar('train/loss_pos', loss_pos, it)
        writer.add_scalar('train/loss_cls', loss_cls, it)
        writer.add_scalar('train/loss_edge', loss_edge, it)
        writer.add_scalar('train/loss_real', loss_real, it)
        writer.add_scalar('train/loss_fake', loss_fake, it)
        writer.add_scalar('train/loss_surf', loss_surf, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad', orig_grad_norm, it)
        writer.flush()

    def validate(it):
        sum_loss, sum_n = np.zeros(5 + 2 + 1), 0   # num of loss
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                loss_list = model.get_loss(
                    pos_real = batch.pos_real,
                    y_real = batch.cls_real.long(),
                    pos_fake = batch.pos_fake,

                    edge_index_real = torch.stack([batch.real_compose_edge_index_0, batch.real_compose_edge_index_1], dim=0),
                    edge_label = batch.real_compose_edge_type,

                    index_real_cps_edge_for_atten = batch.index_real_cps_edge_for_atten,
                    tri_edge_index = batch.tri_edge_index,
                    tri_edge_feat = batch.tri_edge_feat,

                    compose_feature = batch.compose_feature.float(),
                    compose_pos = batch.compose_pos,
                    idx_ligand = batch.idx_ligand_ctx_in_compose,
                    idx_protein = batch.idx_protein_in_compose,
                    
                    y_frontier = batch.ligand_frontier,
                    idx_focal = batch.idx_focal_in_compose,
                    pos_generate = batch.pos_generate,
                    idx_protein_all_mask = batch.idx_protein_all_mask,
                    y_protein_frontier = batch.y_protein_frontier,

                    compose_knn_edge_index = batch.compose_knn_edge_index,
                    compose_knn_edge_feature = batch.compose_knn_edge_feature,
                    real_compose_knn_edge_index = torch.stack([batch.real_compose_knn_edge_index_0, batch.real_compose_knn_edge_index_1], dim=0),
                    fake_compose_knn_edge_index = torch.stack([batch.fake_compose_knn_edge_index_0, batch.fake_compose_knn_edge_index_1], dim=0),
                )
                sum_loss = sum_loss + np.array([torch.nan_to_num(l).item() for l in loss_list]) 
                sum_n += 1
        avg_loss = sum_loss / sum_n

        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss[0])
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss[0])
        else:
            scheduler.step()

        logger.info('[Validate]  Iter %d | Loss %.6f | Loss(Fron) %.6f | Loss(Pos) %.6f | Loss(Cls) %.6f | Loss(Edge) %.6f | Loss(Real) %.6f | Loss(Fake) %.6f  | Loss(Surf) %.6f' % (
            it, *avg_loss,
        ))
        writer.add_scalar('val/loss', avg_loss[0], it)
        writer.add_scalar('val/loss_fron', avg_loss[1], it)
        writer.add_scalar('val/loss_pos', avg_loss[2], it)
        writer.add_scalar('val/loss_cls', avg_loss[3], it)
        writer.add_scalar('val/loss_edge', avg_loss[4], it)
        writer.add_scalar('val/loss_real', avg_loss[5], it)
        writer.add_scalar('val/loss_fake', avg_loss[6], it)
        writer.add_scalar('val/loss_surf', avg_loss[7], it)
        writer.flush()
        return avg_loss

    try:
        model.train()
        for it in range(1, config.train.max_iters+1):
            try:
                train(it)
            except RuntimeError as e:
                logger.error('Runtime Error ' + str(e))
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                validate(it)
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)
                model.train()
    except KeyboardInterrupt:
        logger.info('Terminating...')
        
