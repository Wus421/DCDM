import os
import sys
import shutil
import argparse
from torch_geometric.data import Batch

sys.path.append('.')
from pldatasets.pl_data import FOLLOW_BATCH
import torch
import numpy as np
import torch.utils.tensorboard
from torch.distributions.categorical import Categorical
from easydict import EasyDict
from rdkit import Chem
from models.model import DCDM
from models.bond_predictor import BondPredictor
from utils.sample import seperate_outputs, seperate_outputs_no_traj
from utils.transforms import *
import utils.target_transforms as trans
from utils.misc import *
from utils.reconstruct import *
from pldatasets.pl_data import torchify_dict
from utils.data import PDBProtein
from utils.parser import parse_conf_list
def print_pool_status(pool, logger):
    logger.info('[Pool] Finished %d | Failed %d' % (
        len(pool.finished), len(pool.failed)
    ))


def data_exists(data, prevs):
    for other in prevs:
        if len(data.logp_history) == len(other.logp_history):
            if (data.ligand_context_element == other.ligand_context_element).all().item() and \
                (data.ligand_context_feature_full == other.ligand_context_feature_full).all().item() and \
                torch.allclose(data.ligand_context_pos, other.ligand_context_pos):
                return True
    return False


def pdb_to_pocket_data(pdb_path,ligand_dict):
    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=ligand_dict
    )

    return data


# from PMDM:utils/misc.py
def get_adj_matrix(n_particles):
    rows, cols = [], []
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            rows.append(i)
            cols.append(j)
            rows.append(j)
            cols.append(i)
    rows = torch.LongTensor(rows).unsqueeze(0)
    cols = torch.LongTensor(cols).unsqueeze(0)
    adj = torch.cat([rows, cols], dim=0)
    return adj

# from PMDM:utils/sample.py
class DistributionNodes:
    def __init__(self, histogram):
        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob / np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/sample/sample_DCDM.yml')
    parser.add_argument('--pdb_path', type=str, default='pnd.pdb')
    parser.add_argument('--frag_path', type=str, default='pnd.sdf')
    parser.add_argument('--x_id', type=int, default=0)
    parser.add_argument('--keep_index', nargs='+', type=int)
    parser.add_argument('--min_num_atom', type=int,
                        default=8)
    parser.add_argument('--outdir', type=str, default='./outputs_frag_X')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=0)
    args = parser.parse_args()

    # # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.sample.seed + np.sum([ord(s) for s in args.outdir]))
    # load ckpt and train config
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    train_config = ckpt['config']
    affinity_ckpt = torch.load(config.affinity, map_location=args.device)
    affinity_config = affinity_ckpt['config']

    # # Logging
    log_root = args.outdir.replace('outputs', 'outputs_vscode') if sys.argv[0].startswith('/data') else args.outdir
    log_dir = get_new_log_dir(log_root, prefix=config_name)
    logger = get_logger('sample', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    # Transforms
    # from targetdiff
    protein_featurizer = trans.FeaturizeProteinAtom()
    # ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    ligand_featurizer = FeaturizeMol(train_config.chem.atomic_numbers, train_config.chem.mol_bond_types,
                            use_mask_node=train_config.transform.use_mask_node,
                            use_mask_edge=train_config.transform.use_mask_edge
                            )
    transform_list = [
        protein_featurizer,
        ligand_featurizer
    ]
    if train_config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)
    
    max_size = None
    add_edge = getattr(config.sample, 'add_edge', None)
    
    # # Model
    logger.info('Loading diffusion model...')
    if train_config.model.name == 'diffusion':
    # from targetdiff
        model = DCDM(
            train_config.model,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            num_node_types=ligand_featurizer.num_node_types,
            num_edge_types=ligand_featurizer.num_edge_types
        ).to(args.device)
        affinity_model = DCDM(
            affinity_config.model,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            num_node_types=ligand_featurizer.num_node_types,
            num_edge_types=ligand_featurizer.num_edge_types
        ).to(args.device)
    else:
        raise NotImplementedError
    model.load_state_dict(ckpt['model'])
    model.eval()
    affinity_model.load_state_dict(affinity_ckpt['model'])
    affinity_model.eval()
    
    # # Bond predictor adn guidance
    if 'bond_predictor' in config:
        logger.info('Building bond predictor...')
        ckpt_bond = torch.load(config.bond_predictor, map_location=args.device)
        bond_predictor = BondPredictor(ckpt_bond['config']['model'],
                ligand_featurizer.num_node_types,
                ligand_featurizer.num_edge_types-1 # note: bond_predictor not use edge mask
        ).to(args.device)
        bond_predictor.load_state_dict(ckpt_bond['model'])
        bond_predictor.eval()
    else:
        bond_predictor = None
    if 'guidance' in config.sample:
        guidance = config.sample.guidance  # tuple: (guidance_type[entropy/uncertainty], guidance_scale)
    else:
        guidance = None


    pool = EasyDict({
        'failed': [],
        'finished': [],
    })
    
    # 加载分子片段
    if args.frag_path.endswith('.sdf'):
        rdmol = Chem.MolFromMolFile(args.frag_path, sanitize=False)
    elif args.frag_path.endswith('.mol2'):
        rdmol = Chem.MolFromMol2File(args.frag_path, sanitize=False)
    else:
        raise ValueError
    Chem.SanitizeMol(rdmol)
    rdmol = Chem.RemoveHs(rdmol)
    confs_list = []
    confs_list.append(rdmol)
    smiles = Chem.MolToSmiles(rdmol)
    ligand_dict = parse_conf_list(confs_list, smiles=smiles)
    # from PMDM:PMDM-main/sample_frag.py
    save_results=False
    valid = 0
    stable = 0
    high_affinity=0.0
    num_samples = config.sample.num_mols
    batch_size = config.sample.batch_size
    num_points = args.min_num_atom #random.randint(10,30)
    num_for_pdb = num_points
    context=None
    smile_list = []
    results = []
    protein_files = []
    sa_list = []
    qed_list=[]
    logP_list = []
    Lipinski_list = []
    vina_score_list = []
    rd_vina_score_list = []
    mol_list = []

    start_linker = torchify_dict(ligand_dict)
    # print(start_linker)
    # start_linker['node_type'] = torch.LongTensor([self.ele_to_nodetype[ele.item()] for ele in data.element])
    # start_linker = transform_ligand(start_linker)
    # print(start_linker)
    atomic_numbers = torch.LongTensor([6,7,8,9,15,16,17])
    start_linker['linker_atom_type'] = start_linker['element'].view(-1, 1) == atomic_numbers.view(1, -1)
    # important: define your own keep index
    # keep_index = torch.tensor([29,10,11])
    keep_index = torch.tensor(args.keep_index)
    start_linker['element'] = torch.index_select(start_linker['element'], 0, keep_index)
    # start_linker['atom_feature'] = torch.index_select(start_linker['atom_feature'], 0, keep_index)
    start_linker['linker_atom_type'] = torch.index_select(start_linker['linker_atom_type'], 0, keep_index)
    start_linker['pos'] = torch.index_select(start_linker['pos_all_confs'][0], 0, keep_index)
    # protein_atom_feature = data.protein_atom_feature_full.float()
    # # if 'pocket' in args.ckpt:
    # #     protein_atom_feature = data.protein_atom_feature.float()
    # protein_atom_feature_full = data.protein_atom_feature_full.float()
    # data_list,_ = construct_dataset_pocket(num_samples*1,batch_size,dataset_info,num_points,num_points,start_linker,None,
    # protein_atom_feature,protein_atom_feature_full,data.protein_pos,data.protein_bond_index)

    # # from crossdock_with_pocket
    # n_nodes = {27: 5423, 20: 4887, 28: 4840, 26: 4717, 25: 4537, 21: 4500, 23: 4379, 24: 3716, 30: 3605,
    #         31: 3371, 22: 3346, 32: 3336, 29: 3245, 16: 3098, 19: 3095, 11: 3055, 12: 2902, 10: 2866,
    #         33: 2689, 18: 2624, 14: 2471, 17: 2450, 15: 2297, 13: 2168, 34: 2168, 35: 1881, 9: 1860,
    #         36: 1388, 8: 1043, 37: 931, 38: 916, 39: 752, 6: 709, 40: 691, 44: 691, 7: 655, 41: 630,
    #         42: 504, 43: 360, 46: 213, 48: 187, 45: 185, 47: 119, 5: 111, 54: 93, 49: 73, 4: 45,
    #         52: 42, 50: 38, 55: 35, 67: 28, 51: 21, 66: 20, 58: 18, 56: 12, 57: 9, 61: 9, 59: 5, 63: 5,
    #         3: 4, 53: 3, 65: 2, 106: 2, 71: 1, 62: 1, 86: 1, 77: 1, 68: 1, 98: 1}
    
    # nodes_dist = DistributionNodes(n_nodes)
    # print(nodes_dist.sample(batch_size).tolist())
    # data_list = []

    
    # atom_index={ 6: 0, 7: 1, 8: 2, 9: 3, 15: 4, 16: 5, 17: 6}
    # atom_encoder={'C': 0, 'N': 1, 'O': 2, 'F': 3, 'P': 4, 'S': 5, 'Cl': 6}
    # atom_decoder=['C', 'N', 'O', 'F', 'P', 'S', 'Cl']
    # num_atom = len(atom_decoder)  ## +1 if charge
    # # [6,7,8,9,15,16,17]  # C N O F P S Cl
    # # print('num_atom',num_atom)
    # # num_atom = 20
    
    
    
    
    while len(pool.finished) < config.sample.num_mols:
        if len(pool.failed) > 3 * (config.sample.num_mols):
            logger.info('Too many failed molecules. Stop sampling.')
            break
        
        # prepare batch
        batch_size = args.batch_size if args.batch_size > 0 else config.sample.batch_size
        n_graphs = min(batch_size, (config.sample.num_mols - len(pool.finished))*2)
        # data = data
        data = pdb_to_pocket_data(args.pdb_path,start_linker)
        data = transform(data)
        print(data.node_type)
        # print(data.halfedge_type[0:data.halfedge_index[0][data.halfedge_index[0]==0].size(0)])
        batch = Batch.from_data_list([data.clone() for _ in range(n_graphs)], follow_batch=FOLLOW_BATCH).to(args.device)
        batch_holder = make_data_placeholder_frag_x(n_graphs=n_graphs, min_num_atom=args.min_num_atom, start_linker=start_linker, data=data, device=args.device, max_size=max_size, x_id=args.x_id)
        frag_mask_batch_node, node_type_batch, node_pos_batch, batch_node, halfedge_index, batch_halfedge, halfedge_type_batch, frag_mask_batch_edge = batch_holder['frag_mask_batch_node'], batch_holder['node_type_batch'], batch_holder['node_pos_batch'], batch_holder['batch_node'], batch_holder['halfedge_index'], batch_holder['batch_halfedge'], batch_holder['halfedge_type_batch'], batch_holder['frag_mask_batch_edge']

        # inference
        outputs = model.sample_diffusion_frag_x(
            n_graphs=n_graphs,
            batch_node=batch_node,
            frag_mask_batch_node = frag_mask_batch_node,
            node_type_batch = node_type_batch, 
            node_pos_batch = node_pos_batch,
            halfedge_index=halfedge_index,
            batch_halfedge=batch_halfedge,
            halfedge_type_batch = halfedge_type_batch,
            frag_mask_batch_edge = frag_mask_batch_edge,
            bond_predictor=bond_predictor,
            guidance=guidance,
            protein_pos=batch.protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch.protein_element_batch,
            x_id=args.x_id,
            num_steps=config.sample.num_steps,
            pos_only=config.sample.pos_only,
            center_pos_mode=config.sample.center_pos_mode,
            affinity_model=affinity_model
        )
        outputs = {key:[v.cpu().numpy() for v in value] for key, value in outputs.items()}
        # decode outputs to molecules
        batch_node, halfedge_index, batch_halfedge = batch_node.cpu().numpy(), halfedge_index.cpu().numpy(), batch_halfedge.cpu().numpy()
        # output_list = seperate_outputs_no_traj(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge)
        try:
            output_list = seperate_outputs_no_traj(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge)
        except:
            continue
        gen_list = []
        for i_mol, output_mol in enumerate(output_list):
            mol_info = ligand_featurizer.decode_output(
                pred_node=output_mol['pred'][0],
                pred_pos=output_mol['pred'][1],
                pred_halfedge=output_mol['pred'][2],
                halfedge_index=output_mol['halfedge_index'],
            )  # note: traj is not used
            # mol_info = ligand_featurizer.decode_output(
            #     pred_node=output_mol[0],
            #     pred_pos=output_mol[1],
            #     pred_halfedge=output_mol[2],
            #     halfedge_index=output_mol['halfedge_index'],
            # )  # note: traj is not used
            # print(data)
            
            pred_halfedge = softmax(output_mol['pred'][2], axis=-1)
            edge_type = np.argmax(pred_halfedge, axis=-1)  # omit half for simplicity
            edge_type = torch.tensor(edge_type).to(frag_mask_batch_edge.device)
            # print(batch_halfedge,frag_mask_batch_edge)
            # print(frag_mask_batch_edge[batch_halfedge==0])
            # print(edge_type[frag_mask_batch_edge[batch_halfedge==0]].shape,data.halfedge_type.shape)
            # print(output_mol['pred'][2].shape)
            # print(edge_type.shape,frag_mask_batch_edge.shape,batch_halfedge.shape)
            # exit()
            # print(edge_type[frag_mask_batch_edge[batch_haslfedge==0]][-45:])
            # sdf_faild_dir = log_dir + '_faildSDF'
            # os.makedirs(sdf_faild_dir, exist_ok=True)
            try:
                rdmol = reconstruct_from_generated_with_frag(mol_info,add_edge=add_edge)
            except MolReconsError:
                pool.failed.append(mol_info)
                logger.warning('Reconstruction error encountered.')
                continue
            mol_info['rdmol'] = rdmol
            smiles = Chem.MolToSmiles(rdmol)
            mol_info['smiles'] = smiles
            if '.' in smiles:
                logger.warning('Incomplete molecule: %s' % smiles)
                pool.failed.append(mol_info)
                # Chem.MolToMolFile(mol_info['rdmol'], os.path.join(sdf_faild_dir, '%d.sdf' % len(pool.failed)))
            else:   # Pass checks!
                logger.info('Success: %s' % smiles)
                p_save_traj = np.random.rand()  # save traj
                if p_save_traj <  config.sample.save_traj_prob:
                    traj_info = [ligand_featurizer.decode_output(
                        pred_node=output_mol['traj'][0][t],
                        pred_pos=output_mol['traj'][1][t],
                        pred_halfedge=output_mol['traj'][2][t],
                        halfedge_index=output_mol['halfedge_index'],
                    ) for t in range(len(output_mol['traj'][0]))]
                    mol_traj = []
                    for t in range(len(traj_info)):
                        try:
                            mol_traj.append(reconstruct_from_generated_with_edges(traj_info[t], False, add_edge=add_edge))
                        except MolReconsError:
                            mol_traj.append(Chem.MolFromSmiles('O'))
                    mol_info['traj'] = mol_traj
                gen_list.append(mol_info)
                # pool.finished.append(mol_info)

        # # Save sdf mols
        sdf_dir = log_dir + '_SDF'
        os.makedirs(sdf_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'SMILES.txt'), 'a') as smiles_f:
            for i, data_finished in enumerate(gen_list):
                smiles_f.write(data_finished['smiles'] + '\n')
                rdmol = data_finished['rdmol']
                Chem.MolToMolFile(rdmol, os.path.join(sdf_dir, '%d.sdf' % (i+len(pool.finished))))

                if 'traj' in data_finished:
                    with Chem.SDWriter(os.path.join(sdf_dir, 'traj_%d.sdf' % (i+len(pool.finished)))) as w:
                        for m in data_finished['traj']:
                            try:
                                w.write(m)
                            except:
                                w.write(Chem.MolFromSmiles('O'))
        pool.finished.extend(gen_list)
        print_pool_status(pool, logger)

    torch.save(pool, os.path.join(log_dir, 'samples_all.pt'))
    print("结束："+str(log_dir))