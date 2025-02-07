import sys
import os
import argparse
import pandas as pd
import pickle
from glob import glob
from tqdm.auto import tqdm
import traceback
sys.path.append('.')

from utils.reconstruct import *
from utils.misc import *
from utils.scoring_func import *
from utils.evaluations import *
from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask
from easydict import EasyDict
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 
from pldatasets import get_dataset

cfg_dataset = EasyDict({
        'name': 'pl',
        'root': 'data/',
        'path': 'data/crossdocked_v1.1_rmsd1.0_pocket10/',
        'path_dict':{
            'sdf': 'sdf',
            'summary': 'mol_summary.csv',
            'processed': 'processed.lmdb',},
        'split': 'data/crossdocked_pocket10_pose_split_new.pt',
        'train_smiles': 'train_smiles.pt',
        'train_finger': 'train_finger.pkl',
        'val_smiles': 'val_smiles.pt',
        'val_finger': 'val_finger.pkl',
    })
def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')

def load_mols_from_dataset(dataset_type):
    dataset, subsets = get_dataset(cfg_dataset)
    
    # load sdf
    subset = subsets[dataset_type]
    mol_list = []
    n_all = 0
    n_failed = 0
    failed_list = []
    for idx_data, data in tqdm(enumerate(subset), total=len(subset)):
        data.atom_pos = data.pos_all_confs[0]
        try:
            n_all += 1
            mol = reconstruct_from_generated_with_edges(data)
        except MolReconsError:
            failed_list.append(idx_data)
            n_failed += 1
            continue

        mol_list.append(mol)
    print('Load dataset', dataset_type, 'all:', n_all, 'failed:', n_failed)
    mol_dict = {idx: mol for idx, mol in enumerate(mol_list)}
    
    metrics_dir = os.path.join(cfg_dataset.root, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    df_path = os.path.join(metrics_dir, dataset_type+'.csv')
    if os.path.exists(df_path):
        df = pd.read_csv(df_path, index_col=0)
    else:
        df = pd.DataFrame(index=mol_dict.keys())
        
    return mol_dict, df, metrics_dir, df_path


def load_mols_from_generated(exp_name, result_root):
    # prepare data path
    all_exp_paths = os.listdir(result_root)
    sdf_dir = [path for path in all_exp_paths
                      if (path.startswith(exp_name) and path.endswith('_SDF'))]
    print(sdf_dir)
    assert len(sdf_dir) == 1, f'Found more than one or none sdf directory of sampling with prefix `{exp_name}` and suffix `_SDF` in {result_root}: {sdf_dir}'
    sdf_dir = sdf_dir[0]
    
    sdf_dir = os.path.join(args.result_root, sdf_dir)
    metrics_dir = sdf_dir.replace('_SDF', '')
    df_path = os.path.join(metrics_dir, 'mols.csv')
    mol_names = [mol_name for mol_name in os.listdir(sdf_dir) if (mol_name[-4:] == '.sdf') and ('traj' not in mol_name) ]
    mol_ids = np.sort([int(mol_name[:-4]) for mol_name in mol_names])
        
    # load sdfs
    mol_dict_raw = {mol_id:Chem.MolFromMolFile(os.path.join(sdf_dir, '%d.sdf' % mol_id))
                for mol_id in mol_ids}
    mol_dict = {mol_id:mol for mol_id, mol in mol_dict_raw.items() if mol is not None}
    print('Load success:', len(mol_dict), 'failed:', len(mol_dict_raw)-len(mol_dict))

    # load df
    if os.path.exists(df_path):
        df = pd.read_csv(df_path, index_col=0)
    else:
        df = pd.DataFrame(index=list(mol_dict.keys()))
        df.index.name = 'mol_id'
    return mol_dict, df, metrics_dir, df_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # generated
    parser.add_argument('--from_where', type=str, default='generated',
                        help='be `generated` or `dataset`')
    parser.add_argument('--exp_name', type=str, default='sample_MolDiff_20230602',
                        help='For `generated`, it is the name of the config file of the sampling experiment (e.g., sample_MolDiff)'
                        'For `dataset`, it is one of train/val/test')
    parser.add_argument('--result_root', type=str, default='./outputs',
                        help='The root directory of the generated data and sdf files.')
    # from tardiff
    parser.add_argument('--protein_root', type=str, default='/home/wus/targetdiff/data/test_set')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, choices=['qvina', 'vina_score', 'vina_dock', 'none'],default='vina_dock')
    parser.add_argument('--exhaustiveness', type=int, default=16)
    
    args = parser.parse_args()
    from_where = args.from_where
    metrics_list = [
        'drug_chem',  # qed, sa, logp, lipinski
        'count_prop',  # n_atoms, n_bonds, n_rings, n_rotatable, weight, n_hacc, n_hdon
        'global_3d',  # rmsd_max, rmsd_min, rmsd_median
        'frags_counts',  # cnt_eleX, cnt_bondX, cnt_ringX(size)
        'local_3d',  # bond length, bond angle, dihedral angle
        'similarity', # sim_with_train, uniqueness, diversity
        'ring_type', # cnt_ring_type_{x}, top_n_freq_ring_type
        'targetdiff', # from targetdiff evaluate--main vina
    ]
    if from_where == 'dataset':
        dataset_type = args.exp_name
        mol_dict, df, metrics_dir, df_path = load_mols_from_dataset(dataset_type)
        logger = get_logger('eval_'+dataset_type, metrics_dir)
        metrics_list = ['count_prop', 'frags_counts', 'local_3d', 'ring_type']
    elif from_where == 'generated':
        exp_name = args.exp_name
        mol_dict, df, metrics_dir, df_path = load_mols_from_generated(exp_name, args.result_root)
        logger = get_logger('eval_'+exp_name, metrics_dir)

    # Load generated data
    results_fn_list = glob(os.path.join(os.path.join(args.result_root, args.exp_name), '*result_*.pt'))
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    # if args.eval_num_examples is not None:
    #     results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list)
    logger.info(f'Load generated data done! {num_examples} examples in total.')


    for metric_name in metrics_list:
        logger.info(f'Computing {metric_name} metrics...')
        if metric_name in ['drug_chem', 'count_prop', 'global_3d', 'frags_counts', 'groups_counts']:
            parallel =True
            results_list = get_metric(mol_dict.values(), metric_name, parallel=parallel)
            if list(results_list[0].keys())[0] not in df.columns:
                df = pd.concat([df, pd.DataFrame(results_list, index=mol_dict.keys())], axis=1)
            else:
                df.loc[mol_dict.keys(), results_list[0].keys()] = pd.DataFrame(
                    results_list, index=mol_dict.keys())
            df.to_csv(df_path)
        elif metric_name == 'local_3d':
            local3d = Local3D()
            local3d.get_predefined()
            logger.info(f'Computing local 3d - bond lengths metric...')
            lengths = local3d.calc_frequent(mol_dict.values(), type_='length', parallel=False)
            logger.info(f'Computing local 3d - bond angles metric...')
            angles = local3d.calc_frequent(mol_dict.values(), type_='angle', parallel=False)
            logger.info(f'Computing local 3d - dihedral angles metric...')
            dihedral = local3d.calc_frequent(mol_dict.values(), type_='dihedral', parallel=False)
            save_path = df_path.replace('.csv', '_local3d.pkl')
            local3d = {'lengths': lengths, 'angles': angles, 'dihedral': dihedral}
            with open(save_path, 'wb') as f:
                f.write(pickle.dumps(local3d))
        elif metric_name == 'similarity':
            for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval for every protein')):
                print(r_name)
                r = torch.load(r_name)
                path = os.path.join(cfg_dataset.path, r['data'].ligand_filename)
                sim = SimilarityAnalysis_Protein(path)
                uniqueness = sim.get_novelty_and_uniqueness(mol_dict.values())
                diversity = sim.get_diversity(mol_dict.values())
                uniqueness['diversity'] = diversity
                sim_with_val = sim.get_sim_with_val(mol_dict.values())
                uniqueness['sim_with_val'], similarity_list = sim_with_val
                sim_with_val_ecfp4 = sim.get_sim_with_val_ecfp4(mol_dict.values())
                uniqueness['sim_with_val_ecfp4'], similarity_ecfp4_list = sim_with_val_ecfp4
                # print(similarity_ecfp4_list)
                save_path = df_path.replace('.csv', '_similarity.pkl')
                if list(similarity_list[0].keys())[0] not in df.columns:
                    df = pd.concat([df, pd.DataFrame(similarity_list, index=mol_dict.keys())], axis=1)
                else:
                    df.loc[mol_dict.keys(), similarity_list[0].keys()] = pd.DataFrame(
                        similarity_list, index=mol_dict.keys())
                if list(similarity_ecfp4_list[0].keys())[0] not in df.columns:
                    df = pd.concat([df, pd.DataFrame(similarity_ecfp4_list, index=mol_dict.keys())], axis=1)
                else:
                    df.loc[mol_dict.keys(), similarity_ecfp4_list[0].keys()] = pd.DataFrame(
                        similarity_ecfp4_list, index=mol_dict.keys())
                df.to_csv(df_path)
                with open(save_path, 'wb') as f:
                    f.write(pickle.dumps(uniqueness))
                logger.info(f'Similarity : {uniqueness}')
        elif metric_name == 'ring_type':
            ring_analyzer = RingAnalyzer()
            # cnt of ring type (common in val set)
            # cnt_ring_type = ring_analyzer.get_count_ring(mol_dict.values())
            # if list(cnt_ring_type.keys())[0] not in df.columns:
            #     df = pd.concat([df, pd.DataFrame(cnt_ring_type, index=mol_dict.keys())], axis=1)
            # else:
            #     df.loc[mol_dict.keys(), cnt_ring_type.keys()] = pd.DataFrame(
            #         cnt_ring_type, index=mol_dict.keys())
            # df.to_csv(df_path)
            # top n freq ring type
            freq_dict = ring_analyzer.get_freq_rings(mol_dict.values())
            with open(df_path.replace('.csv', '_freq_ring_type.pkl'), 'wb') as f:
                f.write(pickle.dumps(freq_dict))
        elif metric_name == 'targetdiff':
            validity_dict_all = []
            vina_results_all = []
            vina_results_pose = []
            for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval for every protein')):
                r = torch.load(r_name)
                all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
                all_atom_types = Counter()
                all_pair_dist, all_bond_dist = [], []
                success_pair_dist, success_atom_types = [], Counter()
                for sample_idx, mol in enumerate(tqdm(mol_dict.values(), desc='Eval targetdiff mol')):
                    conf = mol.GetConformer()
                    pos_list = []
                    ele_list = []
                    for i, atom in enumerate(mol.GetAtoms()):
                        pos = conf.GetAtomPosition(i)
                        ele = atom.GetAtomicNum()
                        pos_list.append(list(pos))
                        ele_list.append(ele)
                    # stability check
                    pos_list = np.array(pos_list)
                    ele_list = np.array(ele_list)
                    # pred_atom_type = transforms.get_atomic_number_from_index(ele_list, mode=args.atom_enc_mode)
                    # all_atom_types += Counter(pred_atom_type)
                    r_stable = analyze.check_stability(pos_list, ele_list , debug=True)
                    all_mol_stable += r_stable[0]
                    all_atom_stable += r_stable[1]
                    all_n_atom += r_stable[2]
                    # print(r_stable[0],r_stable[1],r_stable[2])
                    pair_dist = eval_bond_length.pair_distance_from_pos_v(pos_list, ele_list)
                    success_pair_dist += pair_dist
                    success_atom_types += Counter(ele_list)
                    bond_dist = eval_bond_length.bond_distance_from_mol(mol)
                    all_bond_dist += bond_dist
                    validity_dict = {
                    'mol_stable': r_stable[0],
                    'atm_stable': r_stable[1] / r_stable[2],
                    }
                    # print_dict(validity_dict, logger)
                    validity_dict_all.append(validity_dict)
                    # chemical and docking check
                    try:
                        chem_results = scoring_func.get_chem(mol)
                        if args.docking_mode == 'qvina':
                            vina_task = QVinaDockingTask.from_generated_mol(
                                mol, r['data'].ligand_filename, protein_root=args.protein_root)
                            vina_results = vina_task.run_sync()
                            qvina_results = {
                                'Best affinity': vina_results
                            }
                            print_dict(qvina_results, logger)
                            vina_results_all.append(qvina_results)
                        elif args.docking_mode in ['vina_score', 'vina_dock']:
                            vina_task = VinaDockingTask.from_generated_mol(
                                mol, r['data'].ligand_filename, protein_root=args.protein_root)
                            score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
                            minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                            vina_results = {
                                'score_only': score_only_results[0]['affinity'],
                                'minimize': minimize_results[0]['affinity']
                            }                            
                            if args.docking_mode == 'vina_dock':
                                docking_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
                                vina_results['vina_dock'] = docking_results[0]['affinity']
                                # vina_results_pose.append(docking_results[0]['pose'])
                            print_dict(vina_results, logger)
                            vina_results_all.append(vina_results)
                        else:
                            vina_results = {'no vina': 'None'}
                            print_dict(vina_results, logger)
                            vina_results_all.append(vina_results)

                    except Exception as e:
                        traceback.print_exc()
                        logger.warning('Evaluation failed for %s' % f'{example_idx}_{sample_idx}')
                        if args.docking_mode == 'qvina':
                            vina_results = {'Best affinity': 'faild'}
                            vina_results_all.append(vina_results)
                        elif args.docking_mode in ['vina_score', 'vina_dock']:
                            vina_results = {'score_only': 'faild',
                                'minimize': 'faild'}
                            vina_results_all.append(vina_results)
                        else:
                            vina_results = {'no vina': 'faild'}
                            vina_results_all.append(vina_results)
                        continue
                parallel =True
                if list(vina_results_all[0].keys())[0] not in df.columns:
                    df = pd.concat([df, pd.DataFrame(vina_results_all, index=mol_dict.keys())], axis=1)
                else:
                    df.loc[mol_dict.keys(), vina_results_all[0].keys()] = pd.DataFrame(
                        vina_results_all, index=mol_dict.keys())
                df.to_csv(df_path)
                if list(validity_dict_all[0].keys())[0] not in df.columns:
                    df = pd.concat([df, pd.DataFrame(validity_dict_all, index=mol_dict.keys())], axis=1)
                else:
                    df.loc[mol_dict.keys(), validity_dict_all[0].keys()] = pd.DataFrame(
                        validity_dict_all, index=mol_dict.keys())
                df.to_csv(df_path)
                c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
                c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
                logger.info('JS bond distances of complete mols: ') 
                print_dict(c_bond_length_dict, logger)
                success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
                success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
                print_dict(success_js_metrics, logger)
            
            
    logger.info(f'Saving metrics to {df_path}')
    logger.info(f'Done.')

    