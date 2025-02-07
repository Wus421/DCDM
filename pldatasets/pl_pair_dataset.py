import os
import pickle
import lmdb
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import traceback
from rdkit import Chem
from utils.data import PDBProtein, parse_sdf_file
from utils.parser import parse_conf_list
from .pl_data import ProteinLigandData, torchify_dict
import torch
import numpy as np
import pandas as pd

class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='final'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        # self.processed_path = os.path.join(os.path.dirname(self.raw_path),
        #                                    os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_processed_DCDM_{version}.lmdb')
        self.molid2idx_path = self.processed_path[:self.processed_path.find('.lmdb')]+'_molid2idx_DCDM.pt'
        self.transform = transform
        self.db = None
        self.atomic_numbers = [6,7,8,9,15,16,17]
        self.keys = None
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)
        # if not os.path.exists(self.processed_path):
        #     print(f'{self.processed_path} does not exist, begin processing data')
        #     self._process()
        if (not os.path.exists(self.processed_path)) or (not os.path.exists(self.molid2idx_path)):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
            self._precompute_molid2idx()
        self.molid2idx = torch.load(self.molid2idx_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=100*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, _, rmsd) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                try:
                    # data_prefix = '/data/work/jiaqi/binding_affinity'
                    df = pd.read_csv("/home/wus/MolDiff-targetdiff/traindata_duijie.csv") 
                    df1 = pd.read_csv("/home/wus/MolDiff-targetdiff/testdata_duijie.csv") 
                    # np.where(df['ligand_filename'].values=='FA11_HUMAN_388_625_0/4ty6_A_rec_1zhr_ben_lig_tt_docked_4.sdf')[0][0]
                    data_prefix = self.raw_path
                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    path = os.path.join(data_prefix, ligand_fn)
                    #from Moldiff
                    if path.endswith('.sdf'):
                        rdmol = Chem.MolFromMolFile(path, sanitize=False)
                    elif path.endswith('.mol2'):
                        rdmol = Chem.MolFromMol2File(path, sanitize=False)
                    else:
                        raise ValueError
                    Chem.SanitizeMol(rdmol)
                    # rdmol = Chem.RemoveHs(rdmol)
                    confs_list = []
                    confs_list.append(rdmol)
                    smiles = Chem.MolToSmiles(rdmol)
                    ligand_dict = parse_conf_list(confs_list, smiles=smiles)
                    #from endding
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    element = data.element
                    assert np.all([ele in self.atomic_numbers for ele in data.element]), 'unknown element'
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data.smiles = smiles
                    data.mol_id = i
                    data.rmsd = rmsd
                    data.vina_sorce = 1
                    # if len(np.where(df['ligand_filename'].values==data.ligand_filename)[0]) == 1:
                    #     data.vina_sorce = df.iloc[np.where(df['ligand_filename'].values==data.ligand_filename)[0][0]]['score_only']
                    # elif len(np.where(df1['ligand_filename'].values==data.ligand_filename)[0]) == 1:
                    #     data.vina_sorce = df1.iloc[np.where(df1['ligand_filename'].values==data.ligand_filename)[0][0]]['score_only']
                    # else:
                    #     data.vina_sorce = df1.iloc[np.where(df1['ligand_filename'].values==data.ligand_filename)[0][0]]['score_only']  #故意报错跳出
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except Exception as e:
                    # 打印完整的异常信息
                    traceback.print_exc()
                    if 'unknown element' in str(e):
                        print('unknown element is',smiles,element,self.atomic_numbers)
                    num_skipped += 1
                    print('Skipping (%d) %s ' % (num_skipped, ligand_fn))
                    continue
        db.close()
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        return data
    def _precompute_molid2idx(self):
        molid2idx = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except Exception as e:
                # 打印完整的异常信息
                print(i, e)
                traceback.print_exc()
                continue
            mol_id = data.mol_id
            molid2idx[mol_id] = i
        torch.save(molid2idx, self.molid2idx_path)
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    dataset = PocketLigandPairDataset(args.path)
    print(len(dataset), dataset[0])
