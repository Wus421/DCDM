# DCDM
![image](https://github.com/user-attachments/assets/c21e29fc-df4d-4bbf-baf8-3a49021f4494)

# Environment Installation
Before you run DCDM, please follow the instructions I've included below to build the correct operating environment.
```bash
conda create -n DCDM python=3.8
conda activate DCDM
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pyg -c pyg
conda install rdkit openbabel tensorboard pyyaml easydict python-lmdb -c conda-forge

# For Vina Docking
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```
# DATA
The data used for training / evaluating the model are organized in the data Google Drive folder.[data](https://drive.google.com/drive/folders/1lmiUz3hI-Z4XJxz31hky9WWBzP9BX3ca)

To train the model from scratch, you need to download the preprocessed lmdb file and split file:

- crossdocked_v1.1_rmsd1.0_pocket10_processed_DCDM_final.lmdb
- crossdocked_pocket10_pose_split_DCDM.pt

To evaluate the model on the test set, you need to download and unzip the test_set.zip. It includes the original PDB files that will be used in Vina Docking.

The raw data for CrossDocked2020 can be downloaded from [here](https://bits.csb.pitt.edu/files/crossdock2020/).

# Pretrained Models
We have uploaded the pre-trained models on [here](https://drive.google.com/drive/folders/1lmiUz3hI-Z4XJxz31hky9WWBzP9BX3ca). You can download and place them in the root directory.

There are three pre-trained models in total, namely:
- DCDM.pt
- affinity.pt
- bond_predictor.pt

# Training DCDM
Please run the following command to train a new DCDM from scratch:
```bash
python scripts/train_DCDM.py
```
Please run the following command to train a new affinity classification model from scratch:
```bash
python scripts/train_affinity.py
```
Please run the following command to train a new key predictor from scratch:
```bash
python scripts/train_bond.py
```
# Generating Molecules
Here, we provide three examples for molecule generation.
﻿
- To sample molecules based on the first protein from the test set in CrossDocked2020, run the following scripts.
```bash
python scripts/sample_DCDM.py  --data_id {i} # Replace {i} with the index of the data. i should be between 0 and 99 for the test set.
```
- To sample molecules based on a new protein  pocket(a 10A region around the reference ligand), run the following script.
```bash
python scripts/sample_pdb.py --pdb_path examples/***.pdb
```
- To optimize based on the given molecular skeleton, please run the following script.
```bash
python scripts/sample_pdb_frag.py --pdb_path examples/***.pdb --frag_path examples/***.sdf --keep_index 0 1 2 3 4 ...# Replace {keep_index} with the atomic index of the molecule. 
```
# Evaluation
To evaluate the generated molecules, run the following script.
```bash
python scripts/evaluate.py --exp_name sample_filename
```

The docking mode can be chosen from {qvina, vina_score, vina_dock, none}

Note: It will take some time to prepare pqdqt and pqr files when you run the evaluation code with vina_score/vina_dock docking mode for the first time.

# License
The code in this package is licensed under the MIT License. We thanks [TargetDiff](https://github.com/guanjq/targetdiff?tab=readme-ov-file) and [MolDiff](https://github.com/pengxingang/moldiff) for the open source codes.
