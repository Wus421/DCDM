import torch
import numpy as np
from torch_geometric.transforms import Compose
from scipy.special import softmax
from pldatasets.pl_data import ProteinLigandData
from models.transition import *
from utils.misc import *
from utils.evaluation import atom_num


class FeaturizeMol(object):
    def __init__(self, atomic_numbers, mol_bond_types,
                 use_mask_node, use_mask_edge):
        super().__init__()
        self.atomic_numbers = torch.LongTensor(atomic_numbers)
        self.mol_bond_types = torch.LongTensor(mol_bond_types)
        self.num_element = self.atomic_numbers.size(0)
        self.num_bond_types = self.mol_bond_types.size(0)

        self.num_node_types = self.num_element + int(use_mask_node)
        self.num_edge_types = self.num_bond_types + 1 + int(use_mask_edge) # + 1 for the non-bonded edges
        self.use_mask_node = use_mask_node
        self.use_mask_edge = use_mask_edge
        
        self.ele_to_nodetype = {ele: i for i, ele in enumerate(atomic_numbers)}
        self.nodetype_to_ele = {i: ele for i, ele in enumerate(atomic_numbers)}
        
        
        self.follow_batch = ['protein_element','node_type', 'halfedge_type']
        self.exclude_keys = ['orig_keys', 'pos_all_confs', 'smiles', 'num_confs', 'i_conf_list'
                             'bond_index', 'bond_type', 'num_bonds', 'num_atoms','nbh_list']
    
    def __call__(self, data: ProteinLigandData):
        
        data.num_nodes = data.num_atoms
        
        # node type
        assert np.all([ele in self.atomic_numbers for ele in data.element]), 'unknown element'
        data.node_type = torch.LongTensor([self.ele_to_nodetype[ele.item()] for ele in data.element])
        
        # atom pos: sample a conformer from data.pos_all_confs; then move to origin-------atom-pos：从data.pos_all_confs中采样一个conformer；然后移动到原点,这里不需要移动到原点，因为会根据蛋白质的位置进行移动。
        idx = np.random.randint(data.pos_all_confs.shape[0])
        atom_pos = data.pos_all_confs[idx].float()
        # atom_pos = atom_pos - atom_pos.mean(dim=0)

        data.node_pos = atom_pos
        data.i_conf = data.i_conf_list[idx]
        
        # build half edge (not full because perturb for edge_ij should be the same as edge_ji)构建半边缘（未满，因为edge_ij的扰动应与edge_ji相同）
        edge_type_mat = torch.zeros([data.num_nodes, data.num_nodes], dtype=torch.long)
        for i in range(data.num_bonds * 2):  # multiplication by two is for symmtric of bond index
            edge_type_mat[data.bond_index[0, i], data.bond_index[1, i]] = data.bond_type[i]
        halfedge_index = torch.triu_indices(data.num_nodes, data.num_nodes, offset=1)
        halfedge_type = edge_type_mat[halfedge_index[0], halfedge_index[1]]
        assert len(halfedge_type) == len(halfedge_index[0])
        
        data.halfedge_index = halfedge_index
        data.halfedge_type = halfedge_type
        assert (data.halfedge_type > 0).sum() == data.num_bonds
        
        return data
    
    def decode_output(self, pred_node, pred_pos, pred_halfedge, halfedge_index):
        """
        Get the atom and bond information from the prediction (latent space)
        They should be np.array
        pred_node: [n_nodes, n_node_types]
        pred_pos: [n_nodes, 3]
        pred_halfedge: [n_halfedges, n_edge_types]
        """
        # get atom and element
        pred_atom = softmax(pred_node, axis=-1)
        atom_type = np.argmax(pred_atom, axis=-1)
        atom_prob = np.max(pred_atom, axis=-1)
        isnot_masked_atom = (atom_type < self.num_element)
        if not isnot_masked_atom.all():
            edge_index_changer = - np.ones(len(isnot_masked_atom), dtype=np.int64)
            edge_index_changer[isnot_masked_atom] = np.arange(isnot_masked_atom.sum())
        atom_type = atom_type[isnot_masked_atom]
        atom_prob = atom_prob[isnot_masked_atom]
        element = np.array([self.nodetype_to_ele[i] for i in atom_type])
        
        # get pos
        atom_pos = pred_pos[isnot_masked_atom]
        
        # get bond
        if self.num_edge_types == 1:
            return {
                'element': element,
                'atom_pos': atom_pos,
                'atom_prob': atom_prob,
            }
        pred_halfedge = softmax(pred_halfedge, axis=-1)
        edge_type = np.argmax(pred_halfedge, axis=-1)  # omit half for simplicity
        edge_prob = np.max(pred_halfedge, axis=-1)
        
        is_bond = (edge_type > 0) & (edge_type <= self.num_bond_types)  # larger is mask type
        bond_type = edge_type[is_bond]
        bond_prob = edge_prob[is_bond]
        bond_index = halfedge_index[:, is_bond]
        if not isnot_masked_atom.all():
            bond_index = edge_index_changer[bond_index]
            bond_for_masked_atom = (bond_index < 0).any(axis=0)
            bond_index = bond_index[:, ~bond_for_masked_atom]
            bond_type = bond_type[~bond_for_masked_atom]
            bond_prob = bond_prob[~bond_for_masked_atom]

        bond_type = np.concatenate([bond_type, bond_type])
        bond_prob = np.concatenate([bond_prob, bond_prob])
        bond_index = np.concatenate([bond_index, bond_index[::-1]], axis=1)
        
        return {
            'element': element,
            'atom_pos': atom_pos,
            'bond_type': bond_type,
            'bond_index': bond_index,
            
            'atom_prob': atom_prob,
            'bond_prob': bond_prob,
        }


def make_data_placeholder_discriminator(n_graphs, data, device=None, max_size=None):
    # n_nodes_list = np.random.randint(15, 50, n_graphs)
    if max_size is None:  # use statistics from GEOM-Drug dataset
        # n_nodes_list = np.random.normal(24.923464980477522, 5.516291901819105, size=n_graphs)
        # from tagetdiff 
        pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
        n_nodes_list = np.array([atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_graphs)])
        if np.max(n_nodes_list) > data.num_nodes:
            n_nodes_list = np.array([np.max(n_nodes_list) for _ in range(n_graphs)])
        else:
            n_nodes_list = np.array([data.num_nodes for _ in range(n_graphs)])
        # n_nodes_list = np.array([data.num_nodes for _ in range(n_graphs)])
    else:
        n_nodes_list = np.array([max_size] * n_graphs)
    n_nodes_list = n_nodes_list.astype('int64')
    batch_node = np.concatenate([np.full(n_nodes, i) for i, n_nodes in enumerate(n_nodes_list)])
    halfedge_index = []
    batch_halfedge = []
    idx_start = 0
    for i_mol, n_nodes in enumerate(n_nodes_list):
        halfedge_index_this_mol = torch.triu_indices(n_nodes, n_nodes, offset=1)
        halfedge_index.append(halfedge_index_this_mol + idx_start)
        n_edges_this_mol = len(halfedge_index_this_mol[0])
        batch_halfedge.append(np.full(n_edges_this_mol, i_mol))
        idx_start += n_nodes
    
    batch_node = torch.LongTensor(batch_node)
    batch_halfedge = torch.LongTensor(np.concatenate(batch_halfedge))
    halfedge_index = torch.cat(halfedge_index, dim=1)
    
    if device is not None:
        batch_node = batch_node.to(device)
        batch_halfedge = batch_halfedge.to(device)
        halfedge_index = halfedge_index.to(device)

    return {
        # 'n_graphs': n_graphs,
        'batch_node': batch_node,
        'halfedge_index': halfedge_index,
        'batch_halfedge': batch_halfedge,
    }

def make_data_placeholder(n_graphs, data, device=None, max_size=None):
    # n_nodes_list = np.random.randint(15, 50, n_graphs)
    if max_size is None:  # use statistics from GEOM-Drug dataset
        # n_nodes_list = np.random.normal(24.923464980477522, 5.516291901819105, size=n_graphs)
        # from tagetdiff 
        pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
        n_nodes_list = np.array([atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_graphs)])
        if np.max(n_nodes_list) > data.num_nodes:
            n_nodes_list = np.array([np.max(n_nodes_list) for _ in range(n_graphs)])
        else:
            n_nodes_list = np.array([data.num_nodes for _ in range(n_graphs)])
        # n_nodes_list = np.array([data.num_nodes for _ in range(n_graphs)])
    else:
        n_nodes_list = np.array([max_size] * n_graphs)
    n_nodes_list = n_nodes_list.astype('int64')
    batch_node = np.concatenate([np.full(n_nodes, i) for i, n_nodes in enumerate(n_nodes_list)])
    halfedge_index = []
    batch_halfedge = []
    idx_start = 0
    for i_mol, n_nodes in enumerate(n_nodes_list):
        halfedge_index_this_mol = torch.triu_indices(n_nodes, n_nodes, offset=1)
        halfedge_index.append(halfedge_index_this_mol + idx_start)
        n_edges_this_mol = len(halfedge_index_this_mol[0])
        batch_halfedge.append(np.full(n_edges_this_mol, i_mol))
        idx_start += n_nodes
    
    batch_node = torch.LongTensor(batch_node)
    batch_halfedge = torch.LongTensor(np.concatenate(batch_halfedge))
    halfedge_index = torch.cat(halfedge_index, dim=1)
    
    if device is not None:
        batch_node = batch_node.to(device)
        batch_halfedge = batch_halfedge.to(device)
        halfedge_index = halfedge_index.to(device)

    return {
        # 'n_graphs': n_graphs,
        'batch_node': batch_node,
        'halfedge_index': halfedge_index,
        'batch_halfedge': batch_halfedge,
    }

def make_data_placeholder_frag(n_graphs, min_num_atom, start_linker, data, device=None, max_size=None):
    
    pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
    nodes_dist = np.array([atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_graphs)])
    
    
    num_node_frag = 0
    if start_linker is not None:
        print('linker atom number:', len(start_linker['element']))
        num_node_frag = len(start_linker['element'])
    nodesxsample = [i - num_node_frag if i > num_node_frag else min_num_atom for i in nodes_dist.tolist()]
    # print(nodesxsample)
    frag_mask_batch_node = []
    node_type_batch = []
    node_pos_batch = []
    for i, n_particles in enumerate(nodesxsample):
        atom_type = torch.full([n_particles], -1)
        if start_linker is not None:
            node_type = torch.cat([data.node_type, atom_type])
            node_type_batch.extend(node_type)
            pos = torch.randn(n_particles, 3)
            frag_mask = torch.cat(
                [torch.ones(num_node_frag, dtype=torch.long), torch.zeros(n_particles, dtype=torch.long)])
            node_pos = torch.cat([data.pos, pos])
            node_pos_batch.extend(node_pos)
            # data.num_nodes = data.node_type.shape[0]
            frag_mask_batch_node.extend(frag_mask)
    frag_mask_batch_node = torch.stack(frag_mask_batch_node, dim=-1).type(torch.bool)
    node_type_batch = torch.stack(node_type_batch, dim=-1)
    node_pos_batch = torch.stack(node_pos_batch, dim=0)
    
    n_nodes_list = [i if i > num_node_frag else num_node_frag + min_num_atom for i in nodes_dist.tolist()]
    batch_node = np.concatenate([np.full(n_nodes, i) for i, n_nodes in enumerate(n_nodes_list)])
    

    halfedge_index = []
    batch_halfedge = []
    halfedge_type_batch = []
    frag_mask_batch_edge = []
    idx_start = 0
    for i_mol, n_nodes in enumerate(n_nodes_list):
        # print(n_nodes)
        halfedge_index_this_mol = torch.triu_indices(n_nodes, n_nodes, offset=1)
        halfedge_index.append(halfedge_index_this_mol + idx_start)
        n_edges_this_mol = len(halfedge_index_this_mol[0])
        batch_halfedge.append(np.full(n_edges_this_mol, i_mol))
        idx_start_type = 0
        halfedge_type = []
        frag_mask = []
        for i in range(n_nodes-1):
            if i < num_node_frag:
                halfedge_type1 = torch.full([halfedge_index_this_mol[0][halfedge_index_this_mol[0]==i].size(0)-data.halfedge_index[0][data.halfedge_index[0]==i].size(0)], -1)
                halfedge_type.extend(torch.cat([data.halfedge_type[idx_start_type:idx_start_type+data.halfedge_index[0][data.halfedge_index[0]==i].size(0)],halfedge_type1]))
                frag_mask1 = torch.cat(
                [torch.ones(data.halfedge_index[0][data.halfedge_index[0]==i].size(0), dtype=torch.long), torch.zeros(halfedge_index_this_mol[0][halfedge_index_this_mol[0]==i].size(0)-data.halfedge_index[0][data.halfedge_index[0]==i].size(0), dtype=torch.long)])
                frag_mask.extend(frag_mask1)
                idx_start_type = idx_start_type + data.halfedge_index[0][data.halfedge_index[0]==i].size(0)
            else:
                frag_mask1 = torch.zeros(halfedge_index_this_mol[0][halfedge_index_this_mol[0]==i].size(0)-data.halfedge_index[0][data.halfedge_index[0]==i].size(0), dtype=torch.long)
                frag_mask.extend(frag_mask1)
                halfedge_type1 = torch.full([halfedge_index_this_mol[0][halfedge_index_this_mol[0]==i].size(0)], -1)
                halfedge_type.extend(halfedge_type1)

        halfedge_type = torch.stack(halfedge_type, dim=-1)
        halfedge_type_batch.extend(halfedge_type)
        frag_mask = torch.stack(frag_mask, dim=-1).type(torch.bool)
        frag_mask_batch_edge.extend(frag_mask)
        idx_start += n_nodes
    
    batch_node = torch.LongTensor(batch_node)
    batch_halfedge = torch.LongTensor(np.concatenate(batch_halfedge))
    halfedge_index = torch.cat(halfedge_index, dim=1)
    halfedge_type_batch = torch.stack(halfedge_type_batch, dim=-1)
    frag_mask_batch_edge = torch.stack(frag_mask_batch_edge, dim=-1)
    
    if device is not None:
        batch_node = batch_node.to(device)
        batch_halfedge = batch_halfedge.to(device)
        halfedge_index = halfedge_index.to(device)
        halfedge_type_batch = halfedge_type_batch.to(device)
        frag_mask_batch_node = frag_mask_batch_node.to(device)
        frag_mask_batch_edge = frag_mask_batch_edge.to(device)
        node_type_batch = node_type_batch.to(device)
        node_pos_batch = node_pos_batch.to(device)
    

    return {
        # 'n_graphs': n_graphs,
        'frag_mask_batch_node': frag_mask_batch_node,
        'node_type_batch': node_type_batch,
        'node_pos_batch': node_pos_batch,
        'batch_node': batch_node,
        'halfedge_index': halfedge_index,
        'batch_halfedge': batch_halfedge,
        'halfedge_type_batch': halfedge_type_batch,
        'frag_mask_batch_edge': frag_mask_batch_edge,
    }
