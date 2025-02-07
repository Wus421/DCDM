import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from scipy.special import softmax
from tqdm.auto import tqdm
from utils.reconstruct import *
from models.transition import ContigousTransition, GeneralCategoricalTransition
from models.common import GaussianSmearingTime,MLP
from models.diffusion import get_beta_schedule
from models.egnn import EGNN
from models.uni_cross import UniTransformerO2TwoUpdateGeneral
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral_affinity
from models.common import compose_context1, MLP, ShiftedSoftplus
from sklearn.metrics import roc_auc_score

def get_refine_net(refine_net_type,edge_dim, config):
    if refine_net_type == 'uni_o2_cross':
        refine_net = UniTransformerO2TwoUpdateGeneral(
            edge_dim=edge_dim,
            num_blocks=config.denoiser.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            use_gate=config.denoiser.use_gate,
            n_heads=config.n_heads,
            k=config.knn,
            # max_radius=config.max_radius,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup
        )
    elif refine_net_type == 'uni_o2_affinity':
        refine_net = UniTransformerO2TwoUpdateGeneral_affinity(
            edge_dim=edge_dim,
            num_blocks=config.denoiser.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            use_gate=config.denoiser.use_gate,
            n_heads=config.n_heads,
            k=config.knn,
            # max_radius=config.max_radius,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup
        )
    elif refine_net_type == 'egnn':
        refine_net = EGNN(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=1,
            k=config.knn,
            cutoff_mode=config.cutoff_mode
        )
    else:
        raise ValueError(refine_net_type)
    return refine_net

def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = 0.
        pass
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset


# %% categorical diffusion related
def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    # permute_order = (0, -1) + tuple(range(1, len(x.size())))
    # x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return sample_index


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


# %%


# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)


class DCDM(nn.Module):

    def __init__(self, config, protein_atom_feature_dim, num_node_types, num_edge_types):

        super().__init__()
        self.config = config
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.bond_len_loss = getattr(config, 'bond_len_loss', False)
        # variance schedule
        self.model_mean_type = config.model_mean_type  # ['noise', 'C0']
        self.loss_v_weight = config.loss_v_weight
        

        # # define beta and alpha
        self.define_betas_alphas(config.diff)
        
        self.bond_len_loss = getattr(config, 'bond_len_loss', False)
        self.sample_time_method = config.sample_time_method

        # model definition
        # # embedding
        self.hidden_dim = config.hidden_dim
        node_dim = config.node_dim
        edge_dim = config.edge_dim
        time_dim = config.diff.time_dim
        if self.config.node_indicator:
            node_dim = node_dim - 1
            edge_dim = edge_dim - 1
        else:
            node_dim = node_dim
            edge_dim = edge_dim
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, node_dim)
        self.node_embedder = nn.Linear(num_node_types, node_dim-time_dim, bias=False)  # element type
        self.edge_embedder = nn.Linear(num_edge_types, edge_dim-time_dim, bias=False) # bond type
        self.time_emb = nn.Sequential(
            GaussianSmearingTime(stop=self.num_timesteps, num_gaussians=time_dim, type_='linear'),
        )

        # center pos
        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']
        self.refine_net_type = config.model_type
        print(self.refine_net_type)
        self.refine_net = get_refine_net(self.refine_net_type,edge_dim, config)
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, node_dim),
        )
        
        # # decoder
        if self.config.node_indicator:
            self.refine_net = get_refine_net(self.refine_net_type,edge_dim+1, config)
            self.node_decoder = MLP(node_dim+1, num_node_types, node_dim+1)
            self.edge_decoder = MLP(edge_dim+1, num_edge_types, edge_dim+1)
        else:
            self.refine_net = get_refine_net(self.refine_net_type,edge_dim, config)
            self.node_decoder = MLP(node_dim, num_node_types, node_dim)
            self.edge_decoder = MLP(edge_dim, num_edge_types, edge_dim)



    def define_betas_alphas(self, config):
        self.num_timesteps = config.num_timesteps
        self.categorical_space = getattr(config, 'categorical_space', 'discrete')
        
        # try to get the scaling
        if self.categorical_space == 'continuous':
            self.scaling = getattr(config, 'scaling', [1., 1., 1.])
        else:
            self.scaling = [1., 1., 1.]  # actually not used for discrete space (defined for compatibility)

        # # diffusion for pos
        pos_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_pos
        )
        assert self.scaling[0] == 1, 'scaling for pos should be 1'
        self.pos_transition = ContigousTransition(pos_betas)

        # # diffusion for node type
        node_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_atom
        )
        # self.node_betas = nn.Parameter(torch.from_numpy(node_betas).float(), requires_grad=False)
        if self.categorical_space == 'discrete':
            init_prob = config.diff_atom.init_prob
            self.node_transition = GeneralCategoricalTransition(node_betas, self.num_node_types,
                                                            init_prob=init_prob)
        elif self.categorical_space == 'continuous':
            scaling_node = self.scaling[1]
            self.node_transition = ContigousTransition(node_betas, self.num_node_types, scaling_node)
        else:
            raise ValueError(self.categorical_space)

        # # diffusion for edge type
        edge_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_bond
        )
        # self.edge_betas = nn.Parameter(torch.from_numpy(edge_betas).float(), requires_grad=False)
        if self.categorical_space == 'discrete':
            init_prob = config.diff_bond.init_prob
            self.edge_transition = GeneralCategoricalTransition(edge_betas, self.num_edge_types,
                                                            init_prob=init_prob)
        elif self.categorical_space == 'continuous':
            scaling_edge = self.scaling[2]
            self.edge_transition = ContigousTransition(edge_betas, self.num_edge_types, scaling_edge)
        else:
            raise ValueError(self.categorical_space)


    def forward(self, protein_pos, protein_v, batch_protein, h_node_pert, pos_pert, batch_node,
                h_edge_pert, edge_index, batch_edge, t,
                return_all=False, fix_x=False):
        """
        Predict Mol at step `0` given perturbed Mol at step `t` with hidden dims and time step
        """
        # 1 node and edge embedding + time embedding
        time_embed_node = self.time_emb(t.index_select(0, batch_node))
        h_node_pert = torch.cat([self.node_embedder(h_node_pert), time_embed_node], dim=-1)
        time_embed_edge = self.time_emb(t.index_select(0, batch_edge))
        h_edge_pert = torch.cat([self.edge_embedder(h_edge_pert), time_embed_edge], dim=-1)
        h_protein = self.protein_atom_emb(protein_v)
        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            h_node_pert = torch.cat([h_node_pert, torch.ones(len(h_node_pert), 1).to(h_protein)], -1)
            h_edge_pert = torch.cat([h_edge_pert, torch.ones(len(h_edge_pert), 1).to(h_protein)], -1)
        h_all, pos_all, batch_all, mask_ligand = compose_context1(
            h_protein=h_protein,
            h_ligand=h_node_pert,
            pos_protein=protein_pos,
            pos_ligand=pos_pert,
            batch_protein=batch_protein,
            batch_ligand=batch_node,
        )
        outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, h_edge_pert, edge_index, 
                                  node_time=t.index_select(0, batch_node).unsqueeze(-1) / self.num_timesteps,
                                  edge_time=t.index_select(0, batch_edge).unsqueeze(-1) / self.num_timesteps, 
                                  return_all=return_all, fix_x=fix_x)
        final_pos, final_h, final_e = outputs['x'], outputs['h'], outputs['e']

        n_halfedges = final_e.shape[0] // 2
        pred_node = self.node_decoder(final_h[mask_ligand])
        pred_halfedge = self.edge_decoder(final_e[:n_halfedges]+final_e[n_halfedges:])
        pred_pos = final_pos[mask_ligand]

        preds = {
            'pred_ligand_pos': pred_pos,
            'pred_ligand_v': pred_node,
            'pred_ligand_halfedge': pred_halfedge,
            'final_x': final_pos,
            'final_ligand_h': final_h[mask_ligand]
        }
        if return_all:
            final_all_pos, final_all_h, final_all_e = outputs['all_x'], outputs['all_h'], outputs['all_e']
            final_all_ligand_pos = [pos[mask_ligand] for pos in final_all_pos]
            final_all_ligand_v = [self.node_decoder(h[mask_ligand]) for h in final_all_h]
            final_all_ligand_halfedge = [self.edge_decoder(e[:n_halfedges]+e[n_halfedges:]) for e in final_all_e]
            preds.update({
                'layer_pred_ligand_pos': final_all_ligand_pos,
                'layer_pred_ligand_v': final_all_ligand_v,
                'final_all_ligand_halfedge': final_all_ligand_halfedge
            })
        return preds

    # atom type diffusion process
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        # q(vt | vt-1)
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        # alpha_t * vt + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        # compute q(vt | v0)
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)

        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs

    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_index = log_sample_categorical(log_qvt_v0)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        return sample_index, log_sample

    # atom type generative process
    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0

    def kl_v_prior(self, log_x_start, batch):
        num_graphs = batch.max().item() + 1
        log_qxT_prob = self.q_v_pred(log_x_start, t=[self.num_timesteps - 1] * num_graphs, batch=batch)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))
        kl_prior = categorical_kl(log_qxT_prob, log_half_prob)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        return pos0_from_e

    def q_pos_posterior(self, x0, xt, t, batch):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                         extract(self.posterior_mean_ct_coef, t, batch) * xt
        return pos_model_mean

    def kl_pos_prior(self, pos0, batch):
        num_graphs = batch.max().item() + 1
        a_pos = extract(self.alphas_cumprod, [self.num_timesteps - 1] * num_graphs, batch)  # (num_ligand_atoms, 1)
        pos_model_mean = a_pos.sqrt() * pos0
        pos_log_variance = torch.log((1.0 - a_pos).sqrt())
        kl_prior = normal_kl(torch.zeros_like(pos_model_mean), torch.zeros_like(pos_log_variance),
                             pos_model_mean, pos_log_variance)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior
    
    def sample_time(self, num_graphs, device, **kwargs):
        # sample time
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        pt = torch.ones_like(time_step).float() / self.num_timesteps
        return time_step, pt
    
    def add_noise(self, node_type, node_pos, batch_node,
                    halfedge_type, halfedge_index, batch_halfedge,
                    num_mol, t, bond_predictor=None, **kwargs):
            num_graphs = num_mol
            device = node_pos.device

            time_step = t * torch.ones(num_graphs, device=device).long()

            # 2.1 perturb pos, node, edge
            pos_pert = self.pos_transition.add_noise(node_pos, time_step, batch_node)
            node_pert = self.node_transition.add_noise(node_type, time_step, batch_node)
            halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, batch_halfedge)
            # edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
            # batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
            if self.categorical_space == 'discrete':
                h_node_pert, log_node_t, log_node_0 = node_pert
                h_halfedge_pert, log_halfedge_t, log_halfedge_0 = halfedge_pert
            else:
                h_node_pert, h_node_0 = node_pert
                h_halfedge_pert, h_halfedge_0 = halfedge_pert
            return [h_node_pert, pos_pert, h_halfedge_pert]
    
    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):
        # fixed pos variance
        pos_log_variance = extract(self.posterior_logvar, t, batch)
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)
        kl_pos = normal_kl(pos_true_mean, pos_log_variance, pos_model_mean, pos_log_variance)
        kl_pos = kl_pos / np.log(2.)

        decoder_nll_pos = -log_normal(x0, means=pos_model_mean, log_scales=0.5 * pos_log_variance)
        assert kl_pos.shape == decoder_nll_pos.shape
        mask = (t == 0).float()[batch]
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1. - mask) * kl_pos, batch, dim=0)
        return loss_pos

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        return loss_v
    def decode_output(self, pred_node, pred_pos, pred_halfedge, halfedge_index):
        """
        Get the atom and bond information from the prediction (latent space)
        They should be np.array
        pred_node: [n_nodes, n_node_types]
        pred_pos: [n_nodes, 3]
        pred_halfedge: [n_halfedges, n_edge_types]
        """
        # get atom and element
        atomic_numbers = torch.LongTensor([6,7,8,9,15,16,17])
        nodetype_to_ele = {i: ele for i, ele in enumerate(atomic_numbers)}
        pred_atom = softmax(pred_node, axis=-1)
        atom_type = np.argmax(pred_atom, axis=-1)
        atom_prob = np.max(pred_atom, axis=-1)
        isnot_masked_atom = (atom_type < 7)
        if not isnot_masked_atom.all():
            edge_index_changer = - np.ones(len(isnot_masked_atom), dtype=np.int64)
            edge_index_changer[isnot_masked_atom] = np.arange(isnot_masked_atom.sum())
        atom_type = atom_type[isnot_masked_atom]
        atom_prob = atom_prob[isnot_masked_atom]
        element = np.array([nodetype_to_ele[i] for i in atom_type])
        
        # get pos
        atom_pos = pred_pos[isnot_masked_atom]
        
        # get bond
        # if self.num_edge_types == 1:
        #     return {
        #         'element': element,
        #         'atom_pos': atom_pos,
        #         'atom_prob': atom_prob,
        #     }
        pred_halfedge = softmax(pred_halfedge, axis=-1)
        edge_type = np.argmax(pred_halfedge, axis=-1)  # omit half for simplicity
        edge_prob = np.max(pred_halfedge, axis=-1)
        
        is_bond = (edge_type > 0) & (edge_type <= 6)  # larger is mask type
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
    def seperate_labels(self, node_type, node_pos, halfedge_type, n_graphs, batch_node, halfedge_index, batch_halfedge):

        new_outputs = []
        for i_mol in range(n_graphs):
            ind_node = (batch_node == i_mol)
            ind_halfedge = (batch_halfedge == i_mol)
            assert ind_node.sum() * (ind_node.sum()-1) == ind_halfedge.sum() * 2
            new_pred_this = [node_type[ind_node],  # node type
                            node_pos[ind_node],  # node pos
                            halfedge_type[ind_halfedge]]  # halfedge type
                            
            
            halfedge_index_this = halfedge_index[:, ind_halfedge]
            assert ind_node.nonzero()[0].min() == halfedge_index_this.min()
            halfedge_index_this = halfedge_index_this - ind_node.nonzero()[0].min()

            new_outputs.append({
                'labels': new_pred_this,
                'halfedge_index': halfedge_index_this,
            })
        return new_outputs

    def get_diffusion_loss(
            self, protein_pos, protein_v, batch_protein, node_type, node_pos, batch_node,
                halfedge_type, halfedge_index, batch_halfedge, num_mol,time_step=None,
    ):
        num_graphs = num_mol
        protein_pos, node_pos, _ = center_pos(
            protein_pos, node_pos, batch_protein, batch_node, mode=self.center_pos_mode)
        # 1. sample noise levels
        device = node_pos.device
        time_step, _ = self.sample_time(num_graphs, device)

        # 2.1 perturb pos, node, edge
        pos_pert = self.pos_transition.add_noise(node_pos, time_step, batch_node)
        node_pert = self.node_transition.add_noise(node_type, time_step, batch_node)
        halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, batch_halfedge)
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)  # undirected edges
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
        if self.categorical_space == 'discrete':
            h_node_pert, log_node_t, log_node_0 = node_pert
            h_halfedge_pert, log_halfedge_t, log_halfedge_0 = halfedge_pert
        else:
            h_node_pert, h_node_0 = node_pert
            h_halfedge_pert, h_halfedge_0 = halfedge_pert
        
        h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)
        # 3. forward-pass NN, feed perturbed pos and v, output noise
        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            h_node_pert=h_node_pert,
            pos_pert=pos_pert,
            batch_node=batch_node,
            h_edge_pert=h_edge_pert,
            edge_index=edge_index,
            batch_edge=batch_edge,
            t=time_step
        )

        pred_pos, pred_node, pred_halfedge= preds['pred_ligand_pos'], preds['pred_ligand_v'], preds['pred_ligand_halfedge']
        
        # 4. loss
        # 4.1 pos
        loss_pos = F.mse_loss(pred_pos, node_pos)
        if self.bond_len_loss == True:
            bond_index = halfedge_index[:, halfedge_type > 0]
            true_length = torch.norm(node_pos[bond_index[0]] - node_pos[bond_index[1]], dim=-1)
            pred_length = torch.norm(pred_pos[bond_index[0]] - pred_pos[bond_index[1]], dim=-1)
            loss_len = F.mse_loss(pred_length, true_length)
    
        if self.categorical_space == 'discrete':
            # 4.2 node type
            log_node_recon = F.log_softmax(pred_node, dim=-1)
            log_node_post_true = self.node_transition.q_v_posterior(log_node_0, log_node_t, time_step, batch_node, v0_prob=True)
            log_node_post_pred = self.node_transition.q_v_posterior(log_node_recon, log_node_t, time_step, batch_node, v0_prob=True)
            kl_node = self.node_transition.compute_v_Lt(log_node_post_true, log_node_post_pred, log_node_0, t=time_step, batch=batch_node)
            loss_node = torch.mean(kl_node) * 100
            # 4.3 edge type
            log_halfedge_recon = F.log_softmax(pred_halfedge, dim=-1)
            log_edge_post_true = self.edge_transition.q_v_posterior(log_halfedge_0, log_halfedge_t, time_step, batch_halfedge, v0_prob=True)
            log_edge_post_pred = self.edge_transition.q_v_posterior(log_halfedge_recon, log_halfedge_t, time_step, batch_halfedge, v0_prob=True)
            kl_edge = self.edge_transition.compute_v_Lt(log_edge_post_true, 
                            log_edge_post_pred, log_halfedge_0, t=time_step, batch=batch_halfedge)
            loss_edge = torch.mean(kl_edge)  * 100
        else:
            loss_node = F.mse_loss(pred_node, h_node_0)  * 30
            loss_edge = F.mse_loss(pred_halfedge, h_halfedge_0) * 30

        # total
        loss_total = loss_pos + loss_node + loss_edge + (loss_len if self.bond_len_loss else 0)
        
        loss_dict = {
            'loss': loss_total,
            'loss_pos': loss_pos,
            'loss_node': loss_node,
            'loss_edge': loss_edge,
            # 'x0': node_pos,
        }
        if self.bond_len_loss == True:
            loss_dict['loss_len'] = loss_len
        return loss_dict

    @torch.no_grad()
    def likelihood_estimation(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, time_step
    ):
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein')
        assert (time_step == self.num_timesteps).all() or (time_step < self.num_timesteps).all()
        if (time_step == self.num_timesteps).all():
            kl_pos_prior = self.kl_pos_prior(ligand_pos, batch_ligand)
            log_ligand_v0 = index_to_log_onehot(batch_ligand, self.num_classes)
            kl_v_prior = self.kl_v_prior(log_ligand_v0, batch_ligand)
            return kl_pos_prior, kl_v_prior

        # perturb pos and v
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )
        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
        # Vt = a * V0 + (1-a) / K
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)

        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,

            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            time_step=time_step
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']
        if self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_ligand_pos, xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        else:
            raise ValueError

        # atom type
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)

        # t = [T-1, ... , 0]
        kl_pos = self.compute_pos_Lt(pos_model_mean=pos_model_mean, x0=ligand_pos,
                                     xt=ligand_pos_perturbed, t=time_step, batch=batch_ligand)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        return kl_pos, kl_v

    @torch.no_grad()
    def fetch_embedding(self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand):
        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,

            init_ligand_pos=ligand_pos,
            init_ligand_v=ligand_v,
            batch_ligand=batch_ligand,
            fix_x=True
        )
        return preds

    def affinity(self, protein_pos, protein_v, batch_protein, h_node_pert, pos_pert, batch_node,
                h_edge_pert, edge_index, batch_edge, t,
                return_all=False, fix_x=False):
        """
        Predict Mol at step `0` given perturbed Mol at step `t` with hidden dims and time step
        """
        # 1 node and edge embedding + time embedding
        time_embed_node = self.time_emb(t.index_select(0, batch_node))
        h_node_pert = torch.cat([self.node_embedder(h_node_pert), time_embed_node], dim=-1)
        time_embed_edge = self.time_emb(t.index_select(0, batch_edge))
        h_edge_pert = torch.cat([self.edge_embedder(h_edge_pert), time_embed_edge], dim=-1)
        h_protein = self.protein_atom_emb(protein_v)
        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            h_node_pert = torch.cat([h_node_pert, torch.ones(len(h_node_pert), 1).to(h_protein)], -1)
            h_edge_pert = torch.cat([h_edge_pert, torch.ones(len(h_edge_pert), 1).to(h_protein)], -1)
        h_all, pos_all, batch_all, mask_ligand = compose_context1(
            h_protein=h_protein,
            h_ligand=h_node_pert,
            pos_protein=protein_pos,
            pos_ligand=pos_pert,
            batch_protein=batch_protein,
            batch_ligand=batch_node,
        )
        
        outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, h_edge_pert, edge_index, 
                                  node_time=t.index_select(0, batch_node).unsqueeze(-1) / self.num_timesteps,
                                  edge_time=t.index_select(0, batch_edge).unsqueeze(-1) / self.num_timesteps, 
                                  return_all=return_all, fix_x=fix_x)
       
        _, final_h= outputs['x'], outputs['h']
        return final_h
    def get_affinity_loss(
            self, protein_pos, protein_v, batch_protein, node_type, node_pos, batch_node,
                halfedge_type, halfedge_index, batch_halfedge, num_mol, label, time_step=None,
    ):
        num_graphs = num_mol
        protein_pos, node_pos, _ = center_pos(
            protein_pos, node_pos, batch_protein, batch_node, mode=self.center_pos_mode)
        # 1. sample noise levels
        device = node_pos.device
        time_step, _ = self.sample_time(num_graphs, device)

        # 2.1 perturb pos, node, edge
        pos_pert = self.pos_transition.add_noise(node_pos, time_step, batch_node)
        node_pert = self.node_transition.add_noise(node_type, time_step, batch_node)
        halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, batch_halfedge)
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)  # undirected edges
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
        if self.categorical_space == 'discrete':
            h_node_pert, log_node_t, log_node_0 = node_pert
            h_halfedge_pert, log_halfedge_t, log_halfedge_0 = halfedge_pert
        else:
            h_node_pert, h_node_0 = node_pert
            h_halfedge_pert, h_halfedge_0 = halfedge_pert
        
        h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)
        # 3. forward-pass NN, feed perturbed pos and v, output noise
        preds = self.affinity(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            h_node_pert=h_node_pert,
            pos_pert=pos_pert,
            batch_node=batch_node,
            h_edge_pert=h_edge_pert,
            edge_index=edge_index,
            batch_edge=batch_edge,
            t=time_step
        )
        
        preds_affinity = preds
        for i in range(len(label)):
            if label[i] <= -8.:
                label[i] = 1
            else:
                label[i] = 0
        
        label_onehot = torch.nn.functional.one_hot(label.unsqueeze(0).to(torch.int64), 2).float().squeeze()
        loss = F.cross_entropy(preds_affinity,label_onehot)
        accuracy = (preds_affinity.argmax(axis=1)==label.reshape(-1)).sum()/label.shape[0]
        try:
            auc = roc_auc_score(label_onehot.detach().cpu().numpy(), preds_affinity.detach().cpu().numpy())
        except:
            auc = np.array(0.)
        loss_dict = {
            'loss': loss,
            'accuracy': accuracy,
            'auc': auc,
            # 'x0': node_pos,
        }
        return loss_dict
    @torch.no_grad()
    def sample_diffusion(self, n_graphs, batch_node, halfedge_index, batch_halfedge, bond_predictor, guidance, protein_pos, protein_v, batch_protein,
                          num_steps=None, center_pos_mode=None, pos_only=False,affinity_model=None):
        device = batch_node.device
        # # 1. get the init values (position, node types)
        # n_graphs = len(n_nodes_list)
        n_nodes_all = len(batch_node)
        n_halfedges_all = len(batch_halfedge)
        
        node_init = self.node_transition.sample_init(n_nodes_all)
        center_pos1 = scatter_mean(protein_pos, batch_protein, dim=0)
        batch_center_pos = center_pos1[batch_node]
        init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)
        protein_pos, pos_init, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_node, mode=center_pos_mode)
        halfedge_init = self.edge_transition.sample_init(n_halfedges_all)
        if self.categorical_space == 'discrete':
            _, h_node_init, log_node_type = node_init
            _, h_halfedge_init, log_halfedge_type = halfedge_init
        else:
            h_node_init = node_init
            h_halfedge_init = halfedge_init

        # # 1.5 log init
        node_traj = torch.zeros([self.num_timesteps+1, n_nodes_all, h_node_init.shape[-1]],
                                dtype=h_node_init.dtype).to(device)
        pos_traj = torch.zeros([self.num_timesteps+1, n_nodes_all, 3], dtype=pos_init.dtype).to(device)
        halfedge_traj = torch.zeros([self.num_timesteps+1, n_halfedges_all, h_halfedge_init.shape[-1]],
                                    dtype=h_halfedge_init.dtype).to(device)
        node_traj[0] = h_node_init
        pos_traj[0] = pos_init
        halfedge_traj[0] = h_halfedge_init


        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1
        
        # # 2. sample loop
        h_node_pert = h_node_init
        pos_pert = pos_init
        h_halfedge_pert = h_halfedge_init
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
        for i, step in tqdm(enumerate(range(num_steps)[::-1]), total=num_steps,desc="num_timesteps"):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
            h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)
            
            # # 1 inference
            preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            h_node_pert=h_node_pert,
            pos_pert=pos_pert,
            batch_node=batch_node,
            h_edge_pert=h_edge_pert,
            edge_index=edge_index,
            batch_edge=batch_edge,
            t=time_step
            )
            
            pred_node = preds['pred_ligand_v']  # (N, num_node_types)
            pred_pos = preds['pred_ligand_pos']  # (N, 3)
            pred_halfedge = preds['pred_ligand_halfedge']  # (E//2, num_bond_types)
            # # 2 get the t - 1 state
            # pos 
            pos_prev = self.pos_transition.get_prev_from_recon(
                x_t=pos_pert, x_recon=pred_pos, t=time_step, batch=batch_node)
            if self.categorical_space == 'discrete':
                # node types
                log_node_recon = F.log_softmax(pred_node, dim=-1)
                log_node_type = self.node_transition.q_v_posterior(log_node_recon, log_node_type, time_step, batch_node, v0_prob=True)
                node_type_prev = log_sample_categorical(log_node_type)
                h_node_prev = self.node_transition.onehot_encode(node_type_prev)
                
                # halfedge types
                log_edge_recon = F.log_softmax(pred_halfedge, dim=-1)
                log_halfedge_type = self.edge_transition.q_v_posterior(log_edge_recon, log_halfedge_type, time_step, batch_halfedge, v0_prob=True)
                halfedge_type_prev = log_sample_categorical(log_halfedge_type)
                h_halfedge_prev = self.edge_transition.onehot_encode(halfedge_type_prev)
                
            else:
                h_node_prev = self.node_transition.get_prev_from_recon(
                    x_t=h_node_pert, x_recon=pred_node, t=time_step, batch=batch_node)
                h_halfedge_prev = self.edge_transition.get_prev_from_recon(
                    x_t=h_halfedge_pert, x_recon=pred_halfedge, t=time_step, batch=batch_halfedge)

            # # use guidance to modify pos
            if guidance is not None:
                gui_type, gui_scale = guidance
                if (gui_scale > 0):
                    with torch.enable_grad():
                        # print(h_node_pert.size())
                        # final_h = scatter_mean(h_node_pert, batch_node, dim=0, dim_size=n_graphs)
                        h_node_in = h_node_pert.detach().requires_grad_(True)
                        pos_in = pos_pert.detach().requires_grad_(True)
                        pred_bondpredictor = bond_predictor(h_node_in, pos_in, batch_node,
                                    edge_index, batch_edge, time_step)
                        
                        # h_node_in = h_node_pert.detach()
                        # pos_in = pos_pert.detach().requires_grad_(True)
                        pred = affinity_model.affinity(protein_pos=protein_pos,
                                protein_v=protein_v,
                                batch_protein=batch_protein,
                                h_node_pert=h_node_in,
                                pos_pert=pos_in,
                                batch_node=batch_node,
                                h_edge_pert=h_edge_pert,
                                edge_index=edge_index,
                                batch_edge=batch_edge,
                                t=time_step)
                        if gui_type == 'uncertainty':
                            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
                            uncertainty = uncertainty.log().sum()
                            # delta = - torch.autograd.grad(uncertainty, pos_in)[0] * gui_scale    gui_scale=0.0001
                            delta = - torch.autograd.grad(uncertainty, pos_in)[0] * 0.0001
                            label = torch.ones(n_graphs).to(pos_in.device)
                            label_onehot = torch.nn.functional.one_hot(label.unsqueeze(0).to(torch.int64), 2).float().squeeze()
                            entropy = F.cross_entropy(pred,label_onehot, reduction='none')
                            entropy = entropy.log().sum()
                            torch.autograd.set_detect_anomaly(True)
                            pos_delta = - torch.autograd.grad(entropy, pos_in, retain_graph=True)[0] * 0.05
                            h_nodedelta = - torch.autograd.grad(entropy, h_node_in)[0] * 0.0001
                            # print(delta)
                        else:
                            raise NotImplementedError(f'Guidance type {gui_type} is not implemented')
                    pos_prev = pos_prev + delta + pos_delta
                    # pos_prev = pos_prev + delta
                    h_node_prev = h_node_prev + h_nodedelta

            # log update
            node_traj[i+1] = h_node_prev
            pos_traj[i+1] = pos_prev
            halfedge_traj[i+1] = h_halfedge_prev

            # # 3 update t-1
            pos_pert = pos_prev
            h_node_pert = h_node_prev
            h_halfedge_pert = h_halfedge_prev
        # print(pos_traj.size())
        # exit()
        # pred_node = pred_node + h_nodedelta
        # pred_pos = pred_pos + pos_delta
        pred_pos = pred_pos + offset[batch_node]
        return {
            'pred': [pred_node, pred_pos, pred_halfedge],
            'traj': [node_traj, pos_traj, halfedge_traj],

        }
        
    @torch.no_grad()
    def sample_diffusion_frag(self, n_graphs, batch_node,frag_mask_batch_node, node_type_batch, node_pos_batch, halfedge_index, batch_halfedge, halfedge_type_batch, frag_mask_batch_edge, bond_predictor, guidance, protein_pos, protein_v, batch_protein,
                          num_steps=None, center_pos_mode=None, pos_only=False,affinity_model=None):
        
        
        device = batch_node.device
        time_step, _ = self.sample_time(n_graphs, device)
        # 2.1 给frag编码
        log_node_0 = F.one_hot(node_type_batch[frag_mask_batch_node], self.num_node_types).float()
        log_halfedge_0 = F.one_hot(halfedge_type_batch[frag_mask_batch_edge], self.num_edge_types).float()
        
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)  # undirected edges
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)

        
       
        n_nodes_all = len(batch_node)
        n_halfedges_all = len(batch_halfedge)
        node_init = self.node_transition.sample_init(n_nodes_all)
        center_pos1 = scatter_mean(protein_pos, batch_protein, dim=0)
        batch_center_pos = center_pos1[batch_node]
        init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)
        init_ligand_pos[frag_mask_batch_node] = node_pos_batch[frag_mask_batch_node]
        protein_pos, pos_init, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_node, mode=center_pos_mode)
        halfedge_init = self.edge_transition.sample_init(n_halfedges_all)
        if self.categorical_space == 'discrete':
            _, h_node_init, log_node_type = node_init
            _, h_halfedge_init, log_halfedge_type = halfedge_init
        else:
            h_node_init = node_init
            h_halfedge_init = halfedge_init
        h_node_init[frag_mask_batch_node] = log_node_0
        h_halfedge_init[frag_mask_batch_edge] = log_halfedge_0
        # # 1.5 log init
        node_traj = torch.zeros([self.num_timesteps+1, n_nodes_all, h_node_init.shape[-1]],
                                dtype=h_node_init.dtype).to(device)
        pos_traj = torch.zeros([self.num_timesteps+1, n_nodes_all, 3], dtype=pos_init.dtype).to(device)
        halfedge_traj = torch.zeros([self.num_timesteps+1, n_halfedges_all, h_halfedge_init.shape[-1]],
                                    dtype=h_halfedge_init.dtype).to(device)
        node_traj[0] = h_node_init
        pos_traj[0] = pos_init
        halfedge_traj[0] = h_halfedge_init


        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1
        
        # # 2. sample loop
        h_node_pert = h_node_init
        pos_pert = pos_init
        h_halfedge_pert = h_halfedge_init
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
        # # 1 inference
        frag_node = h_node_pert[frag_mask_batch_node]
        frag_pos = pos_pert[frag_mask_batch_node]
        frag_edge = h_halfedge_pert[frag_mask_batch_edge]
        for i, step in tqdm(enumerate(range(num_steps)[::-1]), total=num_steps,desc="num_timesteps"):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
            
            
            h_node_frag_addnoise, _, _ = self.node_transition.add_noise(node_type_batch[frag_mask_batch_node], time_step, batch_node[frag_mask_batch_node])
            h_node_pert[frag_mask_batch_node] = h_node_frag_addnoise
            halfedge_frag_addnoise, _, _ = self.edge_transition.add_noise(halfedge_type_batch[frag_mask_batch_edge], time_step, batch_halfedge[frag_mask_batch_edge])
            
            h_halfedge_pert[frag_mask_batch_edge]= halfedge_frag_addnoise
            h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)
            pos_pert[frag_mask_batch_node] = self.pos_transition.add_noise(pos_init[frag_mask_batch_node], time_step, batch_node[frag_mask_batch_node])
            preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            h_node_pert=h_node_pert,
            pos_pert=pos_pert,
            batch_node=batch_node,
            h_edge_pert=h_edge_pert,
            edge_index=edge_index,
            batch_edge=batch_edge,
            t=time_step
            )
            
            pred_node = preds['pred_ligand_v']  # (N, num_node_types)
            pred_pos = preds['pred_ligand_pos']  # (N, 3)
            pred_halfedge = preds['pred_ligand_halfedge']  # (E//2, num_bond_types)
            
            
            # # 2 get the t - 1 state
            # pos 
            pos_prev = self.pos_transition.get_prev_from_recon(
                x_t=pos_pert, x_recon=pred_pos, t=time_step, batch=batch_node)
            if self.categorical_space == 'discrete':
                # node types
                log_node_recon = F.log_softmax(pred_node, dim=-1)
                log_node_type = self.node_transition.q_v_posterior(log_node_recon, log_node_type, time_step, batch_node, v0_prob=True)
                node_type_prev = log_sample_categorical(log_node_type)
                h_node_prev = self.node_transition.onehot_encode(node_type_prev)
                
                # halfedge types
                log_edge_recon = F.log_softmax(pred_halfedge, dim=-1)
                log_halfedge_type = self.edge_transition.q_v_posterior(log_edge_recon, log_halfedge_type, time_step, batch_halfedge, v0_prob=True)
                halfedge_type_prev = log_sample_categorical(log_halfedge_type)
                h_halfedge_prev = self.edge_transition.onehot_encode(halfedge_type_prev)
                
            else:
                h_node_prev = self.node_transition.get_prev_from_recon(
                    x_t=h_node_pert, x_recon=pred_node, t=time_step, batch=batch_node)
                h_halfedge_prev = self.edge_transition.get_prev_from_recon(
                    x_t=h_halfedge_pert, x_recon=pred_halfedge, t=time_step, batch=batch_halfedge)

            # # use guidance to modify pos
            if guidance is not None:
                gui_type, gui_scale = guidance
                if (gui_scale > 0):
                    with torch.enable_grad():
                        # pos_pert[frag_mask_batch_node] = frag_pos
                        # h_node_pert[frag_mask_batch_node] = frag_node
                        # h_halfedge_pert[frag_mask_batch_edge] = frag_edge
                        # print(h_node_pert.size())
                        # final_h = scatter_mean(h_node_pert, batch_node, dim=0, dim_size=n_graphs)
                        pos_prev[frag_mask_batch_node] = frag_pos
                        h_node_prev[frag_mask_batch_node] = frag_node
                        h_halfedge_prev[frag_mask_batch_edge] = frag_edge
                        h_edge_pert = torch.cat([h_halfedge_prev, h_halfedge_prev], dim=0)
                        h_node_in = h_node_pert.detach().requires_grad_(True)
                        pos_in = pos_pert.detach().requires_grad_(True)
                        pred_bondpredictor = bond_predictor(h_node_in, pos_in, batch_node,
                                    edge_index, batch_edge, time_step)
                        
                        # h_node_in = h_node_pert.detach()
                        # pos_in = pos_pert.detach().requires_grad_(True)
                        pred = affinity_model.affinity(protein_pos=protein_pos,
                                protein_v=protein_v,
                                batch_protein=batch_protein,
                                h_node_pert=h_node_in,
                                pos_pert=pos_in,
                                batch_node=batch_node,
                                h_edge_pert=h_edge_pert,
                                edge_index=edge_index,
                                batch_edge=batch_edge,
                                t=time_step)
                        if gui_type == 'uncertainty':
                            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
                            uncertainty = uncertainty.log().sum()
                            delta = - torch.autograd.grad(uncertainty, pos_in)[0] * 0.0001
                            label = torch.ones(n_graphs).to(pos_in.device)
                            label_onehot = torch.nn.functional.one_hot(label.unsqueeze(0).to(torch.int64), 2).float().squeeze()
                            entropy = F.cross_entropy(pred,label_onehot, reduction='none')
                            entropy = entropy.log().sum()
                            torch.autograd.set_detect_anomaly(True)
                            pos_delta = - torch.autograd.grad(entropy, pos_in, retain_graph=True)[0] * 0.05
                            h_nodedelta = - torch.autograd.grad(entropy, h_node_in)[0] * 0.0001
                        else:
                            raise NotImplementedError(f'Guidance type {gui_type} is not implemented')
                    pos_prev = pos_prev + delta + pos_delta
                    # pos_prev = pos_prev + delta
                    h_node_prev = h_node_prev + h_nodedelta

            # log update
            node_traj[i+1] = h_node_prev
            pos_traj[i+1] = pos_prev
            halfedge_traj[i+1] = h_halfedge_prev

            # # 3 update t-1
            pos_pert = pos_prev
            h_node_pert = h_node_prev
            h_halfedge_pert = h_halfedge_prev
            
            # pos_pert[frag_mask_batch_node] = frag_pos
            # h_node_pert[frag_mask_batch_node] = frag_node
            # h_halfedge_pert[frag_mask_batch_edge] = frag_edge
        # print(pos_traj.size())
        # exit()
        # pred_node = pred_node + h_nodedelta
        # pred_pos = pred_pos + pos_delta
        # 保持片段不变
        # pred_node[frag_mask_batch_node] = frag_node
        # pred_pos[frag_mask_batch_node] = frag_pos
        # pred_halfedge[frag_mask_batch_edge] = frag_edge  # (E//2, num_bond_types)
        # if torch.equal(frag_edge,log_halfedge_0) and torch.equal(pred_halfedge[frag_mask_batch_edge],frag_edge):
        #     print(frag_edge)
        # exit()
        pred_pos = pred_pos + offset[batch_node]
        return {
            'pred': [pred_node, pred_pos, pred_halfedge],
            'traj': [node_traj, pos_traj, halfedge_traj],

        }
        
    @torch.no_grad()
    def sample_diffusion_frag_x(self, n_graphs, batch_node,frag_mask_batch_node, node_type_batch, node_pos_batch, halfedge_index, batch_halfedge, halfedge_type_batch, frag_mask_batch_edge, bond_predictor, guidance, protein_pos, protein_v, batch_protein,
                          x_id,num_steps=None, center_pos_mode=None, pos_only=False,affinity_model=None):
        
        
        device = batch_node.device
        time_step, _ = self.sample_time(n_graphs, device)
        
        log_node_0 = F.one_hot(node_type_batch[frag_mask_batch_node], self.num_node_types).float()
        log_halfedge_0 = F.one_hot(halfedge_type_batch[frag_mask_batch_edge], self.num_edge_types).float()
        
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)  # undirected edges
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)

        
       
        n_nodes_all = len(batch_node)
        n_halfedges_all = len(batch_halfedge)
        node_init = self.node_transition.sample_init(n_nodes_all)
        
        center_batch = [node_pos_batch[x_id] for i in range(n_graphs)]
        center_pos1 = torch.stack(center_batch)
        batch_center_pos = center_pos1[batch_node]
        
        init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)
        init_ligand_pos[frag_mask_batch_node] = node_pos_batch[frag_mask_batch_node]
        protein_pos, pos_init, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_node, mode=center_pos_mode)
       
        halfedge_init = self.edge_transition.sample_init(n_halfedges_all)
        
        if self.categorical_space == 'discrete':
            _, h_node_init, log_node_type = node_init
            _, h_halfedge_init, log_halfedge_type = halfedge_init
            h_halfedge_init, log_halfedge_type = h_halfedge_init.to(h_node_init.device), log_halfedge_type.to(h_node_init.device)
        else:
            h_node_init = node_init
            h_halfedge_init = halfedge_init
        h_node_init[frag_mask_batch_node] = log_node_0
        h_halfedge_init[frag_mask_batch_edge] = log_halfedge_0

        # # 1.5 log init
        node_traj = torch.zeros([self.num_timesteps+1, n_nodes_all, h_node_init.shape[-1]],
                                dtype=h_node_init.dtype).to(device)
        pos_traj = torch.zeros([self.num_timesteps+1, n_nodes_all, 3], dtype=pos_init.dtype).to(device)
        halfedge_traj = torch.zeros([self.num_timesteps+1, len(batch_halfedge), h_halfedge_init.shape[-1]],
                                    dtype=h_halfedge_init.dtype).to(device)
        node_traj[0] = h_node_init
        pos_traj[0] = pos_init
        halfedge_traj[0] = h_halfedge_init


        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1
        
        # # 2. sample loop
        h_node_pert = h_node_init
        pos_pert = pos_init
        h_halfedge_pert = h_halfedge_init
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
       
        frag_node = h_node_pert[frag_mask_batch_node]
        frag_pos = pos_pert[frag_mask_batch_node]
        frag_edge = h_halfedge_pert[frag_mask_batch_edge]
        
        for i, step in tqdm(enumerate(range(num_steps)[::-1]), total=num_steps,desc="num_timesteps"):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
            
            
            h_node_frag_addnoise, _, _ = self.node_transition.add_noise(node_type_batch[frag_mask_batch_node], time_step, batch_node[frag_mask_batch_node])
            h_node_pert[frag_mask_batch_node] = h_node_frag_addnoise
            halfedge_frag_addnoise, _, _ = self.edge_transition.add_noise(halfedge_type_batch[frag_mask_batch_edge], time_step, batch_halfedge[frag_mask_batch_edge])
            
            h_halfedge_pert[frag_mask_batch_edge]= halfedge_frag_addnoise
            h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)
            
            pos_pert[frag_mask_batch_node] = self.pos_transition.add_noise(pos_init[frag_mask_batch_node], time_step, batch_node[frag_mask_batch_node])
            preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            h_node_pert=h_node_pert,
            pos_pert=pos_pert,
            batch_node=batch_node,
            h_edge_pert=h_edge_pert,
            edge_index=edge_index,
            batch_edge=batch_edge,
            t=time_step
            )
            
            pred_node = preds['pred_ligand_v']  # (N, num_node_types)
            pred_pos = preds['pred_ligand_pos']  # (N, 3)
            pred_halfedge = preds['pred_ligand_halfedge']  # (E//2, num_bond_types)
            
            
            # # 2 get the t - 1 state
            # pos 
            pos_prev = self.pos_transition.get_prev_from_recon(
                x_t=pos_pert, x_recon=pred_pos, t=time_step, batch=batch_node)
            if self.categorical_space == 'discrete':
                # node types
                log_node_recon = F.log_softmax(pred_node, dim=-1)
                log_node_type = self.node_transition.q_v_posterior(log_node_recon, log_node_type, time_step, batch_node, v0_prob=True)
                node_type_prev = log_sample_categorical(log_node_type)
                h_node_prev = self.node_transition.onehot_encode(node_type_prev)
                
                # halfedge types
                log_edge_recon = F.log_softmax(pred_halfedge, dim=-1)
                log_halfedge_type = self.edge_transition.q_v_posterior(log_edge_recon, log_halfedge_type, time_step, batch_halfedge, v0_prob=True)
                # log_halfedge_type = self.edge_transition.q_v_posterior(log_edge_recon, log_halfedge_type, time_step, batch_halfedge[~frag_mask_batch_edge], v0_prob=True)
                halfedge_type_prev = log_sample_categorical(log_halfedge_type)
                # if i%100==0:
                #     print(halfedge_type_prev)
                h_halfedge_prev = self.edge_transition.onehot_encode(halfedge_type_prev)
                
            else:
                h_node_prev = self.node_transition.get_prev_from_recon(
                    x_t=h_node_pert, x_recon=pred_node, t=time_step, batch=batch_node)
                h_halfedge_prev = self.edge_transition.get_prev_from_recon(
                    x_t=h_halfedge_pert, x_recon=pred_halfedge, t=time_step, batch=batch_halfedge)

            # # use guidance to modify pos
            if guidance is not None:
                gui_type, gui_scale = guidance
                if (gui_scale > 0):
                    with torch.enable_grad():
                       
                        pos_prev = pos_prev - torch.mean(pos_prev, dim=0) + torch.mean(pos_init, dim=0)
                        pos_prev[frag_mask_batch_node] = frag_pos
                        h_node_prev[frag_mask_batch_node] = frag_node
                        h_halfedge_prev[frag_mask_batch_edge] = frag_edge
                        h_edge_pert = torch.cat([h_halfedge_prev, h_halfedge_prev], dim=0)
                        
                        h_node_in = h_node_prev.detach().requires_grad_(True)
                        pos_in = pos_prev.detach().requires_grad_(True)
                        pred_bondpredictor = bond_predictor(h_node_in, pos_in, batch_node,
                                    edge_index, batch_edge, time_step)
                        

                        pred = affinity_model.affinity(protein_pos=protein_pos,
                                protein_v=protein_v,
                                batch_protein=batch_protein,
                                h_node_pert=h_node_in,
                                pos_pert=pos_in,
                                batch_node=batch_node,
                                h_edge_pert=h_edge_pert,
                                edge_index=edge_index,
                                batch_edge=batch_edge,
                                t=time_step)
                       
                        if gui_type == 'uncertainty':
                            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
                            uncertainty = uncertainty.log().sum()
                            delta = - torch.autograd.grad(uncertainty, pos_in)[0] * 0.0001
                            label = torch.ones(n_graphs).to(pos_in.device)
                            label_onehot = torch.nn.functional.one_hot(label.unsqueeze(0).to(torch.int64), 2).float().squeeze()
                            entropy = F.cross_entropy(pred,label_onehot, reduction='none')
                            entropy = entropy.log().sum()
                            pos_delta = - torch.autograd.grad(entropy, pos_in, retain_graph=True)[0] * 0.05
                            h_nodedelta = - torch.autograd.grad(entropy, h_node_in)[0] * 0.0001
                        else:
                            raise NotImplementedError(f'Guidance type {gui_type} is not implemented')
                    pos_prev = pos_prev + delta + pos_delta
                    h_node_prev = h_node_prev + h_nodedelta

            # log update
            node_traj[i+1] = h_node_prev
            pos_traj[i+1] = pos_prev
            halfedge_traj[i+1] = h_halfedge_prev
            # # 3 update t-1
            pos_pert = pos_prev
            h_node_pert = h_node_prev
            h_halfedge_pert = h_halfedge_prev
            
        pred_node[frag_mask_batch_node] = frag_node
        pred_pos[frag_mask_batch_node] = frag_pos
        pred_halfedge[frag_mask_batch_edge] = frag_edge  # (E//2, num_bond_types)
        pred_pos = pred_pos + offset[batch_node]
        
        return {
            'pred': [pred_node, pred_pos, pred_halfedge],
            'traj': [node_traj, pos_traj, halfedge_traj],

        }

    

# %%
