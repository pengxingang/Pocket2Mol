import torch
from torch.nn import Module, Sequential
from torch.nn import functional as F
from .invariant import GVPerceptronVN, GVLinear
import math

GAUSSIAN_COEF = 1.0 / math.sqrt(2 * math.pi)

class PositionPredictor(Module):
    def __init__(self, in_sca, in_vec, num_filters, n_component):
        super().__init__()
        self.n_component = n_component
        self.gvp = Sequential(
            GVPerceptronVN(in_sca, in_vec, num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1])
        )
        self.mu_net = GVLinear(num_filters[0], num_filters[1], n_component, n_component)
        self.logsigma_net= GVLinear(num_filters[0], num_filters[1], n_component, n_component)
        self.pi_net = GVLinear(num_filters[0], num_filters[1], n_component, 1)

    def forward(self, h_compose, idx_focal, pos_compose):
        h_focal = [h[idx_focal] for h in h_compose]
        pos_focal = pos_compose[idx_focal]
        
        feat_focal = self.gvp(h_focal)
        relative_mu = self.mu_net(feat_focal)[1]  # (N_focal, n_component, 3)
        logsigma = self.logsigma_net(feat_focal)[1]  # (N_focal, n_component, 3)
        sigma = torch.exp(logsigma)
        pi = self.pi_net(feat_focal)[0]  # (N_focal, n_component)
        pi = F.softmax(pi, dim=1)
        
        abs_mu = relative_mu + pos_focal.unsqueeze(dim=1).expand_as(relative_mu)
        return relative_mu, abs_mu, sigma, pi

    def get_mdn_probability(self, mu, sigma, pi, pos_target):
        prob_gauss = self._get_gaussian_probability(mu, sigma, pos_target)
        prob_mdn = pi * prob_gauss
        prob_mdn = torch.sum(prob_mdn, dim=1)
        return prob_mdn


    def _get_gaussian_probability(self, mu, sigma, pos_target):
        """
        mu - (N, n_component, 3)
        sigma - (N, n_component, 3)
        pos_target - (N, 3)
        """
        target = pos_target.unsqueeze(1).expand_as(mu)
        errors = target - mu
        sigma = sigma + 1e-16
        p = GAUSSIAN_COEF * torch.exp(- 0.5 * (errors / sigma)**2) / sigma
        p = torch.prod(p, dim=2)
        return p # (N, n_component)

    def sample_batch(self, mu, sigma, pi, num):
        """sample from multiple mix gaussian
            mu - (N_batch, n_cat, 3)
            sigma - (N_batch, n_cat, 3)
            pi - (N_batch, n_cat)
        return
            (N_batch, num, 3)
        """
        index_cats = torch.multinomial(pi, num, replacement=True)  # (N_batch, num)
        # index_cats = index_cats.unsqueeze(-1)
        index_batch = torch.arange(len(mu)).unsqueeze(-1).expand(-1, num)  # (N_batch, num)
        mu_sample = mu[index_batch, index_cats]  # (N_batch, num, 3)
        sigma_sample = sigma[index_batch, index_cats]
        values = torch.normal(mu_sample, sigma_sample)  # (N_batch, num, 3)
        return values

    def get_maximum(self, mu, sigma, pi):
        """sample from multiple mix gaussian
            mu - (N_batch, n_cat, 3)
            sigma - (N_batch, n_cat, 3)
            pi - (N_batch, n_cat)
        return
            (N_batch, n_cat, 3)
        """
        return mu



