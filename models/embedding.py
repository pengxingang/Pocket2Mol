import torch
from torch.nn import Module, Linear, Embedding
from torch.nn import functional as F


class AtomEmbedding(Module):
    def __init__(self, in_scalar, in_vector,
                 out_scalar, out_vector, vector_normalizer=20.):
        super().__init__()
        assert in_vector == 1
        self.in_scalar = in_scalar
        self.vector_normalizer = vector_normalizer
        self.emb_sca = Linear(in_scalar, out_scalar)
        self.emb_vec = Linear(in_vector, out_vector)

    def forward(self, scalar_input, vector_input):
        vector_input = vector_input / self.vector_normalizer
        assert vector_input.shape[1:] == (3, ), 'Not support. Only one vector can be input'
        sca_emb = self.emb_sca(scalar_input[:, :self.in_scalar])  # b, f -> b, f'
        vec_emb = vector_input.unsqueeze(-1)  # b, 3 -> b, 3, 1
        vec_emb = self.emb_vec(vec_emb).transpose(1, -1)  # b, 1, 3 -> b, f', 3
        return sca_emb, vec_emb
        
