import torch
from torch import nn
import numpy as np
import math
import models.linklink as link
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

import einops
import torch

def vector_gather(vectors, indices):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[N, L, D]
        indices: Tensor[N, K] or Tensor[N]
    Returns:
        Tensor[N, K, D] or Tensor[N, D]
    """
    N, L, D = vectors.shape
    squeeze = False
    if indices.ndim == 1:
        squeeze = True
        indices = indices.unsqueeze(-1)
    N2, K = indices.shape
    assert N == N2
    indices = einops.repeat(indices, "N K -> N K D", D=D)
    out = torch.gather(vectors, dim=1, index=indices)
    if squeeze:
        out = out.squeeze(1)
    return out

class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input

class Query_model(nn.Module):
    def __init__(self, ft_dim, sd_dim, temperature=1, att_func_type='softmax', pool_type='sum', map_func=False):
        '''
        ft_dim: feature dim of image patch or text token
        sd_dim: dim of FDT
        temperature: temperature for softmax or sparsemax
        att_func_type: attention normlization function type
        pool_type: pooling type for attention weights
        '''

        super().__init__()

        #activation 
        assert att_func_type in ['softmax', 'sigmoid', 'sparsemax']
        self.att_func_type = att_func_type

        assert pool_type in ['mean', 'max', 'sum']
        self.pool_type = pool_type

        if self.att_func_type == 'softmax':
            self.att_activation = nn.Softmax(dim=-1)
        elif self.att_func_type == 'sparsemax':
            self.att_activation = Sparsemax(dim=-1)
        else:
            self.att_activation = nn.Sigmoid()

        self.att_dim = sd_dim
        self.temperature = temperature
        self.map_func = map_func
        
        #map patch/text tokens to codebook (query) spaces
        #---note that we donot use mapping for FDT
        if self.map_func: 
            self.q_map = nn.Sequential(
                nn.Linear(ft_dim, sd_dim),
                #nn.GELU(),
            )

    def forward(self, ft, sd, mask=None, return_token_att=False, temperature=1):
        '''
        Args:
            ft: [batch, token_num, ft_dim]
            sd: [FDT_num, sd_dim]
            mask: [batch, token_num]: mask for padded tokens.
            return_token_att: flag for returning attention weights before nomalization.
            used for visualizing FDT.
        Returns:

        '''

        #map image/text token to query space and make sure at the same dim
        if self.map_func:
            q = self.q_map(ft) #bacth, token_num, dim
        else:
            q = ft

        k = sd #sd_num, sd_dim
        k = k.unsqueeze(0) #[1, sd_num, sd_dim]
        k = k.transpose(2, 1) #[1,sd_dim, sd_num]
        
        #-----calculate inner dot
        inner_dot = torch.matmul(q, k) #[bacth, token_num, code_num]

        if return_token_att:
            token_att = inner_dot
        inner_dot = inner_dot / math.sqrt(self.att_dim) #scale dot norm

        #----get attention weights
        att_weight = torch.softmax(inner_dot.permute(0,2,1), dim=-1)
        att_ft = torch.bmm(att_weight, q) #[b, T, sd_dim]

        if return_token_att:
            return token_att, att_ft, sd

        return att_weight, att_ft, sd

class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.rank = link.get_rank()
        ctx.world_size = link.get_world_size()

        #         y = tensor.new(ctx.world_size, *tensor.size())

        y = [tensor.new(*tensor.size()) for _ in range(ctx.world_size)]

        link.allgather(y, tensor)  # call pytorch all togherer

        y = torch.cat(y, 0).view(-1, *tensor.size())

        return y

    @staticmethod
    def backward(ctx, grad_output):
        in_grad = torch.zeros_like(grad_output)
        in_grad.copy_(grad_output)
        # sum grad for gathered tensor
        link.allreduce(in_grad)
        # split
        return in_grad[ctx.rank]

class ClipInfoCELoss(_Loss):
    # def __init__(self, partition_num):
    def __init__(self):
        super(ClipInfoCELoss, self).__init__()
        # self.partition_num = partition_num

    def forward(self, logits_per_image, logits_per_text):
        bs, l_bs = logits_per_image.shape
        if l_bs == bs:
            labels = torch.arange(len(logits_per_image)).cuda()
        else:
            labels = link.get_rank() * bs + torch.arange(0, bs, dtype=torch.long).cuda()

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i+loss_t)/2
        return loss, labels

class NT_Xent(_Loss):
    r"""The normalized temperature-scaled cross entropy loss, based on
    `"A Simple Framework for Contrastive Learning of Visual Representations" <https://arxiv.org/abs/2002.05709>`_
    """

    def __init__(self, batch_size, temperature=1):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(batch_size)

        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        p1 = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(p1.unsqueeze(1), p1.unsqueeze(0)) / self.temperature
        #print("sim:", sim.shape, sim.tolist())
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            self.batch_size * 2, 1
        )
        negative_samples = sim[self.mask].reshape(self.batch_size * 2, -1)

        labels = torch.zeros(self.batch_size * 2).to(z_i.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.criterion(logits, labels)
        #loss /= 2 * self.batch_size
        return loss


class NT_Xent_gather(_Loss):
    r"""The normalized temperature-scaled cross entropy loss, based on
    `"A Simple Framework for Contrastive Learning of Visual Representations" <https://arxiv.org/abs/2002.05709>`_
    """

    def __init__(self, batch_size, temperature=0.1):
        super(NT_Xent_gather, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask_positive = None
        self.mask_negative = None

        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_ib, z_j, z_jb, temperature=None):  # z_ib, z_jb: in-batch
        bs = z_i.shape[0]
        assert bs == self.batch_size
        l_bs = z_ib.shape[0]

        if temperature is None:
            temperature = self.temperature

        p0 = torch.cat((z_i, z_j), dim=0)
        p1 = torch.cat((z_ib, z_jb), dim=0)
        sim = self.similarity_f(p0.unsqueeze(1), p1.unsqueeze(0)) / self.temperature

        if self.mask_positive is None:
            ids = torch.arange(0, bs, dtype=torch.long).to(z_i.device)
            labels = link.get_rank() * bs + torch.arange(0, bs, dtype=torch.long).to(z_i.device)
            # positive samples
            self.mask_positive = torch.zeros([bs*2, l_bs*2]).bool()
            self.mask_positive[ids+bs, labels] = 1
            self.mask_positive[ids, labels+l_bs] = 1
            # negative samples
            self.mask_negative = torch.ones([bs*2, l_bs*2]).bool()
            self.mask_negative[ids, labels] = 0
            self.mask_negative[ids+bs, labels] = 0
            self.mask_negative[ids, labels+l_bs] = 0
            self.mask_negative[ids+bs, labels+l_bs] = 0

        positive_samples = sim[self.mask_positive].reshape(self.batch_size * 2, -1)
        negative_samples = sim[self.mask_negative].reshape(self.batch_size * 2, -1)

        labels = torch.zeros(self.batch_size * 2).to(z_i.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size
        return loss

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res