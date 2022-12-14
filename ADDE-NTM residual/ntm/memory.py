"""An NTM's memory implementation."""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


def _convolve(w, s):
    """Circular convolution implementation."""
    assert s.size(0) == 3
    t = torch.cat([w[-1:], w, w[:1]]).cuda()
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    # nel caso funzionante sia t che w sono torch float32 come s
    return c


class NTMMemory(nn.Module):
    """Memory bank for NTM."""
    def __init__(self, N, M):
        """Initialize the NTM Memory matrix.

        The memory's dimensions are (batch_size x N x M).
        Each batch has it's own memory matrix.

        :param N: Number of rows in the memory.
        :param M: Number of columns/features in the memory.
        """
        super(NTMMemory, self).__init__()

        self.N = N
        self.M = M

    def init_mem_bias(self, memory_init):
        self.mem_bias = torch.tensor(memory_init).cuda()

    def reset(self, batch_size):
        """Initialize memory from bias, for start-of-sequence."""
        self.batch_size = batch_size
        self.memory = self.mem_bias.clone().repeat(1, 1, 1)
        self.memory = self.memory.type(torch.float32)
        self.memory = self.memory.cuda()

    def size(self):
        return self.N, self.M

    def mem_bias_state(self):
        return self.mem_bias.cpu().detach().numpy()

    def memory_state(self):
        return self.memory.cpu().detach().numpy()

    def memory_state_tensor(self):
        return self.memory

    def mem_bias_state_tensor(self):
        return self.mem_bias

    def read(self, w):
        """Read from memory (according to section 3.1)."""
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1).cuda()

    def write(self, w, e, a):
        """write to memory (according to section 3.2)."""
        self.prev_mem = self.memory
        self.memory = torch.Tensor(self.batch_size, self.N, self.M).cuda()
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1)).cuda()
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1)).cuda()
        self.memory = self.prev_mem * (1 - erase) + add

    def address(self, k, ??, g, s, ??, w_prev):
        """NTM Addressing (according to section 3.3).

        Returns a softmax weighting over the rows of the memory matrix.

        :param k: The key vector.
        :param ??: The key strength (focus).
        :param g: Scalar interpolation gate (with previous weighting).
        :param s: Shift weighting.
        :param ??: Sharpen weighting scalar.
        :param w_prev: The weighting produced in the previous time step.
        """
        # Content focus
        wc = self._similarity(k, ??)

        # Location focus
        wg = self._interpolate(w_prev, wc, g)
        ?? = self._shift(wg, s)
        w = self._sharpen(??, ??)

        return w

    def _similarity(self, k, ??):
        k = k.view(self.batch_size, 1, -1)
        w = F.softmax(?? * F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1), dim=1)
        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        result = torch.zeros(wg.size()).cuda()
        for b in range(self.batch_size):
            result[b] = _convolve(wg[b], s[b])
        return result

    def _sharpen(self, ??, ??):
        w = ?? ** ??
        w = torch.div(w, torch.sum(w, dim=1).cuda().view(-1, 1) + 1e-16).cuda()
        return w
