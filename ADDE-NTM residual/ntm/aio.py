"""All in one NTM. Encapsulation of all components."""
import torch
from torch import nn
from .ntm import NTM
from .controller import LSTMController
from .head import NTMReadHead, NTMWriteHead
from .memory import NTMMemory


class EncapsulatedNTM(nn.Module):

    def __init__(self, num_inputs, num_outputs,
                 controller_size, controller_layers, num_heads, N, M):
        """Initialize an EncapsulatedNTM.

        :param num_inputs: External number of inputs.
        :param num_outputs: External number of outputs.
        :param controller_size: The size of the internal representation.
        :param controller_layers: Controller number of layers.
        :param num_heads: Number of heads.
        :param N: Number of rows in the memory bank.
        :param M: Number of cols/features in the memory bank.
        """
        super(EncapsulatedNTM, self).__init__()

        # Save args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_size = controller_size
        self.controller_layers = controller_layers
        self.num_heads = num_heads
        self.N = N
        self.M = M

        # Create the NTM components
        # memory = NTMMemory(N, M)
        # memory = NTMMemory(1, M)
        memory = NTMMemory(N, M)
        controller = LSTMController(num_inputs + M*num_heads, controller_size, controller_layers)
        heads = nn.ModuleList([])
        for i in range(num_heads):
            heads += [
                NTMReadHead(memory, controller_size),
                NTMWriteHead(memory, controller_size)
            ]

        self.ntm = NTM(num_inputs, num_outputs, controller, memory, heads)
        self.memory = memory

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        self.memory.reset(batch_size)
        self.previous_state = self.ntm.create_new_state(batch_size)

    def forward(self, x=None, ind=None, operation=False):
        if x is None:
            x = torch.zeros(self.batch_size, self.num_inputs).cuda()

        o, self.previous_state = self.ntm(x, self.previous_state)
        if operation:
            output_memory = self.memory.memory_state_tensor()
            residual_feat = torch.zeros(ind.shape[0], output_memory.shape[1]).cuda()
            for i in range(0, ind.shape[0]):
                pos_array = torch.zeros(1, output_memory.shape[1]).cuda()
                neg_array = torch.zeros(1, output_memory.shape[1]).cuda()
                for j in range(0, ind.shape[1]):
                    if ind[i][j] == -1:
                        neg_array = torch.reshape(-output_memory[i, :, j], (1, output_memory.shape[1]))
                    elif ind[i][j] == 1:
                        pos_array = torch.reshape(output_memory[i, :, j], (1, output_memory.shape[1]))
                residual_feat[i] = pos_array + neg_array
        else:
            output_memory = None
            residual_feat = None
        return o, self.previous_state, output_memory, residual_feat

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
