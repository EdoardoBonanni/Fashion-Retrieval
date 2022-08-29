#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import utilities
import torch
import numpy as np
from ntm.aio import EncapsulatedNTM
from ntm.dataloader_ntm import dataloader_ntm


def single_manipulation_train(model, optimizer, X, ind, input_memory):
    """Single manipulation of memory."""

    batch_size, inp_seq_len = X.size()

    # New sequence
    # model.model_ntm.memory.init_mem_bias(input_memory)
    model.model_ntm.init_sequence(batch_size)
    # memory_before_manip = model.model_ntm.memory.memory_state_tensor()

    # Feed the sequence
    # for i in range(inp_seq_len):
    #     if i + 1 == inp_seq_len:
    #         _, _, _, residual_feat = model.model_ntm(X[i], ind, True)
    #     else:
    #         _, _, _, _ = model.model_ntm(X[i], ind, False)

    _, _, _, residual_feat = model.model_ntm(X, ind, True)

    # memory_after_manip.cpu().detach().numpy()

    return residual_feat


def single_manipulation_eval(model, X, ind, input_memory):
    """Single manipulation of memory."""

    batch_size, inp_seq_len = X.size()

    # New sequence
    # model.model_ntm.memory.init_mem_bias(input_memory)
    model.model_ntm.init_sequence(batch_size)

    _, _, _, residual_feat = model.model_ntm(X, ind, True)

    return residual_feat


def manipulate_ntm(model, optimizer, ind, train, input_memory):

    dataloader = dataloader_ntm(ind.shape[0], ind)

    for batch_num, x, y in dataloader:
        x.cuda()
        y.cuda()
        if train:
            residual_feat = single_manipulation_train(model, optimizer, x, ind, input_memory)
        else:
            residual_feat = single_manipulation_eval(model, x, ind, input_memory)

    return residual_feat


def init_model(init_weight_memory, batch_size):
    memory_n = init_weight_memory.shape[1]
    memory_m = init_weight_memory.shape[2]
    sequence_max_len = 151
    sequence_min_len = 151
    controller_size = 100
    controller_layers = 1
    num_heads = 1
    sequence_width = 151
    num_batches = 1

    model_ntm = EncapsulatedNTM(sequence_width, sequence_width,
                    controller_size, controller_layers,
                    num_heads,
                    memory_n, memory_m)
    model_ntm.memory.init_mem_bias(init_weight_memory)

    return model_ntm

def memory_state(model):
    return model.model_ntm.memory.memory_state()

def memory_state_tensor(model):
    return model.model_ntm.memory.memory_state_tensor()

def mem_bias_state(model):
    return model.model_ntm.memory.mem_bias_state()

def mem_bias_state_tensor(model):
    return model.model_ntm.memory.mem_bias_state_tensor()