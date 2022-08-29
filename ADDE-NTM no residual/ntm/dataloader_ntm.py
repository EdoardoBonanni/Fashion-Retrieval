import torch

# dataloader for attribute feature
def dataloader_ntm(batch_size, seq=None):
    """

    :param batch_size: Batch size.
    :param seq: Sequence.

    seq_width: The width of each item in the sequence.
    seq_len: Sequence length.
    """

    # seq = torch.transpose(seq, 0, 1)
    # seq = torch.reshape(seq, (seq.shape[0], 1, seq.shape[1]))
    # seq = torch.reshape(seq, (seq.shape[0], seq.shape[1], 1))

    inp = seq.clone()
    outp = seq.clone()

    yield batch_size, inp.float(), outp.float()