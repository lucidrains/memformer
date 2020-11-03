import torch
from operator import itemgetter

def memory_replay_backprop(
    model,
    src,
    tgt,
    src_mask = None,
    tgt_mask = None
):
    b, *_ = src.shape

    # get initial memory and max sequence length from encoder
    mem_init = model.get_initial_mem(b)
    max_seq_len = model.encoder.max_seq_len

    # instantiate memory replay buffer
    replay_buffer = [mem_init]

    # split sequences and masks
    src_segs = src.split(max_seq_len, dim = 1)
    num_segs = len(src_segs)
    src_mask_segs = src_mask.split(max_seq_len, dim = 1) if src_mask is not None else ((None,) * num_segs)

    # for now, assume target sequence and mask is passed at the very last segment
    # todo - allow to tether a target sequence at any point in the segment
    #        and attach custom loss to encoder output
    tgt_segs = ((None,) * (num_segs - 1)) + (tgt,)
    tgt_mask_segs = ((None,) * (num_segs - 1)) + (tgt_mask,)

    # run forwards and gather all memories
    prev_mem = mem_init
    with torch.no_grad():
        for i in range(num_segs - 1):
            src, src_mask = map(itemgetter(i), (src_segs, src_mask_segs))
            _, mem, _ = model(src, src_mask = src_mask, mems = prev_mem)
            replay_buffer.append(mem)
            prev_mem = mem

    # do backpropagation one segment at a time from last step to first
    mem_grad = torch.zeros_like(prev_mem)
    for i in reversed(range(num_segs)):
        src, src_mask, tgt, tgt_mask, mems = map(itemgetter(i), (src_segs, src_mask_segs, tgt_segs, tgt_mask_segs, replay_buffer))
        mems = mems.requires_grad_()

        _, mems_next, tgt_loss = model(src = src, tgt = tgt, src_mask = src_mask, tgt_mask = tgt_mask, mems = mems)
        tgt_loss.backward(retain_graph = True)
        mems_next.backward(mem_grad, retain_graph = True)

        # if not the last step, pass the next memory's gradient back a step
        if i != 0:
            mem_grad.copy_(mems.grad.data)
