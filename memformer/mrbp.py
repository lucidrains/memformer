import torch

def memory_replay_backprop(
    model,
    src,
    tgt,
    src_mask = None,
    tgt_mask = None
):
    b, *_ = src.shape

    mem_init = model.get_initial_mem(b)
    max_seq_len = model.encoder.max_seq_len

    replay_buffer = [mem_init]

    src_segs = src.split(max_seq_len, dim = 1)
    num_segs = len(src_segs)
    src_mask_segs = src_mask.split(max_seq_len, dim = 1) if src_mask is not None else ((None,) * num_segs)

    tgt_segs = ((None,) * (num_segs - 1)) + (tgt,)
    tgt_mask_segs = ((None,) * (num_segs - 1)) + (tgt_mask,)

    prev_mem = mem_init
    with torch.no_grad():
        for i in range(num_segs - 1):
            src, src_mask = map(lambda arr: arr[i], (src_segs, src_mask_segs))
            _, mem, _ = model(src, src_mask = src_mask, mems = prev_mem)
            replay_buffer.append(mem)
            prev_mem = mem

    mem_grad = torch.zeros_like(prev_mem)
    for i in reversed(range(num_segs)):
        src, src_mask, tgt, tgt_mask, mems = map(lambda arr: arr[i], (src_segs, src_mask_segs, tgt_segs, tgt_mask_segs, replay_buffer))
        _, mem, tgt_loss = model(src = src, tgt = tgt, src_mask = src_mask, tgt_mask = tgt_mask)
        tgt_loss.backward(retain_graph = True)
        mem.backward(mem_grad, retain_graph = True)
