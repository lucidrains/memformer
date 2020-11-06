<img src="./memformer.png" width="600px"></img>

## Memformer - Pytorch

Implementation of Memformer, a Memory-augmented Transformer, in Pytorch. It includes memory slots, which are updated with attention, learned efficiently through Memory-Replay BackPropagation (MRBP) through time.

## Install

```bash
$ pip install memformer
```

## Usage

Full encoder / decoder, as in the paper

```python
import torch
from memformer import Memformer

model = Memformer(
    dim = 512,
    enc_num_tokens = 256,
    enc_depth = 2,
    enc_heads = 8,
    enc_max_seq_len = 1024,
    dec_num_tokens = 256,
    dec_depth = 2,
    dec_heads = 8,
    dec_max_seq_len = 1024,
    num_memory_slots = 128
)

src_seg_1 = torch.randint(0, 256, (1, 1024))
src_seg_2 = torch.randint(0, 256, (1, 1024))
src_seg_3 = torch.randint(0, 256, (1, 1024))

tgt = torch.randint(0, 256, (1, 1024))

enc_out1, mems1,    _ = model(src_seg_1) # (1, 1024, 512), (1, 128, 512), _
enc_out2, mems2,    _ = model(src_seg_2, mems = mems1)
enc_out3, mems3, loss = model(src_seg_3, tgt, mems = mems2)

loss.backward()
```

Encoder only

```python
import torch
from memformer import Memformer

model = Memformer(
    dim = 512,
    enc_num_tokens = 256,
    enc_heads = 8,
    enc_depth = 2,
    enc_max_seq_len = 1024,
    num_memory_slots = 128,
    num_mem_updates = 2,
    encoder_only = True       # only use encoder, in which output is encoded output
)

src1 = torch.randint(0, 256, (1, 1024))
src2 = torch.randint(0, 256, (1, 1024))

enc1, mems1 = model(src1) # (1, 1024, 512), (1, 128, 512)
enc2, mems2 = model(src2, mems = mems1)
```

Memory Replay Back-Propagation

```python
import torch
from memformer import Memformer, memory_replay_backprop

model = Memformer(
    dim = 512,
    num_memory_slots = 128,
    enc_num_tokens = 256,
    enc_depth = 2,
    enc_max_seq_len = 1024,
    dec_num_tokens = 256,
    dec_depth = 2,
    dec_max_seq_len = 1024
).cuda()

seq = torch.randint(0, 256, (1, 8192)).cuda()
seq_mask = torch.ones_like(seq).bool().cuda()

tgt = torch.randint(0, 256, (1, 512)).cuda()
tgt_mask = torch.ones_like(tgt).bool().cuda()

# will automatically split the source sequence to 8 segments
memory_replay_backprop(
    model,
    src = seq,
    tgt = tgt,
    src_mask = seq_mask,
    tgt_mask = tgt_mask
)
```

## Citations

```bibtex
@inproceedings{
    anonymous2021memformer,
    title={Memformer: The Memory-Augmented Transformer},
    author={Anonymous},
    booktitle={Submitted to International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=_adSMszz_g9},
    note={under review}
}
```
