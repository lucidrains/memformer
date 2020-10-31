<img src="./memformer.png" width="600px"></img>

## Memformer - Pytorch (wip)

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
    num_tokens = 256,
    dim = 512,
    depth = 2,
    max_seq_len = 1024,
    num_memory_slots = 128,
    num_mem_updates = 2
)

x1 = torch.randint(0, 256, (1, 1024))
y1 = torch.randint(0, 256, (1, 1024))

x2 = torch.randint(0, 256, (1, 1024))
y2 = torch.randint(0, 256, (1, 1024))

tgt_out1, mems1 = model(x1, y1) # (1, 1024, 512), (1, 128, 512)
tgt_out2, mems2 = model(x2, y2, mems = mems1)
```

Encoder only

```python
import torch
from memformer import Memformer

model = Memformer(
    num_tokens = 256,
    dim = 512,
    depth = 2,
    max_seq_len = 1024,
    num_memory_slots = 128,
    num_mem_updates = 2,
    encoder_only = True       # only use encoder, in which output is encoded output
)

x1 = torch.randint(0, 256, (1, 1024))
x2 = torch.randint(0, 256, (1, 1024))

enc1, mems1 = model(x1) # (1, 1024, 512), (1, 128, 512)
enc2, mems2 = model(x2, mems = mems1)
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
