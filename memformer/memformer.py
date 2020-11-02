import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from einops import rearrange, repeat
from collections import namedtuple
from memformer.autoregressive_wrapper import AutoregressiveWrapper

# constants

Results = namedtuple('Results', ['enc_out', 'mem', 'dec_out'])
EncOnlyResults = namedtuple('EncOnlyResults', ['enc_out', 'mem'])

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# positional embedding

class RelativePositionBias(nn.Module):
    def __init__(self, causal = False, num_buckets = 32, max_distance = 128, heads = 8):
        super().__init__()
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qlen, klen):
        device = self.relative_attention_bias.weight.device
        q_pos = torch.arange(qlen, dtype = torch.long, device = device)
        k_pos = torch.arange(klen, dtype = torch.long, device = device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> () h i j')

# main classes

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, causal = False, rel_pos_emb = False):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'
        dim_head = dim // heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.causal = causal
 
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, context = None, pos_emb = None, mask = None, query_mask = None, kv_mask = None, attend_self = False):
        b, n, _, h, scale, device = *x.shape, self.heads, self.scale, x.device

        if attend_self:
            kv_input = torch.cat((x, context), dim = 1)
        else:
            kv_input = default(context, x)

        q = self.to_q(x)
        kv = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        if exists(pos_emb):
            pos_emb_bias = pos_emb(*dots.shape[-2:])
            dots += pos_emb_bias

        if self.causal:
            causal_mask = torch.ones((n, n), device = device).triu_(1).bool()
            dots.masked_fill_(causal_mask, float('-inf'))
            del causal_mask

        if any(map(exists, (query_mask, kv_mask))):
            query_mask = default(query_mask, lambda: torch.ones((b, n), device = device).bool())

            if exists(context):
                kv_mask = default(kv_mask, lambda: torch.ones((b, context.shape[1]), device = device).bool())
            else:
                kv_mask = default(kv_mask, query_mask)

            query_mask = rearrange(query_mask, 'b i -> b () i ()')
            kv_mask = rearrange(kv_mask, 'b j -> b () () j')
            seq_mask = query_mask * kv_mask
            dots.masked_fill_(~seq_mask, float('-inf'))
            del seq_mask

        if exists(mask):
            mask = rearrange(mask, 'b i j -> b () i j')
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Encoder(nn.Module):
    def __init__(self, dim, depth, heads = 8):
        super().__init__()
        self.rel_pos_emb = RelativePositionBias(heads = heads)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, rel_pos_emb = True))),
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim)))
            ]))
    def forward(self, x, context = None, src_mask = None):
        for (self_attn, cross_attn, ff) in self.layers:
            x = self_attn(x, pos_emb = self.rel_pos_emb, query_mask = src_mask)
            x = cross_attn(x, context = context)
            x = ff(x)
        return x

class Decoder(nn.Module):
    def __init__(self, dim, depth, heads = 8):
        super().__init__()
        self.rel_pos_emb = RelativePositionBias(heads = heads, causal = True)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, causal = True, rel_pos_emb = True))),
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim))),
            ]))
    def forward(self, x, context = None, src_mask = None, tgt_mask = None):
        for (self_attn, cross_attn, ff) in self.layers:
            x = self_attn(x, pos_emb = self.rel_pos_emb, query_mask = src_mask)
            x = cross_attn(x, context = context, query_mask = src_mask, kv_mask = tgt_mask)
            x = ff(x)
        return x

class TransformerWrapper(nn.Module):
    def __init__(self, *, num_tokens, max_seq_len, dim, layer_blocks, heads = 8, return_logits = True):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.max_seq_len = max_seq_len
        self.layer_blocks = layer_blocks
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens) if return_logits else nn.Identity()

    def forward(self, x, **kwargs):
        _, n, device = *x.shape, x.device
        x = self.token_emb(x)
        x = self.layer_blocks(x, **kwargs)
        x = self.norm(x)
        return self.to_logits(x)

class Memformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        num_memory_slots,
        num_mem_updates = 1,
        heads = 8,
        encoder_only = False):
        super().__init__()

        self.encoder = TransformerWrapper(
            num_tokens = num_tokens,
            dim = dim,
            max_seq_len = max_seq_len,
            layer_blocks = Encoder(dim, depth, heads),
            return_logits = False
        )

        self.decoder = TransformerWrapper(
            num_tokens = num_tokens,
            dim = dim,
            max_seq_len = max_seq_len,
            layer_blocks = Decoder(dim, depth, heads),
            return_logits = True
        ) if not encoder_only else None

        if exists(self.decoder):
            self.decoder = AutoregressiveWrapper(self.decoder)

        self.num_mem = num_memory_slots
        self.memory_slots = nn.Parameter(torch.randn(num_memory_slots, dim))

        self.num_mem_updates = num_mem_updates
        self.mem_updater = Attention(dim, heads = heads)
        self.gru = nn.GRUCell(dim, dim)
        self.mem_ff = Residual(PreNorm(dim, FeedForward(dim)))

    def forward(self, src, tgt = None, mems = None, src_mask = None, tgt_mask = None):
        b, n, num_mem, device = *src.shape, self.num_mem, src.device
        mems = default(mems, self.memory_slots)

        if mems.ndim == 2:
            mems = repeat(mems, 'n d -> b n d', b = b)

        enc = self.encoder(src, context = mems, src_mask = src_mask)

        if exists(self.decoder) and exists(tgt):
            dec_out = self.decoder(tgt, context = enc, src_mask = tgt_mask, tgt_mask = src_mask, return_loss = True)
        else:
            dec_out = torch.tensor(0., requires_grad = True, device = device)

        # update memory with attention
        mem_mask = torch.eye(num_mem, num_mem, device = device).bool()
        mem_mask = repeat(mem_mask, 'i j -> b i j', b = b)
        mem_mask = F.pad(mem_mask, (0, n), value = True)

        if exists(src_mask):
            src_mask = rearrange(src_mask, 'b j -> b () j')
            mem_enc_mask = F.pad(src_mask, (num_mem, 0), value = True)
            mem_mask &= mem_enc_mask

        for _ in range(self.num_mem_updates):
            prev_mems = mems
            updated_mems = self.mem_updater(mems, enc, mask = mem_mask, attend_self = True)

            next_mems = self.gru(
                rearrange(updated_mems, 'b n d -> (b n) d'),
                rearrange(prev_mems, 'b n d -> (b n) d')
            )

            mems = rearrange(next_mems, '(b n) d -> b n d', b = b)
            mems = self.mem_ff(mems)

        if not exists(self.decoder):
            return EncOnlyResults(enc, mems)

        return Results(enc, mems, dec_out)
