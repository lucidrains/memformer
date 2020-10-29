import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

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

def rel_shift(t):
    b, h, i, j, device, dtype = *t.shape, t.device, t.dtype
    zero_pad = torch.zeros((b, h, i, 1), device = device, dtype = dtype)
    concatted = torch.cat([zero_pad, t], dim = -1)
    shifted = concatted.view(b, h, j + 1, i)[:, :, 1:]
    return shifted.view_as(t)

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, context_len = 0):
        n, device = x.shape[1], x.device
        l = n + context_len
        t = torch.arange(l - 1, -1, -1, device = device).type_as(self.inv_freq)
        sinusoid_inp = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim = -1)
        return emb

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
 
        self.to_pos = nn.Linear(dim, dim_head) if rel_pos_emb else None
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, context = None, pos_emb = None, mask = None, attend_self = False):
        _, n, _, h, scale, device = *x.shape, self.heads, self.scale, x.device

        if attend_self:
            kv_input = torch.cat((x, context), dim = 1)
        else:
            kv_input = default(context, x)

        q = self.to_q(x)
        kv = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        if exists(self.to_pos):
            p = self.to_pos(pos_emb)
            pos_attn = einsum('b h i d, j d -> b h i j', q, p) * scale
            pos_attn = rel_shift(pos_attn)
            dots += pos_attn

        if self.causal:
            causal_mask = torch.ones((n, n), device = device).triu_(1).bool()
            dots.masked_fill_(causal_mask, float('-inf'))
            del causal_mask

        if exists(mask):
            mask = rearrange(mask, 'i j -> () () i j')
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Encoder(nn.Module):
    def __init__(self, dim, depth, heads = 8):
        super().__init__()
        self.sinu_emb = SinusoidalEmbedding(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, rel_pos_emb = True))),
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim)))
            ]))
    def forward(self, x, context = None):
        pos_emb = self.sinu_emb(x)
        for (self_attn, cross_attn, ff) in self.layers:
            x = self_attn(x, pos_emb = pos_emb)
            x = cross_attn(x, context = context)
            x = ff(x)
        return x

class Decoder(nn.Module):
    def __init__(self, dim, depth, heads = 8):
        super().__init__()
        self.sinu_emb = SinusoidalEmbedding(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, causal = True, rel_pos_emb = True))),
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim))),
            ]))
    def forward(self, x, context = None):
        pos_emb = self.sinu_emb(x)
        for (self_attn, cross_attn, ff) in self.layers:
            x = self_attn(x, pos_emb = pos_emb)
            x = cross_attn(x, context = context)
            x = ff(x)
        return x

class TransformerWrapper(nn.Module):
    def __init__(self, *, num_tokens, max_seq_len, dim, layer_blocks, heads = 8, return_logits = True):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
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
        heads = 8):
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
        )

        self.num_mem = num_memory_slots
        self.memory_slots = nn.Parameter(torch.randn(num_memory_slots, dim))

        self.num_mem_updates = num_mem_updates
        self.mem_updater = Attention(dim, heads = heads)
        self.gru = nn.GRUCell(dim, dim)
        self.mem_ff = Residual(PreNorm(dim, FeedForward(dim)))

    def forward(self, src, tgt, mems = None):
        b, n, num_mem, device = *src.shape, self.num_mem, src.device
        mems = default(mems, self.memory_slots)

        if mems.ndim == 2:
            mems = repeat(mems, 'n d -> b n d', b = b)

        enc = self.encoder(src, context = mems)
        out = self.decoder(tgt, context = enc)

        # update memory with attention
        mem_mask = torch.eye(num_mem, num_mem, device = device).bool()
        mem_mask = F.pad(mem_mask, (0, n), value = True)

        for _ in range(self.num_mem_updates):
            prev_mems = mems
            updated_mems = self.mem_updater(mems, enc, mask = mem_mask, attend_self = True)

            next_mems = self.gru(
                rearrange(updated_mems, 'b n d -> (b n) d'),
                rearrange(prev_mems, 'b n d -> (b n) d')
            )

            mems = rearrange(next_mems, '(b n) d -> b n d', b = b)
            mems = self.mem_ff(mems)

        return out, mems
