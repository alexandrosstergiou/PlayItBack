from math import pi, log
from functools import wraps
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from einops.layers.torch import Reduce, Rearrange

from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = ""


# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi # [T, H, W, T, num_bands]
    x = torch.cat([x.sin(), x.cos()], dim = -1) # [T, H, W, T, 2 x num_bands]
    x = torch.cat((x, orig_x), dim = -1) # [T, H, W, T, (2 x num_bands)+1]
    return x

# helper classes

class Contiguous(torch.nn.Module):
    def __init__(self):
        super(Contiguous, self).__init__()

    def forward(self,x):
        return x.contiguous()

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )


    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.temp = 2.0

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def stable_softmax(self,x):
        x = torch.nan_to_num(x)
        x -= reduce(x, '... d -> ... 1', 'max')
        return x.softmax(dim = -1)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        #sim /= self.temp
        #sim = torch.clamp(sim, min=1e-8, max=1e+8)
        attn = self.stable_softmax(sim)
        #attn = torch.nan_to_num(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class
class TemPr(nn.Module):
    def __init__(
        self,*,cfg):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Number of frames (for the network depth).
          max_freq: Maximum frequency, hyperparameter depending on how fine the data is.
          input_channels: Number of channels for each token of the input.
          num_latents: Number of latents, or induced set points, or centroids.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Perceiver paper uses 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()


        self.input_axis = 2
        self.max_freq = cfg.DECODER.MAX_FREQ
        self.num_freq_bands = cfg.DECODER.NUM_FREQ_BANDS
        self.depth = cfg.DECODER.DEPTH
        self.f_loc = cfg.DECODER.FUSION_LOC

        self.fourier_encode_data = cfg.DECODER.FOURIER_ENCODE_DATA
        fourier_channels = (self.input_axis * ((self.num_freq_bands * 2) + 1)) if self.fourier_encode_data else 0
        input_dim = fourier_channels + cfg.DECODER.INPUT_CHANNELS

        self.latents = nn.Parameter(torch.randn(cfg.DECODER.NUM_LATENTS, cfg.DECODER.LATENT_DIM))
        torch.nn.init.kaiming_uniform_(self.latents)
        self.num_classes = cfg.MODEL.NUM_CLASSES

        get_cross_attn = lambda: PreNorm(cfg.DECODER.LATENT_DIM, Attention(cfg.DECODER.LATENT_DIM, input_dim, heads = cfg.DECODER.CROSS_HEADS, dim_head = cfg.DECODER.CROSS_DIM_HEAD, dropout = cfg.DECODER.ATTN_DROPOUT), context_dim = input_dim)
        get_cross_ff = lambda: PreNorm(cfg.DECODER.LATENT_DIM, FeedForward(cfg.DECODER.LATENT_DIM, dropout = cfg.DECODER.FF_DROPOUT))

        get_latent_attn = lambda: PreNorm(cfg.DECODER.LATENT_DIM, Attention(cfg.DECODER.LATENT_DIM, heads = cfg.DECODER.LATENT_HEADS, dim_head = cfg.DECODER.LATENT_DIM_HEAD, dropout = cfg.DECODER.ATTN_DROPOUT))
        get_latent_ff = lambda: PreNorm(cfg.DECODER.LATENT_DIM, FeedForward(cfg.DECODER.LATENT_DIM, dropout = cfg.DECODER.FF_DROPOUT))

        get_depth_attn = lambda: PreNorm(cfg.DECODER.LATENT_DIM, Attention(cfg.DECODER.LATENT_DIM, heads = cfg.DECODER.LATENT_HEADS, dim_head = cfg.DECODER.LATENT_DIM_HEAD, dropout = cfg.DECODER.ATTN_DROPOUT))
        get_depth_ff = lambda: PreNorm(cfg.DECODER.LATENT_DIM, FeedForward(cfg.DECODER.LATENT_DIM, dropout = cfg.DECODER.FF_DROPOUT))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff, get_depth_attn, get_depth_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff, get_depth_attn, get_depth_ff))

        self.layers = nn.ModuleList([])
        for i in range(self.depth):
            should_cache = i > 0 and cfg.DECODER.WEIGHT_TIE_LAYERS
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for _ in range(cfg.DECODER.SELF_PER_CROSS_ATTN):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_ff(**cache_args)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.reduce = nn.Sequential(
            Reduce('s b n d -> s b d', 'mean'),
            Rearrange('s b d -> b s d'),
            nn.LayerNorm(cfg.DECODER.LATENT_DIM)
        ) if cfg.DECODER.FINAL_CLASSIFIER_HEAD else nn.Identity()

        make_contiguous = Contiguous()

        if cfg.DECODER.FUSION == 'mean':
            self.fusion = torch.nn.Sequential(
                Rearrange('b s c -> b c s'),
                make_contiguous,
                nn.AvgPool1d(kernel_size=(cfg.DECODER.DEPTH),beta=(1)),
                Rearrange('b c 1 -> b c'))
        elif cfg.DECODER.FUSION == 'adaptive' and self.f_loc=='features':
            self.fusion = torch.nn.Sequential(
                Rearrange('b s c -> b c s'),
                make_contiguous,
                nn.Conv1d(in_channels=cfg.DECODER.LATENT_DIM,
                          out_channels=cfg.DECODER.LATENT_DIM,
                          kernel_size=(cfg.DECODER.DEPTH),
                          bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                Rearrange('b c 1 -> b c'))
        elif cfg.DECODER.FUSION == 'adaptive' and self.f_loc=='predictions':
            self.fusion = torch.nn.Sequential(
                Rearrange('b s c -> b c s'),
                make_contiguous,
                nn.Conv1d(in_channels=self.num_classes,
                          out_channels=self.num_classes,
                          kernel_size=(cfg.DECODER.DEPTH),
                          groups=self.num_classes,
                          bias=False),
                Rearrange('b c 1 -> b c'))
        elif cfg.DECODER.FUSION == 'adaptive_concat' and self.f_loc=='features':
            self.fusion = torch.nn.Sequential(
                Rearrange('b s c -> b (s c)'),
                nn.Linear(cfg.DECODER.LATENT_DIM * cfg.DECODER.DEPTH, cfg.DECODER.LATENT_DIM*2),
                GEGLU(),
                nn.Dropout(0.1),
                nn.Linear(cfg.DECODER.LATENT_DIM, cfg.DECODER.LATENT_DIM))
                #Rearrange('b c 1 -> b c'))
        else:
            self.fusion = nn.Identity()


        self.fc = nn.Linear(cfg.DECODER.LATENT_DIM, self.num_classes) if cfg.DECODER.FINAL_CLASSIFIER_HEAD else nn.Identity()


    def forward(self, data, mask = None, return_embeddings = False):
        data = rearrange(data, 's b c f t -> s b f t c')
        s, b, *axis, _, device = *data.shape, data.device
        assert len(axis) == self.input_axis, 'input data has %d axes, expected %d'%(len(axis),self.input_axis)

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), axis)) # [F, T]
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim = -1) # [F, T, 3]
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)# [F, T, 3, (2 x num_bands)+1]
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')# [F, T, 3 x ((2 x num_bands)+1)]
            enc_pos = repeat(enc_pos, '... -> s b ...', s = s, b = b)# [S, B, F, T, 3 x ((2 x num_bands)+1)]

            # Create a concat version of the data and the PE
            data = torch.cat((data, enc_pos), dim = -1)# [S, B, F, T, D]

        # flatten spatio-temporal dim
        # [S, B, F, T, D] => [S, B, F x T, D]
        data = rearrange(data, 's b ... d -> s b (...) d')

        # Repeat latents over batch dim
        x = repeat(self.latents, 'n d -> b n d', b = b)

        # layers
        x_list = []

        # Main calls
        for i,(cross_attn, cross_ff, self_attns) in enumerate(self.layers):
            # Cross attention
            x = cross_attn(x, context = data[i], mask = mask) + x
            x = cross_ff(x) + x

            # Latent Transformer
            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

            x_list.append(x)

        # to logits
        x = self.reduce(torch.stack(x_list,dim=0))
        if self.f_loc=='features':
            x = self.fusion(x)
            # class predictions
            pred = self.fc(x)
        else:
            # class predictions
            pred = self.fc(x)
            pred = self.fusion(pred)

        # used for fetching embeddings
        if return_embeddings:
            return pred, x
        return pred




if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    from torchinfo import summary

    ####################################
    ##### N E T W O R K  T E S T S  ####
    ####################################

    from fvcore.common.config import CfgNode
    _C = CfgNode()
    _C.MODEL = CfgNode()
    _C.MODEL.num_classes= 309
    _C.DECODER = CfgNode()
    _C.DECODER.num_freq_bands = 6
    _C.DECODER.depth = 1
    _C.DECODER.max_freq = 10
    _C.DECODER.input_channels = 1
    _C.DECODER.num_latents = 256
    _C.DECODER.latent_dim= 512
    _C.DECODER.cross_heads= 1
    _C.DECODER.latent_heads= 8
    _C.DECODER.cross_dim_head= 64
    _C.DECODER.latent_dim_head= 64
    _C.DECODER.attn_dropout= 0.
    _C.DECODER.ff_dropout= 0.
    _C.DECODER.weight_tie_layers= False
    _C.DECODER.fourier_encode_data= True
    _C.DECODER.self_per_cross_attn= 4
    _C.DECODER.final_classifier_head= True
    _C.DECODER.FUSION= 'mean'

    #--- TEST 1 --- (train -- fp32)
    tmp = torch.rand(_C.DECODER.depth,32,1,512,128).cuda()
    net = torch.nn.DataParallel(TemPr(cfg=_C).cuda())
    out = net(tmp)
    print('--- TEST 1 (train -- fp32) passed ---','input:',tmp.shape,'exited the network with new shape:',out.shape,'\n')
    del out, net, tmp

    tmp = (_C.DECODER.depth,1,128,400)
    net = TemPr(cfg=_C)

    macs, params = get_model_complexity_info(net, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- TEST 2 passed --- ')
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print('\n')

    #--- TEST 6 --- summary
    net = TemPr(cfg=_C).cuda()
    summary(net, (_C.MODEL.depth,16,1,128,400))
    print('--- TEST 6 passed --- \n')
