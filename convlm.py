"""
ConVLM in PyTorch
"""
import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import trunc_normal_, lecun_normal_, to_2tuple
from timm.models.registry import register_model

from helpers import complement_idx

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'deit_small_patch16_304': _cfg(
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        input_size=(3, 304, 304)),
    'deit_small_patch16_288': _cfg(
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        input_size=(3, 288, 288)),
    'deit_small_patch16_272': _cfg(
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        input_size=(3, 272, 272)),
    # patch models (weights from official Google JAX impl)

    # deit models (FB weights)
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0),
    'deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0,
        classifier=('head', 'head_dist')),
}


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.norm_layer = norm_layer

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., keep_rate=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.keep_rate = keep_rate
        assert 0 < keep_rate <= 1, "keep_rate must > 0 and <= 1, got {0}".format(keep_rate)

    def forward(self, x, keep_rate=None, tokens=None,aux_attr = None):
        if keep_rate is None:
            keep_rate = self.keep_rate
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        if aux_attr is not None:
            q[:,:,0,:] = 0.9*q[:,:,0,:]+ 0.1*aux_attr
        token_cls = q[:,:,0,:].reshape(B,-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        left_tokens = N - 1
        if self.keep_rate < 1 and keep_rate < 1 or tokens is not None:  # double check the keep rate
            left_tokens = math.ceil(keep_rate * (N - 1))
            if tokens is not None:
                left_tokens = tokens
            if left_tokens == N - 1:
                return x, None, None, None, left_tokens
            assert left_tokens >= 1
            cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
            _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

            return x, index, idx, cls_attn, left_tokens,token_cls

        return  x, None, None, None, left_tokens,token_cls


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_rate=0.,
                 fuse_token=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop, keep_rate=keep_rate)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.keep_rate = keep_rate
        self.mlp_hidden_dim = mlp_hidden_dim
        self.fuse_token = fuse_token

    def forward(self, x, keep_rate=None, tokens=None, get_idx=False,aux_attr=None):
        if keep_rate is None:
            keep_rate = self.keep_rate  # this is for inference, use the default keep rate
        B, N, C = x.shape

        tmp, index, idx, cls_attn, left_tokens,token_cls = self.attn(self.norm1(x), keep_rate, tokens,aux_attr)
        x = x + self.drop_path(tmp)

        if index is not None:
            # B, N, C = x.shape
            non_cls = x[:, 1:]
            x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]

            if self.fuse_token:
                compl = complement_idx(idx, N - 1)  # [B, N-1-left_tokens]

                #ablation study
                # index_rand = torch.LongTensor(random.sample(range(compl.shape[1]), int(compl.shape[1]/4))).sort()[0].cuda()
                # compl = compl.index_select(1, index_rand)

                non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]

                non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
                extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
                x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
            else:
                x = torch.cat([x[:, 0:1], x_others], dim=1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        n_tokens = x.shape[1] - 1
        if get_idx and index is not None:
            return x, n_tokens, idx,token_cls
        return x, n_tokens, None,token_cls


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

class AttributeTrans(nn.Module):
    def __init__(self,att_dim,embed_dim) -> None:
        super(AttributeTrans,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(att_dim,embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim,embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2,embed_dim),
            nn.Sigmoid(),

        )
        self.apply(weights_init)

    def forward(self,x):
        out = self.fc(x)
        return out

class Token_cls_reg(nn.Module):
    def __init__(self,embed_dim,att_dim):
        super(Token_cls_reg,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim,att_dim),
            nn.ReLU(),
            nn.Linear(att_dim,att_dim),
            nn.ReLU(),
        )

    def forward(self,x):
        return self.fc(x)


class ConVLM(nn.Module):
    """ ConVLM """

    def __init__(self,img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=1024, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', keep_rate=(1, ), fuse_token=False,dataset=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.img_size = img_size
        if len(keep_rate) == 1:
            keep_rate = keep_rate * depth
        self.keep_rate = keep_rate
        self.depth = depth
        self.first_shrink_idx = depth
        for i, s in enumerate(keep_rate):
            if s < 1:
                self.first_shrink_idx = i
                break
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        #self.att = att
        self.dataset = dataset

        self.aux_head = nn.Sequential(
            nn.Linear(embed_dim,embed_dim*2,bias=False),
            nn.ReLU(),
            nn.Linear(embed_dim*2,512,bias=False),
            nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool1d(output_size=512)
        self.num_heads = num_heads
        if self.dataset == 'CAM16':
            atttovis_list = [AttributeTrans(1024,embed_dim) for _ in range(3)]
            self.atttovis = nn.ModuleList(atttovis_list)
            token_reg_list = [Token_cls_reg(embed_dim,1024) for _ in range(3)]
            self.token_cls_reg = nn.ModuleList(token_reg_list)
        elif self.dataset == 'RCC_DHMC'':
            atttovis_list = [AttributeTrans(1024,embed_dim) for _ in range(3)]
            self.atttovis = nn.ModuleList(atttovis_list)
            token_reg_list = [Token_cls_reg(embed_dim,1024) for _ in range(3)]
            self.token_cls_reg = nn.ModuleList(token_reg_list)
        else:
            atttovis_list = [AttributeTrans(1024,embed_dim) for _ in range(3)]
            self.atttovis = nn.ModuleList(atttovis_list)
            token_reg_list = [Token_cls_reg(embed_dim,1024) for _ in range(3)]
            self.token_cls_reg = nn.ModuleList(token_reg_list)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                keep_rate=keep_rate[i], fuse_token=fuse_token)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    @property
    def name(self):
        return "EViT"

    def forward_features(self, x, keep_rate=None, tokens=None, get_idx=False,label = None,att=None):
        _, _, h, w = x.shape
        if not isinstance(keep_rate, (tuple, list)):
            keep_rate = (keep_rate, ) * self.depth
        if not isinstance(tokens, (tuple, list)):
            tokens = (tokens, ) * self.depth
        assert len(keep_rate) == self.depth
        assert len(tokens) == self.depth
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        # for input with another resolution, interpolate the positional embedding.
        # used for finetining a ViT on images with larger size.
        pos_embed = self.pos_embed
        #print("value before matrix multiplication")
        #print("value np.shape(pos_embed)", pos_embed)
        if x.shape[1] != pos_embed.shape[1]:
            assert h == w  # for simplicity assume h == w
            real_pos = pos_embed[:, self.num_tokens:]
            #print("real_pos.shape ", real_pos.shape, "real_pos.shape[1]", real_pos.shape[1])
            #print("value inside if condition")

            hw = int(math.sqrt(real_pos.shape[1]))
            #print("value after first multiplication")
            #print("x.shape ", x.shape)
            #print("self.num_tokens ", self.num_tokens)
            #print ("x.shape[1] - self.num_tokens ", x.shape[1] - self.num_tokens)
            true_hw = int(math.sqrt(x.shape[1] - self.num_tokens))
            #print("value after 2nd multiplication", true_hw)
            real_pos = real_pos.transpose(1, 2).reshape(1, self.embed_dim, hw, hw)
            new_pos = F.interpolate(real_pos, size=true_hw, mode='bicubic', align_corners=False)
            new_pos = new_pos.reshape(1, self.embed_dim, -1).transpose(1, 2)
            pos_embed = torch.cat([pos_embed[:, :self.num_tokens], new_pos], dim=1)
            #print("value afor pos_embed ", pos_embed)
        x = self.pos_drop(x + pos_embed)
        #print ("value for x ", x)
        left_tokens = []
        idxs = []
        aux = []
        p_count = 0
        token_cls_reg = []
        for i, blk in enumerate(self.blocks):
            #print("inside seconf for loop")
            if self.keep_rate[i] < 1 and self.training and att is not None:
               aux_attr = self.atttovis[p_count](att[label]).reshape(-1,self.num_heads, self.embed_dim // self.num_heads)
               aux.append(aux_attr.reshape(-1,self.embed_dim))
               x, left_token, idx,token_cls = blk(x, keep_rate[i], tokens[i], get_idx,aux_attr)
               token_cls_reg_p = self.token_cls_reg[p_count](token_cls)
               token_cls_reg.append(token_cls_reg_p)
               p_count += 1
            else:
                x, left_token, idx,token_cls = blk(x, keep_rate[i], tokens[i], get_idx)
            left_tokens.append(left_token)

            if idx is not None:
                idxs.append(idx)
        x = self.norm(x)
        if self.dist_token is None:
            x_aux= (self.aux_head(x[:,1:])).max(1)[0]
            return self.pre_logits(x[:, 0]), left_tokens, idxs,x_aux,aux,token_cls_reg
        else:
            return x[:, 0], x[:, 1], idxs

    def forward(self, x, keep_rate=None, tokens=None, get_idx=False,label=None,att=None):
        x, _, idxs, x_aux, aux,token_cls_reg= self.forward_features(x, keep_rate, tokens, get_idx,label,att)
        if get_idx:
            return x, idxs
        return x,x_aux,aux,token_cls_reg



def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('aux'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: ConVLM, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_evit(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    default_cfg.update(kwargs)
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        ConVLM, variant, pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs)
    return model


@register_model
def ConVLM_base(pretrained=False, base_keep_rate=0.7, drop_loc=(3, 6, 9), **kwargs):
    keep_rate = [1] * 12
    print ("Context guided vision transformer")
    for loc in drop_loc:
        keep_rate[loc] = base_keep_rate
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=12, num_heads=12, keep_rate=keep_rate)
    model_kwargs.update(kwargs)
    model = _create_evit('deit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_evit('deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_evit('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


# -------------------------------------------------------------
# EViT prototype models
@register_model
def deit_small_patch16_shrink_base(pretrained=False, base_keep_rate=0.7, drop_loc=(3, 6, 9), **kwargs):
    keep_rate = [1] * 12
    for loc in drop_loc:
        keep_rate[loc] = base_keep_rate
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, keep_rate=keep_rate)
    model_kwargs.update(kwargs)
    model = _create_evit('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def deit_base_patch16_shrink_base(pretrained=False, base_keep_rate=0.7, drop_loc=(3, 6, 9), **kwargs):
    keep_rate = [1] * 12
    for loc in drop_loc:
        keep_rate[loc] = base_keep_rate
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=12, num_heads=12, keep_rate=keep_rate)
    model_kwargs.update(kwargs)
    model = _create_evit('deit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


# -------------------------------------------------------------
# Some example EViT models
@register_model
def deit_small_patch16_224_shrink_base(pretrained=False, base_keep_rate=0.7, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6,
                        keep_rate=(1, 1, 1, base_keep_rate) + (1, 1, base_keep_rate) + (1, 1, base_keep_rate) + (1, 1), **kwargs)
    model = _create_evit('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def deit_small_patch16_224_shrink(pretrained=False, base_keep_rate=0.5, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6,
                        keep_rate=(1, 1, 1, 0.7) + (1, 1, 0.7) + (1, 1, 0.7) + (1, 1), **kwargs)
    model = _create_evit('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def deit_small_patch16_272_shrink(pretrained=False, base_keep_rate=0.5, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6,
                        keep_rate=(1, 1, 1, 0.7) + (1, 1, 0.7) + (1, 1, 0.7) + (1, 1), **kwargs)
    model = _create_evit('deit_small_patch16_272', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def deit_small_patch16_224_shrink05(pretrained=False, base_keep_rate=0.5, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6,
                        keep_rate=(1, 1, 1, 0.5) + (1, 1, 0.5) + (1, 1, 0.5) + (1, 1), **kwargs)
    model = _create_evit('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def deit_small_patch16_288_shrink06(pretrained=False, base_keep_rate=0.6, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6,
                        keep_rate=(1, 1, 1, 0.6) + (1, 1, 0.6) + (1, 1, 0.6) + (1, 1), **kwargs)
    model = _create_evit('deit_small_patch16_288', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def deit_small_patch16_304_shrink05(pretrained=False, base_keep_rate=0.5, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6,
                        keep_rate=(1, 1, 1, 0.5) + (1, 1, 0.5) + (1, 1, 0.5) + (1, 1), **kwargs)
    model = _create_evit('deit_small_patch16_304', pretrained=pretrained, **model_kwargs)
    return model


# -------------------------------------------------------------
@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=12, num_heads=12, **kwargs)
    model = _create_evit('deit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    """ DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=12, num_heads=12, **kwargs)
    model = _create_evit('deit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model
