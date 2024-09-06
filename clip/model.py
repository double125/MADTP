from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.utils import *
import io
import os

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=True
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, sd_dim=768):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        
        self.query_model = Query_model(ft_dim=d_model, sd_dim=sd_dim, temperature=1, att_func_type='sparsemax', pool_type='max', map_func=True)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)[0]

    def Reduce_token(self, x, reduce_num=0, temperature=0, self_attn=None, cls_attn=None, token_attn=None, max_keep=1):
        token_num = x.shape[-2]
        if self_attn is not None:
            self_attn_w = self_attn[:,:,1:,1:].max(1)[0]  # B, N, N
            self_attn_w = self_attn_w.sum(dim=1)
            self_attn_w = self_attn_w / (self_attn_w.sum(dim=1, keepdim=True) + 1e-8)
        
        if token_attn is not None:
            token_attn_w = token_attn.max(2)[0] # B, N
            token_attn_w = token_attn_w / (token_attn_w.sum(dim=1, keepdim=True) + 1e-8)

        Importance_score = (self_attn_w + token_attn_w + cls_attn) / 3.0
        #Importance_score = cls_attn
        if cls_attn is not None and token_attn is not None:

            token_attn /= temperature 
            token_attn = torch.softmax(token_attn, dim=1).permute(0,2,1)
            
            score_weight = torch.bmm(token_attn, Importance_score.unsqueeze(-1))
            threshold = torch.min(score_weight, dim=1)[0]

            idx = Importance_score > threshold

            topk_num = torch.max(idx.sum(dim=1)).item() 
            reduce_num = token_num - topk_num
            
            if topk_num <= max_keep or reduce_num <= 1:
                return x
            # Merge Token
            # no sorted
            _, indices = Importance_score.topk(topk_num, dim=-1, largest=True, sorted=False)
            x_topk = vector_gather(x, indices)
            Importance_score_weight, indices_sort = Importance_score.topk(token_num, dim=-1, largest=True, sorted=True)
            x_sort = vector_gather(x, indices_sort)
            x_reduce = x_sort[:,topk_num:,:]
            Importance_score_weight = Importance_score_weight[:,topk_num:]
            Importance_score_weight = Importance_score_weight / ( Importance_score_weight.sum(dim=1, keepdim=True) + 1e-8)
            x_combine = torch.bmm(Importance_score_weight.unsqueeze(1), x_reduce)
            x = torch.cat([x_topk, x_combine], dim=1)

        return x

    def forward(self, inputs):
        # x: (N, B, C)
        x, space_dict, temperature, sd_ft_all, max_keep = inputs
        if space_dict is not None:
            patch_ft = x.permute(1, 0, 2)[:,1:,:] #(B, N-1, C)
            token_attn, sd_ft, _ = self.query_model(patch_ft, space_dict, return_token_att=True)
            if sd_ft_all is None:
                sd_ft_all = sd_ft
            else:
                sd_ft_all += sd_ft

        x = x + self.attention(self.ln_1(x))

        ## get attention value for token reduce
        self_attn = self.attn.get_attention_map()
        cls_attn = self.attn.get_cls_attn()

        ## Attention-based token reduce module
        if space_dict is not None and temperature > 0:
            cls_ft = x.permute(1, 0, 2)[:,:1,:]
            patch_ft = x.permute(1, 0, 2)[:,1:,:]
            patch_ft = self.Reduce_token(patch_ft, 0, temperature, self_attn=self_attn, cls_attn=cls_attn, token_attn=token_attn, max_keep=max_keep)
            x = torch.cat([cls_ft, patch_ft], dim=1).permute(1, 0, 2)
            
        x = x + self.mlp(self.ln_2(x)) if not hasattr(self, 'alpha') else x + self.mlp[2](self.mlp[1](self.mlp[0](self.ln_2(x)) * self.alpha))
        return x, space_dict, temperature, sd_ft_all, max_keep


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, sd_dim=768):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, sd_dim=sd_dim) for _ in range(layers)])

    def forward(self, x: torch.Tensor, space_dict=None, temperature=0, sd_ft_all=None, max_keep=1):
        return self.resblocks((x, space_dict, temperature, sd_ft_all, max_keep))


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,  sd_dim=768):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, sd_dim=sd_dim)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, space_dict=None, temperature=0, max_keep=1):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        sd_img_ft_all = None
        x = x.permute(1, 0, 2)  # NLD -> LND
        if space_dict is not None:
            x, _, _, sd_img_ft_all, _ = self.transformer(x, space_dict, temperature, sd_img_ft_all, max_keep)
        else:
            x = self.transformer(x)[0]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x, sd_img_ft_all


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # evaluate
                 evaluate: bool,
                 config=None
                 ):
        super().__init__()

        self.context_length = context_length

        ## add fdt
        if config is None:
            self.sd_num = 100
            self.sd_dim = 768
        else:
            self.sd_num = config['sd_num']
            self.sd_dim = config['sd_dim']
        
        self.space_dict = nn.Parameter(torch.randn(self.sd_num, self.sd_dim))  
        self.criterion = nn.CosineEmbeddingLoss()
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                sd_dim=self.sd_dim
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                sd_dim=self.sd_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            sd_dim=self.sd_dim
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if not evaluate:
            self.initialize_parameters()

        self.tokenize = None
        self.vision_layers = vision_layers
        self.transformer_layers = transformer_layers

        # momentum modules
        self.momentum = 0.995
        self.visual_m = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                sd_dim=self.sd_dim
        )
        self.transformer_m = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            sd_dim=self.sd_dim
        )
        self.token_embedding_m = nn.Embedding(vocab_size, transformer_width)
        self.ln_final_m = LayerNorm(transformer_width)
        self.model_pairs = [[self.visual,self.visual_m],
                            [self.transformer,self.transformer_m],
                            [self.token_embedding,self.token_embedding_m],
                            [self.ln_final,self.ln_final_m],
                           ]    
         
        self.text_projection_m = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.positional_embedding_m = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.params_pairs = [[self.text_projection,self.text_projection_m],
                            [self.positional_embedding,self.positional_embedding_m],
                           ]    
        self.copy_params()  

        # create the queue
        self.queue_size = 57600
        self.embed_dim = embed_dim
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1,self.queue_size),-100))
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))  
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        # matching 
      
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, space_dict=None, temperature=0):
        return self.visual(image.type(self.dtype), space_dict=space_dict, temperature=temperature)

    def encode_text(self, text, space_dict=None, temperature=0):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        sd_txt_ft_all = None

        max_keep = text.argmax(dim=-1).max() + 2 
        x, _, _, sd_txt_ft_all, _ = self.transformer(x, space_dict, temperature, sd_txt_ft_all, max_keep)

        #x = self.transformer(x)[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x, sd_txt_ft_all

    def encode_image_m(self, image, space_dict=None, temperature=0):
        return self.visual_m(image.type(self.dtype), space_dict=space_dict, temperature=temperature)

    def encode_text_m(self, text, space_dict=None, temperature=0):
        x = self.token_embedding_m(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding_m.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        sd_txt_ft_all = None

        max_keep = text.argmax(dim=-1).max() + 2
        x, _, _, sd_txt_ft_all, _ = self.transformer(x, space_dict, temperature, sd_txt_ft_all, max_keep)

        #x = self.transformer_m(x)[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection_m

        return x, sd_txt_ft_all


    def forward(self, image, caption, alpha, idx, temperature=0):
        with torch.no_grad():
            self.logit_scale.clamp_(0, 4.6052)
        
        idx = idx.view(-1,1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)  
        pos_idx = torch.eq(idx, idx_all).float()       
        sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)
        image_features, sd_img_ft = self.encode_image(image, space_dict=self.space_dict, temperature=temperature)
        text = self.tokenize(caption).to(image.device)
        text_features, sd_txt_ft = self.encode_text(text, space_dict=self.space_dict, temperature=temperature)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_features_m, sd_img_ft_m = self.encode_image_m(image, space_dict=self.space_dict, temperature=temperature)
            text_features_m, sd_txt_ft_m = self.encode_text_m(text, space_dict=self.space_dict, temperature=temperature)
            image_features_m = image_features_m / image_features_m.norm(dim=1, keepdim=True)
            text_features_m = text_features_m / text_features_m.norm(dim=1, keepdim=True)
            image_features_m_all = torch.cat([image_features_m.t(),self.image_queue.clone().detach()],dim=1)
            text_features_m_all = torch.cat([text_features_m.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_m = logit_scale * image_features_m @ text_features_m_all
            sim_t2i_m = logit_scale * text_features_m @ image_features_m_all

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        # cosine similarity as logits
        logits_per_image = logit_scale * image_features @ text_features_m_all
        logits_per_text = logit_scale * text_features @ image_features_m_all

        loss_i2t = -torch.sum(F.log_softmax(logits_per_image, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(logits_per_text, dim=1)*sim_t2i_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2

        idxs = concat_all_gather(idx)
        self._dequeue_and_enqueue(image_features, text_features, idxs)    

        loss_fdt = loss_ita
        loss_fdt_m = loss_ita
        if temperature !=0 and sd_img_ft is not None and sd_txt_ft is not None:
            #l2 normalization
            sd_img_ft = sd_img_ft / (sd_img_ft.norm(dim=-1, keepdim=True) + 1e-10)
            sd_txt_ft = sd_txt_ft / (sd_txt_ft.norm(dim=-1, keepdim=True) + 1e-10)

            sd_img_ft = sd_img_ft.reshape(-1, self.sd_dim)
            sd_txt_ft = sd_txt_ft.reshape(-1, self.sd_dim)
            labels = torch.ones(sd_img_ft.shape[0]).to(sd_txt_ft.device).long()
            loss_fdt = self.criterion(sd_img_ft, sd_txt_ft, labels)

        if temperature !=0 and sd_img_ft_m is not None and sd_txt_ft_m is not None:
            sd_img_ft_m = sd_img_ft_m / (sd_img_ft_m.norm(dim=-1, keepdim=True) + 1e-10)
            sd_txt_ft_m = sd_txt_ft_m / (sd_txt_ft_m.norm(dim=-1, keepdim=True) + 1e-10)
            
            sd_img_ft_m = sd_img_ft_m.reshape(-1, self.sd_dim)
            sd_txt_ft_m = sd_txt_ft_m.reshape(-1, self.sd_dim)
            labels = torch.ones(sd_img_ft_m.shape[0]).to(sd_txt_ft_m.device).long()
            loss_fdt_m = self.criterion(sd_img_ft_m, sd_txt_ft_m, labels)

        return loss_ita, loss_fdt, loss_fdt_m


    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        

        batch_size = image_feats.shape[0]
        ptr = int(self.ptr_queue)
        assert self.queue_size % batch_size == 0  # for simplicity

        if ptr % batch_size != 0:
            ptr = (ptr // batch_size) * batch_size

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size # move pointer

        self.ptr_queue[0] = ptr  

    @torch.no_grad()    
    def copy_params(self):
        remove_alpha = lambda model: [param for name, param in list(model.named_parameters()) if not ('alpha' in name)]
        for model_pair in self.model_pairs:           
            for param, param_m in zip(remove_alpha(model_pair[0]), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    
        # self.text_projection_m.data.copy_(self.text_projection.data)
        # self.text_projection_m.requires_grad = False
        for model_pair in self.params_pairs:
            param, param_m = model_pair[0], model_pair[1]
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient    

    @torch.no_grad()        
    def _momentum_update(self):
        remove_alpha = lambda model: [param for name, param in list(model.named_parameters()) if not ('alpha' in name)]
        for model_pair in self.model_pairs:           
            for param, param_m in zip(remove_alpha(model_pair[0]), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
        # self.text_projection_m.data = self.text_projection_m.data * self.momentum + self.text_projection.data * (1. - self.momentum)
        for model_pair in self.params_pairs:       
            param, param_m = model_pair[0], model_pair[1]
            param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()       
    def reset_queue(self):
        self.image_queue = torch.randn(self.embed_dim, self.queue_size).to(device=self.image_queue.device)
        self.text_queue = torch.randn(self.embed_dim, self.queue_size).to(device=self.text_queue.device)
        self.idx_queue = torch.full((1,self.queue_size),-100).to(device=self.idx_queue.device)
        self.ptr_queue = torch.zeros(1, dtype=torch.long).to(device=self.ptr_queue.device)
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, evaluate: bool = False, config=None):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        evaluate, config
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    return model


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output      