from telnetlib import PRAGMA_HEARTBEAT
from models.med import BertConfig
from models.nlvr_encoder import BertModel
from models.vit import interpolate_pos_embed
from models.blip import create_vit, init_tokenizer, is_url

from timm.models.hub import download_cached_file

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
import os
import io
import functools
from models.utils import *

class BLIP_NLVR(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 480,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                   
                 evaluate=False,
                 config=None
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        self.layers = 12 if vit == 'base' else 24
        ## add fdt
        if config is None:
            self.sd_num = 100
            self.sd_dim = 768
            self.batch_size = 16
        else:
            self.sd_num = config['sd_num']
            self.sd_dim = config['sd_dim']
            self.batch_size = config['batch_size_train']
        self.space_dict = nn.Parameter(torch.randn(self.sd_num, self.sd_dim))
        self.criterion = nn.CosineEmbeddingLoss()
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1, evaluate=evaluate, sd_dim=self.sd_dim)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        med_config.evaluate = evaluate
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False, sd_dim=self.sd_dim) 

        self.cls_head = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 2)
                )  

    def forward(self, image, text, targets, temperature=0, train=True):
        image_embeds, sd_img_ft = self.visual_encoder(image, space_dict=self.space_dict, temperature=temperature)
        
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        image0_embeds, image1_embeds = torch.split(image_embeds,targets.size(0))     
        text = self.tokenizer(text, padding='longest', return_tensors="pt").to(image.device) 
        text.input_ids[:,0] = self.tokenizer.enc_token_id        

        output, sd_txt_ft = self.text_encoder(text.input_ids, 
                                   attention_mask = text.attention_mask, 
                                   encoder_hidden_states = [image0_embeds,image1_embeds],
                                   encoder_attention_mask = [image_atts[:image0_embeds.size(0)],
                                                             image_atts[image0_embeds.size(0):]],        
                                   return_dict = True,
                                   space_dict = self.space_dict,
                                   temperature=temperature,
                                  )  
        hidden_state = output.last_hidden_state[:,0,:]        
        prediction = self.cls_head(hidden_state)

        if train:            
            loss_ori = F.cross_entropy(prediction, targets)
            loss_fdt = loss_ori
            if temperature != 0 and sd_img_ft is not None and sd_txt_ft is not None:
                #l2 normalization
                sd_img0_ft, sd_img1_ft = torch.split(sd_img_ft,targets.size(0))
                sd_img_ft = (sd_img0_ft + sd_img1_ft) / 2
                sd_img_ft = sd_img_ft / (sd_img_ft.norm(dim=-1, keepdim=True) + 1e-10)
                sd_txt_ft = sd_txt_ft / (sd_txt_ft.norm(dim=-1, keepdim=True) + 1e-10)

                sd_img_ft = sd_img_ft.reshape(-1, self.sd_dim)
                sd_txt_ft = sd_txt_ft.reshape(-1, self.sd_dim)
                labels = torch.ones(sd_img_ft.shape[0]).to(sd_txt_ft.device).long()
                loss_fdt = self.criterion(sd_img_ft, sd_txt_ft, labels)
            
            return loss_ori, loss_fdt
        else:
            return prediction

    
    def forward_throughput(self, image, text, targets):
        
        image_embeds = self.visual_encoder(image) 

        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device) 
        image0_embeds, image1_embeds = torch.split(image_embeds,targets.size(0))     

        output = self.text_encoder(text.input_ids, 
                                   attention_mask = text.attention_mask, 
                                   encoder_hidden_states = [image0_embeds,image1_embeds],
                                   encoder_attention_mask = [image_atts[:image0_embeds.size(0)],
                                                             image_atts[image0_embeds.size(0):]],        
                                   return_dict = True,
                                  )  
        hidden_state = output.last_hidden_state[:,0,:]        
        prediction = self.cls_head(hidden_state)

        return prediction
        
def blip_nlvr(pretrained='',**kwargs):
    model = BLIP_NLVR(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    return model  

    
def load_checkpoint(model, url_or_filename, client=None):
    if client is not None:
        with io.BytesIO(client.get(os.path.join('s3://BucketName/ProjectName', url_or_filename), enable_cache=True)) as f:
            checkpoint = torch.load(f, map_location='cpu')
    elif is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
    state_dict = checkpoint['model']
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    
    for key in list(state_dict.keys()):
        if 'crossattention.self.' in key:
            new_key0 = key.replace('self','self0')
            new_key1 = key.replace('self','self1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]
        elif 'crossattention.output.dense.' in key:
            new_key0 = key.replace('dense','dense0')
            new_key1 = key.replace('dense','dense1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]  

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return (model, msg)


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))