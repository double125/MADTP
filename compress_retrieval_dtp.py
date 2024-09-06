'''
 * Copyright (c) 2023, Dachuan Shi.
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * For full license text, see LICENSE.txt file in the repo root
 * By Dachuan Shi
'''
import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
from pathlib import Path
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip_retrieval import blip_retrieval
import utils
from utils import cosine_lr_schedule, print_params_and_flops
from data import create_dataset, create_sampler, create_loader

from fvcore.nn import FlopCountAnalysis
from torch.cuda.amp import autocast as autocast

def train(model, data_loader, optimizer, epoch, device, config, scaler=None, temperature=0):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.7f}'))
    metric_logger.add_meter('temperature', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_fdt', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_fdt_m', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i,(image, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True)   
        idx = idx.to(device,non_blocking=True)   
       
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        if scaler is not None:
            with autocast():
                loss_ita, loss_itm, loss_fdt, loss_fdt_m = model(image, caption, alpha=alpha, idx=idx, temperature=temperature)                  
                loss = loss_ita + loss_itm + 0.1 * loss_fdt + 0.1 * loss_fdt_m
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_ita, loss_itm, loss_fdt, loss_fdt_m = model(image, caption, alpha=alpha, idx=idx, temperature=temperature)                  
            loss = loss_ita + loss_itm + 0.1 * loss_fdt + 0.1 * loss_fdt_m

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    

        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss_fdt=loss_fdt.item())
        metric_logger.update(loss_fdt_m=loss_fdt_m.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(temperature=temperature)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, data_loader, device, config, temperature=0):
    # evaluate
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50
    GFLOPS = 0
    len_data_loader = len(data_loader)
    print('Computing features for evaluation...')

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []  
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
        text_output, _ = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text', space_dict=model.space_dict, temperature=temperature)  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_embed)   
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
    
    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_ids[:,0] = model.tokenizer.enc_token_id

    print('Computing image features for evaluation...')
    image_feats = []
    image_embeds = []
    for image, caption, img_id in tqdm(data_loader): 
        image = image.to(device) 
        image_feat, _ = model.visual_encoder(image, space_dict=model.space_dict, temperature=temperature)   
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)   
        
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)

        ## calculate Gflops
        idx = img_id.to(device) 
        alpha = config['alpha'] 

        flops = FlopCountAnalysis(model.to(device), inputs=(image, caption, alpha, idx, temperature, False,))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        flops.tracer_warnings("none")
        B = image.shape[0]
        GFLOPS += flops.total() / B / 1e9
    GFLOPS = GFLOPS / len_data_loader
    print("Current Temperature:", temperature)
    print("Averaged GFLOPS:", GFLOPS)
    
    image_embeds = torch.cat(image_embeds,dim=0)
    min_len = 0
    for image_feat in image_feats:
        if min_len < image_feat.shape[1]:
            min_len = image_feat.shape[1]
    image_feats_update = []
    for image_feat in image_feats:
        feat_len = image_feat.shape[1]
        pad_len = min_len - feat_len
        if pad_len > 0:
            pad_feat = image_feat[:,0,:].unsqueeze(1).repeat(1,pad_len,1)
            image_feat = torch.cat([image_feat, pad_feat], dim=1)
        image_feats_update.append(image_feat)
    image_feats = torch.cat(image_feats_update,dim=0)
    print("image_feats:", image_feats.shape)
    
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[start+i].repeat(config['k_test'],1,1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[topk_idx], 
                                    attention_mask = text_atts[topk_idx],
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                    space_dict=model.space_dict, temperature=temperature
                                   )[0]
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[start+i,topk_idx] = score + topk_sim
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[start+i].repeat(config['k_test'],1), 
                                    attention_mask = text_atts[start+i].repeat(config['k_test'],1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                    space_dict=model.space_dict, temperature=temperature
                                   )[0]
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i,topk_idx] = score + topk_sim

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy(), GFLOPS

@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result

@torch.no_grad()
def calculate_temperature(model, data_loader, device, config, Cur_Gflops, Target_Gflops):
    model.eval() 
    temperature = 0
    while Target_Gflops - Cur_Gflops > 10 or Cur_Gflops - Target_Gflops > 10:
        ## temperature change
        if Cur_Gflops > Target_Gflops:
            if Cur_Gflops - Target_Gflops > 100:
                temperature += 1
            elif Cur_Gflops - Target_Gflops > 50:
                temperature += 0.5
            elif Cur_Gflops - Target_Gflops > 30:
                temperature += 0.3
            elif Cur_Gflops - Target_Gflops > 20:
                temperature += 0.2
            elif Cur_Gflops - Target_Gflops > 10:
                temperature += 0.1
            elif Cur_Gflops - Target_Gflops > 5:
                temperature += 0.05
            else:
                temperature += 0.02
        else:
            if Target_Gflops - Cur_Gflops > 100:
                temperature -= 1
            elif Target_Gflops - Cur_Gflops > 50:
                temperature -= 0.5
            elif Target_Gflops - Cur_Gflops > 30:
                temperature -= 0.3
            elif Target_Gflops - Cur_Gflops > 20:
                temperature -= 0.2
            elif Target_Gflops - Cur_Gflops > 10:
                temperature -= 0.1
            elif Target_Gflops - Cur_Gflops > 5:
                temperature -= 0.05
            else:
                temperature -= 0.02
        print("Current Temperature:", temperature)
        GFLOPS = 0
        count_num = 20
        for idx, (image, caption, img_id) in enumerate(data_loader):
            if idx > count_num:
                break 
            ## calculate Gflops
            img_id = img_id.to(device,non_blocking=True) 
            image = image.to(device,non_blocking=True) 
            alpha = config['alpha'] 
            flops = FlopCountAnalysis(model.to(device), inputs=(image, caption, alpha, img_id, temperature, False,))
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            flops.tracer_warnings("none")
            B = image.shape[0]  
            try:
                GFLOPS += flops.total() / B / 1e9
            except:
                continue
        Cur_Gflops = GFLOPS / count_num
        print("Cur_Gflops:", Cur_Gflops)
        
    return Cur_Gflops, temperature
    
def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    config['pretrained'] = args.pretrained
    config['max_epoch'] = args.epoch
    config['p'] = args.p

    #### Dataset #### 
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None,None,None])

    #### Model ####
    temperature = 1.0
    if not args.evaluate:
        print("Creating model for token pruning")
        model = blip_retrieval(pretrained=config['pretrained'],
                                image_size=config['image_size'], vit=config['vit'],
                                vit_grad_ckpt=config['vit_grad_ckpt'],
                                vit_ckpt_layer=config['vit_ckpt_layer'],
                                queue_size=config['queue_size'],
                                negative_all_rank=config['negative_all_rank'], config=config,
                                )
        model = model.to(device)
        print_params_and_flops('retrieval', model, device) 
        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    else:
        print("Creating model for evaluation")
        model = blip_retrieval(pretrained='', 
                                image_size=config['image_size'], vit=config['vit'],
                                vit_grad_ckpt=config['vit_grad_ckpt'],
                                vit_ckpt_layer=config['vit_ckpt_layer'],
                                queue_size=config['queue_size'],
                                negative_all_rank=config['negative_all_rank'], config=config,
                                evaluate=True)

        checkpoint = torch.load(config['pretrained'])
        model.load_state_dict(checkpoint['model'], strict=False)
        temperature = checkpoint["temperature"]
        model = model.to(device)
        model_without_ddp = model
    
    # calculate temperature
    Ori_Gflops = 153.2
    Target_Gflops = Ori_Gflops * (1 - config['p'])
    
    if not args.evaluate:
        print("Original model Gflops:", Ori_Gflops)
        print("Target model Gflops:", Target_Gflops)
        print('Target compression ratio: {}%'.format(config['p']*100))    
        # compression ratio -> init temperature
        # sample_loader = create_loader([test_dataset],[None],
        #                             batch_size=[config['batch_size_test']],num_workers=[8],
        #                             is_trains=[False], 
        #                             collate_fns=[None])[0]
        # _, temperature = calculate_temperature(model_without_ddp, sample_loader, device, config, Ori_Gflops, Target_Gflops)

    best = 0
    best_epoch = 0
    Cur_Gflops = Ori_Gflops
    scaler = torch.cuda.amp.GradScaler() if (not args.evaluate and args.amp) else None
    for epoch in range(0, config['max_epoch']):
        if epoch > 0:
            ## temperature change
            if Cur_Gflops > Target_Gflops:
                if Cur_Gflops - Target_Gflops > 50:
                        temperature += 0.5
                elif Cur_Gflops - Target_Gflops > 30:
                    temperature += 0.3
                elif Cur_Gflops - Target_Gflops > 20:
                    temperature += 0.2
                elif Cur_Gflops - Target_Gflops > 10:
                    temperature += 0.1
                elif Cur_Gflops - Target_Gflops > 5:
                    temperature += 0.05
                elif Cur_Gflops - Target_Gflops > 2:
                    temperature += 0.02
                else:
                    temperature += 0.01
            else:
                if Target_Gflops - Cur_Gflops > 50:
                    temperature -= 0.5
                elif Target_Gflops - Cur_Gflops > 30:
                    temperature -= 0.3
                elif Target_Gflops - Cur_Gflops > 20:
                    temperature -= 0.2
                elif Target_Gflops - Cur_Gflops > 10:
                    temperature -= 0.1
                elif Target_Gflops - Cur_Gflops > 5:
                    temperature -= 0.05
                elif Target_Gflops - Cur_Gflops > 2:                    
                    temperature -= 0.02
                else:
                    temperature -= 0.01

        print("Temperature:", temperature)

        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train_stats = train(model, train_loader, optimizer, epoch, device, config, scaler=scaler, temperature=temperature)  
        

        #score_val_i2t, score_val_t2i, _ = evaluate(model_without_ddp, val_loader, device, config, temperature=temperature)
        score_test_i2t, score_test_t2i, Cur_Gflops = evaluate(model_without_ddp, test_loader, device, config, temperature=temperature)
        dist.barrier()
        
        if utils.is_main_process():  

            #val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)  
            #print(val_result)
            test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt) 
            print(test_result)
            if not args.evaluate and test_result['r_mean'] > best and Cur_Gflops - Target_Gflops < 10.0:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    # 'config': config,
                    'epoch': epoch,
                    "temperature": temperature,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                best = test_result['r_mean']        
                best_epoch = epoch  
                
            if args.evaluate:                
                log_stats = {
                            #**{f'val_{k}': v for k, v in val_result.items()},
                            **{f'test_{k}': v for k, v in test_result.items()},
                            'Cur_Gflops': round(Cur_Gflops, 2),                  
                            }
                with open(os.path.join(args.output_dir, "evaluate.txt"),"w") as f:
                    f.write(json.dumps(log_stats) + "\n")     
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            #**{f'val_{k}': v for k, v in val_result.items()},
                            **{f'test_{k}': v for k, v in test_result.items()},  
                            'epoch': epoch,
                            'Cur_Gflops': round(Cur_Gflops, 2),
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            print("LOG: ", log_stats)
        if args.evaluate:
            break

        dist.barrier()
        torch.cuda.empty_cache()

    if utils.is_main_process():   
        print("LOG: best epoch: %d"%best_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')        
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--pretrained', default='pretrained/model_base_retrieval_coco.pth', type=str)
    parser.add_argument('--epoch', default=5, type=int, help='number of epochs')
    parser.add_argument('--p', default=0.5, type=float, help='total compression ratio')
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)