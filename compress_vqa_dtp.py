'''
 * Copyright (c) 2023, Dachuan Shi.
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * For full license text, see LICENSE.txt file in the repo root
 * By Dachuan Shi
'''
import argparse
import os
from regex import B
import ruamel_yaml as yaml
import numpy as np
import random
from pathlib import Path
import json

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip_vqa import blip_vqa
import utils
from utils import cosine_lr_schedule, print_params_and_flops
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn
from data.utils import save_result

from fvcore.nn import FlopCountAnalysis
from torch.cuda.amp import autocast as autocast

def train(model, data_loader, optimizer, epoch, device, scaler=None, temperature=0):
    # train
    model.train()  

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.7f}'))
    metric_logger.add_meter('temperature', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_fdt', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ori', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i,(image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True) 
        if scaler is not None:
            with autocast():   
                loss_ori, loss_fdt = model(image, question, answer, train=True, n=n, weights=weights, temperature=temperature)
                loss = loss_ori + 0.1 * loss_fdt        
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_ori, loss_fdt = model(image, question, answer, train=True, n=n, weights=weights, temperature=temperature)  
            loss = loss_ori + 0.1 * loss_fdt      
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())  
        metric_logger.update(loss_ori=loss_ori.item())  
        metric_logger.update(loss_fdt=loss_fdt.item())
        metric_logger.update(temperature=temperature)  

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, data_loader, device, config, temperature=0):
    # test
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    GFLOPS = 0
    len_data_loader = len(data_loader)
    result = []
    
    if config['inference']=='rank':   
        answer_list = data_loader.dataset.answer_list
        answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
        answer_candidates.input_ids[:,0] = model.tokenizer.bos_token_id
        
    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             

        if config['inference']=='generate':
            answers = model(image, question, train=False, inference='generate', temperature=temperature) 
            
            for answer, ques_id in zip(answers, question_id):
                ques_id = int(ques_id.item())       
                result.append({"question_id":ques_id, "answer":answer})             
            
        elif config['inference']=='rank':    
            answer_ids = model(image, question, answer_candidates, train=False, inference='rank', k_test=config['k_test'],
                        temperature=temperature)      

            for ques_id, answer_id in zip(question_id, answer_ids):
                result.append({"question_id":int(ques_id.item()), "answer":answer_list[answer_id]})

        ## calculate Gflops
        flops = FlopCountAnalysis(model.to(device), inputs=(image, question, None, temperature, False, None, None, 'generate', ))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        flops.tracer_warnings("none")
        B = image.shape[0]  
        GFLOPS += flops.total() / B / 1e9
    GFLOPS = GFLOPS / len_data_loader
    print("Current Temperature:", temperature)
    print("Averaged GFLOPS:", GFLOPS)   

    return result, GFLOPS

@torch.no_grad()
def calculate_temperature(model, data_loader, device, Cur_Gflops, Target_Gflops):
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
        print("Current Temperature:", temperature)
        GFLOPS = 0
        count_num = 20
        for idx, (image, caption, _) in enumerate(data_loader):
            image = image.to(device)  
            if idx > count_num:
                break 
            flops = FlopCountAnalysis(model.to(device), inputs=(image, caption, temperature,))
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            flops.tracer_warnings("none")
            B = image.shape[0]  
            GFLOPS += flops.total() / B / 1e9
        Cur_Gflops = GFLOPS / count_num
        print("Cur_Gflops:", Cur_Gflops)
        if Cur_Gflops - Target_Gflops < 10:
            break
        
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
    config['w_sp_attn'] = args.w_sp_attn / args.world_size
    config['w_sp_mlp'] = args.w_sp_mlp  /args.world_size
    config['max_epoch'] = args.epoch
    config['p'] = args.p
    if not args.evaluate:
        print('Target compression ratio: {}%'.format(config['p']*100))

    #### Dataset #### 
    print("Creating vqa datasets")
    datasets = create_dataset('vqa', config)
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]
    
    train_loader, test_loader = create_loader(datasets,samplers,
                                              batch_size=[config['batch_size_train'],config['batch_size_test']],
                                              num_workers=[4,4],is_trains=[True, False], 
                                              collate_fns=[vqa_collate_fn,None]) 
    
    #### Model ####
    temperature = 0.5
    if not args.evaluate: 
        print("Creating model for token pruning")
        model = blip_vqa(pretrained=config['pretrained'], image_size=config['image_size'], 
                        vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], config=config)
        model = model.to(device)
        print_params_and_flops('vqa', model, device)
        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    else:
        print("Creating model for evaluation")
        model = blip_vqa(pretrained='', image_size=config['image_size'], 
                        vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], config=config, evaluate=True)
        checkpoint = torch.load(config['pretrained'])
        model.load_state_dict(checkpoint['model'], strict=False)
        temperature = checkpoint["temperature"]
        model = model.to(device)
        model_without_ddp = model


    # Ori_Gflops, Target_Gflops = 0, 0
    # if not args.evaluate:
    #     with torch.no_grad():
    #         vqa_result, Ori_Gflops = evaluate(model_without_ddp, test_loader, device, config)  
    #         dist.barrier()      
    #         result_file = save_result(vqa_result, args.result_dir, 'vqa_result')
    #     Target_Gflops = Ori_Gflops * (1 - config['p'])
    #     print("Original model Gflops:", Ori_Gflops)
    #     print("Target model Gflops:", Target_Gflops)

    #     ### calculate temperature
    #     _, temperature = calculate_temperature(model_without_ddp, train_loader, device, Ori_Gflops, Target_Gflops)

    Ori_Gflops = 186.1
    Target_Gflops = Ori_Gflops * (1 - config['p'])
    
    Cur_Gflops = Ori_Gflops 
    scaler = torch.cuda.amp.GradScaler() if (not args.evaluate and args.amp) else None
    for epoch in range(0, config['max_epoch']):
        if epoch > 0:
            ## temperature change
            if Cur_Gflops > Target_Gflops:
                if Cur_Gflops - Target_Gflops > 50:
                    temperature += 0.25
                elif Cur_Gflops - Target_Gflops > 30:
                    temperature += 0.15
                elif Cur_Gflops - Target_Gflops > 10:
                    temperature += 0.1
                elif Cur_Gflops - Target_Gflops > 5:
                    temperature += 0.05
                elif Cur_Gflops - Target_Gflops > 2:
                    temperature += 0.01
            else:
                if Target_Gflops - Cur_Gflops > 50:
                    temperature -= 0.25
                elif Target_Gflops - Cur_Gflops > 30:
                    temperature -= 0.15
                elif Target_Gflops - Cur_Gflops > 10:
                    temperature -= 0.1
                elif Target_Gflops - Cur_Gflops > 5:
                    temperature -= 0.05
                elif Target_Gflops - Cur_Gflops > 2:
                    temperature -= 0.01
        print("Temperature:", temperature)

        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            print("Starting train model ...")
            train_stats = train(model, train_loader, optimizer, epoch, device, scaler=scaler, temperature=temperature)    
        
            if utils.is_main_process():     
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            'Cur_Gflops': round(Cur_Gflops, 2),
                            }                
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                print("LOG: ", log_stats)
                        
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'epoch': epoch,
                    "temperature": temperature,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  

        dist.barrier()         
        vqa_result = evaluate(model_without_ddp, test_loader, device, config, temperature=temperature)[0]        
        result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d'%epoch)
        
        if args.evaluate:
            break   

        dist.barrier()
        torch.cuda.empty_cache()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/vqa.yaml') 
    parser.add_argument('--output_dir', default='output/VQA')
    parser.add_argument('--evaluate', action='store_true')      
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--pretrained', default='pretrained/model_base_vqa_capfilt_large.pth', type=str)
    parser.add_argument('--w_sp_attn', default=1.44e-2, type=float, help='regularization coefficient for attn')
    parser.add_argument('--w_sp_mlp', default=5e-4, type=float, help='regularization coefficient for mlp')
    parser.add_argument('--epoch', default=5, type=int, help='number of epoches')
    parser.add_argument('--p', default=0.5, type=float, help='total compression ratio')
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config)