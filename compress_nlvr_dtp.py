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
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip_nlvr import blip_nlvr
import utils
from utils import cosine_lr_schedule, print_params_and_flops
from data import create_dataset, create_sampler, create_loader

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

    for _,(image0, image1, text, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device) 
        
        if scaler is not None:
            with autocast():
                loss_ori, loss_fdt = model(images, text, targets=targets, temperature=temperature, train=True)
                loss = loss_ori + 0.1 * loss_fdt
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else: 
            loss_ori, loss_fdt = model(images, text, targets=targets, temperature=temperature, train=True)
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
def evaluate(model, data_loader, device, temperature=0):
    # test
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50
    GFLOPS = 0
    len_data_loader = len(data_loader)

    for image0, image1, text, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)   
        
        prediction = model(images, text, targets=targets, temperature=temperature, train=False)  
 
        _, pred_class = prediction.max(1)
        accuracy = (targets==pred_class).sum() / targets.size(0)
        
        metric_logger.meters['acc'].update(accuracy.item(), n=image0.size(0))

        flops = FlopCountAnalysis(model.to(device), inputs=(images, text, targets, temperature, False,))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        flops.tracer_warnings("none")
        B = targets.shape[0] 
        GFLOPS += flops.total() / B / 1e9
    GFLOPS = GFLOPS / len_data_loader
    print("Current Temperature:", temperature)
    print("Averaged GFLOPS:", GFLOPS)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}, GFLOPS

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
    print("Creating nvlr dataset")
    datasets = create_dataset('nlvr', config) 
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True,False,False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]
    
    batch_size=[config['batch_size_train'],config['batch_size_test'],config['batch_size_test']]
    train_loader, val_loader, test_loader = create_loader(datasets,samplers,batch_size=batch_size, num_workers=[4,4,4],is_trains=[True, False,False], collate_fns=[None,None,None])

    #### Model ####
    temperature = 1.0
    if not args.evaluate:
        print("Creating model for token pruning")
        model = blip_nlvr(pretrained=config['pretrained'], image_size=config['image_size'], 
                        vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], config=config)
        model = model.to(device)
        print_params_and_flops('nlvr', model, device)
        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    else:
        print("Creating model for evaluation")
        model = blip_nlvr(pretrained='', image_size=config['image_size'], 
                            vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], evaluate=True, config=config)
        checkpoint = torch.load(config['pretrained'])
        model.load_state_dict(checkpoint['model'], strict=False)
        temperature = checkpoint["temperature"]
        model = model.to(device)
        model_without_ddp = model

    # calculate temperature
    Ori_Gflops = 132.54
    Target_Gflops = Ori_Gflops * (1 - config['p'])
    if not args.evaluate:
        print("Original model Gflops:", Ori_Gflops)
        print("Target model Gflops:", Target_Gflops)
        print('Target compression ratio: {}%'.format(config['p']*100))

    best = 0
    best_epoch = 0
    Cur_Gflops = Ori_Gflops
    scaler = torch.cuda.amp.GradScaler() if (not args.evaluate and args.amp) else None
    for epoch in range(0, config['max_epoch']):
        if epoch > 0:
            ## temperature change   
            if Cur_Gflops > Target_Gflops:
                if Cur_Gflops - Target_Gflops > 30:
                    temperature += 1
                elif Cur_Gflops - Target_Gflops > 10:
                    temperature += 0.5
                elif Cur_Gflops - Target_Gflops > 5:
                    temperature += 0.25
                # elif Cur_Gflops - Target_Gflops > 2:
                #     temperature += 0.1
                elif Cur_Gflops - Target_Gflops > 1:
                    temperature += 0.1
                else:
                    temperature += 0.01
            else:
                if Target_Gflops - Cur_Gflops > 30:
                    temperature -= 1
                elif Target_Gflops - Cur_Gflops > 10:
                    temperature -= 0.5
                elif Target_Gflops - Cur_Gflops > 5:
                    temperature -= 0.25
                # elif Target_Gflops - Cur_Gflops > 2:
                #     temperature -= 0.1
                elif Target_Gflops - Cur_Gflops > 1:
                    temperature -= 0.1
                else:
                    temperature -= 0.01
        print("Temperature:", temperature)

        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train_stats = train(model, train_loader, optimizer, epoch, device, scaler=scaler, temperature=temperature)

        val_stats, Cur_Gflops = evaluate(model_without_ddp, val_loader, device, temperature=temperature)
        test_stats, _ = evaluate(model_without_ddp, test_loader, device, temperature=temperature)  
        
        if utils.is_main_process():  
            if args.evaluate:                
                log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'Cur_Gflops': round(Cur_Gflops, 2),
                            }
                with open(os.path.join(args.output_dir, "evaluate.txt"),"w") as f:
                    f.write(json.dumps(log_stats) + "\n")     
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'val_{k}': v for k, v in val_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'Cur_Gflops': round(Cur_Gflops, 2),
                            }
                if float(test_stats['acc']) > best and Cur_Gflops - Target_Gflops < 5.0:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                        # 'config': config,
                        'epoch': epoch,
                        "temperature": temperature,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 

                    best = float(test_stats['acc'])
                    best_epoch = epoch
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
    parser.add_argument('--config', default='./configs/nlvr.yaml')
    parser.add_argument('--output_dir', default='output/NLVR')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--pretrained', default='pretrained/model_base_nlvr.pth', type=str)
    parser.add_argument('--epoch', default=15, type=int, help='number of epochs')
    parser.add_argument('--p', default=0.5, type=float, help='total compression ratio')
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)