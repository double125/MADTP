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
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip import blip_decoder
import utils
from utils import cosine_lr_schedule, print_params_and_flops
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval

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
    print_freq = 100
    for i, (image, caption, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)       

        if scaler is not None:
            with autocast():
                loss_ori, loss_fdt = model(image, caption, temperature=temperature, train=True) 
                loss = loss_ori + 0.1 * loss_fdt
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_ori, loss_fdt = model(image, caption, temperature=temperature, train=True) 
            loss = loss_ori + 0.1 * loss_fdt

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(temperature=temperature)
        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_ori=loss_ori.item())
        metric_logger.update(loss_fdt=loss_fdt.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, data_loader, device, config, temperature=0):
    # evaluate
    model.eval() 
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 50
    GFLOPS = 0
    len_data_loader = len(data_loader)

    result = []
    for image, _, image_id in metric_logger.log_every(data_loader, print_freq, header): 
        
        image = image.to(device)       
        
        captions = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
                                  min_length=config['min_length'], 
                                  temperature=temperature)
        
        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})

        ## calculate Gflops
        flops = FlopCountAnalysis(model.to(device), inputs=(image, caption, temperature, False,))
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
def calculate_temperature(model, data_loader, device, config, Cur_Gflops, Target_Gflops):
    model.eval() 
    temperature = 1.0
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
        for idx, (image, caption, _) in enumerate(data_loader):
            if idx > count_num:
                break 
            image = image.to(device)  
            flops = FlopCountAnalysis(model.to(device), inputs=(image, caption, temperature, False,))
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            flops.tracer_warnings("none")
            B = image.shape[0]  
            GFLOPS += flops.total() / B / 1e9
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
    print("Creating captioning dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('caption_coco', config)
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset,val_dataset,test_dataset], [True,False,False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size']]*3,num_workers=[4,4,4],
                                                          is_trains=[True, False, False], collate_fns=[None,None,None])
    #### Model ####
    temperature = 1.0
    if not args.evaluate: 
        print("Creating model for token pruning")
        model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                            vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                            prompt=config['prompt'], config=config)
        model = model.to(device)   
        print_params_and_flops('caption', model, device)
        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module    
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    else:
        print("Creating model for evaluation")
        model = blip_decoder(pretrained='', image_size=config['image_size'], vit=config['vit'], 
                            vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                            prompt=config['prompt'],
                            evaluate=True, config=config)

        checkpoint = torch.load(config['pretrained'])
        model.load_state_dict(checkpoint['model'], strict=False)
        temperature = checkpoint["temperature"]
        model = model.to(device)
        model_without_ddp = model   
    
    # calculate temperature
    Ori_Gflops = 65.7
    Target_Gflops = Ori_Gflops * (1 - config['p'])

    if not args.evaluate:
        print("Original model Gflops:", Ori_Gflops)
        print("Target model Gflops:", Target_Gflops)
        print('Target compression ratio: {}%'.format(config['p']*100))    
        # compression ratio -> init temperature
        sample_loader = create_loader([test_dataset],[None],
                                      batch_size=[config['batch_size']],num_workers=[8],
                                      is_trains=[False], 
                                      collate_fns=[None])[0]
        _, temperature = calculate_temperature(model_without_ddp, sample_loader, device, config, Ori_Gflops, Target_Gflops)

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
        
        # if not args.evaluate:        
        #     if args.distributed:
        #         train_loader.sampler.set_epoch(epoch)
        #     cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
        #     train_stats = train(model, train_loader, optimizer, epoch, device, scaler=scaler, temperature=temperature) 

        #val_result, _ = evaluate(model_without_ddp, val_loader, device, config, temperature=temperature)  
        #val_result_file = save_result(val_result, args.result_dir, 'val_epoch%d'%epoch, remove_duplicate='image_id')       

        print("eval model on test dataset...")
        test_result, Cur_Gflops = evaluate(model_without_ddp, test_loader, device, config, temperature=temperature)  
        test_result_file = save_result(test_result, args.result_dir, 'test_epoch%d'%epoch, remove_duplicate='image_id')  

        if utils.is_main_process():   
            #coco_val = coco_caption_eval(config['coco_gt_root'],val_result_file,'val')
            coco_test = coco_caption_eval(config['coco_gt_root'],test_result_file,'test')
            
            if args.evaluate:            
                log_stats = {#**{f'val_{k}': v for k, v in coco_val.eval.items()},
                             **{f'test_{k}': v for k, v in coco_test.eval.items()},
                             'Cur_Gflops': round(Cur_Gflops, 2),                       
                            }
                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")                   
            else:             
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    # 'config': config,
                    'epoch': epoch,
                    "temperature": temperature,
                }

                if coco_test.eval['CIDEr'] + coco_test.eval['SPICE'] > best and Cur_Gflops - Target_Gflops < 5.0:
                    best = coco_test.eval['CIDEr'] + coco_test.eval['SPICE']
                    best_epoch = epoch                
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                    
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             #**{f'val_{k}': v for k, v in coco_val.eval.items()},
                             **{f'test_{k}': v for k, v in coco_test.eval.items()},
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
    parser.add_argument('--config', default='./configs/caption_coco.yaml')
    parser.add_argument('--output_dir', default='output/Caption_coco')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--pretrained', default='pretrained/model_base_caption_capfilt_large.pth', type=str)
    parser.add_argument('--epoch', default=5, type=int, help='number of epochs')
    parser.add_argument('--p', default=0.5, type=float, help='total compression ratio')
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)