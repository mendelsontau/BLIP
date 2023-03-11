'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip_retrieval_vg import blip_retrieval_vg
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
import sys
sys.path.insert(0, "/home/gamir/DER-Roei/alon/BLIP")
from laion_dataset import get_data, augment_laion_pairs
from vg_dataset import VgDatasetText, get_vg_loader, get_vg_val_loader
from Winoground.evaluate_winoground import evaluate_winoground, blip_processor
from vsr.evaluate_vsr import evaluate_vsr
from VL_CheckList.example_models.blip.engine import BLIP
from VL_CheckList.vl_checklist.evaluate import Evaluate
from torchvision import transforms as transforms

from loralib import mark_only_lora_as_trainable
from torchvision.transforms.functional import convert_image_dtype
from detr.util.box_ops import box_cxcywh_to_xyxy
from torchvision.utils import draw_bounding_boxes


def remove_repetitions(object_indexes, label_predictions_list):
    new_object_indexes = []
    seen_labels = []
    for idx in object_indexes:
        label = label_predictions_list[idx]
        if label in seen_labels:
            continue
        else:
            seen_labels.append(label)
            new_object_indexes.append(idx)
    return new_object_indexes

def organize_batch_classes(object_descriptions, valid_objects, vg_bbs, args, device):
    class_tokens = []
    object_samples = []
    tgt_boxes = []
    for i in range (valid_objects.shape[0]):
        valid_samples = valid_objects[i].item()
        invalid_samples = args.objects - valid_samples
        valid_for_sample = valid_samples * [True] + invalid_samples*[False]
        object_samples.append(valid_for_sample)

        if valid_samples == 0:
            tgt_boxes.append(torch.tensor([]).to(device=device, non_blocking=True))
        else:
            mask = torch.tensor(valid_for_sample).unsqueeze(1).expand(-1,4)
            boxes_for_sample = torch.masked_select(vg_bbs[i],mask).view(-1,4).to(device=device, non_blocking=True)
            tgt_boxes.append(boxes_for_sample)
    
    
    tgt_labels = []
    for i in range(len(object_descriptions)):
        labels_for_sample = []
        for j in range(len(object_descriptions[i])):
            if object_samples[i][j] == False:
                continue
            desc = object_descriptions[i][j]
            exists = False
            for k in range (len(class_tokens)):
                if desc == class_tokens[k]:
                    exists = True
                    labels_for_sample.append(k)
                    break
            if not exists:
                labels_for_sample.append(len(class_tokens))
                class_tokens.append(desc)
        tgt_labels.append(torch.tensor(labels_for_sample).type(torch.int64).to(device=device, non_blocking=True))
    targets = [{"labels": l, "boxes": b} for l,b in zip(tgt_labels,tgt_boxes)]
    return class_tokens, targets


def organize_batch_classes_relations(relation_descriptions, valid_relations, vg_bbs, args, device):
    class_tokens = []
    relation_samples = []
    tgt_boxes = []
    for i in range (valid_relations.shape[0]):
        valid_samples = valid_relations[i].item()
        invalid_samples = args.relations - valid_samples
        valid_for_sample = valid_samples * [True] + invalid_samples*[False]
        relation_samples.append(valid_for_sample)

        if valid_samples == 0:
            tgt_boxes.append(torch.tensor([]).to(device=device, non_blocking=True))
        else:
            mask = torch.tensor(valid_for_sample).unsqueeze(1).expand(-1,4)
            boxes_for_sample = torch.masked_select(vg_bbs[i],mask).view(-1,4).to(device=device, non_blocking=True)
            tgt_boxes.append(boxes_for_sample)
    
    
    tgt_labels = []
    for i in range(len(relation_descriptions)):
        labels_for_sample = []
        for j in range(len(relation_descriptions[i])):
            if relation_samples[i][j] == False:
                continue
            desc = relation_descriptions[i][j]
            exists = False
            for k in range (len(class_tokens)):
                if desc == class_tokens[k]:
                    exists = True
                    labels_for_sample.append(k)
                    break
            if not exists:
                labels_for_sample.append(len(class_tokens))
                class_tokens.append(desc)
        tgt_labels.append(torch.tensor(labels_for_sample).type(torch.int64).to(device=device, non_blocking=True))
    targets = [{"labels": l, "boxes": b} for l,b in zip(tgt_labels,tgt_boxes)]
    return class_tokens, targets

def evaluate_auxiliary(model,batch,args,epoch):
    inv_trans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.26862954, 1/0.26130258, 1/0.27577711 ]),
                                transforms.Normalize(mean = [ -0.48145466, -0.4578275, -0.40821073 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    device = torch.device(args.device)
    model.eval()
    vg_images, _, valid_objects, bounding_boxes, object_descriptions_t = batch
    #randomlist = random.sample(range(0, 16), 8)
    #vg_images = vg_images[randomlist]
    #valid_objects = valid_objects[randomlist]
    #bounding_boxes = bounding_boxes[randomlist]
    object_descriptions = [list(x) for x in zip(*object_descriptions_t)]
    #object_descriptions = [object_descriptions[i] for i in randomlist]
    object_descriptions, targets = organize_batch_classes(object_descriptions,valid_objects,bounding_boxes,args,device)
    num_object_descs = len(object_descriptions)
    no_object_rows_to_add = model.num_matcher_classes - num_object_descs
    vg_images = vg_images.to(device=device, non_blocking=True)
    with torch.no_grad():
        text = model.tokenizer(object_descriptions,padding='max_length', truncation=True, max_length=35, 
                                    return_tensors="pt").to(device)
        object_descriptions += no_object_rows_to_add*["random row"] + ["no object"]
        image_embeds = model.visual_encoder(vg_images)        
        object_tokens = image_embeds[:,1 : 1 + args.object_tokens ,:]
        text_output_m = model.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')    
        text_feat = F.normalize(model.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1)
        random_rows = model.random_row
        no_object_row = model.no_object_row.to(device)
        random_rows = random_rows.expand(no_object_rows_to_add,-1).to(device)
        objects_descs_feat_m = torch.cat([text_feat,random_rows,no_object_row])
        label_embeddings = model.class_head(object_tokens)
        label_probs = label_embeddings @ objects_descs_feat_m.t() / model.temp 
        bb_predictions = model.bb_head(object_tokens).sigmoid()
        predictions_dict = {"pred_logits" : label_probs, "pred_boxes": bb_predictions}
        loss_dict = model.vgcriterion(predictions_dict, targets)
        label_predictions = torch.argmax(label_probs, dim = -1)
        bb_predictions = box_cxcywh_to_xyxy(bb_predictions) * 224
    bb_predictions = bb_predictions.detach().cpu()
    label_predictions = label_predictions.detach().cpu()
    vg_images = vg_images.cpu()
    visualizations_folder = os.path.join(args.output_dir,"visualizations")
    if not os.path.exists(visualizations_folder):
        os.mkdir(visualizations_folder)
    visualizations_folder_epoch = os.path.join(visualizations_folder,str(epoch))
    if not os.path.exists(visualizations_folder_epoch):
        os.mkdir(visualizations_folder_epoch)
    gt_labels_index = 0
    for i in range(vg_images.shape[0]):
        full_image_path = os.path.join(visualizations_folder_epoch, "img_bb_" + str(i) +  ".jpg")
        full_image_path_gt = os.path.join(visualizations_folder_epoch, "img_bb_gt_" + str(i) +  ".jpg")
        just_image_path = os.path.join(visualizations_folder_epoch, "img_" + str(i) +  ".jpg")
        img = vg_images[i]
        img = inv_trans(img)
        img = torch.clamp(img,min=0.0,max=1.0)
        img = convert_image_dtype(img,torch.uint8)
        #bb_gt = bounding_boxes[i, : valid_objects[i].item(),:]
        #bb_gt = box_cxcywh_to_xyxy(bb_gt) * 224
        #labels_gt = object_descriptions[gt_labels_index : gt_labels_index + valid_objects[i].item()]
        #gt_labels_index += valid_objects[i].item()
        #bb_img = draw_bounding_boxes(img,bb_gt,labels_gt)
        #new_image = transforms.ToPILImage()(bb_img)
        #new_image.save(full_image_path_gt)
        label_predictions_list = label_predictions[i].tolist()
        object_indexes = [j for j in range(len(label_predictions_list)) if label_predictions_list[j] != model.num_matcher_classes]
        object_indexes = remove_repetitions(object_indexes, label_predictions_list)
        bb = bb_predictions[i,object_indexes,:]
        label = label_predictions[i,object_indexes]
        labels = [object_descriptions[j] for j in label.tolist()]
        #bb_img = draw_bounding_boxes(img,bb,labels,colors = "white")
        #new_image = transforms.ToPILImage()(bb_img)
        #new_image.save(full_image_path)
        img = transforms.ToPILImage()(img)
        img.save(just_image_path)
    return loss_dict


def train(model, data_loader, optimizer, epoch, device, config, args, vg_data_loader = None):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(data_loader.num_batches, delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if args.negatives or args.laion_augmentations:
        metric_logger.add_meter('loss_neg', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if args.vg_loss_lambda > 0.0:
        metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_bbox', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_giou', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_sg', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('ce_correct', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    if args.laion_augmentations:
        f = open(os.path.join("../../../datasets/vg","relations_annotations.json"))
        relations_annotations = json.load(f)
        f = open(os.path.join("../../../datasets/vg","attributes_annotations.json"))
        attributes_annotations = json.load(f)

    if vg_data_loader != None:
        vg_iter = iter(vg_data_loader)

    for i,(image, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        idx = [int(id) for id in idx]
        idx = torch.IntTensor(idx)

        laion_negs = None
        laion_neg_mask = None
        if args.laion_augmentations:
            laion_negs, laion_neg_mask = augment_laion_pairs(caption, relations_annotations, attributes_annotations)
            laion_neg_mask = torch.tensor(laion_neg_mask).to(device,non_blocking=True)
        neg_mask = None
        objects_descs = None
        targets = None
        relations_descs = None
        relations_targets = None
        if vg_data_loader != None:
            try:
                vg_batch = next(vg_iter)
            except StopIteration:
                vg_iter = iter(vg_data_loader)
                vg_batch = next(vg_iter)
            if args.vg_loss_lambda > 0.0:
                if args.relations > 0:
                    if args.negatives:
                        vg_image, vg_text, valid_objects, bounding_boxes, object_descriptions_t, valid_relations, relations_bounding_boxes, relation_descriptions_t, neg_text, neg_mask, vg_idx =  vg_batch
                        object_descriptions = [list(x) for x in zip(*object_descriptions_t)]
                        relation_descriptions = [list(x) for x in zip(*relation_descriptions_t)]
                        vg_text += neg_text
                        neg_mask = neg_mask.to(device,non_blocking=True)
                    else:
                        vg_image, vg_text, valid_objects, bounding_boxes, object_descriptions_t, valid_relations, relations_bounding_boxes, relation_descriptions_t, vg_idx =  vg_batch
                        object_descriptions = [list(x) for x in zip(*object_descriptions_t)]
                        relation_descriptions = [list(x) for x in zip(*relation_descriptions_t)]
                    relations_descs, relations_targets = organize_batch_classes_relations(relation_descriptions,valid_relations,relations_bounding_boxes,args,device)
                else:
                    if args.negatives:
                        vg_image, vg_text, valid_objects, bounding_boxes, object_descriptions_t, neg_text, neg_mask, vg_idx =  vg_batch
                        object_descriptions = [list(x) for x in zip(*object_descriptions_t)]
                        vg_text += neg_text
                        neg_mask = neg_mask.to(device,non_blocking=True)
                    else:
                        vg_image, vg_text, valid_objects, bounding_boxes, object_descriptions_t, vg_idx =  vg_batch
                        object_descriptions = [list(x) for x in zip(*object_descriptions_t)]
                objects_descs, targets = organize_batch_classes(object_descriptions, valid_objects, bounding_boxes, args, device)
            else:
                if args.negatives:
                    vg_image, vg_text, neg_text, neg_mask, vg_idx = vg_batch
                    vg_text += neg_text
                    neg_mask = neg_mask.to(device,non_blocking=True) 
                else:
                    vg_image, vg_text, vg_idx = vg_batch

            
        

            caption += vg_text
            image = torch.cat([image,vg_image])
            idx = torch.cat([idx,vg_idx])

        
        image = image.to(device,non_blocking=True)   
        idx = idx.to(device,non_blocking=True)

        vg_batch_size = 0 if vg_data_loader == None else vg_image.shape[0]
       
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/data_loader.num_batches)

        loss_ita, loss_itm, loss_neg, loss_dict, weight_dict = model(image, caption, alpha=alpha, idx=idx, vg_batch_size=vg_batch_size, ignore_mask=neg_mask, objects_descs = objects_descs, targets = targets, relations_descs = relations_descs, relations_targets=relations_targets, laion_negs = laion_negs, laion_neg_mask = laion_neg_mask)
        loss = loss_ita + loss_itm
        if args.negatives or args.laion_augmentations:
            loss += loss_neg
        if args.vg_loss_lambda > 0.0:
            loss_ce = loss_dict["loss_ce"]
            loss_bbox = loss_dict["loss_bbox"]
            loss_giou = loss_dict["loss_giou"]
            ce_correct = loss_dict["ce_correct"]
            class_error = loss_dict["class_error"]
            loss_dict.pop("ce_correct")
            loss_sg = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) * args.vg_loss_lambda                  
            loss +=  loss_sg
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        if args.negatives or args.laion_augmentations:
            metric_logger.update(loss_neg=loss_neg.item())
        if args.vg_loss_lambda > 0.0:
            metric_logger.update(loss_ce=loss_ce.item())
            metric_logger.update(loss_bbox=loss_bbox.item())
            metric_logger.update(loss_giou=loss_giou.item())
            metric_logger.update(loss_sg=loss_sg.item())
            metric_logger.update(ce_correct=ce_correct)
            metric_logger.update(class_error=class_error)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  



def main(args, config):
    utils.init_distributed_mode(args) 
 
    
    device = torch.device(args.device)

    processor = blip_processor(config["image_size"])

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    args.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    args.world_size = utils.get_world_size()

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank() 

    if utils.is_main_process():
        params_file = os.path.join(args.output_dir, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                f.write(f"{name}: {val}\n")   

    #### Model #### 
    print("Creating model")
    model = blip_retrieval_vg(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                             vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                             queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'], args = args)

    if args.lora != -1:
        mark_only_lora_as_trainable(model)
    
    if args.lock:
        for param in model.parameters():
            param.requires_grad = False

    if args.object_tokens > 0:
        model.visual_encoder.object_tokens.requires_grad_()

    if args.relation_tokens > 0:
        model.visual_encoder.relation_tokens.requires_grad_()
    
    if args.prompt_attention:
        if not args.prompts_lora > 0:
            for a in model.visual_encoder.blocks:
                for param in a.attn.qkv_prompts.parameters():
                    param.requires_grad_()
                for param in a.attn.proj_prompts.parameters():
                    param.requires_grad_()
    
    if args.prompt_attention_full:
            for b in model.visual_encoder.blocks:
                if not args.prompts_lora > 0:
                    for param in b.mlp_prompts.parameters():
                        param.requires_grad_()
                for param in b.norm1_prompts.parameters():
                    param.requires_grad_()
                for param in b.norm2_prompts.parameters():
                    param.requires_grad_()

    if args.vg_loss_lambda > 0.0:
        model.random_row.requires_grad_()
        model.no_object_row.requires_grad_()
        for param in model.bb_head.parameters():
            param.requires_grad_()
        for param in model.class_head.parameters():
            param.requires_grad_()
        if args.relations > 0:
            model.no_relation_row.requires_grad_()
            for param in model.rel_bb_head.parameters():
                param.requires_grad_()
            for param in model.rel_class_head.parameters():
                param.requires_grad_()

    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name)

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay']) 

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            if 'epoch' in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = 0 if args.evaluate else checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))        

    #### laion Dataset #### 
    print("Creating laion dataset")
    data = get_data(args, epoch=start_epoch)
    train_loader = data["train"].dataloader

    #### vg Dataset ####
    vg_dataloader = None 
    if args.vg_data:
        print("Creating vg dataset")
        vg_train_dataset = VgDatasetText(args.vg_data, "train", processor, args.objects, args.vg_loss_lambda, args.negatives, args.relations, args.no_dense_ablation, args.no_relation_ablation, args.size_ablation)
        vg_dataloader = get_vg_loader(vg_train_dataset, args, args.vg_batch_size)
        if args.vg_loss_lambda > 0.0 and args.auxiliary_frequency > 0:
            vg_val_dataset = VgDatasetText(args.vg_data, "val", processor, args.objects, args.vg_loss_lambda)
            vg_val_dataloader = get_vg_val_loader(vg_val_dataset, args, 16)
            vg_val_iterator = iter(vg_val_dataloader)
            vg_val_batch = next(vg_val_iterator)
   
    
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()  
    dist.barrier()  

    for epoch in range(start_epoch, config['max_epoch']):    
        if not args.evaluate:        
            if args.distributed:
                data["train"].set_epoch(epoch)
                if vg_dataloader != None:
                    vg_dataloader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            
            if args.vg_data:
                train_stats = train(model, train_loader, optimizer, epoch, device, config, args,vg_data_loader=vg_dataloader)
            else:
                train_stats = train(model, train_loader, optimizer, epoch, device, config, args)
            

            #log train stats
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                        }
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
            
            #save checkpoint
            if not args.checkpoint_frequency > 0:
                checkpoint_dict = {
                    "epoch": epoch + 1,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.output_dir, f"epoch_latest.pt")
                )
            else:
                if (epoch + 1) % args.checkpoint_frequency == 0:
                    checkpoint_dict = {
                        "epoch": epoch + 1,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    torch.save(
                        checkpoint_dict,
                        os.path.join(args.output_dir, f"epoch_{epoch + 1}.pt")
                    )

                
        if utils.is_main_process():
            if args.evaluate:
                if args.winoground_frequency > 0:
                    processor = blip_processor(config["image_size"])
                    winoground_dict = evaluate_winoground(model_without_ddp, processor,device)
                    winoground_folder = os.path.join(args.output_dir,"winoground")
                    if not os.path.exists(winoground_folder):
                        os.mkdir(winoground_folder)
                    winoground_dict_path = os.path.join(winoground_folder,"winoground_" + str(epoch))
                    with open(os.path.join(winoground_dict_path), 'w',encoding='utf-8') as f:
                        json.dump(winoground_dict, f)


                if args.vlchecklist_frequency > 0:
                    vl_model = BLIP(f'epoch {epoch}',model_without_ddp, processor, device)
                    vl_eval = Evaluate(config_file="VL_CheckList/configs/blip1.yaml", model = vl_model,epoch = epoch,args = args)
                    vl_eval.start()
                    vl_eval = Evaluate(config_file="VL_CheckList/configs/blip2.yaml", model = vl_model,epoch = epoch,args = args)
                    vl_eval.start()
                    vl_eval = Evaluate(config_file="VL_CheckList/configs/blip3.yaml", model = vl_model,epoch = epoch,args = args)
                    vl_eval.start()
                    vl_eval = Evaluate(config_file="VL_CheckList/configs/blip4.yaml", model = vl_model,epoch = epoch,args = args)
                    vl_eval.start()
                
                if args.auxiliary_frequency > 0:
                    val_loss_dict = evaluate_auxiliary(model_without_ddp,vg_val_batch,args,epoch)
                    vg_stats = {**{f'val_{k}': v.item() if torch.is_tensor(v) else v for k, v in val_loss_dict.items() }                  
                                }
                    with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                        f.write(json.dumps(vg_stats) + "\n") 
                if args.vsr_frequency > 0:
                    vsr_folder = os.path.join(args.output_dir,"vsr")
                    vsr_dict_path1 = os.path.join(vsr_folder,"vsr_" + str(epoch))
                    vsr_dict_path2 = os.path.join(vsr_folder,"vsr_meta_" + str(epoch))
                    results_by_cat, results_by_meta_cat = evaluate_vsr(model_without_ddp,blip_processor(config["image_size"]),device)
                    if not os.path.exists(vsr_folder):
                        os.mkdir(vsr_folder)
                    with open(os.path.join(vsr_dict_path1), 'w',encoding='utf-8') as f:
                        json.dump(results_by_cat, f)
                    with open(os.path.join(vsr_dict_path2), 'w',encoding='utf-8') as f:
                        json.dump(results_by_meta_cat, f)
            else:
                if args.winoground_frequency > 0 and (epoch + 1) % args.winoground_frequency == 0:
                    processor = blip_processor(config["image_size"])
                    winoground_dict = evaluate_winoground(model_without_ddp, processor,device)
                    winoground_folder = os.path.join(args.output_dir,"winoground")
                    if not os.path.exists(winoground_folder):
                        os.mkdir(winoground_folder)
                    winoground_dict_path = os.path.join(winoground_folder,"winoground_" + str(epoch))
                    with open(os.path.join(winoground_dict_path), 'w',encoding='utf-8') as f:
                        json.dump(winoground_dict, f)


                if args.vlchecklist_frequency > 0 and (epoch + 1) % args.vlchecklist_frequency == 0:
                    vl_model = BLIP(f'epoch {epoch}',model_without_ddp, processor, device)
                    vl_eval = Evaluate(config_file="VL_CheckList/configs/blip1.yaml", model = vl_model,epoch = epoch,args = args)
                    vl_eval.start()
                    vl_eval = Evaluate(config_file="VL_CheckList/configs/blip2.yaml", model = vl_model,epoch = epoch,args = args)
                    vl_eval.start()
                    vl_eval = Evaluate(config_file="VL_CheckList/configs/blip3.yaml", model = vl_model,epoch = epoch,args = args)
                    vl_eval.start()
                    vl_eval = Evaluate(config_file="VL_CheckList/configs/blip4.yaml", model = vl_model,epoch = epoch,args = args)
                    vl_eval.start()
                
                if args.auxiliary_frequency > 0 and (epoch + 1) % args.auxiliary_frequency == 0:
                    val_loss_dict = evaluate_auxiliary(model_without_ddp,vg_val_batch,args,epoch)
                    vg_stats = {**{f'val_{k}': v.item() if torch.is_tensor(v) else v for k, v in val_loss_dict.items() }                  
                                }
                    with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                        f.write(json.dumps(vg_stats) + "\n")


                    
        if args.evaluate: 
            break

        dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/laion_vg.yaml')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument("--name", default="test")
    parser.add_argument('--evaluate', action='store_true')        
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--train-data', default = '../../../datasets/laion/images/{34230..72957}.tar', type=str)
    parser.add_argument('--train-num-samples', default = 0, type=int)
    parser.add_argument('--dataset-type', default = "auto", type=str)
    parser.add_argument('--workers', default = 1, type=int)
    parser.add_argument('--vg-data', default = None, type=str)
    parser.add_argument('--vg-loss-lambda', default = 0.0, type=float)
    parser.add_argument('--negatives', action='store_true')
    parser.add_argument('--batch-size', default = 32, type=int)
    parser.add_argument('--vg-batch-size', default = 8, type=int)
    parser.add_argument('--objects', default = 0, type=int)
    parser.add_argument('--object-tokens', default = 0, type=int)
    parser.add_argument('--relations', default = 0, type=int)
    parser.add_argument('--relation-tokens', default = 0, type=int)
    parser.add_argument('--winoground-frequency', default = 0, type=int)
    parser.add_argument('--vlchecklist-frequency', default = 0, type=int)
    parser.add_argument('--auxiliary-frequency', default = 0, type=int)
    parser.add_argument('--checkpoint-frequency', default = 0, type=int)
    parser.add_argument('--vsr-frequency', default = 0, type=int)
    parser.add_argument('--lora', default = -1, type=int)
    parser.add_argument('--text-lora', action='store_true')
    parser.add_argument('--image-lora', action='store_true')
    parser.add_argument('--prompts-lora', default = -1, type=int)
    parser.add_argument('--resume', default = None, type=str)
    parser.add_argument('--lr', default = 0.0001, type=float)
    parser.add_argument('--prompt-attention', action='store_true')
    parser.add_argument('--prompt-attention-full', action='store_true')
    parser.add_argument('--mask-layers', default=None, type=str)
    parser.add_argument('--lock', action='store_true')
    parser.add_argument('--epochs', default = 0, type=int)
    parser.add_argument("--loss-ce", default = 1.0, type=float)
    parser.add_argument("--laion-augmentations", action='store_true')
    parser.add_argument("--no-dense-ablation", default = 0, type=int)
    parser.add_argument("--no-relation-ablation", action='store_true')
    parser.add_argument("--random-graph-ablation", action='store_true')
    parser.add_argument("--size-ablation", default = 1.0, type=float)

    


    
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config["init_lr"] = args.lr
    if args.epochs != 0:
        config["max_epoch"] = args.epochs
    args.output_dir = os.path.join(args.output_dir,args.name)
    Path(os.path.join(args.output_dir)).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w')) 


    if args.train_num_samples == 0:
        args.train_num_samples = int(750000 * args.batch_size / 32) 
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"{name}: {val}\n") 


    
    main(args, config)