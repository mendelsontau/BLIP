import logging
import json
import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
import math
import copy

def repair_text(text):
    text_arr = text.split()
    new_text_arr = []
    for i in range(len(text_arr) - 1):
        if text_arr[i] != text_arr[i+1]:
            new_text_arr.append(text_arr[i])
    new_text_arr.append(text_arr[-1])
    new_text = " ".join(new_text_arr)
    return new_text

def create_text_from_graph(walks, all_objects_dict):
    text = ""
    for relations in walks:
        r = relations[0]
        if r["object_id"] == -1:
            if "attributes" in all_objects_dict[str(r["subject_id"])]:
                text += all_objects_dict[str(r["subject_id"])]["attributes"][0] + " " + all_objects_dict[str(r["subject_id"])]["names"][0]
            else:
                text += all_objects_dict[str(r["subject_id"])]["names"][0]
        else:               
            if "attributes" in all_objects_dict[str(r["subject_id"])]:
                text += all_objects_dict[str(r["subject_id"])]["attributes"][0] + " " + all_objects_dict[str(r["subject_id"])]["names"][0] + " " + r["predicate"] + " "
            else:
                text += all_objects_dict[str(r["subject_id"])]["names"][0] + " " + r["predicate"] + " "
            if len(relations) != 1:
                for i in range (1, len(relations)):
                    r = relations[i]
                    if "attributes" in all_objects_dict[str(r["subject_id"])]:
                        text += all_objects_dict[str(r["subject_id"])]["attributes"][0] + " " + all_objects_dict[str(r["subject_id"])]["names"][0] + " " + r["predicate"] + " "
                    else:
                        text += all_objects_dict[str(r["subject_id"])]["names"][0] + " " + r["predicate"] + " "
            r = relations[-1]
            if "attributes" in all_objects_dict[str(r["object_id"])]:
                text += all_objects_dict[str(r["object_id"])]["attributes"][0] + " " + all_objects_dict[str(r["object_id"])]["names"][0]
            else:
                text += all_objects_dict[str(r["object_id"])]["names"][0]
        text += ". "
    return text

def apply_negative_type_4(all_objects_list, all_objects_dict, attributes_annotations):
    success = False
    random.shuffle(all_objects_list)
    for chosen_object in all_objects_list:
        object_id = chosen_object["object_id"]
        if "attributes" not in chosen_object:
            continue
        for att in chosen_object["attributes"]:
            if att not in attributes_annotations:
                continue
            att_annotation = attributes_annotations[att]
            if "negations" not in att_annotation:
                continue
            negatives = att_annotation["negations"]
            if len(negatives) == 0:
                continue
            success = True
            chosen_negative = random.choice(negatives)
            all_objects_dict[str(object_id)]["attributes"][0] = chosen_negative
            break
        if success == True:
            break
    return success, all_objects_dict


def apply_negative_type_3(all_objects_list, all_objects_dict, attributes_annotations):
    success = False
    random.shuffle(all_objects_list)
    for chosen_object in all_objects_list:
        if success == True:
            break
        first_object_id = chosen_object["object_id"]
        if "attributes" not in chosen_object:
            continue
        for att in chosen_object["attributes"]:
            if success == True:
                break
            if att not in attributes_annotations:
                continue
            att_annotation = attributes_annotations[att]
            if "negations" not in att_annotation:
                continue
            negatives = att_annotation["negations"]
            if len(negatives) == 0:
                continue
            for chosen_object2 in all_objects_list:
                if success == True:
                    break
                second_object_id = chosen_object2["object_id"]
                if second_object_id == first_object_id:
                    continue
                if "attributes" not in chosen_object2:
                    continue
                for att2 in chosen_object2["attributes"]:
                    if att2 in negatives and all_objects_dict[str(first_object_id)]["names"][0] != all_objects_dict[str(second_object_id)]["names"][0]:
                        all_objects_dict[str(first_object_id)]["attributes"][0] = att2
                        all_objects_dict[str(second_object_id)]["attributes"][0] = att
                        success = True
                        break
    return success, all_objects_dict


        


def apply_negative_type_1(walks, relations_annotations):
    #find all relationships in the graph
    all_relations = []
    for w in range(len(walks)):
        walk = walks[w]
        for r in range(len(walk)):
            rel = walk[r]
            if rel["object_id"] != -1:
                all_relations.append((rel,w,r))

    random.shuffle(all_relations)
    for rand_rel in all_relations:
        success = False
        rel = rand_rel[0]
        rel_w = rand_rel[1]
        rel_r = rand_rel[2]
        subject_id = rel["subject_id"]
        object_id = rel["object_id"]
        predicate = rel["predicate"]
        if predicate in relations_annotations:
            if relations_annotations[predicate]["symmetry"] == "yes":
                success = True
                walks[rel_w][rel_r]["subject_id"] = object_id
                walks[rel_w][rel_r]["object_id"] = subject_id
        
        if success:
            break
    return success, walks

def apply_negative_type_2(walks, relations_annotations):
    #find all relationships in the graph
    all_relations = []
    for w in range(len(walks)):
        walk = walks[w]
        for r in range(len(walk)):
            rel = walk[r]
            if rel["object_id"] != -1:
                all_relations.append((rel,w,r))

    random.shuffle(all_relations)
    for rand_rel in all_relations:
        success = False
        rel = rand_rel[0]
        rel_w = rand_rel[1]
        rel_r = rand_rel[2]
        predicate = rel["predicate"]
        if predicate in relations_annotations:
            if relations_annotations[predicate]["negations"] != "" :
                negations = relations_annotations[predicate]["negations"].split(",")
                new_predicate = random.choice(negations)
                success = True
                walks[rel_w][rel_r]["predicate"] = new_predicate
        
        if success:
            break
    return success, walks

class VgDatasetText(Dataset):
    def __init__(self, vg_path, split, transforms, num_objects, vg_loss_lambda, negatives = False):
        f = open(os.path.join(vg_path,"vg_150k_" + split +  ".json"))
        self.data = json.load(f)
        f = open(os.path.join(vg_path,"relations_annotations.json"))
        self.relations_annotations = json.load(f)
        f = open(os.path.join(vg_path,"attributes_annotations.json"))
        self.attributes_annotations = json.load(f)
        self.vg_path = vg_path
        self.clip_image_size = 224
        self.transforms = transforms
        self.split = split
        self.num_objects = num_objects
        self.negatives = negatives
        self.only_text = True if vg_loss_lambda == 0.0 else False
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #load image
        image_url = self.data[idx]["image_data"]["url"]
        crop_dimensions = self.data[idx]["to_crop"]
        min_x = crop_dimensions[0]
        min_y = crop_dimensions[1]
        max_x = crop_dimensions[2]
        max_y = crop_dimensions[3]
        image_h = max_y - min_y
        image_w = max_x - min_x
        url_parts = image_url.split("/")
        folder = url_parts[5]
        filename = url_parts[6]
        image_path = os.path.join(self.vg_path,folder,folder,filename)

        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.crop(crop_dimensions)
        image = self.transforms(image)

        all_objects_dict = self.data[idx]["objects"]
        objects = []
        for id in all_objects_dict:
            objects.append(all_objects_dict[id])
        if not self.only_text:
            missing_objects = self.num_objects - len(objects)
            valid_objects = torch.tensor(self.num_objects - missing_objects, dtype=torch.long)
        
        #generate text
        walks = self.data[idx]["relations"]
        text = create_text_from_graph(walks, all_objects_dict)

        #generate negatives
        if self.negatives and self.split == "train":
            start_index = random.randint(0,3)
            for i in range(start_index,start_index + 4):
                negative_type = i % 4
                if negative_type == 0:
                    success, new_walks = apply_negative_type_1(copy.deepcopy(walks), self.relations_annotations)
                    new_all_objects_dict = all_objects_dict
                if negative_type == 1:
                    success, new_walks = apply_negative_type_2(copy.deepcopy(walks), self.relations_annotations)
                    new_all_objects_dict = all_objects_dict
                if negative_type == 2:
                    success, new_all_objects_dict = apply_negative_type_3(copy.deepcopy(objects), copy.deepcopy(all_objects_dict), self.attributes_annotations)
                    new_walks = walks
                if negative_type == 3:
                    success, new_all_objects_dict = apply_negative_type_4(copy.deepcopy(objects), copy.deepcopy(all_objects_dict), self.attributes_annotations)
                    new_walks = walks
                if success == True:
                    break
            neg_text = ""
            neg_mask = torch.tensor(0.0)
            if success:
                neg_text = create_text_from_graph(new_walks,new_all_objects_dict)
                neg_mask = torch.tensor(1.0)

        
        #prepare bounding boxes
        if not self.only_text:
            objects_bbs = [[ob["x"],ob["y"],ob["w"], ob["h"]] for ob in objects]
            for obj in objects_bbs:
                new_x1 = obj[0]
                new_y1 = obj[1]
                new_x2 = obj[0] + obj[2]
                new_y2 = obj[1] + obj[3]
                if obj[0] < min_x:
                    new_x1 = min_x
                if obj[1] < min_y:
                    new_y1 = min_y
                if obj[0] + obj[2] > max_x:
                    new_x2 = max_x
                if obj[1] + obj[3] > max_y:
                    new_y2 = max_y
                obj[0] = new_x1 - min_x
                obj[1] = new_y1 - min_y
                obj[2] = new_x2 - new_x1
                obj[3] = new_y2 - new_y1
                
            bounding_boxes = [[(ob[0] + 0.5*ob[2])/image_w,(ob[1] + 0.5*ob[3])/image_h,min((ob[2])/image_w,1.0),min((ob[3])/image_h,1.0)] for ob in objects_bbs]
            if missing_objects > 0:
                bounding_boxes += [[0.0,0.0,0.0,0.0] for i in range(missing_objects)]
            bounding_boxes = torch.tensor(bounding_boxes)

            #prepare object descriptions
            object_descriptions = [obj["attributes"][0] + " " + obj["names"][0] if "attributes" in obj else obj["names"][0] for obj in objects]
            object_descriptions = [repair_text(desc)for desc in object_descriptions]
            if missing_objects > 0:
                object_descriptions += ["" for i in range(missing_objects)]
            if self.split == "val":
                max_chars = 150
                object_text = [[ord(c) for c in s] for s in object_descriptions]
                padding = [[int(36) for i in range(max_chars - len(s))] for s in object_descriptions]
                object_texts = [lst1 + lst2 for lst1, lst2 in zip(object_text,padding)]
                text_lengths = [len(s) for s in object_descriptions]

        if self.only_text:
            if self.negatives:
                return image, text, neg_text, neg_mask, idx
            else:
                return image, text, idx
        if self.split == "train" and self.negatives:
            return image, text, valid_objects, bounding_boxes, object_descriptions, neg_text, neg_mask, idx
        elif self.split == "train" and not self.negatives:
            return image, text, valid_objects, bounding_boxes, object_descriptions, idx
        else:
            return image, text, valid_objects, bounding_boxes, object_descriptions, torch.tensor(object_texts), text_lengths


def get_vg_loader(dataset, args, vg_batch_size):
    sampler = DistributedSampler(dataset) if args.distributed else None
    shuffle = sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=vg_batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True
    )
    return dataloader

def get_vg_val_loader(dataset, args, vg_batch_size):
    dataloader = DataLoader(
        dataset,
        batch_size=vg_batch_size,
        num_workers=args.workers,
        pin_memory=True
    )
    return dataloader

