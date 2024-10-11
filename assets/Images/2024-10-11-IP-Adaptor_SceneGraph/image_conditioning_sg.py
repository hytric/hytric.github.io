import datetime
import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import random
import h5py
import numpy as np
import PIL
from PIL import Image
import tempfile

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor
from torchvision import transforms
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from cgip.cgip import CGIPModel
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,vocab, h5_path, image_dir, image_size=256, max_objects=10, max_samples=None,
                 include_relationships=True, use_orphaned_objects=True, size=512, t_drop_rate=0.05, i_drop_rate=0.05, 
                 ti_drop_rate=0.05):
        super().__init__()
        
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        
        with open(vocab, 'r') as f:
            vocab = json.load(f)
        self.image_dir = image_dir
        self.image_size = (image_size, image_size)
        self.vocab = vocab
        self.num_objects = len(vocab['object_idx_to_name'])
        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects = max_objects
        self.max_samples = max_samples
        self.include_relationships = include_relationships

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.data = {}
        with h5py.File(h5_path, 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    self.image_paths = list(v)
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))
                    
    def __getitem__(self, idx):
        item = self.set_sg(idx)
        image_id = self.image_paths[idx].decode('utf-8')
        
        # read image
        image = item[0]
        sg = item
        
        return {
            "image": image,
            "text_input_ids": image_id,
            "clip_image": sg,
            "idx" : str(self.image_paths[idx], encoding="utf-8")
        }

    def __len__(self):
        num = self.data['object_names'].size(0)
        if self.max_samples is not None:
            return min(self.max_samples, num)
        return num

    def set_sg(self, index):
        img_path = os.path.join(self.image_dir, str(self.image_paths[index], encoding="utf-8"))
        
        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))

        image = image * 2 - 1

        obj_idxs_with_rels = set()
        obj_idxs_without_rels = set(range(self.data['objects_per_image'][index].item()))
        for r_idx in range(self.data['relationships_per_image'][index]):
            s = self.data['relationship_subjects'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            obj_idxs_with_rels.add(s)
            obj_idxs_with_rels.add(o)
            obj_idxs_without_rels.discard(s)
            obj_idxs_without_rels.discard(o)

        obj_idxs = list(obj_idxs_with_rels)
        obj_idxs_without_rels = list(obj_idxs_without_rels)
        if len(obj_idxs) > self.max_objects - 1:
            obj_idxs = random.sample(obj_idxs, self.max_objects)
        if len(obj_idxs) < self.max_objects - 1 and self.use_orphaned_objects:
            num_to_add = self.max_objects - 1 - len(obj_idxs)
            num_to_add = min(num_to_add, len(obj_idxs_without_rels))
            obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)
        O = len(obj_idxs) + 1

        objs = torch.LongTensor(O).fill_(-1)

        boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(O, 1)
        obj_idx_mapping = {}
        for i, obj_idx in enumerate(obj_idxs):
            objs[i] = self.data['object_names'][index, obj_idx].item()
            x, y, w, h = self.data['object_boxes'][index, obj_idx].tolist()
            x0 = float(x) / WW
            y0 = float(y) / HH
            x1 = float(x + w) / WW
            y1 = float(y + h) / HH
            boxes[i] = torch.FloatTensor([x0, y0, x1, y1])
            obj_idx_mapping[obj_idx] = i

        objs[O - 1] = self.vocab['object_name_to_idx']['__image__']

        triples = []
        for r_idx in range(self.data['relationships_per_image'][index].item()):
            if not self.include_relationships:
                break
            s = self.data['relationship_subjects'][index, r_idx].item()
            p = self.data['relationship_predicates'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            s = obj_idx_mapping.get(s, None)
            o = obj_idx_mapping.get(o, None)
            if s is not None and o is not None:
                triples.append([s, p, o])

        in_image = self.vocab['pred_name_to_idx']['__in_image__']
        for i in range(O - 1):
            triples.append([i, in_image, O - 1])

        triples = torch.LongTensor(triples)
        return image, objs, boxes, triples
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = [example["text_input_ids"] for example in data]
    idx = [example["idx"] for example in data]
    temp = []
    for example in data:
        temp.append(example["clip_image"])
    clip_images = vg_collate_fn(temp)

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "Scene Graph": clip_images, 
        "idx" : idx
    }
    
def vg_collate_fn(batch):
    all_imgs, all_objs, all_boxes, all_triples = [], [], [], []
    all_obj_to_img, all_triple_to_img = [], []
    obj_offset = 0
    for i, (img, objs, boxes, triples) in enumerate(batch):
        all_imgs.append(img[None])
        O, T = objs.size(0), triples.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))
        obj_offset += O

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_triples = torch.cat(all_triples)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)

    out = (all_imgs, all_objs, all_boxes, all_triples, all_obj_to_img, all_triple_to_img)
    return out

class Resize(object):
    def __init__(self, size, interp=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
            H, W = size
            self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)

def save_graph_info(vocab, objs, triples, path):
    f = open(path, 'w')
    data = f'''
    objs : 
    {objs}
    
    
    triples :
    {triples}
    
    
    ### vocab ###
    {vocab['object_name_to_idx']}
    
    {vocab["attribute_name_to_idx"]}
    
    {vocab["pred_name_to_idx"]}
    '''
    f.write(data)
    f.close() 

def draw_scene_graph(objs, triples, vocab=None, **kwargs):
    output_filename = kwargs.pop('output_filename', 'graph.png')
    orientation = kwargs.pop('orientation', 'V')
    edge_width = kwargs.pop('edge_width', 6)
    arrow_size = kwargs.pop('arrow_size', 1.5)
    binary_edge_weight = kwargs.pop('binary_edge_weight', 1.2)
    ignore_dummies = kwargs.pop('ignore_dummies', True)

    if orientation not in ['V', 'H']:
        raise ValueError('Invalid orientation "%s"' % orientation)
    rankdir = {'H': 'LR', 'V': 'TD'}[orientation]

    if vocab is not None:
        assert torch.is_tensor(objs)
        assert torch.is_tensor(triples)
        objs_list, triples_list = [], []
        for i in range(objs.size(0)):
            objs_list.append(vocab['object_idx_to_name'][objs[i].item()])
        for i in range(triples.size(0)):
            s = triples[i, 0].item()
            # p = vocab['pred_name_to_idx'][triples[i, 1].item()]
            p = triples[i, 1].item()
            o = triples[i, 2].item()
            triples_list.append([s, p, o])
        objs, triples = objs_list, triples_list

    lines = [
        'digraph{',
        'graph [size="5,3",ratio="compress",dpi="300",bgcolor="transparent"]',
        'rankdir=%s' % rankdir,
        'nodesep="0.5"',
        'ranksep="0.5"',
        'node [shape="box",style="rounded,filled",fontsize="48",color="none"]',
        'node [fillcolor="lightpink1"]',
    ]

    for i, obj in enumerate(objs):
        if ignore_dummies and obj == '__image__':
            continue
        lines.append('%d [label="%s"]' % (i, obj))

    next_node_id = len(objs)
    lines.append('node [fillcolor="lightblue1"]')
    for s, p, o in triples:
        p = vocab['pred_idx_to_name'][p]
        if ignore_dummies and p == '__in_image__':
            continue
        lines += [
            '%d [label="%s"]' % (next_node_id, p),
            '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
                s, next_node_id, edge_width, arrow_size, binary_edge_weight),
            '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
                next_node_id, o, edge_width, arrow_size, binary_edge_weight)
        ]
        next_node_id += 1
    lines.append('}')

    ff, dot_filename = tempfile.mkstemp()
    with open(dot_filename, 'w') as f:
        for line in lines:
            f.write('%s\n' % line)
    os.close(ff)

    output_format = os.path.splitext(output_filename)[1][1:]
    os.system('dot -T%s %s > %s' % (output_format, dot_filename, output_filename))
    return None


def convert(objs, triples, vocab):
    obj_vocab = {v: k for k, v in vocab['object_name_to_idx'].items()}  
    rel_vocab = {v: k for k, v in vocab["pred_name_to_idx"].items()}  

    objs_En = []
    
    for obj_idx in objs:
        obj_name = obj_vocab.get(obj_idx.item(), "unknown") 
        objs_En.append(obj_name)
        
    sentences = []
    for subj_idx, pred_idx, obj_idx in triples:
        subj_name = objs_En[subj_idx]  
        obj_name = objs_En[obj_idx]   
        pred_name = rel_vocab.get(pred_idx.item(), "related to")  

        sentence = f"{subj_name} {pred_name} {obj_name}"
        sentences.append(sentence)
    
    text = ". ".join(sentences) + "."

    text = text.replace("__in_image__", "").replace("__image__", "").strip()

    text = ' '.join(text.split())

    return text
        

# -----------------------------------------------------------------------------------------

with open("/home/jskim/Graph_VIdeo_generation/dataset/vg/vocab.json", 'r') as f:
    vocab = json.load(f)

# dataloader
train_dataset = MyDataset(vocab='/home/jskim/Graph_VIdeo_generation/dataset/vg/vocab.json', 
                            h5_path='/home/jskim/Graph_VIdeo_generation/dataset/vg/train.h5', 
                            image_dir='/home/jskim/Graph_VIdeo_generation/dataset/vg/frames',  
                            )
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=collate_fn,
    batch_size=1,
    num_workers=1,
)

captions_dict = {}

from tqdm import tqdm
for step, batch in enumerate(tqdm(train_dataloader, desc="Captioning Progress", unit="batch")):
    # print("\n")
    # print(f"{batch['idx']}")
    # print(f"images\t: {batch['images'].shape}")
    # print(f"text\t: {batch['text_input_ids']}")
    # print(f"SG")
    # print(f"|-- objs\t: {batch['Scene Graph'][1]}")
    # print(f"|-- triples\t: {batch['Scene Graph'][3]}")
    # save_graph_info(vocab, batch['Scene Graph'][1], batch['Scene Graph'][3], "/home/jskim/T2I/IP-Adapter/make_sg/graph_info.txt")
    # draw_scene_graph(batch['Scene Graph'][1], batch['Scene Graph'][3], vocab, output_filename="/home/jskim/T2I/IP-Adapter/make_sg/graph.png")
    
    caption = convert(batch['Scene Graph'][1], batch['Scene Graph'][3], vocab)
    captions_dict[str(batch['text_input_ids'])] = caption
    pass
    
with open('image_captions_sg.json', 'w', encoding='utf-8') as json_file:
    json.dump(captions_dict, json_file, ensure_ascii=False, indent=4)

print("Image captions have been successfully saved to 'image_captions.json'.")

print("\nExample Captions:")
for i, (img_index, caption) in enumerate(list(captions_dict.items())[:10]):
    print(f"{i+1}. Image Path: {img_index}, Caption: {caption}")