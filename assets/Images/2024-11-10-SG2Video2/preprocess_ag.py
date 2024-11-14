import argparse
import os
import imageio
import numpy as np
from PIL import Image
import pickle
import json
import random
import h5py
from collections import Counter, defaultdict

VG_DIR = '/home/jskim/Graph_VIdeo_generation/dataset/action-genome'

parser = argparse.ArgumentParser()

# Input data
parser.add_argument('--frame_list', default=os.path.join(VG_DIR, 'frame_list.txt'))
parser.add_argument('--objects_attributes_pkl', default=os.path.join(VG_DIR, 'object_bbox_and_relationship.pkl'))
parser.add_argument('--object_aliases', default=os.path.join(VG_DIR, 'object_classes.txt'))
parser.add_argument('--relationship_aliases', default=os.path.join(VG_DIR, 'relationship_classes.txt'))
parser.add_argument('--person_bbox', default=os.path.join(VG_DIR, 'person_bbox.pkl'))

# Arguments for images
parser.add_argument('--min_image_size', default=200, type=int)
parser.add_argument('--train_split', default='train')

# Arguments for objects
parser.add_argument('--min_object_instances', default=2000, type=int)
parser.add_argument('--min_attribute_instances', default=2000, type=int)
parser.add_argument('--min_object_size', default=2, type=int)
parser.add_argument('--min_objects_per_image', default=2, type=int)
parser.add_argument('--max_objects_per_image', default=10, type=int)
parser.add_argument('--max_attributes_per_image', default=20, type=int)

# Arguments for relationships
parser.add_argument('--min_relationship_instances', default=500, type=int)
parser.add_argument('--min_relationships_per_image', default=0, type=int)
parser.add_argument('--max_relationships_per_image', default=100, type=int)

# Output
parser.add_argument('--output_vocab_json', default=os.path.join(VG_DIR, 'vocab.json'))
parser.add_argument('--output_train_id_json', default=os.path.join(VG_DIR, 'train_id_to_image.json'))
parser.add_argument('--output_test_id_json', default=os.path.join(VG_DIR, 'test_id_to_image.json'))
parser.add_argument('--output_type_json', default=os.path.join(VG_DIR, 'type.json'))
parser.add_argument('--base_path', default="")
parser.add_argument('--output_h5_dir', default=VG_DIR)

def load_image(path):
    return imageio.imread(path)

def resize_image(image, size):
    return np.array(Image.fromarray(image).resize(size, Image.BILINEAR))

def load_frame_list(path):
    with open(path, 'r') as f:
        frame_list = f.readlines()
    return [frame.strip() for frame in frame_list]

def load_pickle_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_txt_file(path):
    with open(path, 'r') as f:
        data = f.readlines()
    return [line.strip() for line in data]

def load_json_file(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def create_image_id_to_image_dict(frame_list):
    image_id_to_image = {idx: {'image_id': idx, 'file_name': file_name} for idx, file_name in enumerate(frame_list)}
    return image_id_to_image

# 객체 이름과 일치하는 인덱스를 찾는 함수
def get_object_id_by_name(name, vocab):
    try:
        idx = name.replace('/', '')
        return vocab['object_idx_to_name'].index(idx)
    except ValueError:
        print('No ID : '+ idx + name)
        return None

def create_object_id_to_object_dict(objects_attributes, person_bbox, vocab):
    '''
    object_id_to_object = {idx: {'object_id': idx, 'name': name, 'bbox': [0,0,0,0]}}
    '''
    object_id_to_object = {}
    idx = 1

    for key, objects in objects_attributes.items():
        video = key.split('/')
        bbox = None
        if key in person_bbox:
            if person_bbox[key]['bbox'].size > 0:
                bbox = tuple(person_bbox[key]['bbox'][0])
                objects.append({'class': 'person', 'bbox': bbox, 'attention_relationship': None, 'spatial_relationship': None, 'contacting_relationship': None, 'metadata': {'tag': video[1] + '/person/' + video[0], 'set': None}, 'visible': True})
            else:
                bbox = [0,0,0,0]
                objects.append({'class': 'person', 'bbox': bbox, 'attention_relationship': None, 'spatial_relationship': None, 'contacting_relationship': None, 'metadata': {'tag': video[1] + '/person/' + video[0], 'set': None}, 'visible': True})
            

        valid_objects = []
        for obj in objects:
            name = obj['class']
            name_id = get_object_id_by_name(name, vocab)
            object_id_to_object[idx] = {'name_id': name_id, 'name': name, 'bbox': obj['bbox']}
            obj['object_idx'] = idx  # Add idx to the original objects_attributes
            idx += 1
            valid_objects.append(obj)

        # Update the objects_attributes to only include valid objects
        if valid_objects:
            objects_attributes[key] = valid_objects
        else:
            print(key)
            

    return object_id_to_object
    
def create_object_vocab(object_aliases_path):
    # 객체 별칭 파일 로드
    object_aliases = load_txt_file(object_aliases_path)

    # 특별한 객체 "__image__"을 추가
    object_aliases = ['__image__'] + object_aliases

    # object_name_to_idx 사전 생성
    object_name_to_idx = {name: idx for idx, name in enumerate(object_aliases)}

    # object_idx_to_name 리스트 생성
    object_idx_to_name = object_aliases

    # 결과 어휘 사전
    vocab = {
        'object_name_to_idx': object_name_to_idx,
        'object_idx_to_name': object_idx_to_name
    }

    return vocab

def create_rel_vocab(relationship_aliases_path):
    # 관계 별칭 파일 로드
    relationship_aliases = load_txt_file(relationship_aliases_path)

    # 특별한 관계 "__in_image__"을 추가
    relationship_aliases = ['__in_image__'] + relationship_aliases

    # relationship_name_to_idx 사전 생성
    relationship_name_to_idx = {name: idx for idx, name in enumerate(relationship_aliases)}

    # relationship_idx_to_name 리스트 생성
    relationship_idx_to_name = relationship_aliases

    # 결과 어휘 사전 갱신
    vocab = {
        'pred_name_to_idx': relationship_name_to_idx,
        'pred_idx_to_name': relationship_idx_to_name
    }

    return vocab

def extract_relationships(objects_attributes):
    objects = []
    relationships = []
    
    for frame in objects_attributes:
        objs = objects_attributes[frame]
        for obj in objs:
            if obj.get('class'):
                if obj['class'] not in objects:
                    objects.append(obj['class'])
            if obj.get('attention_relationship'):
                for rel in obj['attention_relationship']:
                    if rel not in relationships:
                        relationships.append(rel)
            if obj.get('spatial_relationship'):
                for rel in obj['spatial_relationship']:
                    if rel not in relationships:
                        relationships.append(rel)
            if obj.get('contacting_relationship'):
                for rel in obj['contacting_relationship']:
                    if rel not in relationships:
                        relationships.append(rel)
                
    return objects, relationships

from collections import defaultdict

import random

def set_splits(image_id_to_image):
    # Step 1: Collect image IDs by video
    video_name = {}
    for i in image_id_to_image:
        video = image_id_to_image[i]['file_name'].split('/')[0]
        image_id = image_id_to_image[i]['image_id']
        if video in video_name:
            video_name[video].append(image_id)
        else:
            video_name[video] = [image_id]
    
    # Step 2: Split videos into train, val, and test sets
    videos = list(video_name.keys())
    random.shuffle(videos)
    
    num_train = int(len(videos) * 0.98)
    num_val = int(len(videos) * 0.01) 
    num_test = len(videos) - num_train - num_val
    
    train_videos = videos[:num_train]
    val_videos = videos[num_train:num_train + num_val]
    test_videos = videos[num_train + num_val:]
    
    splits = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for video in train_videos:
        splits['train'].extend(video_name[video])
    
    for video in val_videos:
        splits['val'].extend(video_name[video])
    
    for video in test_videos:
        splits['test'].extend(video_name[video])
    
    return splits



def create_relationship_dict(image_id_to_image, objects_attributes, person_bbox, vocab):
    relationships_data = []

    for image in image_id_to_image.values():
        file_name = image['file_name']
        image_id = image['image_id']
        
        if file_name not in objects_attributes or not objects_attributes[file_name]:
            print(file_name + ' : there is no file')
        
        # person 객체를 찾습니다.
        person = next((obj for obj in objects_attributes[file_name] if obj['class'] == 'person'), None)
        
        if person is None or 'bbox' not in person:
            print(file_name + ' : there is no person')
        else:
            objects = [obj for obj in objects_attributes[file_name] if obj['class'] != 'person']
        
        image_relationships = []

        for obj in objects:
            if 'bbox' not in obj:
                print(file_name + ' : there is no obj')
            elif obj['bbox'] == None:
                continue
            for rel_type in ['attention_relationship', 'spatial_relationship', 'contacting_relationship']:
                if obj[rel_type] is not None:
                    for predicate in obj[rel_type]:
                        predicate_key = predicate.replace('_', '')
                        relationship_id = vocab['pred_name_to_idx'].get(predicate_key, None)
                        if relationship_id is None:
                            raise ValueError(f"Unknown predicate '{predicate}' encountered.")
                        
                        relationship = {
                            "predicate": predicate_key,
                            "object": {
                                "name": obj['class'],
                                "h": obj['bbox'][3],
                                "object_id": obj['object_idx'],
                                "synsets":  [obj['class']],
                                "w": obj['bbox'][2],
                                "y": obj['bbox'][1],
                                "x": obj['bbox'][0]
                            },
                            "relationship_id": relationship_id,
                            "synsets": [predicate_key],
                            "subject": {
                                "name": person['class'],
                                "h": person['bbox'][3],
                                "object_id": person['object_idx'],
                                "synsets":  [person['class']],
                                "w": person['bbox'][2],
                                "y": person['bbox'][1],
                                "x": person['bbox'][0]
                            }
                        }
                        image_relationships.append(relationship)

        relationships_data.append({"relationships": image_relationships, "image_id": image_id})

    return relationships_data

def create_object_dict(image_id_to_image, objects_attributes):
    object_data = []

    for image in image_id_to_image.values():
        file_name = image['file_name']
        image_id = image['image_id']
        
        image_object = []

        for obj in objects_attributes[file_name]:
            if obj['bbox'] == None:
                x, y, width, height = 0,0,0,0
                continue
            else:
                x, y, width, height = obj['bbox']
            object_info = {
                "object_id": obj['object_idx'],
                "names": [obj['class']],
                "synsets": [obj['class']],
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "merged_object_ids": []
            }
            image_object.append(object_info)

        object_data.append({"objects": image_object, "image_id": image_id})

    return object_data


def create_attributes_dict(image_id_to_image, objects_attributes):
    object_data = []

    for image in image_id_to_image.values():
        file_name = image['file_name']
        image_id = image['image_id']
        
        image_object = []

        for obj in objects_attributes[file_name]:
            if obj['bbox'] == None:
                x, y, width, height = 0,0,0,0
                continue
            else:
                x, y, width, height = obj['bbox']
            object_info = {
                'synsets': [obj['class']],
                'h': height,
                'object_id': obj['object_idx'],
                'names': [obj['class']],
                'w': width,
                'attributes': [obj['class']],
                'y': y,
                'x': x
            }

            image_object.append(object_info)

        object_data.append({"attributes": image_object, "image_id": image_id})

    return object_data

def data_management(image_id_to_image, objects_attributes):
    '''
    person이 없는 경우, person에 bbox가 없는 경우(None) dataset에서 삭제
    '''
    
    invalid_image_ids = []
    invalid_file_names = []

    for image in image_id_to_image.values():
        image_id = image['image_id']
        file_name = image['file_name']
        image_objects = objects_attributes.get(file_name, [])

        # 'person' 클래스가 있는지 확인하고 bbox가 None인지 확인
        person_found = False
        for obj in image_objects:
            if obj['class'] == 'person':
                if obj.get('bbox') is not None:  # bbox가 None이 아닌지 확인
                    person_found = True
                    break
        
        if not person_found:
            invalid_image_ids.append(image_id)
            invalid_file_names.append(file_name)
    
    # invalid_image_ids에 있는 이미지 ID를 image_id_to_image에서 삭제
    for invalid_id in invalid_image_ids:
        if invalid_id in image_id_to_image:
            del image_id_to_image[invalid_id]
    
    # invalid_file_names에 있는 파일 이름을 objects_attributes에서 삭제
    for file_name in invalid_file_names:
        if file_name in objects_attributes:
            del objects_attributes[file_name]

    # 걸러진 이미지 수 프린트
    print(f"{len(invalid_image_ids)} images were removed from the dataset.")


def create_attribute_vocab(args, image_ids, attributes, vocab):
  image_ids = set(image_ids)
  print('Making attribute vocab from %d training images' % len(image_ids))
  attribute_name_counter = Counter()
  for image in attributes:
    if image['image_id'] not in image_ids:
      continue
    for attribute in image['attributes']:
      names = set()
      try:
        for name in attribute['attributes']:
          names.add(name)
        attribute_name_counter.update(names)
      except KeyError:
        pass
  attribute_names = []
  for name, count in attribute_name_counter.most_common():
    if count >= args.min_attribute_instances:
      attribute_names.append(name)
  print('Found %d attribute categories with >= %d training instances' %
        (len(attribute_names), args.min_attribute_instances))

  attribute_name_to_idx = {}
  attribute_idx_to_name = []
  for idx, name in enumerate(attribute_names):
    attribute_name_to_idx[name] = idx
    attribute_idx_to_name.append(name)
  vocab['attribute_name_to_idx'] = attribute_name_to_idx
  vocab['attribute_idx_to_name'] = attribute_idx_to_name

def save_id(image_id_to_image, train_id, test_id):
    frame_id = {}
    for idx, image_id in enumerate(train_id):
        video = image_id_to_image[image_id]['file_name'].split('/')[0]
        if video in frame_id:
            frame_id[video].append(int(idx))  # int64 타입을 Python의 int 타입으로 변환
        else:
            frame_id[video] = [int(idx)]  # int64 타입을 Python의 int 타입으로 변환
    print('Writing image_id_to_image to "%s"' % args.output_train_id_json)
    print(len(image_id_to_image))
    with open(args.output_train_id_json, 'w') as f:
        json.dump(frame_id, f)

    frame_id = {}
    for idx, image_id in enumerate(test_id):
        video = image_id_to_image[image_id]['file_name'].split('/')[0]
        if video in frame_id:
            frame_id[video].append(int(idx))  # int64 타입을 Python의 int 타입으로 변환
        else:
            frame_id[video] = [int(idx)]  # int64 타입을 Python의 int 타입으로 변환
    print('Writing image_id_to_image to "%s"' % args.output_test_id_json)
    with open(args.output_test_id_json, 'w') as f:
        json.dump(frame_id, f)


def get_image_paths(image_id_to_image, image_ids):
    paths = []
    for image_id in image_ids:
        path = os.path.join(args.base_path, image_id_to_image[image_id]['file_name'])
        paths.append(path)

    return paths

def save_image_id_to_image(splits, image_id_to_image):
    temp_splits = []
    for split, image_ids in splits.items():
        for i in image_ids:
            temp_splits.append({
                    'image_ids':image_ids, 
                    'frame_path': image_id_to_image[image_ids]['file_name'],
                    'type': split
                })
            
    print('Writing type to "%s"' % args.output_type_json)
    with open(args.output_type_json, 'w') as f:
        json.dump(temp_splits, f)  


def encode_graphs(args, splits, objects, relationships, vocab, object_id_to_obj, attributes):

  image_id_to_objects = {} 
  for image in objects:
    image_id = image['image_id']
    image_id_to_objects[image_id] = image['objects']
  image_id_to_relationships = {}
  for image in relationships:
    image_id = image['image_id']
    image_id_to_relationships[image_id] = image['relationships']
  image_id_to_attributes = {} 
  for image in attributes:
    image_id = image['image_id']
    image_id_to_attributes[image_id] = image['attributes']

  numpy_arrays = {}
  for split, image_ids in splits.items(): 
    skip_stats = defaultdict(int)
    final_image_ids = []
    object_ids = []
    object_names = []
    object_boxes = []
    objects_per_image = []
    relationship_ids = []
    relationship_subjects = []
    relationship_predicates = []
    relationship_objects = []
    relationships_per_image = []
    attribute_ids = []
    attributes_per_object = []
    object_attributes = []
    for image_id in image_ids:
      image_object_ids = []
      image_object_names = []
      image_object_boxes = []
      object_id_to_idx = {}
      for obj in image_id_to_objects[image_id]:
        object_id = obj['object_id']
        if object_id not in object_id_to_obj:
          continue
        obj = object_id_to_obj[object_id]
        object_id_to_idx[object_id] = len(image_object_ids)
        image_object_ids.append(object_id)
        image_object_names.append(obj['name_id'])
        image_object_boxes.append(obj['bbox'])
      num_objects = len(image_object_ids)
      too_few = num_objects < args.min_objects_per_image
      too_many = num_objects > args.max_objects_per_image
      if too_few:
        skip_stats['too_few_objects'] += 1
        continue
      if too_many:
        skip_stats['too_many_objects'] += 1
        continue
      image_rel_ids = []
      image_rel_subs = []
      image_rel_preds = []
      image_rel_objs = []
      for rel in image_id_to_relationships[image_id]:
        relationship_id = rel['relationship_id']
        pred = rel['predicate']
        pred_idx = vocab['pred_name_to_idx'].get(pred, None)
        if pred_idx is None:
          continue
        sid = rel['subject']['object_id']
        sidx = object_id_to_idx.get(sid, None)
        oid = rel['object']['object_id']
        oidx = object_id_to_idx.get(oid, None)
        if sidx is None or oidx is None:
          continue
        image_rel_ids.append(relationship_id)
        image_rel_subs.append(sidx)
        image_rel_preds.append(pred_idx)
        image_rel_objs.append(oidx)
      num_relationships = len(image_rel_ids)
      too_few = num_relationships < args.min_relationships_per_image
      too_many = num_relationships > args.max_relationships_per_image
      if too_few:
        skip_stats['too_few_relationships'] += 1
        continue
      if too_many:
        skip_stats['too_many_relationships'] += 1
        continue

      obj_id_to_attributes = {}
      num_attributes = []
      for obj_attribute in image_id_to_attributes[image_id]:
        obj_id_to_attributes[obj_attribute['object_id']] = obj_attribute.get('attributes', None)
      for object_id in image_object_ids:
        attributes = obj_id_to_attributes.get(object_id, None)
        if attributes is None:
          object_attributes.append([-1] * args.max_attributes_per_image)
          num_attributes.append(0)
        else:
          attribute_ids = []
          for attribute in attributes:
            if attribute in vocab['attribute_name_to_idx']:
              attribute_ids.append(vocab['attribute_name_to_idx'][attribute])
            if len(attribute_ids) >= args.max_attributes_per_image:
              break
          num_attributes.append(len(attribute_ids))
          pad_len = args.max_attributes_per_image - len(attribute_ids)
          attribute_ids = attribute_ids + [-1] * pad_len
          object_attributes.append(attribute_ids)

      # Pad object info out to max_objects_per_image
      while len(image_object_ids) < args.max_objects_per_image:
        image_object_ids.append(-1)
        image_object_names.append(-1)
        image_object_boxes.append([-1, -1, -1, -1])
        num_attributes.append(-1)

      # Pad relationship info out to max_relationships_per_image
      while len(image_rel_ids) < args.max_relationships_per_image:
        image_rel_ids.append(-1)
        image_rel_subs.append(-1)
        image_rel_preds.append(-1)
        image_rel_objs.append(-1)

      final_image_ids.append(image_id)
      object_ids.append(image_object_ids)
      object_names.append(image_object_names)
      object_boxes.append(image_object_boxes)
      objects_per_image.append(num_objects)
      relationship_ids.append(image_rel_ids)
      relationship_subjects.append(image_rel_subs)
      relationship_predicates.append(image_rel_preds)
      relationship_objects.append(image_rel_objs)
      relationships_per_image.append(num_relationships)
      attributes_per_object.append(num_attributes)

    print('Skip stats for split "%s"' % split)
    for stat, count in skip_stats.items():
      print(stat, count)
    print()
    numpy_arrays[split] = {
      'image_ids': np.asarray(final_image_ids),
      'object_ids': np.asarray(object_ids),
      'object_names': np.asarray(object_names),
      'object_boxes': np.asarray(object_boxes),
      'objects_per_image': np.asarray(objects_per_image),
      'relationship_ids': np.asarray(relationship_ids),
      'relationship_subjects': np.asarray(relationship_subjects),
      'relationship_predicates': np.asarray(relationship_predicates),
      'relationship_objects': np.asarray(relationship_objects),
      'relationships_per_image': np.asarray(relationships_per_image),
      'attributes_per_object': np.asarray(attributes_per_object),
      'object_attributes': np.asarray(object_attributes),
    }
    for k, v in numpy_arrays[split].items():
      if v.dtype == np.int64:
        numpy_arrays[split][k] = v.astype(np.int32)
  return numpy_arrays

def load_aliases(alias_path):
  aliases = {}
  print('Loading aliases from "%s"' % alias_path)
  with open(alias_path, 'r') as f:
    for line in f:
      line = [s.strip() for s in line.split(',')]
      for s in line:
        aliases[s] = line[0]
  return aliases

def main(args):
    image_id_to_image = {}
    relationships = {}
    vocab = {}
    splits = {}

    embedding = False
    obj_aliases = load_aliases(args.object_aliases)
    rel_aliases = load_aliases(args.relationship_aliases)

    print('### Upload Data ###')

    # 프레임 리스트 로드
    frame_list = load_frame_list(args.frame_list)
    image_id_to_image = create_image_id_to_image_dict(frame_list)
    # video_list = make_video_frame(image_id_to_image)

    person_bbox = load_pickle_file(args.person_bbox)
    objects_attributes = load_pickle_file(args.objects_attributes_pkl)

    # object_aliases = load_txt_file(args.object_aliases)
    vocab = create_object_vocab(args.object_aliases)

    object_id_to_obj = create_object_id_to_object_dict(objects_attributes, person_bbox, vocab)
    
    # 'name_id' 값의 개수 카운트
    name_id_count = {}

    # 각 객체에서 'name_id'의 개수를 셈
    for obj in object_id_to_obj.values():
        name_id = obj.get('name_id', None)
        if name_id is not None:
            if name_id in name_id_count:
                name_id_count[name_id] += 1
            else:
                name_id_count[name_id] = 1

    # 결과 출력
    print(name_id_count)

    # bbox 비어있는거 빼는 함수
    if embedding:
        data_management(image_id_to_image, objects_attributes)
    splits = set_splits(image_id_to_image)

    relationship_vocab = create_rel_vocab(args.relationship_aliases)
    vocab.update(relationship_vocab)
    
    print(f'train: {len(splits["train"])} / val: {len(splits["val"])} / test: {len(splits["test"])}')
    

    print('### Data Preprocessing ###')

    relationships = create_relationship_dict(image_id_to_image, objects_attributes, person_bbox, vocab)
    objects = create_object_dict(image_id_to_image, objects_attributes)
    attributes = create_attributes_dict(image_id_to_image, objects_attributes)
    
    #--------------------------- 정리해야하는 코드 ---------------------------
    train_ids = splits[args.train_split]
    create_attribute_vocab(args, train_ids, attributes, vocab)
    #------------------------------------------------------------------------
    
    # 분포
    import numpy as np
    from collections import Counter
    
    relationships_list = []
    for i in relationships:
        for j in i['relationships']:
            relationships_list.append(j['predicate'])
    aa = Counter(relationships_list)
    print("relationships_list:")
    print(aa)
    
    objects_list = []
    for i in objects:
        for j in i['objects']:
            objects_list.append(j['names'][0])
    bb = Counter(objects_list)
    print("objects_list:")
    print(bb)

    print('Set relationships, object, attributes')

    numpy_arrays = encode_graphs(args, splits, objects, relationships, vocab, object_id_to_obj, attributes)
    
    print('### final dataset ###')
    print(f'train: {len(numpy_arrays["train"]["image_ids"])} / val: {len(numpy_arrays["val"]["image_ids"])} / test: {len(numpy_arrays["test"]["image_ids"])}\n')
    
    print('Writing HDF5 output files')
    for split_name, split_arrays in numpy_arrays.items():
        image_ids = list(split_arrays['image_ids'].astype(int))
        h5_path = os.path.join(args.output_h5_dir, '%s.h5' % split_name)
        print('Writing file "%s"' % h5_path)
        with h5py.File(h5_path, 'w') as h5_file:
            for name, ary in split_arrays.items():
                print('Creating datset: ', name, ary.shape, ary.dtype)
                h5_file.create_dataset(name, data=ary)
            print('Writing image paths')
            image_paths = get_image_paths(image_id_to_image, image_ids)
            path_dtype = h5py.special_dtype(vlen=str)
            path_shape = (len(image_paths),)
            path_dset = h5_file.create_dataset('image_paths', path_shape,
                                                dtype=path_dtype)
            for i, p in enumerate(image_paths):
                path_dset[i] = p
        print()

    save_id(image_id_to_image, numpy_arrays["train"]["image_ids"], numpy_arrays["test"]["image_ids"])

    print('Writing vocab to "%s"' % args.output_vocab_json)
    with open(args.output_vocab_json, 'w') as f:
        json.dump(vocab, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)