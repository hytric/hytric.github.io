import os
import numpy as np
from PIL import Image
import torch
import h5py
import json
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm

# GPU 가용성 확인 및 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 사전 학습된 이미지 캡셔닝 모델 로드 및 GPU로 이동
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=False, device_map={"": 0}, torch_dtype=torch.float32
)

data = {}
captions_dict = {}

# HDF5 파일에서 데이터 읽기
with h5py.File("/home/jskim/Graph_VIdeo_generation/dataset/vg/train.h5", 'r') as f:
    for k, v in f.items():
        if k == 'image_paths':
            image_paths = list(v)
        else:
            data[k] = torch.IntTensor(np.asarray(v))

# 각 이미지에 대해 캡션 생성
for index in tqdm(image_paths, desc="Generating captions"):
    img_path = os.path.join('/home/jskim/Graph_VIdeo_generation/dataset/vg/frames', str(index, encoding="utf-8"))

    # 이미지 파일이 존재하는지 확인
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue

    try:
        with open(img_path, 'rb') as f:
            with Image.open(f) as image:
                WW, HH = image.size
                image = image.convert('RGB')

        # Prepare the image for the captioning model
        inputs = processor(image, return_tensors="pt").to(device)

        # Remove 'max_length' from inputs to prevent the warning
        if 'max_length' in inputs:
            del inputs['max_length']

        # Generate the caption
        out = model.generate(
            pixel_values=inputs['pixel_values'],
            num_beams=5,             # Apply beam search
            temperature=1.2,         # Increase creativity with temperature
            top_p=0.9,               # Use nucleus sampling for more diverse results
            top_k=50,                # Add diversity with top-k sampling
            length_penalty=2.0       # Set length_penalty to favor longer captions
        )

        # Decode the generated caption
        caption = processor.decode(out[0], skip_special_tokens=True)

        # 캡션을 딕셔너리에 저장
        captions_dict[str(index)] = caption  # encoding="utf-8" 제거

    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
    
    # GPU 메모리 관리 (필요시)
    torch.cuda.empty_cache()

# 캡션을 JSON 파일로 저장
with open('image_captions.json', 'w', encoding='utf-8') as json_file:
    json.dump(captions_dict, json_file, ensure_ascii=False, indent=4)

print("Image captions have been successfully saved to 'image_captions.json'.")

# 이미지 캡션 예시 10개 출력
print("\nExample Captions:")
for i, (img_index, caption) in enumerate(list(captions_dict.items())[:10]):
    print(f"{i+1}. Image Path: {img_index}, Caption: {caption}")
