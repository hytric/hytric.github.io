import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import json
import re
import string

from tqdm.auto import tqdm

def normalize_answer(s):
    def remove_(text):
        ''' 불필요한 기호 제거 '''
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\(사진\)', ' ', text)
        text = re.sub(r'△', ' ', text)
        text = re.sub(r'▲', ' ', text)
        return text

    def white_space_fix(text):
        '''연속된 공백일 경우 하나의 공백으로 대체'''
        return ' '.join(text.split())

    def remove_punc(text):
        '''구두점 제거'''
        exclude = set(string.punctuation) - {'%'}  # 퍼센트 기호를 제거한 구두점 집합
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        '''소문자 전환'''
        return text.lower()

    # 전처리 함수들을 차례로 적용
    return white_space_fix(remove_punc(lower(remove_(s))))


# CSV 파일 경로
csv_file = './dacon_dataset/train.csv'
data = pd.read_csv(csv_file)

json_data = []

# 데이터 전처리
for _, row in tqdm(data.iterrows(), total=data.shape[0]):
    context = normalize_answer(row['context'])  # context 열에 전처리 함수 적용
    # context = row['context']  # context 열에 전처리 함수 적용
    question = row['question']
    answer = row['answer']


    json_data.append({
        "context": context,
        "question": question,
        "answer": answer
    })

# JSON 형식으로 저장
json_string = json.dumps(json_data, ensure_ascii=False, indent=4)
with open('train_json.json', 'w', encoding='utf-8') as file:
    file.write(json_string)

print("Done!")

# ---- argmented Data 이건 나중에 해보자


# with open('./train_json.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)
#
# def augment_data(data_list):
#     # 데이터를 랜덤하게 선택
#     pair = random.sample(data_list, 2)
#
#     # 연결한 문구를 랜덤하게 선택
#     conjunctions = [" 그리고 ", " 또한 ", " 과 ", " 와 "]
#     conjunction = random.choice(conjunctions)
#
#     # 두 데이터를 이어붙임
#     combined_question = pair[0]['question'] + conjunction + pair[1]['question']
#     combined_answer = pair[0]['answer'] + conjunction + pair[1]['answer']
#
#     new_data = {
#         "question": combined_question,
#         "answer": combined_answer
#     }
#     return new_data
#
# augmented_data = [augment_data(data) for _ in range(500)]
# augmented_full_data = data + augmented_data
#
# with open('./augmented_data.json', 'w', encoding='utf-8') as file:
#     json.dump(augmented_full_data, file, ensure_ascii=False, indent=4)


