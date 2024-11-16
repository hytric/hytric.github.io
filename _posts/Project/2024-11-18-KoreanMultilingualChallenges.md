---
title: "[프로젝트] Korean Audio, Multilingual Hubert translate Training Challenges"
last_modified_at: 2024-11-18
categories:
  - Project
excerpt: "기존 Unit based audio Multilingual translate으로 제안된 논문에 Korean을 추가"
use_math: true
classes: wide
---

> Inha univ.  |  기존 Unit based audio Multilingual translate으로 제안된 논문에 Korean을 추가  
> ECE Capston design

> 
 
<br>

# Korean Audio, Multilingual Hubert translate Training Challenges

![Slide10.jpg](/assets/Images/2024-11-18-KoreanMultilingualChallenges/Slide10.jpg)

## Hubert 학습 과정 중 오류

**Training graph**

![image.png](/assets/Images/2024-11-18-KoreanMultilingualChallenges/image.png)

학습 중간 중에 loss 값이 갑자기 튀어서 시작으로 돌아감

Multilingual으로 hubert를 학습할 때 다음과 같은 현상 발생, 언어를 1개만 사용하면 정상적으로 학습이 진행됨

데이터 셋을 잘 선별, 조합 해야함
 
<br>

**Sampling 결과** : unit 분포가 고루 나와야하는데 그렇지 못함

- 영어 unit 분포 (기존 모델)
    
    
![unit_frequency.png](/assets/Images/2024-11-18-KoreanMultilingualChallenges/unit_frequency.png)
    
![unit_distribution.png](/assets/Images/2024-11-18-KoreanMultilingualChallenges/unit_distribution.png)
    
    Top 10 most common units and their counts:
    Unit ID 6: 581402 occurrences
    Unit ID 333: 334992 occurrences
    Unit ID 366: 303016 occurrences
    Unit ID 321: 302634 occurrences
    Unit ID 148: 302559 occurrences
    Unit ID 497: 292957 occurrences
    Unit ID 81: 280082 occurrences
    Unit ID 246: 274967 occurrences
    Unit ID 499: 270654 occurrences
    Unit ID 544: 268308 occurrences
    
    Total units: 61678104
    Unique units: 1000
    Diversity ratio: 0.00%
    
    Average maximum consecutive repeat of the same unit: 17.33
     
<br>

- 1번 시도 100km unit 분포 한글
    
    
![unit_distribution (1).png](/assets/Images/2024-11-18-KoreanMultilingualChallenges/unit_distribution_(1).png)
    
![unit_frequency (1).png](/assets/Images/2024-11-18-KoreanMultilingualChallenges/unit_frequency_(1).png)
    
    Top 10 most common units and their counts:
    Unit ID 156: 612918 occurrences
    Unit ID 159: 576338 occurrences
    Unit ID 102: 262760 occurrences
    Unit ID 0: 110868 occurrences
    Unit ID 16: 98076 occurrences
    Unit ID 172: 76559 occurrences
    Unit ID 31: 67843 occurrences
    Unit ID 228: 48923 occurrences
    Unit ID 23: 36542 occurrences
    Unit ID 146: 33211 occurrences
    
    Total units: 2150601
    Unique units: 116
    Diversity ratio: 0.01%
    
    Average maximum consecutive repeat of the same unit: 10.63
     
<br>

- 1번 시도 100km unit 분포 영어
    
    
![image.png](/assets/Images/2024-11-18-KoreanMultilingualChallenges/image%201.png)
    
![image.png](/assets/Images/2024-11-18-KoreanMultilingualChallenges/image%202.png)
    
    Top 10 most common units and their counts:
    Unit ID 156: 16935621 occurrences
    Unit ID 159: 9664653 occurrences
    Unit ID 102: 8959726 occurrences
    Unit ID 31: 4072103 occurrences
    Unit ID 0: 2924761 occurrences
    Unit ID 228: 2419386 occurrences
    Unit ID 16: 2219868 occurrences
    Unit ID 172: 1766505 occurrences
    Unit ID 23: 1593109 occurrences
    Unit ID 47: 1135941 occurrences
     
    Total units: 61678104
    Unique units: 145
    Diversity ratio: 0.00%
     
    Average maximum consecutive repeat of the same unit: 22.35
     
<br>

- kr-hubert 모델 활용
    
    
![unit_frequency (2).png](/assets/Images/2024-11-18-KoreanMultilingualChallenges/unit_frequency_(2).png)
    
![unit_distribution (2).png](/assets/Images/2024-11-18-KoreanMultilingualChallenges/unit_distribution_(2).png)
    
    Top 10 most common units and their counts:
    Unit ID 200: 20688124 occurrences
    Unit ID 102: 13252254 occurrences
    Unit ID 208: 11483470 occurrences
    Unit ID 290: 6566346 occurrences
    Unit ID 653: 3652414 occurrences
    Unit ID 52: 3160903 occurrences
    Unit ID 545: 2270142 occurrences
    Unit ID 185: 1967716 occurrences
    Unit ID 441: 1909765 occurrences
    Unit ID 444: 1372218 occurrences
    
    Total units: 89532680
    Unique units: 745
    Diversity ratio: 0.00%
    
    Average maximum consecutive repeat of the same unit: 37.83
    

데이터 셋을 배경 음이 적고 다양한 화자의 음성이 반영될 수 있도록 데이터 셋을 구성
 
<br>

## Transformer학습 중 오류

![image.png](/assets/Images/2024-11-18-KoreanMultilingualChallenges/image%203.png)

빠르게 학습이 이루어짐 → overfitting 학습을 위한 데이터 양이 많이 필요함
 
<br>

### 1차 Proposal

![page_1_high_res.png](/assets/Images/2024-11-18-KoreanMultilingualChallenges/page_1_high_res.png)

![page_2_high_res.png](/assets/Images/2024-11-18-KoreanMultilingualChallenges/page_2_high_res.png)
 
<br>

### 2차 Progress

![page_1_high_res_v3.png](/assets/Images/2024-11-18-KoreanMultilingualChallenges/page_1_high_res_v3.png)

![page_2_high_res_v3.png](/assets/Images/2024-11-18-KoreanMultilingualChallenges/page_2_high_res_v3.png)

![page_3_high_res_v3.png](/assets/Images/2024-11-18-KoreanMultilingualChallenges/page_3_high_res_v3.png)

![page_4_high_res_v3.png](/assets/Images/2024-11-18-KoreanMultilingualChallenges/page_4_high_res_v3.png)