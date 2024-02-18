# sentence_transformers 꼭 설치

import csv
from sentence_transformers import losses
from torch.utils.data import DataLoader
from sentence_transformers import util
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample
import torch

f = open('dict.tsv','r', encoding='utf-8') 
rdr = csv.reader(f, delimiter='\t')
mapper = []
for row in rdr:
    mapper.append(row[1])
f.close()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_count = 800 # hyper_params
test_count = 200
batch_size = 16
epochs = 32

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device=device) # pre-trained 모델 불러오기
model.eval() # gpu 너무 많이 잡아 먹어서 달아놓은 코드

train_examples = []
f = open('corrected_train_data_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
rdr = csv.reader(f)
for row in rdr:
    train_examples.append(InputExample(texts=[row[0], row[1], row[2]]))
f.close()

test_examples = []
f = open('corrected_test_data_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
rdr = csv.reader(f)
for row in rdr:
    test_examples.append(row)
f.close()

train_dataset = SentencesDataset(train_examples, model) # train_dataset 생성
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size) # DataLoader 초기화
train_loss = losses.TripletLoss(model=model) # loss 정의. TripletLoss로

# model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=100) # fit
# torch.save(model.state_dict(), "corrected_model_v1_epochs_" + str(epochs) + "_batchsize_" + str(batch_size) +  "_dataset_"  + str(train_count+test_count) +  ".pt") # 모델 저장

model_state_dict = torch.load("corrected_model_v1_epochs_" + str(epochs) + "_batchsize_" + str(batch_size) +  "_dataset_"  + str(train_count+test_count) +  ".pt", map_location=device)
model.load_state_dict(model_state_dict)

f = open('corrected_raw.csv', 'r', encoding='utf-8')
rdr = csv.reader(f, delimiter='\t')
test_data = [[] for _ in list(range(len(mapper)))]
for row in rdr:
    test_data[int(row[1])].append(str(row[0]))
f.close()

import numpy as np
'''
TODO: 문제상황 해결 -> 현재 모델 성능 너무 안 나옴
현재 생각: 평균 벡터를 구해서 이걸 가지고 cos sim을 돌리는 게 아니라, 각각의 벡터를 cos_sim을 돌리고, 그걸 평균 내는 게 더 좋지 않을까?
cos이 linearly sth? 한 게 아니니까...

아니면, 각각의 벡터를 cos_sim 돌리고, 상위 몇 개의 벡터를 가진 순으로 추천하는 것도 괜찮을 것 같음.
1안, 2안 둘 다 해보는 게 좋을 것 같음. 이 코드는 2안
'''
vectors = [[] for _ in list(range(len(mapper)))]
for i in list(range(len(test_data))):
    for j in list(range(len(test_data[i]))):
        vector = model.encode(test_data[i][j])
        vectors[i].append(vector)
    print(i, "done")

acc = 0
hitsAt3 = 0
hitsAt5 = 0
hitsAt10 = 0
rankingBasedMetric = 0

for i in list(range(test_count)):
    text = test_examples[i][0] #입력 리뷰
    answer = int(test_examples[i][1]) # answer label

    targetVector = model.encode([text]) # targetVector는 데스트 할 text string의 sentence vector
    results = []
    answerList = []
    cosCalculatedVectors = [[] for _ in list(range(len(mapper)))]
    meanCosCalVec = []

    for j in list(range(len(vectors))):
        for k in list(range(len(vectors[j]))):
            similarities = util.cos_sim(vectors[j][k], targetVector) # compute similarity between sentence vectors
            cosCalculatedVectors[j].append(float(similarities))

        cosCalculatedVectors[j].sort(reverse=True)
        cosCalculatedVectors[j] = cosCalculatedVectors[j][:3] # repr 몇 개 뽑을 지 결정.
        meanCosCalVec.append((j, mapper[j], np.mean(cosCalculatedVectors[j])))
        # if(i == 0):
        #     print(text, (j, mapper[j], np.mean(cosCalculatedVectors[j])))
    meanCosCalVec.sort(key = lambda x : -x[2])

    for _ in meanCosCalVec:
        answerList.append(_[0])

    # if(answer == answerList[0]):
    #     print(answer, answerList)
    
    # print(answer, answerList[:20])
    if(answerList[0] == answer): # acc 계산
        acc = acc+1
    if(answer in answerList[:3]): # hits@3 계산
        hitsAt3 = hitsAt3+1
    if(answer in answerList[:5]): # hits@5 계산
        hitsAt5 = hitsAt5+1
    if(answer in answerList[:10]): # hits@10 계산
        hitsAt10 = hitsAt10+1
    # rankingBasedMetric 계산. 정답을 n개 중에서 1등으로 예측하면 1점, 2등으로 예측하면 (1-1/n)점, 3등으로 예측하면 (1-2/n)점, ... 이런 식으로 계산
    rankingBasedMetric = rankingBasedMetric + (1 - answerList.index(answer) / len(answerList)) 

# 전체 metric은 최댓값이 1입니다.
acc = acc / test_count
hitsAt3 = hitsAt3 / test_count
hitsAt5 = hitsAt5 / test_count
hitsAt10 = hitsAt10 / test_count
rankingBasedMetric = rankingBasedMetric / test_count

# 출력.
print("acc", acc)
print("hitsAt3", hitsAt3)
print("hitsAt5", hitsAt5)
print("hitsAt10", hitsAt10)
print("rankingBasedMetric", rankingBasedMetric)