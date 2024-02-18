import csv
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_count = 1 # hyper_params
test_count = 2
batch_size = 16
epochs = 1

f = open('dict.tsv','r', encoding='utf-8') 
rdr = csv.reader(f, delimiter='\t')
mapper = []
for row in rdr:
    mapper.append(row[1])
f.close()

f = open('data.tsv','r', encoding='utf-8')
rdr = csv.reader(f, delimiter='\t')
test_data = [[] for _ in list(range(len(mapper)))]
for row in rdr:
    test_data[int(row[1])].append(str(row[0]))
f.close()

test_examples = []
f = open('test.csv','r', encoding='utf-8')
rdr = csv.reader(f)
for row in rdr:
    test_examples.append(row)
f.close()

f = open('data.tsv','r', encoding='utf-8')
rdr = csv.reader(f, delimiter='\t')
test_data = [[] for _ in list(range(len(mapper)))]
temp_data = []
for row in rdr:
    test_data[int(row[1])].append(str(row[0]))
    temp_data.append(row)
f.close()

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device=device) # pre-trained 모델 불러오기
model.eval() # gpu 너무 많이 잡아 먹어서 달아놓은 코드

import os
from collections import OrderedDict

model_state_dict = torch.load("test_model" +  ".pt", map_location=device)

try:
    model.load_state_dict(model_state_dict)
except RuntimeError:
    model.load_state_dict(model_state_dict, strict=False)

import numpy as np

vectors = [[] for _ in list(range(len(mapper)))]
try:
    f = open('test' + '_vectors_' + str(train_count+test_count) + '.csv','r', encoding='utf-8', newline='')
    rdr = csv.reader(f)
    for i, row in enumerate(rdr):
        floatCastedRow = []
        for _row in row:
            floatCastedRow.append(float(_row))
            # if(i == 0):
            #     print(_row, float(_row), 0.3396976888179779, "int(temp_data[i][1])", int(temp_data[i][1]))
        vectors[int(temp_data[i][1])].append((np.array(floatCastedRow), int(temp_data[i][1])))
    f.close()

except FileNotFoundError:
    f = open('test' + '_vectors_' + str(train_count+test_count) + '.csv','w', encoding='utf-8', newline='')
    writer = csv.writer(f)

    for i in list(range(len(mapper))): # 각 강의평마다 주어진 강의평 벡터의 평균을 각 강의와 매핑합니다.
        for j in list(range(len(test_data[i]))):
            vector = model.encode(test_data[i][j])
            listedVector = vector.tolist()
            writer.writerow(listedVector)
        print(i, "encoding done")

    f.close()

    f = open('test' + '_vectors_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
    rdr = csv.reader(f)
    for i, row in enumerate(rdr):
        floatCastedRow = []
        for _row in row:
            floatCastedRow.append(float(_row))
            # if(i == 0):
            #     print(_row, float(_row))
        vectors[int(temp_data[i][1])].append((np.array(floatCastedRow), int(temp_data[i][1])))
    f.close()

dimList = []
vectorData = [[] for _ in list(range(len(mapper)))]
for i in list(range(len(vectors))):
    for j in list(range(len(vectors[i]))):
        vectorData[i].append(vectors[i][j][0])
        dimList.append(np.array(vectors[i][j][0]).shape)

targetText = "재미있는 소설." #입력 리뷰
targetVector = model.encode([targetText]) # targetVector는 데스트 할 text string의 sentence vector

vectors = [] 
for i in list(range(len(mapper))): # 각 리뷰마다 주어진 리뷰 벡터의 평균을 각 도서와 매핑합니다.
    vector = model.encode(test_data[i])
    vector = np.mean(vector, axis=0)
    vectors.append(vector)
    print(i, "mean cal. done")

acc = 0
hitsAt3 = 0
hitsAt5 = 0
hitsAt10 = 0
rankingBasedMetric = 0

for i in list(range(test_count)):
    targetVector = model.encode([targetText]) # targetVector는 데스트 할 text string의 sentence vector
    results = []
    answerList = []
    for j in list(range(len(mapper))):
        similarities = util.cos_sim(vectors[j], targetVector) # compute similarity between sentence vectors
        results.append((j, mapper[j], float(similarities)))
    results.sort(key = lambda x : -x[2])

print(targetText, "검색 결과: ")
print("번호" + " " + "도서명" + " " + "저자" + " " + "score")
print("="*45)
for result in results[:10]:
    printedString = [str(result[0]), result[1].split('_')[0], result[1].split('_')[1], str(round(result[2], 3))]

    print(printedString[0] + " " + printedString[1] + " " + printedString[2] + " " + printedString[3])

# import matplotlib.pyplot as plt
# plt.hist(simList[results[0][0]], bins=20)
# plt.show()