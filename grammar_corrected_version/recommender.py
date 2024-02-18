import csv
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_count = 800 # hyper_params
test_count = 200
batch_size = 16
epochs = 32

f = open('dict.tsv','r', encoding='utf-8') 
rdr = csv.reader(f, delimiter='\t')
mapper = []
for row in rdr:
    mapper.append(row[1])
f.close()

test_examples = []
f = open('corrected_test_data_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
rdr = csv.reader(f)
for row in rdr:
    test_examples.append(row)
f.close()

f = open('corrected_raw.csv', 'r', encoding='utf-8')
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

model_state_dict = torch.load("corrected_model_v1_epochs_" + str(epochs) + "_batchsize_" + str(batch_size) +  "_dataset_"  + str(train_count+test_count) +  ".pt", map_location=device)

try:
    model.load_state_dict(model_state_dict)
except RuntimeError:
    model.load_state_dict(model_state_dict, strict=False)

import numpy as np

vectors = [[] for _ in list(range(len(mapper)))]
try:
    f = open('corrected_vectors_' + str(train_count+test_count) + '.csv','r', encoding='utf-8', newline='')
    rdr = csv.reader(f)
    for i, row in enumerate(rdr):
        floatCastedRow = []
        for _row in row:
            floatCastedRow.append(np.double(_row))
            # if(i == 0):
            #     print(_row, float(_row), 0.3396976888179779, "int(temp_data[i][1])", int(temp_data[i][1]))
        vectors[int(temp_data[i][1])].append((np.array(floatCastedRow), int(temp_data[i][1])))
        # if(i == 0):
        #     print(vectors[i][-1])
    f.close()

except FileNotFoundError:
    f = open('corrected_vectors_' + str(train_count+test_count) + '.csv','w', encoding='utf-8', newline='')
    writer = csv.writer(f)

    for i in list(range(len(mapper))):
        for j in list(range(len(test_data[i]))):
            vector = model.encode(test_data[i][j])
            listedVector = vector.tolist()
            writer.writerow(listedVector)
        print(i, "encoding done")

    f.close()

    f = open('corrected_vectors_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
    rdr = csv.reader(f)
    for i, row in enumerate(rdr):
        floatCastedRow = []
        for _row in row:
            floatCastedRow.append(np.double(_row))
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

# ---------------------------------------------------------------------------------------
# func_input 메소드로 원하는 도서에 대한 설명을 입력받고 리턴합니다.
# 입력받은 문자열은 string_input을 통해 리턴됩니다.
# ---------------------------------------------------------------------------------------
def func_input():
    string_input = input("입력: ")

    return string_input

targetText = func_input() #입력 리뷰
targetVector = model.encode([targetText]) # targetVector는 데스트 할 text string의 sentence vector

acc = 0
hitsAt3 = 0
hitsAt5 = 0
hitsAt10 = 0
rankingBasedMetric = 0

targetVector = model.encode([targetText]).astype(np.double) # targetVector는 데스트 할 text string의 sentence vector
results = []
answerList = []
cosCalculatedVectors = [[] for _ in list(range(len(mapper)))]
meanCosCalVec = []

for j in list(range(len(vectors))):
    for k in list(range(len(vectors[j]))):
        # print(type(vectors[j][k][0]), type(targetVector))
        # print(type(vectors[j][k][0][0]), type(targetVector[0][0]))
        # print(vectors[j][k][0].shape, targetVector.shape)
        similarities = util.cos_sim(vectors[j][k][0], targetVector) # compute similarity between sentence vectors
        cosCalculatedVectors[j].append(float(similarities))

    cosCalculatedVectors[j].sort(reverse=True)
    cosCalculatedVectors[j] = cosCalculatedVectors[j][:3]
    results.append((j, mapper[j], np.mean(cosCalculatedVectors[j])))
    # if(i == 0):
    #     print(targetText, (j, mapper[j], np.mean(cosCalculatedVectors[j])))
results.sort(key = lambda x : -x[2])

for _ in results:
    answerList.append(_[0])

# if(answer == answerList[0]):
#     print(answer, answerList)

string_output = [[] for _ in list(range(len(results)))]
print(targetText, "검색 결과: ")
print("번호" + " " + "도서명" + " " + "저자" + " " + "score")
print("="*45)
for result in results[:]:
    printedString = [str(result[0]), result[1].split('_')[0], result[1].split('_')[1], str(round(result[2], 3))]
    string_output.append(printedString)
    print(printedString[0] + " " + printedString[1] + " " + printedString[2] + " " + printedString[3])

# ---------------------------------------------------------------------------------------
# func_output 메소드로 결과 리턴합니다.
# 리턴 값인 string_output은 [도서명, 저자, 유사도] 리스트로 구성되어 있는 리스트입니다.
# string_output은 유사도 기준으로 내림차순 정렬되어 있습니다.
# ---------------------------------------------------------------------------------------
def func_output():
    string_output = []
    for result in results[:]:
        string_output.append([result[1].split('_')[0], result[1].split('_')[1], str(round(result[2], 3))])

    return string_output
# import matplotlib.pyplot as plt
# plt.hist(simList[results[0][0]], bins=20)
# plt.show()