import csv
 
# f = open('NL_BO_TAG_CLOUD4_202112.csv', 'r', encoding='cp949')
# rdr = csv.reader(f, delimiter='\t')

# books = []
# pre_tags = []

# for line in rdr:
#     books.append(line[0])
#     pre_tags.append([line[0], line[1]])
# f.close()   

# print(books[0])
# print(pre_tags[0])

# books = list(set(books))
# tags = [[] for _ in list(range(len(books)))]

# for pre_tag in pre_tags:
#     tags[books.index(pre_tag[0])].append(pre_tag[1])

# f = open('cold_start_tags.csv', 'w', encoding='utf-8', newline='')
# wr = csv.writer(f)
# for i in list(range(len(books))):
#     temp = [i, books[i]]
    
#     for j in list(range(len(tags[i]))):
#         temp.append(tags[i][j])
    
#     wr.writerow(temp)
# f.close()

# -----------------------------------------------------------------------------

# f = open('cold_start_tags.csv', 'r', encoding='utf-8')
# rdr = csv.reader(f)
# books = []
# for line in rdr:
#     books.append(int(line[1].replace(' ', '')))
# f.close()

# f = open('cold_start_tags.csv', 'r', encoding='utf-8')
# rdr = csv.reader(f)
# tags = []
# for i, line in enumerate(rdr):
#     tags.append([i, books[i], line[2:]])
# f.close()   

# pref_pre_data = []
# male_pref_pre_data = []
# female_pref_pre_data = []

# f = open('NL_BO_BEST_LOAN_BOOK_HISTORY_202110.csv', 'r', encoding='cp949')
# rdr = csv.reader(f, delimiter='\t')
# for i, line in enumerate(rdr):
#     if(i == 0):
#         continue
#     pref_pre_data.append(line)

#     if(line[19] == "남성"):
#         male_pref_pre_data.append(line[3])
#     elif(line[19] == "여성"):
#         female_pref_pre_data.append(line[3])
# f.close()

# print(pref_pre_data[0])
# print(male_pref_pre_data[0])
# print(female_pref_pre_data[0])

# male_pref_data = []
# for data in male_pref_pre_data:
#     try:
#         male_pref_data.append(tags[books.index(int(data))][2])
#     except ValueError:
#         continue

# female_pref_data = []
# for data in female_pref_pre_data:
#     try:
#         female_pref_data.append(tags[books.index(int(data))][2])
#     except ValueError:
#         continue

# # print(male_pref_data[0])
# # print(female_pref_data[0])

# male_pref_data = sum(male_pref_data, [])
# female_pref_data = sum(female_pref_data, [])

# from collections import Counter
# male_pref_tags = Counter(male_pref_data)
# female_pref_tags = Counter(female_pref_data)

# # print(male_pref_tags)
# # print(female_pref_tags)

# f = open('male_start_tags.csv', 'w', encoding='utf-8', newline='')
# wr = csv.writer(f)
# for key, item in male_pref_tags.items(): 
#     wr.writerow([key, item])
# f.close()

# f = open('female_start_tags.csv', 'w', encoding='utf-8', newline='')
# wr = csv.writer(f)
# for key, item in female_pref_tags.items(): 
#     wr.writerow([key, item])
# f.close()

# -----------------------------------------------------------------------------

# generation_pref_pre_data = []

# f = open('NL_AGE_ACCTO_BOOK_KWRD_LIST_202110.csv', 'r', encoding='utf-8')
# rdr = csv.reader(f)
# for i, line in enumerate(rdr):
#     if(i == 0):
#         continue
#     generation_pref_pre_data.append(line)
# f.close()

# print(generation_pref_pre_data[0])

# generation_pref_data = [[] for _ in range(6)]
# for data in generation_pref_pre_data:
#     if(data[2] == "20대"):
#         generation_pref_data[1].append(data[4])
#     elif(data[2] == "30대"):
#         generation_pref_data[2].append(data[4])
#     elif(data[2] == "40대"):
#         generation_pref_data[3].append(data[4])
#     elif(data[2] == "50대"):
#         generation_pref_data[4].append(data[4])
#     elif(data[2] == "60대 이상"):
#         generation_pref_data[5].append(data[4])
#     else:
#         generation_pref_data[0].append(data[4])

# print(generation_pref_data[0])

# f = open('gen_start_tags.csv', 'w', encoding='utf-8', newline='')
# wr = csv.writer(f)
# for data in generation_pref_data: 
#     wr.writerow(data)
# f.close()

# -----------------------------------------------------------------------------

# books = []
# book_tags = [[] for _ in range(48)]
# male_pref_tags = []
# female_pref_tags = []
# gen_pref_tags = [[] for _ in range(6)]

# f = open('dict.tsv','r', encoding='utf-8') 
# rdr = csv.reader(f, delimiter='\t')
# for row in rdr:
#     books.append(row[1])
# f.close()

# f = open('bestseller_tags.txt', 'r', encoding='utf-8')
# lines = f.readlines()
# for i, line in enumerate(lines):
#     book_tags[i] = line.split(' ')
#     book_tags[i][-1].replace('\n', '')
# f.close()

# f = open('male_start_tags.csv', 'r', encoding='utf-8')
# rdr = csv.reader(f)
# for line in rdr:
#     male_pref_tags.append(line[0])
# f.close()

# f = open('female_start_tags.csv', 'r', encoding='utf-8')
# rdr = csv.reader(f)
# for line in rdr:
#     female_pref_tags.append(line[0])
# f.close()

# f = open('gen_start_tags.csv', 'r', encoding='utf-8')
# rdr = csv.reader(f)
# for i, line in enumerate(rdr):
#     gen_pref_tags[i] = line
# f.close()

# print(male_pref_tags[0])
# print(female_pref_tags[0])
# print(gen_pref_tags[0][0])
# print(book_tags[0][0])

# # 10대 이하 남성: 0, 10대 이하 여성: 1, 20대 남성: 2, 20대 여성: 3, ...
# scores = [[[i, 0] for i in list(range(len(books)))] for _ in range(12)]

# print(len(scores))
# print(scores[0])

# for i in list(range(len(books))):
#     for j in list(range(len(book_tags[i]))):
#         if(book_tags[i][j] in male_pref_tags):
#             if(book_tags[i][j] in gen_pref_tags[0]):
#                 scores[0][i][1] = scores[0][i][1]+1
#             if(book_tags[i][j] in gen_pref_tags[1]):
#                 scores[1][i][1] = scores[1][i][1]+1
#             if(book_tags[i][j] in gen_pref_tags[2]):
#                 scores[2][i][1] = scores[2][i][1]+1
#             if(book_tags[i][j] in gen_pref_tags[3]):
#                 scores[3][i][1] = scores[3][i][1]+1
#             if(book_tags[i][j] in gen_pref_tags[4]):
#                 scores[4][i][1] = scores[4][i][1]+1
#             if(book_tags[i][j] in gen_pref_tags[5]):
#                 scores[5][i][1] = scores[5][i][1]+1

#         if(book_tags[i][j] in female_pref_tags):
#             if(book_tags[i][j] in gen_pref_tags[0]):
#                 scores[6][i][1] = scores[6][i][1]+1
#             if(book_tags[i][j] in gen_pref_tags[1]):
#                 scores[7][i][1] = scores[7][i][1]+1
#             if(book_tags[i][j] in gen_pref_tags[2]):
#                 scores[8][i][1] = scores[8][i][1]+1
#             if(book_tags[i][j] in gen_pref_tags[3]):
#                 scores[9][i][1] = scores[9][i][1]+1
#             if(book_tags[i][j] in gen_pref_tags[4]):
#                 scores[10][i][1] = scores[10][i][1]+1
#             if(book_tags[i][j] in gen_pref_tags[5]):
#                 scores[11][i][1] = scores[11][i][1]+1

# for i in range(12):
#     scores[i].sort(key=lambda x: -x[1])

#     print("---------------------------------")
#     print(i)

#     for j in list(range(len(books))):
#         print(j, books[scores[i][j][0]], scores[i][j][1])

# -----------------------------------------------------------------------------

books = []
f = open('dict.tsv','r', encoding='utf-8') 
rdr = csv.reader(f, delimiter='\t')
for row in rdr:
    books.append(row[1])
f.close()

star_data = open("star_rate.txt", 'r', encoding='utf-8')
stars = []
star_rate = []
lines = star_data.readlines()

for i, line in enumerate(lines):
    temp = line.split(' ')
    temp[-1].replace('\n', '')

    float_temp = []
    for j in list(range(len(temp))):
        float_temp.append(float(temp[j]))
    stars.append(float_temp[0])
    star_rate.append(float_temp[1:])

star_data.close()

# 10대 이하 여성: 0, 20대 여성: 1, ... ,60대 이상 여성: 5, 10대 이하 남성: 6, ... ,60대 이상 남성: 11
scores = [[[i, 0] for i in list(range(len(books)))] for _ in range(12)]

for i in list(range(len(books))):
    for j in range(12):
        scores[j][i][1] = stars[i]*star_rate[i][j]

print(scores[0])

recommend_list = [['' for _ in range(len(books))] for i in range(12)]
recommend_scores = [[0 for _ in range(len(books))] for i in range(12)]
for i in range(12):
    scores[i].sort(key=lambda x: -x[1])

    print("---------------------------------")
    print(i)

    for j in list(range(len(books))):
        print(j, books[scores[i][j][0]], scores[i][j][1])
        recommend_list[i][j] = books[scores[i][j][0]]
        recommend_scores[i][j] = scores[i][j][1]

f = open('cold_start_recommend_list.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
for i in list(range(12)):
    temp = []
    for j in list(range(len(books))):
        temp.append(recommend_list[i][j])
    wr.writerow(temp)
f.close()

f = open('cold_start_recommend_scores.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
for i in list(range(12)):
    temp = []
    for j in list(range(len(books))):
        temp.append(recommend_scores[i][j])
    wr.writerow(temp)
f.close()
