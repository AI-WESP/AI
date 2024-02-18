import csv

genre = []
f = open('bestseller_genre_modified.csv','r', encoding='utf-8') 
rdr = csv.reader(f)
for row in rdr:
    for _row in row:
        if(_row != ''):
            genre.append(_row)
f.close()

genre = list(set(genre))

print(genre)

f = open("bestseller_genre_modified_v2.txt", 'w', encoding='utf-8')
for _genre in genre:
    f.write(_genre+'\n')
f.close()