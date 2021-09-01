import csv
from collections import Counter

f1 = open('out2.csv', 'r', encoding='utf-8')
f2 = open('out3.csv', 'r', encoding='utf-8')
f3 = open('out4.csv', 'r', encoding='utf-8')
f4 = open('out5.csv', 'r', encoding='utf-8')
f5 = open('out6.csv', 'r', encoding='utf-8')

rdr1 = csv.reader(f1)
rdr2 = csv.reader(f2)
rdr3 = csv.reader(f3)
rdr4 = csv.reader(f4)
rdr5 = csv.reader(f5)

rdrs = [rdr1, rdr2, rdr3, rdr4, rdr5]

result = open('result.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(result)
x = 0
y = 0
for r1, r2, r3, r4, r5 in zip(rdr1, rdr2, rdr3, rdr4, rdr5):
    tmp = [r1[1], r2[1], r3[1], r4[1], r5[1]]
    cnt = Counter(tmp).most_common()
    # print(cnt)
    # print(tmp)
    if len(cnt) > 1 and cnt[0][1] == cnt[1][1]:
        wr.writerow([r1[0], cnt[0][0]])
        x += 1
    else:
        wr.writerow([r1[0], r1[1]])
        y += 1
print(x, y)




