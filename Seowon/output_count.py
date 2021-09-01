import csv
f = open('output___2.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
cnt = [0]*18

for i, line in enumerate(rdr):
  if i>0:
    cnt[int(line[1])-1] += 1

print(cnt)

f.close()

print("mask/incorrect/unmask : ", sum(cnt[:6]), sum(cnt[6:12]), sum(cnt[12:]))
male = sum(cnt[:3]) + sum(cnt[6:9]) + sum(cnt[12:15])
print("male/female : ", male, sum(cnt)-male)
y, m, o = 0, 0, 0
for i, age in enumerate(cnt):
  if i%3 == 0:
    y += age
  elif i%3 == 1:
    m += age
  else:
    o += age
print("~30/31~59/60 : ", y, m, o)