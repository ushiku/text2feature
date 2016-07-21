from collections import deque
import re

"""
テストデータ
str ="私はリンゴを食べる。"
heads = ["001","002","003","004","005","006","007"]
tails = ["002","005","004","005","006","007","0"]
words = ["私","は","リンゴ","を","食べ","る","。"]
poss = ["代名詞","助詞","名詞","助詞","動詞","語尾","補助記号"]
"""

"""
係り受けを与えていないarr

['私', 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 'は', 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 'リ', 'ン', 'ゴ', 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 'を', 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, '食', 'べ', 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0, 'る', 0]
[0, 0, 0, 0, 0, 0, 0, 0, 0, '。']

"""

"""
skip表(cnt配列)
[0, 1, 2, 5, 6, 8, 9]
"""

# 入力部
# strも得る（どうやって？）

heads = []
tails = []
words = []
poss = []

file = open('edatest.txt', 'r')
for line in file:
    line = line.strip()
    if re.match('ID', line):
        continue
    else:
        units = line.split()
        heads.append(int(units[0]))
        tails.append(int(units[1]))
        words.append(units[2])
        poss.append(units[3])
        continue

str = "".join(words)


"""出力表の作成"""
arr = [[0 for i in range(len(str))] for j in range(len(words))]

"""空白カウントcnt"""
cnt = []
spacenum = 0
for i in range(len(words)):
    if i != 0:
        spacenum = cnt[i-1] + len(words[i-1])
    cnt.append(spacenum)

"""文を出力表に入れる:キューの利用"""

ch = list(str)
queue = deque([])
for i in range(len(ch)):  #　rangeをオーバーしているらしい # でも入力はできてる
    queue.append(ch[i])

for i in range(len(str)):
    for j in range(cnt[i], cnt[i] + len(words[i])):
        arr[i][j] = queue.popleft()

"""head,tailsの下処理"""
headsnum = []
tailsnum = []
for head in heads:              # よくわからないけど81行目から84行目を一気に読ませるとおかしくなる　#でも分けてやると通る
    headsnum.append(int(head))
for tail in tails:
    tailsnum.append(int(tail))

"""係り受け入力"""

for i in range(len(words)):
    if tailsnum[i] != 0:
        arr[i][cnt[tailsnum[i]-1]] = 'D'

for i in range(len(words)):
    for j in range(cnt[i]+len(words[i]), cnt[tailsnum[i]-1]):
        arr[i][j] = '-'

"""
タテはいまいち、わからない
for i in range(len(words)):
    for j in range(len(str)):
        if arr[i][j] == 'D':
            for k in range(i, arr[k][j] != 0):
                arr[i][k] = '|'
"""

"""print"""
for line in arr:
    print(line)



