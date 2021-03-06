from collections import deque
import re

'''ＥＤＡファイルを受け取り、すべて表として出力します。'''

def textmap_for_eda_file(eda) :
  for i in range (len(eda)):
    for j in range (len(eda[i])):
      tails = []
      words = []
      for k in range(len(eda[i][j])):
        tails = tails + [eda[i][j][k][1]]
        words = words + [eda[i][j][k][2]]
      textmap(words,tails) 



def textmap(words,tails):

    str = "".join(words)
    arr = [[0 for i in range(len(str))] for j in range(len(words))]
    cnt = []
    spacenum = 0
    for i in range(len(words)):
        if i != 0:
            spacenum = cnt[i - 1] + len(words[i - 1])
        cnt.append(spacenum)
    ch = list(str)
    queue = deque([])
    for i in range(len(ch)):
        queue.append(ch[i])
    for i in range(len(words)):
        for j in range(cnt[i], cnt[i] + len(words[i])):
            arr[i][j] = queue.popleft()
    for i in range(len(words)):
        if tails[i] != 0:
            arr[i][cnt[tails[i] - 1]] = 'D'
    for i in range(len(words)):
        for j in range(cnt[i] + len(words[i]), cnt[tails[i] - 1]):
            arr[i][j] = '-'
    for i in range(len(words)):
        for j in range(len(str)):
            if arr[i][j] == 'D':
                while True:
                    if arr[i + 1][j] != 0:
                        break
                    else:
                        arr[i + 1][j] = '|'
    for line in arr:
        print(line)



'''
eda形式のサンプル

a = [ [1,2,'私','代名詞',0] , [2,5,'は','助詞',0],[3,4,'リンゴ','名詞',0],[4,5,'を','助詞',0],[5,6,'食べ','動詞',0],[6,7,'る','語尾',0],[7,0,'。','補助記号',0] ]
b = [ [1,2,'私','代名詞',0] , [2,5,'は','助詞',0],[3,4,'ミカン','名詞',0],[4,5,'を','助詞',0],[5,6,'食べ','動詞',0],[6,7,'る','語尾',0],[7,0,'。','補助記号',0] ]
c = [ [1,2,'私','代名詞',0] , [2,5,'は','助詞',0],[3,4,'アンズ','名詞',0],[4,5,'を','助詞',0],[5,6,'食べ','動詞',0],[6,7,'る','語尾',0],[7,0,'。','補助記号',0] ]
d = [ [1,2,'私','代名詞',0] , [2,5,'は','助詞',0],[3,4,'ブドウ','名詞',0],[4,5,'を','助詞',0],[5,6,'食べ','動詞',0],[6,7,'る','語尾',0],[7,0,'。','補助記号',0] ]

A = [a,b,c,d] # 文書Ａ

e = [ [1,2,'あなた','代名詞',0] , [2,5,'は','助詞',0],[3,4,'リンゴ','名詞',0],[4,5,'を','助詞',0],[5,6,'食べ','動詞',0],[6,7,'る','語尾',0],[7,0,'。','補助記号',0] ]
f = [ [1,2,'あなた','代名詞',0] , [2,5,'は','助詞',0],[3,4,'ミカン','名詞',0],[4,5,'を','助詞',0],[5,6,'食べ','動詞',0],[6,7,'る','語尾',0],[7,0,'。','補助記号',0] ]
g = [ [1,2,'あなた','代名詞',0] , [2,5,'は','助詞',0],[3,4,'アンズ','名詞',0],[4,5,'を','助詞',0],[5,6,'食べ','動詞',0],[6,7,'る','語尾',0],[7,0,'。','補助記号',0] ]
h = [ [1,2,'あなた','代名詞',0] , [2,5,'は','助詞',0],[3,4,'ブドウ','名詞',0],[4,5,'を','助詞',0],[5,6,'食べ','動詞',0],[6,7,'る','語尾',0],[7,0,'。','補助記号',0] ]

B = [e,f,g,h] # 文書Ｂ

i = [ [1,2,'彼女','代名詞',0] , [2,5,'は','助詞',0],[3,4,'リンゴ','名詞',0],[4,5,'を','助詞',0],[5,6,'食べ','動詞',0],[6,7,'る','語尾',0],[7,0,'。','補助記号',0] ]
j = [ [1,2,'彼女','代名詞',0] , [2,5,'は','助詞',0],[3,4,'ミカン','名詞',0],[4,5,'を','助詞',0],[5,6,'食べ','動詞',0],[6,7,'る','語尾',0],[7,0,'。','補助記号',0] ]
k = [ [1,2,'彼女','代名詞',0] , [2,5,'は','助詞',0],[3,4,'アンズ','名詞',0],[4,5,'を','助詞',0],[5,6,'食べ','動詞',0],[6,7,'る','語尾',0],[7,0,'。','補助記号',0] ]
l = [ [1,2,'彼女','代名詞',0] , [2,5,'は','助詞',0],[3,4,'ブドウ','名詞',0],[4,5,'を','助詞',0],[5,6,'食べ','動詞',0],[6,7,'る','語尾',0],[7,0,'。','補助記号',0] ]

C = [i,j,k,l] #文書C

# 文書集合eda
eda = [A,B,C]

'''
'''
    textmap(words,tails)について
    words[],tails[]を受け取り、依存構造を図示した表を出力します。
    処理にqueueを使用しています。
    ex.
        words=['私', 'は', 'リンゴ', 'を', '食べ', 'る', '。']
        tails=[2, 5, 4, 5, 6, 7, 0]
    を受け取って
        ['私', 'D', 0, 0, 0, 0, 0, 0, 0, 0]
        [0, 'は', '-', '-', '-', '-', 'D', 0, 0, 0]
        [0, 0, 'リ', 'ン', 'ゴ', 'D', '|', 0, 0, 0]
        [0, 0, 0, 0, 0, 'を', 'D', 0, 0, 0]
        [0, 0, 0, 0, 0, 0, '食', 'べ', 'D', 0]
        [0, 0, 0, 0, 0, 0, 0, 0, 'る', 'D']
        [0, 0, 0, 0, 0, 0, 0, 0, 0, '。']
    を出力する。
'''

