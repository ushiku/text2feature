import numpy as np
import re
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def text2dip(text):
    '''
    text は EDAの出力1columずつlist化したもの.  depbigramをとって、text_word形式に
    '''
    heads = []
    tails = []
    words = []
    poss = []
    for line in text:
        line = line.strip()
        units = line.split(' ')
        heads.append(int(units[0]))
        tails.append(int(units[1]))
        words.append(units[2])
        poss.append(units[3])
    depbigram_unigram = ''
    depbigram = ''  # depを利用したbigram
    depbigram = '^' + words[0]
    for tail, word  in zip(tails, words):
        if tail == -1:
            depbigram = depbigram + ' ' + word + '$'
            depbigram_unigram = depbigram
            for word in words:
                depbigram_unigram = depbigram_unigram + ' ' + word
        else:
            depbigram = depbigram + ' ' + word + words[tail -1]
    return depbigram, depbigram_unigram


def ReturnBIGGEST(n, array):
    '''
    nと,vector_matrixを渡すことで、その中で最大の類似度の組み合わせを表示する  
    '''
    sim_vector = []
    vector1 = array[n].todense()
    vector1 = np.squeeze(np.asarray(vector1))
    for vector2 in array.todense():
        vector2 = np.squeeze(np.asarray(vector2))
        sim_vector.append(1-cosine(vector1, vector2))  # ここcosineが1-cosine距離で定式している?
    
    print("original:",text_word[n])
    for a in range(0, 5):  # 上位n個を出す(n未満の配列には対応しないので注意)
#        while np.nanmax(sim_vector) == 1:  # 完全一致は出さない
 #           text_word[np.nanargmax(sim_vector)] = -1
        print("simirality:", np.nanmax(sim_vector), "answer ", a, ":", text_word[np.nanargmax(sim_vector)])
        sim_vector[np.nanargmax(sim_vector)] = -1
    print()
    return 0    

def caluculate(vectorizer, text):
    X = vectorizer.fit_transform(text)
    #print(X.toarray())
    #print('feature一覧:', vectorizer.get_feature_names())
#    ReturnBIGGEST(7, X)
    ReturnBIGGEST(4, X)
    ReturnBIGGEST(504, X)
    ReturnBIGGEST(1004, X)
    ReturnBIGGEST(1504, X)
#    ReturnBIGGEST(9, X)
    return 0

    
text_full = []  # かかり受け情報を含んだそのままのEDAの結果をlist化
text_word = []  # scikit-learn ように ['This is a pen', 'That is a pen']というlist
fulls = []
words = ''
file = open('../corpus/full.eda', 'r')
for line in file:
    line = line.strip()
    if re.match('ID', line):
        continue
    if line == '':
        text_full.append(fulls)
        fulls = []
        text_word.append(words.strip())
        words = ''
        continue
    units = line.split(' ')
    fulls.append(line)
    words = words + ' ' +  units[2]
text_full.append(fulls)
text_word.append(words.strip())
text_word_dep = [] 
text_word_dep_uni = []
for text in text_full:
    depbigram, depbigram_unigram = text2dip(text)
    text_word_dep.append(depbigram.strip())
    text_word_dep_uni.append(depbigram_unigram.strip())

print('tfidf')
caluculate(TfidfVectorizer(), text_word)  # tfidfモデル
print('dep')
caluculate(CountVectorizer(), text_word_dep)  # depモデル
print('depbigram-tfidf')
caluculate(TfidfVectorizer(), text_word_dep)  # depbigram-tfidfモデル
print('depbigram-unigram')
caluculate(CountVectorizer(), text_word_dep_uni)  # depbigram-unigramモデル
