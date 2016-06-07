import numpy as np
import re
import fileinput
import gensim
import numpy
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
        if tail == -1 or 0:
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
        
    sim_vector[n] = -1  # 自分自身は無視
    print("original:",text_word[n])
    for a in range(0, number):  # 上位n個を出す(n未満の配列には対応しないので注意)
        print("simirality:", np.nanmax(sim_vector), "answer ", a, ":", text_word[np.nanargmax(sim_vector)])
        sim_vector[np.nanargmax(sim_vector)] = -1
    print()
    return 0    

def caluculate(vectorizer, text):
    X = vectorizer.fit_transform(text)
    #print(X.toarray())
    #print('feature一覧:', vectorizer.get_feature_names())
#    print(X)
    ReturnBIGGEST(0, X)
#    ReturnBIGGEST(1, X)
    return 0

def doc2vec_sim(input_text, text_word_for_doc2vec):  # doc2vecのモデルを使って類似度を計算。未知語はないものとして
    model = gensim.models.doc2vec.Doc2Vec.load('../../word2vec/corpus/doc2vec.model')
    sim_vector = []
    filtered_words = []
    input_text = list( set(input_text) & set(model.vocab.keys()) )  #  未知語削除
    for words_for_doc2vec in text_word_for_doc2vec:
        filtered_words = list( set(words_for_doc2vec) & set(model.vocab.keys()) )
        filtered_sim = model.n_similarity(input_text, filtered_words)
        if type(filtered_sim) == numpy.ndarray: #  なぜか、simがどちらかが空（未知語しかない）ときにndarrayを返すため
            filtered_sim = 0
        sim_vector.append(filtered_sim)
    print("original:", text_word[0])
    for a in range(0, number):  # 上位n個を出す(n未満の配列には対応しないので注意)
        print("simirality:", np.nanmax(sim_vector), "answer ", a, ":", text_word[np.nanargmax(sim_vector)])
        sim_vector[np.nanargmax(sim_vector)] = -1
    print()
    return 0

    return 0
    
number = 5  # 上位何個を出すか?
text_full = []  # かかり受け情報を含んだそのままのEDAの結果をlist化
text_word = []  # scikit-learn ように ['This is a pen', 'That is a pen']というlist
text_word_for_doc2vec = []  # doc2vec ように[['this', 'is'], [..]]
fulls = []
words = ''
words_for_doc2vec = []
input_text = '../../../text2feature/corpus/sample.eda'
corpus = '../../../text2feature/corpus/full.eda'
data = [input_text, corpus]
for line in fileinput.input(data):
    line = line.strip()
    if re.match('ID', line):
        continue
    if line == '':
        text_full.append(fulls)
        fulls = []
        text_word.append(words.strip())
        text_word_for_doc2vec.append(words_for_doc2vec)
        words = ''
        words_for_doc2vec = []
        continue
    units = line.split(' ')
    fulls.append(line)
    words = words + ' ' +  units[2]
    words_for_doc2vec.append(units[2])
text_full.append(fulls)
text_word.append(words.strip())
text_word_for_doc2vec.append(words_for_doc2vec)
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
caluculate(TfidfVectorizer(), text_word_dep_uni)  # depbigram-unigramモデル
print('doc2vec')
doc2vec_sim(text_word_for_doc2vec[0], text_word_for_doc2vec)  # doc2vec
