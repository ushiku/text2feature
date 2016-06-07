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


def caluculate(vectorizer, text):
    '''
    nと,vector_matrixを渡すことで、その中で最大の類似度の組み合わせを表示する  
    '''
    array = vectorizer.fit_transform(text)
    sim_vector = []
    vector1 = array[0].todense()
    vector1 = np.squeeze(np.asarray(vector1))
    for vector2 in array.todense():
        vector2 = np.squeeze(np.asarray(vector2))
        sim_vector.append(1-cosine(vector1, vector2))  # ここcosineが1-cosine距離で定式している?
    sim_vector[0] = -1  # 自分自身は無視
    print("original:", corpus_list[1][n])
    for a in range(0, number):  # 上位n個を出す(n未満の配列には対応しないので注意)
        print("simirality:", np.nanmax(sim_vector), "answer ", a, ":", corpus_list[1][np.nanargmax(sim_vector)])
        sim_vector[np.nanargmax(sim_vector)] = -1
    print()
    return 0    


def doc2vec_sim(input_text, text_word_for_doc2vec):  # doc2vecのモデルを使って類似度を計算。未知語はないものとして
    model = gensim.models.doc2vec.Doc2Vec.load('../model/doc2vec.model')
    sim_vector = []
    filtered_words = []
    input_text = sorted(list( set(model.vocab.keys() & set(input_text) )), key=input_text.index)  #  未知語削除
    print(input_text)
    for words_for_doc2vec in text_word_for_doc2vec:
        filtered_words = sorted(list( set(words_for_doc2vec) & set(model.vocab.keys())), key=words_for_doc2vec.index)
        filtered_sim = model.n_similarity(input_text, filtered_words)
#        print(input_text, filtered_words, "|sim=:", filtered_sim)
        if type(filtered_sim) == numpy.ndarray: #  なぜか、simがどちらかが空（未知語しかない）ときにndarrayを返すため
            filtered_sim = 0
        sim_vector.append(filtered_sim)
    print("original:", input_list[1][0])
    for a in range(0, number):  # 上位n個を出す(n未満の配列には対応しないので注意)
        print("simirality:", np.nanmax(sim_vector), "answer ", a, ":", corpus_list[1][np.nanargmax(sim_vector)])
        sim_vector[np.nanargmax(sim_vector)] = -1
    print()
    return 0


def eda2list(file_eda):
    '''
    edaファイルを展開して、listを吐く
    '''
    text_full = []  # かかり受け情報を含んだそのままのEDAの結果をlist化
    text_word = []  # scikit-learn ように ['This is a pen', 'That is a pen']というlist
    text_word_for_doc2vec = []  # doc2vec ように[['this', 'is'], [..]]
    fulls = []
    words = ''
    words_for_doc2vec = []
    for line in open(file_eda, 'r'):
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
    return [text_full, text_word, text_word_for_doc2vec, text_word_dep, text_word_dep_uni]


number = 5  # 上位何個を出すか?

input_text = '../corpus/sample.eda'
corpus = '../corpus/full.eda'
data = [input_text, corpus]
corpus_list = eda2list(corpus)
input_list = eda2list(input_text)
for n in range(0, 5):
    input_list[n].extend(corpus_list[n])
    corpus_list[n] = input_list[n]
print('tfidf')
caluculate(TfidfVectorizer(), corpus_list[1])  # tfidfモデル
print('dep')
caluculate(CountVectorizer(), corpus_list[3])  # depモデル
print('depbigram-tfidf')
caluculate(TfidfVectorizer(), corpus_list[3])  # depbigram-tfidfモデル
print('depbigram-unigram')
caluculate(TfidfVectorizer(), corpus_list[4])  # depbigram-unigramモデル
print('doc2vec')
doc2vec_sim(input_list[2][0], corpus_list[2])  # doc2vec
