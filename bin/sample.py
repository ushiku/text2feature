from module.doc2vec import Doc2Vec
from module.dep2feature import Dep2Feature
import module.text2dep as text2dep
import pickle

#テキストファイル(hoge.txt)とedaの学習モデルを引数として渡す
#output = text2dep.text2dep('../corpus/hoge.txt',  kytea_model='../model/model.bin', eda_model='../model/bccwj-20140727.etm')
#f = open("../model/eda.dump", "wb")
#pickle.dump(output, f)            # ファイルに保存
#f.close()

#かかり受けに時間がかかるので、読み込み用
f = open('../model/eda.dump', 'rb')
input_eda = pickle.load(f)
f.close()

#input_text = open('../corpus/sample.eda')

#eda形式は、open()でもよい
corpus_eda = open('../corpus/full.eda')

#eda形式をリスト化する
#sample_list = Dep2Feature.eda2list(corpus)
#input_list = Dep2Feature.eda2list(input_text)

input_vector, corpus_vector = Dep2Feature.vectorizer(input_eda, corpus_eda, feature='word', vectorizer = 'tfidf')
print(input_vector, corpus_vector)
#Dep2Feature.calculate(input_list, sample_list, feature='word', number=5, vectorizer = 'tfidf') # tfidfモデル
#Dep2Feature.calculate(input_list, sample_list, feature='word_dep', number=5, vectorizer = 'count')
#Dep2Feature.calculate(input_list, sample_list, feature='word_dep', number=5, vectorizer = 'tfidf')
#Dep2Feature.calculate(input_list, sample_list, feature='word_dep_uni', number=5, vectorizer = 'tfidf')

