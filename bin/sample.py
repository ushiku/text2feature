from module.doc2vec import Doc2Vec
from module.dep2feature import Dep2Feature
import module.text2dep as text2dep
import pickle
import pdb
#テキストファイル(hoge.txt)とedaの学習モデルを引数として渡す
#input_eda = text2dep.text2dep('../corpus/hoge.txt',  kytea_model='../model/model.bin', eda_model='../model/bccwj-20140727.etm')
#input_eda = text2dep.text2dep('../corpus/hoge.txt',  kytea_model='../model/model.bin', eda_model='../model/bccwj-20140727.etm')
#f = open("../model/eda.dump", "wb")
#pickle.dump(input_eda, f)            # ファイルに保存
#f.close()

#かかり受けに時間がかかるので、読み込み用
f = open('../model/eda.dump', 'rb')
input_eda = pickle.load(f)
f.close()

#input_text = open('../corpus/sample.eda')
# TODO: Tf:idf:


corpus_eda = []
for line in open('../corpus/full.eda'):
    corpus_eda.append(line.strip())
input_vector, corpus_vector, array = Dep2Feature.vectorizer(input_eda, corpus_eda, feature='word', vectorizer = 'count')  # vectorizer 

tf = Dep2Feature.calculate_tf(array, 1)
idf = Dep2Feature.calculate_idf(array)

tf_idf = tf * idf
input_vector, corpus_vector, array = Dep2Feature.vectorizer(input_eda, corpus_eda, feature='word', vectorizer = 'tfidf')  # vectorizer


pdb.set_trace()
#Dep2Feature.sim_example(input_vector, corpus_vector, input_eda, corpus_eda)  # cos距離の実演
#Dep2Feature.calculate(input_list, sample_list, feature='word', number=5, vectorizer = 'tfidf') # tfidfモデル
#Dep2Feature.calculate(input_list, sample_list, feature='word_dep', number=5, vectorizer = 'count')
#Dep2Feature.calculate(input_list, sample_list, feature='word_dep', number=5, vectorizer = 'tfidf')
#Dep2Feature.calculate(input_list, sample_list, feature='word_dep_uni', number=5, vectorizer = 'tfidf')

