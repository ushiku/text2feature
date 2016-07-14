from module.doc2vec import Doc2Vec
from module.dep2feature import Dep2Feature
import module.text2dep as text2dep
import sys
import pickle
import pdb
#テキストファイル(hoge.txt)とedaの学習モデルを引数として渡す


#input_list = ['../corpus/input/1.txt', '../corpus/input/2.txt']
#input_eda, input_eda_raw = text2dep.text2dep(input_list,  kytea_model='../model/model.bin', eda_model='../model/bccwj-20140727.etm')
#print(input_eda)
#f = open("../model/eda.dump", "wb")
#pickle.dump(input_eda, f)            # ファイルに保存
#f.close()

#かかり受けに時間がかかるので、読み込み用
f = open('../model/eda.dump', 'rb')
input_eda = pickle.load(f)
f.close()

#corpus_list = sys.argv[1:]
#print(corpus_list)
#corpus_eda, corpus_eda_raw = text2dep.text2dep(corpus_list,  kytea_model='../model/model.bin', eda_model='../model/bccwj-20140727.etm')
#f = open("../model/corpus_eda.dump", "wb")
#pickle.dump(corpus_eda, f)            # ファイルに保存
#f.close()

f = open('../model/corpus_eda.dump', 'rb')
corpus_eda = pickle.load(f)
f.close()

#for line in open('../corpus/full.eda'):  # この形でもeda形式に落とせる
#    corpus_eda.append(line.strip())

OBJ = Dep2Feature()  # インスタンス作成
input_doc2vec = OBJ.vectorize_doc2vec(input_eda)
corpus_doc2vec = OBJ.vectorize_doc2vec(corpus_eda)
input_vector, corpus_vector = OBJ.vectorize(input_eda, corpus_eda, unigram = 1, dep_bigram = 1, dep_trigram = 1, vectorizer = 'tfidf')

#idf = OBJ.calculate_idf()  # idfも引っ張ってこれる
#tf = OBJ.calculate_tf(3)  # tfも持ってこれる

Dep2Feature.sim_example(input_vector, corpus_vector, input_eda, corpus_eda)  # cos距離の実演
Dep2Feature.sim_example(input_doc2vec, corpus_doc2vec, input_eda, corpus_eda)
pdb.set_trace()

