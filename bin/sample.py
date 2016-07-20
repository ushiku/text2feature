import sys
import pdb
import pickle
from module.dep2feature import Dep2Feature
from module.text2dep import Text2dep


#input_list = ['../corpus/input/1.txt', '../corpus/input/2.txt']
#input_eda, input_eda_raw = Text2dep().t2f(input_list,  kytea_model='../model/model.bin', eda_model='../model/bccwj-20140727.etm')
#f = open("../model/eda.dump", "wb")
#pickle.dump(input_eda, f)            # ファイルに保存
#f.close()

#かかり受けに時間がかかるので、読み込み用
f = open('../model/eda.dump', 'rb')
input_eda = pickle.load(f)
f.close()

#corpus_list = sys.argv[1:]
#corpus_eda, corpus_eda_raw = text2dep.text2dep(corpus_list,  kytea_model='../model/model.bin', eda_model='../model/bccwj-20140727.etm')
#f = open("../model/corpus_eda.dump", "wb")
#pickle.dump(corpus_eda, f)            # ファイルに保存
#f.close()

#かかり受けに時間がかかるので、読み込み用
f = open('../model/corpus_eda.dump', 'rb')
corpus_eda = pickle.load(f)
f.close()

OBJ = Dep2Feature(input_eda, corpus_eda)  # インスタンス作成
#input_doc2vec = OBJ.vectorize_doc2vec(input_eda, '../model/doc2vec.model')  # doc2vecを使いベクトル化
#corpus_doc2vec = OBJ.vectorize_doc2vec(corpus_eda, '../model/doc2vec.model')
input_vector1, corpus_vector1 = OBJ.vectorize(unigram = 1, dep_bigram = 0, dep_trigram = 0, vectorizer = 'count')  # Vecotrize
input_vector2, corpus_vector2 = OBJ.vectorize(unigram = 1, bigram=1, dep_bigram = 0, dep_trigram = 0, vectorizer = 'count')
input_vector3, corpus_vector3 = OBJ.vectorize(unigram = 1, bigram=1, trigram=1, dep_trigram = 0, vectorizer = 'count')
input_vector4, corpus_vector4 = OBJ.vectorize(unigram = 1, bigram=1, trigram=1, dep_bigram = 1, dep_trigram = 1, vectorizer = 'count')
input_vector5, corpus_vector5 = OBJ.vectorize(unigram=1, vectorizer = 'tfidf')
#idf = OBJ.calculate_idf()  # idfも引っ張ってこれる
#tf = OBJ.calculate_tf(3)  # tfも持ってこれる
OBJ.sim_example(input_vector1, corpus_vector1, number = 3)  # cos距離の例
OBJ.sim_example(input_vector2, corpus_vector2, number = 3)
OBJ.sim_example(input_vector3, corpus_vector3, number = 3)
OBJ.sim_example(input_vector4, corpus_vector4, number = 3)
OBJ.sim_example(input_vector5, corpus_vector5, number = 3)


pdb.set_trace()
