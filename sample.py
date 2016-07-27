import sys
import pdb
import pickle
from text2feature.dep2feature import Dep2Feature
from text2feature.text2dep import Text2dep

# ファイルの読み込み
#input_list = ['corpus/input/1.txt', 'corpus/input/2.txt']
#input_eda, input_eda_raw = Text2dep().t2f(input_list,  kytea_model='model/model.bin', eda_model='model/bccwj-20140727.etm')
#f = open("model/eda.dump", "wb")
#pickle.dump(input_eda, f)            # ファイルに保存
#f.close()


#かかり受けに時間がかかるので、読み込み用。計算用の大規模corpusの読み込み
f = open('model/corpus_eda.dump', 'rb')
corpus_eda = pickle.load(f)
f.close()

eda_file_path_list = ['corpus/sample.eda', 'corpus/sample.eda']
input_eda = Text2dep.load_eda(eda_file_path_list)

OBJ = Dep2Feature(input_eda, corpus_eda)  # インスタンス作成
input_vector1, corpus_vector1 = OBJ.vectorize(unigram = 1, bigram = 1, trigram = 1, dep_bigram = 0, dep_trigram = 0, vectorizer = 'tfidf')  # Vecotrize

idf = OBJ.calculate_idf()  # idfも引っ張ってこれる
tf = OBJ.calculate_tf(1)  # tfも持ってこれる

sim_vector_cos = OBJ.sim_example_cos(input_vector1, corpus_vector1)
sim_vector_dic = OBJ.sim_example_cos(input_vector1, corpus_vector1)
sim_vector_sim = OBJ.sim_example_cos(input_vector1, corpus_vector1)
sim_vector_jac = OBJ.sim_example_cos(input_vector1, corpus_vector1)

OBJ.sim_print(OBJ.eda2unigram(input_eda), OBJ.eda2unigram(corpus_eda), sim_vector_cos)
OBJ.sim_print(OBJ.eda2unigram(input_eda), OBJ.eda2unigram(corpus_eda), sim_vector_dic)
OBJ.sim_print(OBJ.eda2unigram(input_eda), OBJ.eda2unigram(corpus_eda), sim_vector_sim)
OBJ.sim_print(OBJ.eda2unigram(input_eda), OBJ.eda2unigram(corpus_eda), sim_vector_jac)
