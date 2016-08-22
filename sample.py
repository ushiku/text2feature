import sys
import pdb
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from text2feature.dep2feature import Dep2Feature
from text2feature.text2dep import Text2dep


m = Text2dep()
input_list = ['input1.txt', 'input2.txt']

input_kytea = m.kytea(input_list, kytea_model='model/model.bin')
input_eda1 = m.eda(input_kytea, eda_model='model/bccwj-20140727.etm')

input_list = ['input3.txt']
input_eda2 = m.t2f(input_list, kytea_model='model/model.bin', eda_model='model/bccwj-20140727.etm')

input_eda3 = m.load_eda(['input4.eda'])

input_eda4 = m.kytea2eda(input_kytea)

D2F = Dep2Feature([input_eda1, input_eda2, input_eda3], unigram = 1, bigram = 1, vectorizer=CountVectorizer())  # 辞書作成

vectors = D2F.vectorize([input_eda1, input_eda2])
input_vector1 = vectors[0]
input_vector2 = vectors[1]


sim_matrix_cos = D2F.sim_example_cos(input_vector1, input_vector2)
sim_matrix_dic = D2F.sim_example_dic(input_vector1, input_vector2)
sim_matrix_sim = D2F.sim_example_sim(input_vector1, input_vector2)
sim_matrix_jac = D2F.sim_example_jac(input_vector1, input_vector2)

print('cos類似度')
D2F.sim_print(D2F.eda2unigram(input_eda1), D2F.eda2unigram(input_eda2), sim_matrix_cos, number = 2)
