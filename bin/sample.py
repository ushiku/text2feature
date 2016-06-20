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
input_text = pickle.load(f)
f.close()

#d2v.make_model('/mnt/mqs01/data/ushiku/BCCWJ/full.word', 'model_sample')

#input_text = '../corpus/sample.eda'

#eda形式は、open()でもよい
corpus = open('../corpus/full.eda')

#eda形式をリスト化する
sample_list = Dep2Feature.eda2list(corpus)
input_list = Dep2Feature.eda2list(input_text)


print(input_list)
#print(Dep2Feature.caluculate(input_list, sample_list, feature='word', number=5, vectorizer = 'tfidf'))  # tfidfモデル
#print(Dep2Feature.caluculate(input_list, sample_list, feature='word_dep', number=5, vectorizer = 'count'))
#print(Dep2Feature.caluculate(input_list, sample_list, feature='word_dep', number=5, vectorizer = 'tfidf'))
#print(Dep2Feature.caluculate(input_list, sample_list, feature='word_dep_uni', number=5, vectorizer = 'tfidf'))

