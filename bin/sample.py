from module.doc2vec import Doc2Vec
from module.dep2feature import Dep2Feature
import module.text2dep as text2dep
import pickle
#テキストファイル(hoge.txt)とedaの学習モデルを引数として渡す
#output = text2dep.text2dep('../corpus/hoge.txt',  kytea_model='../model/model.bin', eda_model='../model/bccwj-20140727.etm')
#f = open("../model/eda.dump", "wb")
#pickle.dump(output, f)            # ファイルに保存
#f.close()

f = open('../model/eda.dump', 'rb')
l = pickle.load(f)
f.close()
eda = l.split('\n')
eda.pop(-1)
eda.pop(-1)
#for a in open('../corpus/sample.eda'):
#    print(a)
    


#
#d2v.make_model('/mnt/mqs01/data/ushiku/BCCWJ/full.word', 'model_sample')

#input_text = '../corpus/sample.eda'
input_text = eda
corpus = open('../corpus/full.eda')
sample_list = Dep2Feature.eda2list(corpus)
input_list = Dep2Feature.eda2list(input_text)

a = []
b = []
c = []
d = []
e = []
list_i = []
for each in input_list:
    a.append(each[0])
    b.append(each[1])
    c.append(each[2])
    d.append(each[3])
    e.append(each[4])
    a.append(b.append(c.append(d.append(e))))
    list_i.append(a)
print(list_i)
print(input_list)
print(Dep2Feature.caluculate(input_list, sample_list, feature='word', number=5, vectorizer = 'tfidf'))  # tfidfモデル
#print(Dep2Feature.caluculate(input_list, sample_list, feature='word_dep', number=5, vectorizer = 'count'))
#print(Dep2Feature.caluculate(input_list, sample_list, feature='word_dep', number=5, vectorizer = 'tfidf'))
#print(Dep2Feature.caluculate(input_list, sample_list, feature='word_dep_uni', number=5, vectorizer = 'tfidf'))

