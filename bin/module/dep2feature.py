import numpy as np
import re
import fileinput
import gensim
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


class Dep2Feature:
    '''
    かかり受けから素性を生成する
    '''
    def __init__(self):                  # コンストラクタ
        self.name = ""
        self.corpus_list = []

    @classmethod 
    def eda2full(self, eda):
        ''' 
        かかり受け情報を含んだそのままのEDAの結果をlist化
        '''
        text_full = []
        fulls = []
        for line in eda:
            line = line.strip()
            if re.match('ID', line):
                continue
            if line == '':
                text_full.append(fulls)
                fulls = []
                continue
            fulls.append(line)
        text_full.append(fulls)
        return text_full


    @classmethod
    def eda2word(self, eda):
        '''
        scikit-learn ように ['This is a pen', 'That is a pen']というlist
        '''
        text_word = []
        words = ''
        for line in eda:
            line = line.strip()
            if re.match('ID', line):
                continue
            if line == '':
                text_word.append(words.strip())
                words = ''
                continue
            units = line.split(' ')
            words = words + ' ' +  units[2]
        text_word.append(words.strip()) 
        return text_word
    
    @classmethod
    def eda2word_dep(self, eda):
        '''
        かかり受けのbigramモデル
        '''
        text_full = self.eda2full(eda)
        text_word_dep = []
        for text in text_full:
            depbigram, depbigram_unigram = self.text2dip(text)
            text_word_dep.append(depbigram.strip())
        return text_word_dep
        
    @classmethod
    def eda2list(self, eda):
        '''
        edaファイルを展開して、listを吐く
        '''
        text_full = []  # かかり受け情報を含んだそのままのEDAの結果をlist化
        text_word = []  # scikit-learn ように ['This is a pen', 'That is a pen']というlist
        text_word_for_doc2vec = []  # doc2vec ように[['this', 'is'], [..]]
        fulls = []
        words = ''
        words_for_doc2vec = []
        for line in eda:
#        for line in open(file_eda, 'r'):
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
            depbigram, depbigram_unigram = self.text2dip(text)
            text_word_dep.append(depbigram.strip())
            text_word_dep_uni.append(depbigram_unigram.strip())
        return_list = (text_full, text_word, text_word_for_doc2vec, text_word_dep, text_word_dep_uni)
        return return_list

    @classmethod
    def text2dip(self, text):
        '''
        text は EDAの出力1columずつlist化したもの.  depbigramと、それにunigramを末尾に加えたdepbigram_unigramを吐く
        '''
        heads, tails, words, poss = [], [], [], []
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


    @classmethod
    def vectorizer(self, input_eda, corpus_eda, feature='word', vectorizer='count'):
        '''
        input_listをcorpus_listを使ってvectorizeする
        '''
        input_length = len(self.eda2full(input_eda))
        words = [0]
        words.extend(self.eda2word(input_eda))
        words.extend(self.eda2word(corpus_eda))
        words.pop(0)
        if feature == 'word':
            print('word')
            text = [0]
            text.extend(self.eda2word(input_eda))
            text.extend(self.eda2word(corpus_eda))
            text.pop(0)
        elif feature == 'word_dep':
            print('word_dep')
            text = [0]
            text.extend(self.eda2word_dep(input_eda))
            text.extend(self.eda2word_dep(corpus_eda))
            text.pop(0)
#        elif feature == 'word_dep_uni':
#            print('word_dep_uni')
#            text = [0]
#            text.extend(input_list[4])
#            text.extend(corpus_list[4])
#            text.pop(0)
        else:
            print("無効な素性です")
            return 0
        print(vectorizer)
        if vectorizer == 'count':
            vectorizer = CountVectorizer()
        elif vectorizer == 'tfidf':
            vectorizer = TfidfVectorizer()
        else:
            print("無効なVectorizerです")
            return 0
        array = vectorizer.fit_transform(text)
        input_vector = array[:input_length].todense()
        input_vector = np.squeeze(np.asarray(input_vector))
        corpus_vector = array[input_length + 1:].todense()
        return input_vector, corpus_vector


    @classmethod
    def calculate(self, input_list, corpus_list, feature='word', number=5, vectorizer='count'):
        '''
        input_listをもらって、一つづつ、calculate_backに回す
        TODO: 入力をvectorにするべき
        '''
        one_input = []
        for input_number in range(0, len(input_list[0])):
            for units in input_list:
                one_input.append(units[input_number])
            one_output = self.calculate_back(one_input, corpus_list, feature, number, vectorizer)
            print(one_input[1])
            one_input = []
            print(one_output[0])
            for a in one_output[1]:
                print(a)
        return 0


    @classmethod
    def calculate_back(self, one_input, corpus_list, feature='word', number=5, vectorizer='count'):
        '''
        one_inputとcorpus_listを使ってベクトル化して、ベクトルとcorpusないの類似度の高いものを吐く
        '''
#        print(corpus_list[1])
        words = [0]
        words.extend([one_input[1]])
        words.extend(corpus_list[1])
        words.pop(0)
        if feature == 'word':
            print('word')
            text = [0]
            text.extend([one_input[1]])
            text.extend(corpus_list[1])
            text.pop(0)
        elif feature == 'word_dep':
            print('word_dep')
            text = [0]
            text.extend([one_input[3]])
            text.extend(corpus_list[3])
            text.pop(0)
        elif feature == 'word_dep_uni':
            print('word_dep_uni')
            text = [0]
            text.extend([one_input[4]])
            text.extend(corpus_list[4])
            text.pop(0)
        else:
            print("無効な素性です")
            return 0
        print(vectorizer)
        if vectorizer == 'count':
            vectorizer = CountVectorizer()
        elif vectorizer == 'tfidf':
            vectorizer = TfidfVectorizer()
        else:
            print("無効なVectorizerです")
            return 0
        array = vectorizer.fit_transform(text)
        sim_vector, sim_list = [], []
        input_vector = array[0].todense()
        input_vector = np.squeeze(np.asarray(input_vector))
        for corpus_vector in array.todense():
            corpus_vector = np.squeeze(np.asarray(corpus_vector))
            sim_vector.append(1-cosine(input_vector, corpus_vector))  # ここcosineが1-cosine距離で定式している?
#        sim_vector[0] = -1  # 自分自身は無視
        for count in range(0, number):  # 上位n個を出す(n未満の配列には対応しないので注意)
            ans_sim = [np.nanmax(sim_vector), words[np.nanargmax(sim_vector)]]
            sim_list.append(ans_sim)
            sim_vector[np.nanargmax(sim_vector)] = -1
        return input_vector, sim_list


    def doc2vec_sim(self, input_text, text_word_for_doc2vec):  # doc2vecのモデルを使って類似度を計算。未知語はないものとして
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
