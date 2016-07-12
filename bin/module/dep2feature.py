import numpy as np
import re
import fileinput
import math
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
#        self.corpus_list = []


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


# unigram, bigram, trigram, depbigram, deptrigramの3つ

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
    def eda2bigram(self, eda):
        '''
        eda2wordようにbigram
        '''
        text_word = []
        words = ''
        head = '^'
        tail = '$'
        for line in eda:
            line = line.strip()
            if re.match('ID', line):
                continue
            if line == '':
                text_word.append(words.strip())
                words = ''
                continue
            units = line.split(' ')
            words = words + ' ' + head+units[2]
            head = units[2]
        words = words + ' ' + head + tail
        text_word.append(words.strip())
        return text_word
    

    @classmethod
    def eda2trigram(self, eda):
        '''
        trigramモデル
        '''
        text_word = []
        words = ''
        head1, head2 = '^', '^'
        tail1, tail2 = '$', '$'
        for line in eda:
            line = line.strip()
            if re.match('ID', line):
                continue
            if line == '':
                text_word.append(words.strip())
                words = ''
                continue
            units = line.split(' ')
            words = words + ' ' + head1 + head2 + units[2]
            head1 = head2
            head2 = units[2]
        words = words + ' ' + head1 + head2 + tail1
        words = words + ' ' + head2 + tail1 + tail2
        text_word.append(words.strip())
        return text_word


    @classmethod
    def eda2word_dep(self, eda):
        '''
        かかり受けのbigramモデル. eda2wordにたいして、depのbigramをとる
        '''
        text_full = self.eda2full(eda)
        text_word_dep = []
        for text in text_full:
            depbigram, depbigram_unigram = self.text2dip(text)
            text_word_dep.append(depbigram.strip())
        return text_word_dep


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
        elif feature == 'bigram':
            print('bigram')
            text = [0]
            text.extend(self.eda2bigram(input_eda))
            text.extend(self.eda2bigram(corpus_eda))
            text.pop(0)
        elif feature == 'trigram':
            print('trigram')
            text = [0]
            text.extend(self.eda2trigram(input_eda))
            text.extend(self.eda2trigram(corpus_eda))
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
        corpus_vector = array[input_length :].todense()
#        input_vector = array[:input_length]
#        corpus_vector = array[:output_vector]
        return input_vector, corpus_vector, array

    
    @classmethod
    def calculate_idf(self, array):
        '''
        count_vectorizeしたものから、idf_arrayを作成する(原理的にはtfidf_vectorizeしたものでもいけるはず)
        '''
        array = array.toarray()
        list_for_count = [0] * len(array[0])
        for document in array:
            number = 0  # 今アクセスしている要素番号
            for word_score in document:
                if word_score > 0:  # count
                    list_for_count[number] += 1
                number += 1
        idf_list = []
        for count in list_for_count:
            idf_list.append(math.log(len(array[0])/count))
        idf_list = np.array(idf_list)
        return idf_list
        

    @classmethod
    def calculate_tf(self, array, number):
        '''
        count_vectorizeの一部を渡すことでtfを作成する。(原理的に、こっちはCount_vectorizeのみ)
        '''
        doc_array = array.toarray()[number]
        total_word_count = 0
        tf_list = []
        for word_count in doc_array:
            total_word_count += word_count
            tf_list.append(word_count)
        tf_list = tf_list/total_word_count
        return tf_list


    @classmethod
    def sim_example(self, input_vector, corpus_vector, input_eda, corpus_eda, number=5):
        '''
        input_vectorをもらって、corpus_vectorとの類似度の大きいものを返す
        '''
        input_word = self.eda2word(input_eda)
        corpus_word = self.eda2word(corpus_eda)
        for input_one, input_sent in zip(input_vector, input_word):
            print("input=", input_sent)
            sim_vector = []
            sim_list = []
            for corpus_one in corpus_vector:
                corpus_one = np.squeeze(np.asarray(corpus_one))
                sim_vector.append(1-cosine(input_one, corpus_one))  # ここcosineが1-cosine距離で定式している?

            for count in range(0, number):  # 上位n個を出す(n未満の配列には対応しないので注意)
                ans_sim = [np.nanmax(sim_vector), np.nanargmax(sim_vector)]
                sim_vector[np.nanargmax(sim_vector)] = -1
                print('No.', count, ' sim=', ans_sim[0], ' ', corpus_word[ans_sim[1]])
            print()
        return 0


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
