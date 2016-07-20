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
    def __init__(self, input_eda, corpus_eda):                  # コンストラクタ
        self.input_eda = input_eda
        self.corpus_eda = corpus_eda
        self.name = ""
        self.vectorizer = ""


    @classmethod 
    def load_eda(self, eda_file_path):
        ''' 
        かかり受け情報を含んだそのままのEDAの結果をlist化
        '''
        text_full = []
        fulls = []
        for line in open(eda_file_path, 'r'):
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


# unigram, bigram, trigram, depbigram, deptrigramの計5つ
    @classmethod
    def eda2unigram(self, eda):
        '''
        eda形式からunigramを返す。[[私 は 元気. 肝 座っている], [ . . . ]]
        '''
        text_word = []  # 各articleごとのunigram
        words = ''
        for article in eda:
            for sentence in article:
                for line in sentence:
                    units = line.strip().split(' ')
                    words = words + ' ' + units[2]
            text_word.append(words.strip())
            words = ''
        return text_word


    @classmethod
    def eda2bigram(self, eda):
        '''
        eda形式からbigramを返す。[[私はは元気 元気. 肝 座 っている], [ . . . ]]
        '''
        text_word = []  # 各articleごとのunigram
        words = ''
        head = '^'
        tail = '$'
        for article in eda:
            for sentence in article:
                for line in sentence:
                    units = line.strip().split(' ')
                    words = words + ' ' + head + units[2]
                    head = units[2]
                words = words + ' ' + units[2] + tail
                head = '^'
            text_word.append(words.strip())
            words = ''
        return text_word


    @classmethod
    def eda2trigram(self, eda):
        '''
        eda形式からbigramを返す。[[私はは元気 元気. 肝 座 っている], [ . . . ]]
        '''
        text_word = []  # 各articleごとのunigram
        words = ''
        head1, head2 = '^', '^'
        tail1, tail2 = '$', '$'
        for article in eda:
            for sentence in article:
                for line in sentence:
                    units = line.strip().split(' ')
                    words = words + ' ' + head1 + head2 + units[2]
                    head1 = head2
                    head2 = units[2]
                words = words + ' ' + head1 + head2 +tail1
                words = words + ' ' + head1 + head2 + units[2]
                head1, head2 = '^', '^'
            text_word.append(words.strip())
            words = ''
        return text_word


    @classmethod
    def eda2dep_bigram(self, eda):
        '''
        かかり受けのbigramモデル. eda2wordにたいして、depのbigramをとる
        '''
        text_word = []
        for article in eda:
            for sentence in article:
                if sentence == []:
                    continue
                dep_bigram =  self.text2dep_bigram(sentence)
                text_word.append(dep_bigram.strip())
        return text_word


    @classmethod
    def text2dep_bigram(self, text):
        '''
        depbigramを吐く. eda2dep_bigramの実行部分
        '''
        dep_bigram = ''
        heads, tails, words, poss = [], [], [], []

        for line in text:
            line = line.strip()
            units = line.split(' ')
            heads.append(int(units[0]))
            tails.append(int(units[1]))
            words.append(units[2])
            poss.append(units[3])
        dep_bigram = '^' + words[0]
        for tail, word in zip(tails, words):
            if tail == -1 or 0:
                dep_bigram = dep_bigram + ' ' + word + '$'
            else:
                dep_bigram = dep_bigram + ' ' + word + words[tail -1]
        return dep_bigram

    
    @classmethod
    def eda2dep_trigram(self, eda):
        '''
        depのtrigramをとる
        '''
        text_word = []
        for article in eda:
            for sentence in article:
                if sentence == []:
                    continue
                dep_bigram =  self.text2dep_trigram(sentence)
                text_word.append(dep_bigram.strip())
        return text_word


    @classmethod
    def text2dep_trigram(self, text):
        '''
        eda2dep_trigramのかかり受け部分
        '''
        dep_trigram = ''
        heads, tails, words, poss = [], [], [], []
        
        for line in text:
            line = line.strip()
            units = line.split(' ')
            heads.append(int(units[0]))
            tails.append(int(units[1]))
            words.append(units[2])
            poss.append(units[3])
        if len(words) >= 2:  # 一つのときはこの動作を行わない
            dep_trigram = '^' + words[0] + words[1]
        dep_bigram = dep_trigram + ' ' + '^' + '^' + words[0] 
        for tail, word in zip(tails, words):
            if tail == -1 or 0:
                dep_trigram = dep_trigram + ' ' + word + '$' + '$'  # 1個後ろもない
            elif tails[tail-1] == -1 or 0:
                dep_trigram = dep_trigram + ' ' + word + words[tail-1] + '$'  # 2個後ろがない
            else:
                dep_trigram = dep_trigram + ' ' + word + words[tail-1] + words[tails[tail-1]-1]  # 2個後ろまで
        return dep_trigram
        

    def vectorize_doc2vec(self, input_eda):
        '''
        input_listをdoc2vecを利用してvectorizeする
        '''
        model = gensim.models.doc2vec.Doc2Vec.load('../model/doc2vec.model')
        input_vector = []
        first_flag = 1
        for text in text_full:
            words = []
            for line in text:
                line = line.strip()
                units = line.split(' ')
                words.append(units[2])
            if first_flag == 1:
                input_vector = model.infer_vector(words)
                first_flag = 0
            else:
                input_vector = np.vstack((input_vector, model.infer_vector(words)))
        return input_vector
    

    def doc2vec(self, model_path):
        input_eda = self.vectorize_doc2vec(self.input_eda, model_path)
        corpus_eda = self.vectorize_doc2vec(self.corpus_eda, model_path)
        return input_eda, corpus_eda


    @classmethod
    def vectorize_doc2vec(self, input_eda, model_path):
        '''
        input_listをdoc2vecを利用してvectorizeする
        '''
        model = gensim.models.doc2vec.Doc2Vec.load(model_path)
        input_vector = []
        first_flag = 1
        for article in input_eda:
            words = []
            for sentence in article:
                for line in sentence:
                    line = line.strip()
                    units = line.split(' ')
                    words.append(units[2])
            if first_flag == 1:
                input_vector = model.infer_vector(words)
                first_flag = 0
            else:
                input_vector = np.vstack((input_vector, model.infer_vector(words)))
        return input_vector


    def vectorize(self, unigram=1, bigram=0, trigram=0, dep_bigram=0, dep_trigram=0, vectorizer='count'):
        '''
        input_listをcorpus_listを使ってvectorizeする
        '''
        words = [0]
        words.extend(self.eda2unigram(self.input_eda))
        words.extend(self.eda2unigram(self.corpus_eda))
        words.pop(0)
        input_length = len(self.input_eda)
        corpus_length = len(self.corpus_eda)
        
        text_list = []
        if unigram == 1:
#            print('unigram')
            text = [0]
            text.extend(self.eda2unigram(self.input_eda))
            text.extend(self.eda2unigram(self.corpus_eda))
            text.pop(0)
            text_list.append(text)
        if bigram == 1:
#            print('bigram')
            text = [0]
            text.extend(self.eda2bigram(self.input_eda))
            text.extend(self.eda2bigram(self.corpus_eda))
            text.pop(0)
            text_list.append(text)
        if trigram == 1:
#            print('trigram')
            text = [0]
            text.extend(self.eda2trigram(self.input_eda))
            text.extend(self.eda2trigram(self.corpus_eda))
            text.pop(0)
            text_list.append(text)
        if dep_bigram == 1:
#            print('dep_bigram')
            text = [0]
            text.extend(self.eda2dep_bigram(self.input_eda))
            text.extend(self.eda2dep_bigram(self.corpus_eda))
            text.pop(0)
            text_list.append(text)
        if dep_trigram == 1:
#            print('dep_trigram')
            text = [0]
            text.extend(self.eda2dep_trigram(self.input_eda))
            text.extend(self.eda2dep_trigram(self.corpus_eda))
            text.pop(0)
            text_list.append(text)
        if text_list == []:
 #           print('Error:素性が選択されていません')
            return 0
        
        text_mixed = []
        for text in text_list:
            if text_mixed == []:
                text_mixed = text
            else:
                for line_text_mixed, line_text in zip(text_mixed, text):
                    text_mixed[text_mixed.index(line_text_mixed)] = line_text_mixed + ' ' + line_text

        print(vectorizer)
        self.count_array = CountVectorizer().fit_transform(text_mixed)  # tf計算用

        if vectorizer == 'count':
            self.vectorizer = CountVectorizer()
        elif vectorizer == 'tfidf':
            self.vectorizer = TfidfVectorizer()
        else:
            print("Error:無効なVectorizerです")
            return 0
        array = self.vectorizer.fit_transform(text_mixed)   # インスタンス変数にアクセスはインスタンスメソッドのみ
        input_vector = array[:input_length].todense()
        input_vector = np.squeeze(np.asarray(input_vector))
        corpus_vector = array[input_length :].todense()
        corpus_vector = np.squeeze(np.asarray(corpus_vector))
        return input_vector, corpus_vector

    
    def calculate_idf(self):
        '''
        count_vectorizeしたものから、idf_arrayを作成する(原理的にはtfidf_vectorizeしたものでもいけるはず)
        '''
        array = self.count_array.toarray()
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
        

    def calculate_tf(self, number):
        '''
        count_vectorizeの一部を渡すことでtfを作成する。numberはinputの何ファイル目かに対応
        '''
        doc_array = self.count_array.toarray()[number]
        total_word_count = 0
        tf_list = []
        for word_count in doc_array:
            total_word_count += word_count
            tf_list.append(word_count)
        tf_list = tf_list/total_word_count
        return tf_list


    def sim_example(self, input_vector, corpus_vector, number=5):
        '''
        input_vectorをもらって、corpus_vectorとの類似度の大きいものを返す
        '''
        input_word = self.eda2unigram(self.input_eda)
        corpus_word = self.eda2unigram(self.corpus_eda)
        for input_one, input_sent in zip(input_vector, input_word):
            print("input=", input_sent)
            sim_vector = []
            sim_list = []
            for corpus_one in corpus_vector:
                corpus_one = np.squeeze(np.asarray(corpus_one))
                sim_vector.append(1-cosine(input_one, corpus_one))  # ここcosineが1-cosine距離で定式している?
            for count in range(0, number):  # 上位n個を出す(n未満の配列には対応しないので注意)
                ans_sim = [np.nanmax(sim_vector), np.nanargmax(sim_vector)]
                print('配列:', np.nanargmax(sim_vector), 'No.', count, ' sim=', ans_sim[0], ' ', corpus_word[ans_sim[1]])
                sim_vector[np.nanargmax(sim_vector)] = -1
            print()
        return 0

