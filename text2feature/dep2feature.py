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
    かかり受けから素性を生成する。
    '''
    def __init__(self, input_eda, corpus_eda):                  # コンストラクタ
        self.input_eda = input_eda
        self.corpus_eda = corpus_eda
        self.vectorizer = ""

# unigram, bigram, trigram, depbigram, deptrigramの計5つ
    @classmethod
    def _eda2unigram(self, eda):
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
    def _eda2bigram(self, eda):
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
    def _eda2trigram(self, eda):
        '''
        eda形式からtrigramを返す。
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
                words = words + ' ' + head1 + head2 + tail1
                words = words + ' ' + head1 + head2 + units[2]
                head1, head2 = '^', '^'
            text_word.append(words.strip())
            words = ''
        return text_word

    @classmethod
    def _eda2dep_bigram(self, eda):
        '''
        かかり受けのbigramモデル. eda2wordにたいして、depのbigramをとる
        '''
        text_word = []
        for article in eda:
            for sentence in article:
                if sentence == []:
                    continue
                dep_bigram = self._text2dep_bigram(sentence)
                text_word.append(dep_bigram.strip())
        return text_word

    @classmethod
    def _text2dep_bigram(self, text):
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
                dep_bigram = dep_bigram + ' ' + word + words[tail - 1]
        return dep_bigram

    @classmethod
    def _eda2dep_trigram(self, eda):
        '''
        depのtrigramをとる
        '''
        text_word = []
        for article in eda:
            for sentence in article:
                if sentence == []:
                    continue
                dep_bigram = self._text2dep_trigram(sentence)
                text_word.append(dep_bigram.strip())
        return text_word

    @classmethod
    def _text2dep_trigram(self, text):
        '''
        deptrigramを吐く. eda2dep_trigramの実行部分
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

#    def vectorize_doc2vec(self, input_eda, model_path):
#        '''
#        input_listをdoc2vecを利用してvectorizeする
#        '''
#        model = gensim.models.doc2vec.Doc2Vec.load(model_path)
#        input_vector = []
#        first_flag = 1
#        for text in text_full:
#            words = []
#            for line in text:
#                line = line.strip()
#                units = line.split(' ')
#                words.append(units[2])
#            if first_flag == 1:
#                input_vector = model.infer_vector(words)
#                first_flag = 0
#            else:
#                input_vector = np.vstack((input_vector, model.infer_vector(words)))
#        return input_vector

#    def doc2vec(self, model_path):
#        input_eda = self.vectorize_doc2vec(self.input_eda, model_path)
#        corpus_eda = self.vectorize_doc2vec(self.corpus_eda, model_path)
#        return input_eda, corpus_eda

    def vectorize(self, unigram=1, bigram=0, trigram=0, dep_bigram=0, dep_trigram=0, vectorizer='count', token_pattern = u'(?u)\\b\\w\\w+\\b'):
        '''
        input_listをcorpus_listを使ってvectorizeする
        '''
        words = [0]
        words.extend(self._eda2unigram(self.input_eda))
        words.extend(self._eda2unigram(self.corpus_eda))
        words.pop(0)
        input_length = len(self.input_eda)
        corpus_length = len(self.corpus_eda)
        text_list = []
        if unigram == 1:
            text = [0]
            text.extend(self._eda2unigram(self.input_eda))
            text.extend(self._eda2unigram(self.corpus_eda))
            text.pop(0)
            text_list.append(text)
        if bigram == 1:
            text = [0]
            text.extend(self._eda2bigram(self.input_eda))
            text.extend(self._eda2bigram(self.corpus_eda))
            text.pop(0)
            text_list.append(text)
        if trigram == 1:
            text = [0]
            text.extend(self._eda2trigram(self.input_eda))
            text.extend(self._eda2trigram(self.corpus_eda))
            text.pop(0)
            text_list.append(text)
        if dep_bigram == 1:
            text = [0]
            text.extend(self._eda2dep_bigram(self.input_eda))
            text.extend(self._eda2dep_bigram(self.corpus_eda))
            text.pop(0)
            text_list.append(text)
        if dep_trigram == 1:
            text = [0]
            text.extend(self._eda2dep_trigram(self.input_eda))
            text.extend(self._eda2dep_trigram(self.corpus_eda))
            text.pop(0)
            text_list.append(text)
        if text_list == []:
            print('Error:素性が選択されていません')
            return 0
        text_mixed = []
        for text in text_list:
            if text_mixed == []:
                text_mixed = text
            else:
                for line_text_mixed, line_text in zip(text_mixed, text):
                    text_mixed[text_mixed.index(line_text_mixed)] = line_text_mixed + ' ' + line_text
        self.count_array = CountVectorizer(token_pattern=token_pattern).fit_transform(text_mixed)  # tf計算用
        if vectorizer == 'count':
            self.vectorizer = CountVectorizer(token_pattern=token_pattern)
        elif vectorizer == 'tfidf':
            self.vectorizer = TfidfVectorizer(token_pattern=token_pattern)  # token_pattern=u'(?u)\\b\\w+\\b'
        else:
            print("Error:無効なVectorizerです")
            return 0
        array = self.vectorizer.fit_transform(text_mixed)   # インスタンス変数にアクセスはインスタンスメソッドのみ
        input_vector = array[:input_length].todense()
        input_vector = np.atleast_2d(np.squeeze(np.asarray(input_vector)))
        corpus_vector = array[input_length:].todense()
        corpus_vector = np.atleast_2d(np.squeeze(np.asarray(corpus_vector)))
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

    def sim_print(self, input_word, corpus_word, sim_matrix, number=5):
        
        for input_sent, sim_vector in zip(input_word, sim_matrix):
            print("input=", input_sent)
            for count in range(0, number):  # 上位n個を出す(n未満の配列には対応しないので注意)
                ans_sim = [np.nanmax(sim_vector), np.nanargmax(sim_vector)]
                print('配列番号:', np.nanargmax(sim_vector), 'No.', count, 'sim=', ans_sim[0])
                print('output=', corpus_word[ans_sim[1]])
                src_set = set(input_sent.split())
                tag_set = set(corpus_word[ans_sim[1]].split())
                print('共通部分', list(src_set & tag_set))
                print()
                sim_vector[np.nanargmax(sim_vector)] = -1
            print()
        return 0

    def sim_example_cos(self, input_vector, corpus_vector):
        '''
        input_vectorをもらって、corpus_vectorとの類似度の大きいものを返す(cos_simmirarity)
        返り値はsim_vector
        '''
        sim_matrix = []
        for input_one in input_vector:
            sim_vector = []
            sim_list = []
            for corpus_one in corpus_vector:
                corpus_one = np.squeeze(np.asarray(corpus_one))
                sim_vector.append(1-cosine(input_one, corpus_one))  # ここcosineが1-cosine距離で定式している?
            sim_matrix.append(sim_vector)
        return sim_matrix

    def sim_example_jac(self, input_vector, corpus_vector):
        '''
        input_vectorをもらって、corpus_vectorとの類似度の大きいものを返す(jaccard係数)
        doc2vecのベクトルには対応していないので注意.
        '''
        sim_matrix = []
        for input_one in input_vector:
            sim_vector = []
            sim_list = []
            for corpus_one in corpus_vector:
                input_word_number_list = np.where(input_one > 0)
                corpus_word_number_list = np.where(corpus_one > 0)
                common = np.intersect1d(input_word_number_list, corpus_word_number_list)
                either = np.union1d(input_word_number_list[0], corpus_word_number_list[0])
                sim_vector.append(len(common)/len(either)) # jaccard係数(共通部分の要素数/全体部分の要素数)
            sim_matrix.append(sim_vector)
        return sim_matrix

    def sim_example_sim(self, input_vector, corpus_vector):
        '''
        input_vectorをもらって、corpus_vectorとの類似度の大きいものを返す(simpson係数)
        doc2vecのベクトルには対応していないので注意.
        '''
        sim_matrix = []
        for input_one in input_vector:
            sim_vector = []
            sim_list = []
            for corpus_one in corpus_vector:
                input_word_number_list = np.where(input_one > 0)
                corpus_word_number_list = np.where(corpus_one > 0)
                common = np.intersect1d(input_word_number_list, corpus_word_number_list)
                min_number = min(len(input_word_number_list[0]), len(corpus_word_number_list[0]))
                sim_vector.append(len(common)/min_number) # simpson係数(共通部分の要素数/少ない要素数)
            sim_matrix.append(sim_vector)
        return sim_matrix

    def sim_example_dic(self, input_vector, corpus_vector):
        '''
        input_vectorをもらって、corpus_vectorとの類似度の大きいものを返す(dice係数)
        doc2vecのベクトルには対応していないので注意.
        '''
        sim_matrix = []
        for input_one in input_vector:
            sim_vector = []
            sim_list = []
            for corpus_one in corpus_vector:
                input_word_number_list = np.where(input_one > 0)
                corpus_word_number_list = np.where(corpus_one > 0)
                common = np.intersect1d(input_word_number_list, corpus_word_number_list)
                sum_number = len(input_word_number_list[0]) + len(corpus_word_number_list[0])
                sim_vector.append(2 * len(common)/sum_number) # dice係数(2 * 共通部分の要素数/要素数の和)
            sim_matrix.append(sim_vector)
        return sim_matrix
