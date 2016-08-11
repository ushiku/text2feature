### Documentation

```
input_list = ['corpus/input/1.txt', 'corpus/input/2.txt']
input_eda, input_eda_raw = Text2dep().t2f(input_list,  kytea_model='model/model.bin', eda_model='model/bccwj-20140727.etm')

f = open("model/eda.dump", "wb")
pickle.dump(input_eda, f)            # ファイルに保存
f.close()
```
入力として、まず、コーパスを形態素解析、構文解析します。コーパスは複数のファイルからなり、一つ一つのファイルは記事などの意味ある区切りを想定しています。
kyteaで形態素解析、edaで構文解析を行いますが、その時にモデルが必要です。
- KyTea: http://www.phontron.com/kytea/index-ja.html
- EDA: http://www.ar.media.kyoto-u.ac.jp/tool/EDA/

にそれぞれモデルが存在しているので利用すると良いでしょう。アノテーション済みのデータがあれば学習を行うこともできますが、ここでは深く説明しません。
構文解析は、モデルの読み込みに時間がかかるので、できれば結果をシリアライズするのが望ましいです。

```
f = open('model/corpus_eda.dump', 'rb')
corpus_eda = pickle.load(f)
f.close()
```
あらかじめ、シリアライズしていた大規模なコーパスの構文解析済みデータを読み込みます.

```
eda_file_path_list = ['corpus/sample.eda', 'corpus/sample.eda']
input_eda = Text2dep.load_eda(eda_file_path_list)
```
EDAをCUI上で実行した結果を保存している場合、load_edaを使って読み込むことも可能です。

```
OBJ = Dep2Feature([input_eda, corpus_eda], bigram = 1, vectorizer=CountVectorizer())  # インスタンス作成

vector_list = OBJ.vectorize([input_eda, corpus_vector])

```
インスタンスを作成しますが、引数として、辞書を作成するためのcorpusのリストを与えます。
このインスタンスで、vectorizeメソッドを実行することで、引数のリストをそれぞれベクトルにします.
unigram~dep_trigramに関しては、0 or 1の値、vectorizerに関しては,'CountVectorizer(), TfIdfVectorizer()'を許容します。(importしておいてください) unigram, bigram、trigramに関しては、連続する、1単語、2単語、3単語をそれぞれ素性にします。(なお、1単語に関しては、助詞などの語は捨てています)
dep_bigramは、構文解析による結果を利用し、かかり元とかかり先のつながりを連続する単語として素性にしています。dep_trigram に関しても、同様でかかり先のさらにかかり先までを含めて、連続する3単語と見て素性にしています。
これらの素性はそれぞれ、同時に利用することができます。

countは、単純に語の出現頻度をとったもの。tfidfは、tfidfによって単語ごとに重みをつけます。ただし、単純なtfidfではなく、正規化がされています。

```
idf = OBJ.calculate_idf()  # idfも引っ張ってこれる
tf = OBJ.calculate_tf(1)  # tfも持ってこれる
```
tf, idfのそれぞれの値に関しては、上記の関数で持って来れます。tfの引数は、n番目の記事を指定しています。


```
sim_vector_cos = OBJ.sim_example_cos(input_vector, corpus_vector)
OBJ.sim_print(OBJ.eda2unigram(input_eda), OBJ.eda2unigram(corpus_eda), sim_vector_cos, number = 5)
```
vectorizeを確かめるために、cos類似度を出す関数です。
input_vectorのそれぞれの記事に対して、corpus_vectorの中から類似度の高い上位5件を表示します。


```
OBJ.sim_example_jac(input_vector, corpus_vector)  # jaccard係数
OBJ.sim_example_sim(input_vector, corpus_vector)  # simpson係数
OBJ.sim_example_dic(input_vector, corpus_vector)  # dice係数
```

