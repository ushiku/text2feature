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
OBJ = Dep2Feature(input_eda, corpus_eda)  # インスタンス作成
input_vector, corpus_vector = OBJ.vectorize(unigram = 1, bigram = 0, trigram = 0, dep_bigram = 0, dep_trigram = 0, vectorizer = 'tfidf')  # Vecotrize
```
インスタンスを作成しますが、引数として、構文解析済みの入力のデータと、大規模なコーパスのデータを与えます。
このインスタンスで、vectorize関数を実行することで、それぞれをvectorにします。
unigram~dep_trigramに関しては、0 or 1の値、vectorizerに関しては,'tfidf' or 'count'を許容します。 unigram, bigram、trigramに関しては、連続する、1単語、2単語、3単語をそれぞれ素性にします。(なお、1単語に関しては、助詞などの語は捨てています)
dep_bigramは、構文解析による結果を利用し、かかり元とかかり先のつながりを連続する単語として素性にしています。dep_trigram に関しても、同様でかかり先のさらにかかり先までを含めて、連続する3単語と見て素性にしています。
これらの素性はそれぞれ、同時に利用することができます。

countは、単純に語の出現頻度をとったもの。tfidfは、tfidfによって単語ごとに重みをつけます。ただし、単純なtfidfではなく、正規化がされています。

```
idf = OBJ.calculate_idf()  # idfも引っ張ってこれる
tf = OBJ.calculate_tf(1)  # tfも持ってこれる
```
tf, idfのそれぞれの値に関しては、上記の関数で持って来れます。tfの引数は、n番目の記事を指定しています。


```
OBJ.sim_example_cos(input_vector, corpus_vector, number = 5)  # cos距離の例を表示する
```
vectorizeを確かめるために、cos類似度を出す関数です。 
input_vectorのそれぞれの記事に対して、corpus_vectorの中から類似度の高い上位5件を表示します。


```
OBJ.sim_example_jac(input_vector, corpus_vector, number = 5)  # cos距離の例を表示する
```
同様に、jaccard係数での類似度の高い5件を表示します。
