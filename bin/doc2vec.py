import gensim
labeledSentences = gensim.models.doc2vec.TaggedLineDocument('corpus/ALL-train.word')
model = gensim.models.doc2vec.Doc2Vec(labeledSentences, size=100, window=8, min_count=5, workers=4)
model.save('corpus/doc2vec.model')
