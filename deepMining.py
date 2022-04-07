import pandas as pd
import gensim

def prepareForDoc2Vec (l):
    for i, line in enumerate(l):
        # For training data, add tags
        yield gensim.models.doc2vec.TaggedDocument(line, [i])

file = pd.read_csv("/Users/joshgun.sirajzade/Documents/C2DH/AWAC2/AWAC2_webpage_feminism.csv")

guardian = file[file['domain']=="theguardian.com"]

#print(file)

from gensim.models import Word2Vec
from gensim.models import Doc2Vec

corpus_text = '\n'.join(guardian['content'])
sentences = corpus_text.split('\n')
sentences = [line.lower().split(' ') for line in sentences]

#print(sentences)
print(sentences[2])
model = Word2Vec(sentences=sentences, vector_size=50, window=5, workers=8)
#model.save("/Users/joshgun.sirajzade/Documents/C2DH/AWAC2/word2vec.model")

#print(model.corpus_total_words)
s = "feminism"
sims = model.wv.most_similar(s, topn=10)
print(sims)
documentsForDoc2Vec = list(prepareForDoc2Vec(sentences))
print(documentsForDoc2Vec[1])
modelDoc = Doc2Vec(documents=documentsForDoc2Vec, vector_size=10, window=5, min_count=1, workers=4)

print("-----------------")
print("here comes the prediction: ")
womanKeyWords = ['diversity', 'and', 'equality', 'women', 'on', 'company']
prediction = modelDoc.predict_output_word(womanKeyWords)
print(prediction)
print("here comes the vector: ")
vector = modelDoc.infer_vector(womanKeyWords)
print(vector)
print("the end!")

print("let us find most similar documents: ")

sims = modelDoc.dv.most_similar([vector], topn=len(modelDoc.dv))
print (sims)
print (documentsForDoc2Vec[142])
