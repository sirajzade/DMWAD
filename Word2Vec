import pandas as pd
import gensim

guardian = file[file['domain']=="theguardian.com"]

s = "feminism"

from gensim.models import Word2Vec

corpus_text = '\n'.join(guardian['content'])
sentences = corpus_text.split('\n')
sentences = [line.lower().split(' ') for line in sentences]
print(sentences)

model = Word2Vec(sentences=sentences, vector_size=50, window=5, workers=8)
model.save(r"C:\Users\wildkde\Desktop\Archived Unleashed\AWAC Lux\feminism.mod")

print(model.corpus_total_words)
sims = model.wv.most_similar(s, topn=10, )
print(sims)
