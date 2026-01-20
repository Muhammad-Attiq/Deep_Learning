import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gensim.downloader as api
from gensim.models import KeyedVectors

plt.style.use('ggplot')

model = api.load("glove-wiki-gigaword-100")

model['bread']
model['croissant']

model.most_similar('usa')
model.most_similar('banana')
model.most_similar('croissant')
model.most_similar(negative='banana')

result = model.most_similar(positive=['woman', 'king'], negative=['man'])
print("{}: {:.4f}".format(*result[0]))

def analogy(x1, x2, y1):
    result = model.most_similar(positive=[y1, x2], negative=[x1])
    return result[0][0]

analogy('man', 'king', 'woman')
analogy('australia', 'beer', 'russia')
analogy('pencil', 'sketching', 'camera')
analogy('obama', 'clinton', 'reagan')
analogy('tall', 'tallest', 'long')
