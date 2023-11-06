# %%
from preprocessing import *
from language_model import *

with open("quijote.txt", "r") as f:
    quijote = f.readlines()

corpus = preprocess_corpus(quijote)

from sklearn.model_selection import train_test_split
corpus_train, corpus_test = train_test_split(corpus, test_size=0.3)

vocab, index_sents = get_vocabulary(corpus_train)

n_grams = get_n_grams(index_sents, 2)
freq_n_grams = Counter(n_grams)

# %%
def average_perplexity(corpus, n, model, vocab):
    perplexities = 0
    for sentence in corpus:
        perplexities += perplexity(sentence, n, model, vocab)

    return perplexities // len(corpus)

# %%
bigram_model = get_model(index_sents, vocab, n=2, l=1)
trigram_model = get_model(index_sents, vocab, n=3, l=1)
# %%
print("Perplejidades promedio de los modelos:")
perplexity_3 = average_perplexity(corpus_test, 3, trigram_model, vocab)
perplexity_2 = average_perplexity(corpus_test, 2, bigram_model, vocab)

print("Bigramas: ", perplexity_2)
print("Trigramas: ", perplexity_3)
# %%
