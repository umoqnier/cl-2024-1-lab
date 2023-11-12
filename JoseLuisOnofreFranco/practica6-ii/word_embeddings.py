# %% [markdown]
# # Word Embeddings II

# %% [markdown]
# En esta práctica, se va a analizar diferentes técnicas para reducir la dimensión de los vectores que codifican a una palabra, de un modelo ya generado por Word2Vec. Estas ténicas son:
# - PCA
# - SVD
# - TSNE

# %%
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

# %%
model_path = "eswiki-large-vs500-w6-SKIP_GRAM.model"
model = Word2Vec.load(model_path)

# %%
np.random.seed(13)


# %%
def get_vector_sampling(model, n) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
    -----------
    model:
        a Word2Vec model
    
    n: int
        is number of vectors to get
        
    Returns:
    --------
    tuple
        where the first element is the array of word embeddings and the
        second one is the array of words of those vectors
    """
    vocab = model.wv.index_to_key
    random_words = np.random.choice(vocab, size=n, replace=False)
    vectors = np.array([ model.wv[word] for word in random_words ])
    return vectors, random_words

def plot_embeddings(vectors, words):
    plt.clf()
    plt.figure(figsize=(15,15))
    plt.scatter(vectors[:,0], vectors[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, vectors):
        plt.text(x+0.03, y+0.03, word)
    plt.show()


# %% [markdown]
# Para hacer la comparación, se va a utilizar un muestreo de 100 véctores elegidos al azar.

# %%
vectors, words = get_vector_sampling(model, 100)

# %% [markdown]
# A continuación, se muestra las gráficas de los vectores en 2D al aplicar cada una de las ténicas de reducción de dimensionalidad.

# %% [markdown]
# ## PCA

# %%
pca_vectors = PCA(n_components=2, random_state=42).fit_transform(vectors)
    
plot_embeddings(pca_vectors, words)
# %% [markdown]
# Se puede observar aquí que muchas palabras del muestreo se mantienen juntas al centro. Se observa que palabras que pertenecen al mismo Idioma, como el inglés, pertenecen cercanas entre sí, aunque no necesariamente todas. A simple vista, las palabras que están cercanas no parece tener características en común.

# %% [markdown]
# ## SVD


# %%
svd_vectors = TruncatedSVD(n_components=2, random_state=42).fit_transform(vectors)
    
plot_embeddings(svd_vectors, words)

# %% [markdown]
# Lo que se observa es que hay una porción grande de palabras que se agrupan hacia un punto, pero la otra porción está más dispersa. No se observan bien las relaciones entre los embeddings a primera vista.
#
# Hay palabras que cambian drásticamente su posición, como es el caso de *cívico-militar*, *formula_75*, respecto a PCA.

# %% [markdown]
# ## TSNE

# %%
tsne_vectors = TSNE(n_components=2, random_state=42).fit_transform(vectors)
    
plot_embeddings(tsne_vectors, words)
# %% [markdown]
# Con esta ténica, parece que las palabras están más dispersas en comparación los otras dos. Se observan grupos de tienen cosas en común, como *chinchillas, cigüeñas, caballas* que son animales, o *ammonities, arenaria* que son plantas marinas. Con esta técnica, parece que es más fácil observar los embeddings que posiblemente estén cercanos respecto a los vectores originales, que con las otras técnicas.


# %% [markdown]
# Para hacer una comparación, se tomaron 10 palabras al azar y se calculó la distancia media entre las distancias de las 100 palabras.

# %%
def distance(a, b) -> float:
    return np.linalg.norm(a - b)

def mean_distance(index, vectors) -> float:
    v = vectors[index]
    distances = [ distance(v, u) for u in vectors]
    return np.mean(distances)


# %%
indices = np.random.choice(range(100), size=10, replace=False)
words_sample = words[indices]

# Distances from the original embeddings
original_distances = [ np.mean(model.wv.distances(word, words.tolist())) for word in words_sample ]
pca_distances = [ mean_distance(index, pca_vectors) for index in indices ]
svd_distances = [ mean_distance(index, svd_vectors) for index in indices ]
tsne_distances = [ mean_distance(index, tsne_vectors) for index in indices ]


data = { "words": words_sample, 
        "original": original_distances, 
        "PCA": pca_distances, 
        "SVD": svd_distances, 
        "TSNE": tsne_distances}
df = pd.DataFrame(data)

df

# %% [markdown]
# A pesar de que los datos se visualizan mejor en TSNE, SVD y PCA tienden a conservar mejor esa relación de la distancia entre las palabras. Por lo que TSNE captura otras características que permiten una mejor visualización de los embeddings, pero dejando por fuera otras.
