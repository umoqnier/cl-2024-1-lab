import numpy as np
from gensim.models import word2vec
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from random import seed, randrange
import matplotlib.pyplot as plt
semilla = 1234
np.random.seed(semilla)
seed(semilla)
"""
Importar el modelo
"""
directory = "./"
model_name = "eswiki-large-vs100-w2-CBOW.model"
model_path = directory + model_name
def load_model(model_path):
    try:
        return word2vec.Word2Vec.load(model_path)
    except:
        print(f"[WARN] Model not found in path {model_path}")
        return None

# A indicación de mi profesora de Supercómputo, ya o le pondré
# colorcitos a las prácticas porque no son buenas prácticas :c
if __name__ == '__main__':
    print("Práctica 6 - Word Embeddings - Lenin Pavón")
    print("Cargando el modelo " + model_name)
    model = word2vec.Word2Vec.load(model_path)
    word_vectors = model.wv
    tokens, dim = word_vectors.vectors.shape[0], word_vectors.vectors.shape[1]
    print(f"El modelo tiene {tokens} tokens y una dimensión vectorial de {dim}")
    n_palabras = 100
    idx_palabras = [randrange(0,tokens-1) for i in range(n_palabras)]
    palabras = [word_vectors.index_to_key[i] for i in idx_palabras]
    vecs = word_vectors[palabras]
    # print(palabras) # Descomentar para ver la lista de palabras
    print("Reduciendo la dimensionalidad")
    # Todos los modelos tienen por defecto una dimensión = 2
    print("Aplicando Principal Component Analysis (PCA)")
    pca = PCA(n_components=2)
    vecs_pca = pca.fit_transform(vecs)
    fig, ax = plt.subplots(1,3, figsize=(21,7))
    fig.suptitle(f'Reducción de la dimensionalidad de {model_name}')
    ax[0].title.set_text('PCA')
    ax[0].set_xlabel('Componente principal - 1')
    ax[0].set_ylabel('Componente principal - 2')
    x = [v[0] for v in vecs_pca]
    y = [v[1] for v in vecs_pca]
    for i, txt in enumerate(palabras):
        ax[0].annotate(txt,(x[i],y[i]))
    ax[0].scatter(x, y)
    
    print("t-Distributed Stochastic Neighbor Embedding (T-SNE)")
    tsne = TSNE(n_components=2)
    vecs_tsne = tsne.fit_transform(vecs)
    
    ax[1].title.set_text('T-SNE')
    ax[1].set_xlabel('Eje x')
    ax[1].set_ylabel('Eje y')
    x = [v[0] for v in vecs_tsne]
    y = [v[1] for v in vecs_tsne]
    for i, txt in enumerate(palabras):
        ax[1].annotate(txt,(x[i],y[i]))
    ax[1].scatter(x, y)
    
    print("Singular Value Descomposition (SVD)")
    svd = TruncatedSVD(n_components=2)
    vecs_svd = svd.fit_transform(vecs)
    ax[2].title.set_text('SVD')
    ax[2].set_xlabel('Eje x')
    ax[2].set_ylabel('Eje y')
    x = [v[0] for v in vecs_svd]
    y = [v[1] for v in vecs_svd]
    ax[2].scatter(x, y)
    for i, txt in enumerate(palabras):
        ax[2].annotate(txt,(x[i],y[i]))
    plt.show()