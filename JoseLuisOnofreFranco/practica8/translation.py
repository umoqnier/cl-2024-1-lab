# %% [markdown]
# # Neural Machine Translation (NMT)

# %% [markdown]
# ## Dependencias y configuración inicial

# %%
import requests as req

# %%
from google.colab import drive
drive.mount('/content/drive')


# %%
# %cd /content/drive/MyDrive/nmt0/

# %%
# !git clone https://github.com/ymoslem/MT-Preparation.git

# %%
# !pip3 install -r MT-Preparation/requirements.txt

# %%
# !pip3 install --upgrade -q sentencepiece

# %%
# !pip install OpenNMT-py

# %% [markdown]
# ## Desarrollo

# %%
def get_corpus_files(base_url, files):
    corpus = []
    for file in files:
        response = req.get(base_url + file).text
        corpus.append((file, response))

    return corpus


def get_parallel_corpus(corpus_info: dict):
    base_url = "https://raw.githubusercontent.com/"
    base_url += "AmericasNLP/americasnlp2021/main/data/"+corpus_info["name"]+"/"

    files1 = [ f"{file}.{corpus_info['lang1_code']}" for file in ["dev", "train"]]
    files2 = [ f"{file}.{corpus_info['lang2_code']}" for file in ["dev", "train", "test"]]
    files = files1 + files2
    lang1 = get_corpus_files(base_url, files1)
    lang2 = get_corpus_files(base_url, files2)

    return (lang1, lang2)


# %%
corpus_info = {
    "name": "guarani-spanish",
    "lang1_code": "gn",
    "lang2_code": "es"
}

# %% [markdown]
# Se obtiene el corpus paralelo de guarani-español, en donde se obtiene:
#
# - `.train, .dev` para el guaraní.
# - `.train, .dev, .test` para el español

# %%
guarani, spanish = get_parallel_corpus(corpus_info)


# %% [markdown]
# Para poder usar MT-Preparation, es necesario guardar los corpus en archivos.

# %%
def write_corpus(lang_corpus):
    for name, corpus in lang_corpus:
        with open(name, "w") as f:
            f.write(corpus)


# %%
write_corpus(guarani)
write_corpus(spanish)

# %%
# !ls

# %% [markdown]
# Se aplica el filtrado tanto para los archivos `train` y `dev` de cada lengua.

# %%
# !python3 MT-Preparation/filtering/filter.py train.es train.gn es gn

# %%
# !python3 MT-Preparation/filtering/filter.py dev.es dev.gn es gn

# %% [markdown]
# El siguiente paso es crear las subwords para todos los archivos generados.

# %%
# !python3 MT-Preparation/subwording/1-train_unigram.py train.es-filtered.es train.gn-filtered.gn

# %%
# !python3 MT-Preparation/subwording/2-subword.py source.model target.model train.es-filtered.es train.gn-filtered.gn

# %%
# !python3 MT-Preparation/subwording/2-subword.py source.model target.model dev.es-filtered.es dev.gn-filtered.gn

# %%
# !ls

# %% [markdown]
# Entonces, podemos construir el vocabulario. Estos archivos van a estar en `source.onmt.vocab` y `target.onmt.vocab`

# %%
# Creación del archivo de configuración
# Usando valores pequeños en vista de que tenemos un corpus limitado
# Para datasets grandes deberian aumentar los valores:
# train_steps, valid_steps, warmup_steps, save_checkpoint_steps, keep_checkpoint
SRC_DATA_NAME = "es-filtered.es.subword"
TARGET_DATA_NAME = "gn-filtered.gn.subword"


# %%
config = f'''# config.yaml

## Where the samples will be written
save_data: run

# Rutas de archivos de entrenamiento
#(previamente aplicado subword tokenization)
data:
    corpus_1:
        path_src: train.{SRC_DATA_NAME}
        path_tgt: train.{TARGET_DATA_NAME}
        transforms: [filtertoolong]
    valid:
        path_src: dev.{SRC_DATA_NAME}
        path_tgt: dev.{TARGET_DATA_NAME}
        transforms: [filtertoolong]

# Vocabularios (serán generados por `onmt_build_vocab`)
src_vocab: source.onmt.vocab
tgt_vocab: target.onmt.vocab

# Tamaño del vocabulario
#(debe concordar con el parametro usado en el algoritmo de subword tokenization)
src_vocab_size: 50000
tgt_vocab_size: 50000

# Filtrado sentencias de longitud mayor a n
# actuara si [filtertoolong] está presente
src_seq_length: 150
src_seq_length: 150

# Tokenizadores
src_subword_model: source.model
tgt_subword_model: target.model

# Archivos donde se guardaran los logs y los checkpoints de modelos
log_file: train.log
save_model: models/model.enes

# Condición de paro si no se obtienen mejoras significativas
# despues de n validaciones
early_stopping: 4

# Guarda un checkpoint del modelo cada n steps
save_checkpoint_steps: 1000

# Mantiene los n ultimos checkpoints
keep_checkpoint: 3

# Reproductibilidad
seed: 3435

# Entrena el modelo maximo n steps
# Default: 100,000
train_steps: 3000

# Corre el set de validaciones (*.dev) despues de n steps
# Defatul: 10,000
valid_steps: 1000

warmup_steps: 1000
report_every: 100

# Numero de GPUs y sus ids
world_size: 1
gpu_ranks: [0]

# Batching
bucket_size: 262144
num_workers: 0
batch_type: "tokens"
batch_size: 4096
valid_batch_size: 2048
max_generator_batches: 2
accum_count: [4]
accum_steps: [0]

# Configuración del optimizador
model_dtype: "fp16"
optim: "adam"
learning_rate: 2
# warmup_steps: 8000
decay_method: "noam"
adam_beta2: 0.998
max_grad_norm: 0
label_smoothing: 0.1
param_init: 0
param_init_glorot: true
normalization: "tokens"

# Configuración del Modelo
encoder_type: transformer
decoder_type: transformer
position_encoding: true
enc_layers: 6
dec_layers: 6
heads: 8
hidden_size: 512
word_vec_size: 512
transformer_ff: 2048
dropout_steps: [0]
dropout: [0.1]
attention_dropout: [0.1]
'''

with open("/content/drive/MyDrive/nmt0/config.yaml", "w+") as config_yaml:
  config_yaml.write(config)

# %%
# %%time
# !onmt_build_vocab -config config.yaml -n_sample -1 -num_threads 2

# %%
# %%time
# !onmt_train -config config.yaml

# %% [markdown]
# Después se realiza la traducción del test. Primero hay que obtener el corpus del target language

# %%
target_code = "gn"
response = req.get(f"https://raw.githubusercontent.com/AmericasNLP/americasnlp2021/main/test_data/test.{target_code}")
with open(f"test.{target_code}", "w") as f:
  f.write(response.text)

# %%
# !ls test*

# %%
# !python3 MT-Preparation/filtering/filter.py test.es test.gn es gn

# %%
# !python3 MT-Preparation/subwording/2-subword.py source.model target.model test.es-filtered.es test.gn-filtered.gn

# %%
# %%time
# !onmt_translate -model models/model.enes_step_3000.pt -src test.es-filtered.es.subword -output gn.practice.translated -gpu 0 -min_length 1

# %% [markdown]
# Se obtiene el resultado al hacer el desubword

# %%
# !python3 MT-Preparation/subwording/3-desubword.py target.model gn.practice.translated

# %% [markdown]
# ## Evaluación

# %% [markdown]
# Para hacer la evaluación, se hará uso de los métodos que tiene el shared task.

# %%
# !git clone https://github.com/AmericasNLP/americasnlp2021

# %%
# !python3 americasnlp2021/evaluate.py --sys gn.practice.translated.desubword --ref test.es-filtered.es

# %% [markdown]
# Los resultados fueron los siguientes:
#
# | Model     | BLEU  | ChrF (0-1) |
# |-----------|-------|------------|
# | Baseline  | 3.26  | 0.22       |
# | Practica  | 0.26  | 11.89      |
#

# %% [markdown]
# ## Extra

# %% [markdown]
# **¿Cómo se diferencia de BLEU? (ChrF)**
#
# BLEU toma en cuenta los n-gramas a nivel palabra, mientras que ChrF lo hace a nivel cáracter, lo que beneficia a lenguajes que tienen una morfología muy rica. Otra diferencia es la penalización a palabras cortas, donde BLEU destaca en esto.
#
# **¿Porqué es reelevante utilizar otras medidas de evaluación además de BLEU?**
#
# Porque las diversidad de las lenguas. Un caso puede ser la morfología, que en un BLEU casos como (dormí, dormía), donde no son iguales, pero se acercan en el significado. Otra cosa serían los sinónimos, que bien no captura bien BLEU: *Yo tomé una pluma*, *Yo agarré una pluma*, *Yo cogí un boligrafo* están muy cercanos en significado, pero evaluando a nivel palabra no se captura eso.
#
#
