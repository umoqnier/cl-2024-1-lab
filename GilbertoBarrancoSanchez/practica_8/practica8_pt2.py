# -*- coding: utf-8 -*-
"""practica8_pt2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hSDtpiHlZErzoVg0RaZCwX_PXp02NhL4
"""

# Creación del archivo de configuración
# Usando valores pequeños en vista de que tenemos un corpus limitado
# Para datasets grandes deberian aumentar los valores:
# train_steps, valid_steps, warmup_steps, save_checkpoint_steps, keep_checkpoint
SRC_DATA_NAME = "nah-filtered.nah.subword"
TARGET_DATA_NAME = "es-filtered.es.subword"

config = f'''# config.yaml

## Where the samples will be written
save_data: run

# Rutas de archivos de entrenamiento
#(previamente aplicado subword tokenization)
data:
    corpus_1:
        path_src: {SRC_DATA_NAME}.train
        path_tgt: {TARGET_DATA_NAME}.train
        transforms: [filtertoolong]
    valid:
        path_src: {SRC_DATA_NAME}.dev
        path_tgt: {TARGET_DATA_NAME}.dev
        transforms: [filtertoolong]

# Vocabularios (serán generados por `onmt_build_vocab`)
src_vocab: source.vocab
tgt_vocab: target.vocab

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

with open("/config.yaml", "w+") as config_yaml:
  config_yaml.write(config)

# Revisa si la GPU está activa.
!nvidia-smi -L

# Revisa si la GPU es visible para PyTorch

import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

gpu_memory = torch.cuda.mem_get_info(0)
print("Free GPU memory:", gpu_memory[0]/1024**2, "out of:", gpu_memory[1]/1024**2)