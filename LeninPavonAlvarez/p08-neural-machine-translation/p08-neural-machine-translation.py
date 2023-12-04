from nmt.filtering.filter import prepare
import sentencepiece as spm
from os import rename
from itertools import product
from shutil import copy


import requests # Obtención de corpus

def make_file(name:str, text:str, mode:str = "x") -> bool:
    try:
        f = open(name,mode=mode)
        f.write(text)
        f.close()
        return True
    except:
        return False

def get_corpora(l1: str="nah", l2: str="es", tipos:list = ['train', 'dev', 'test']) -> bool:
    for lang in [l1,l2]:
        for tipo in ['train', 'dev', 'test']:
            try:
                file_name = f"https://raw.githubusercontent.com/AmericasNLP/americasnlp2021/main/data/nahuatl-spanish/{tipo}.{lang}"
                r = requests.get(file_name)
                make_file(f"{tipo}.{lang}",r.text)
            except:
                pass
    # Obteniendo test.nah
    file_name = f"https://raw.githubusercontent.com/AmericasNLP/americasnlp2021/main/test_data/test.nah"
    r = requests.get(file_name)
    make_file(f"test.nah",r.text,mode="w")
    return True

# Código del otro wey que no es el ayudante pero que lo separa raro 
# y por eso necesito ponerlo yo acá
def train_unigram(train_source_file_tok,train_target_file_tok):
    # Source subword model

    source_train_value = '--input='+train_source_file_tok+' --model_prefix=source --vocab_size=50000 --hard_vocab_limit=false --split_digits=true'
    spm.SentencePieceTrainer.train(source_train_value)
    print("Done, training a SentencepPiece model for the Source finished successfully!")


    # Target subword model

    target_train_value = '--input='+train_target_file_tok+' --model_prefix=target --vocab_size=50000 --hard_vocab_limit=false --split_digits=true'
    spm.SentencePieceTrainer.train(target_train_value)
    print("Done, training a SentencepPiece model for the Target finished successfully!")
    
def pre_training(L1:str = "es",L2:str = "nah",tipos:list = ['train','dev','test']):
    # Unigram para train, dev, test 
    for tipo in tipos:
        # Por alguna razón el filtrado elimina
        # todas las líneas de test
        # por lo que no limpiaremos el archivo
        if tipo != 'test':
            prepare(f"{tipo}.{L1}", f"{tipo}.{L2}", L1, L2)
        else:
            copy(f"{tipo}.{L1}",f"{tipo}.{L1}-filtered.{L1}")
            copy(f"{tipo}.{L2}",f"{tipo}.{L2}-filtered.{L2}")
        train_unigram(f"{tipo}.{L1}-filtered.{L1}", f"{tipo}.{L2}-filtered.{L2}")
        # Subword para train, dev, test
        for n1,n2 in product(["source","target"],["model","vocab"]):
            rename(f"{n1}.{n2}",f"{tipo}.{n1}.{n2}")
    return

def subword(source_model,target_model,source_raw,target_raw):
    source_subworded = source_raw + ".subword"
    target_subworded = target_raw + ".subword"

    print("Source Model:", source_model)
    print("Target Model:", target_model)
    print("Source Dataset:", source_raw)
    print("Target Dataset:", target_raw)


    sp = spm.SentencePieceProcessor()


    # Subwording the train source

    sp.load(source_model)

    with open(source_raw, encoding='utf-8') as source, open(source_subworded, "w+", encoding='utf-8') as source_subword:
        for line in source:
            line = line.strip()
            line = sp.encode_as_pieces(line)
            # line = ['<s>'] + line + ['</s>']    # add start & end tokens; optional
            line = " ".join([token for token in line])
            source_subword.write(line + "\n")

    print("Done subwording the source file! Output:", source_subworded)


    # Subwording the train target

    sp.load(target_model)

    with open(target_raw, encoding='utf-8') as target, open(target_subworded, "w+", encoding='utf-8') as target_subword:
        for line in target:
            line = line.strip()
            line = sp.encode_as_pieces(line)
            # line = ['<s>'] + line + ['</s>']    # add start & end tokens; unrequired for OpenNMT
            line = " ".join([token for token in line])
            target_subword.write(line + "\n")

    print("Done subwording the target file! Output:", target_subworded)

if __name__ == '__main__':
    TIPOS = ['train','dev','test']
    L1,L2 = "es","nah"
    # Obtener archivos
    get_corpora()
    pre_training()
    for tipo in TIPOS:
        subword(f"{tipo}.source.model",
                f"{tipo}.target.model",
                f"{tipo}.{L1}-filtered.{L1}",
                f"{tipo}.{L2}-filtered.{L2}"
                )
    # No hacemos split porque los tomamos ya spliteados
    # Entrenamiento modelo unigrama
    SRC_DATA_NAME = f"{L1}-filtered.{L1}.subword"
    TARGET_DATA_NAME = f"{L2}-filtered.{L2}.subword"

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
    make_file("config.yaml", config, "w")
    # TODO: Crear yaml
    # Onmt
    # Vocabulary
    # onmt_build_vocab -config config.yaml -n_sample -1 -num_threads 2
    # Creates source and target vocab
    
       