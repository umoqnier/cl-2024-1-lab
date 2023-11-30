- Son necesarias las herramientas de preprocesamiento y evaluación de los repositorios:
	https://github.com/ymoslem/MT-Preparation.git
	https://github.com/ymoslem/MT-Evaluation.git

- Es necesario instalar lo que sigue:
	OpenNMT-py
	sentencepiece

- Mediante la siguiente línea se instalan los requerimientos para traducción y evaluación:
	pip3 install -r MT-Preparation/requirements.txt
	pip3 install -r MT-Evaluation/requirements.txt

- Los corpus utilizados pertenecen a los repositorios:
	https://raw.githubusercontent.com/AmericasNLP/americasnlp2021/main/data/nahuatl-spanish/train.nah
	https://raw.githubusercontent.com/AmericasNLP/americasnlp2021/main/data/nahuatl-spanish/dev.nah
	https://raw.githubusercontent.com/AmericasNLP/americasnlp2021/main/data/nahuatl-spanish/train.es
	https://raw.githubusercontent.com/AmericasNLP/americasnlp2021/main/data/nahuatl-spanish/dev.es

## Aquí se ejecuta practica8_pt1.py
- A continuación, mediante estas ejecución, se filtran las líneas de los documentos es y nah:
	python3 MT-Preparation/filtering/filter.py es nah es nah

- Luego se entrena un tokenizador a partir de los documentos anteriores:
	python3 MT-Preparation/subwording/1-train_unigram.py es-filtered.es nah-filtered.nah

- Con el modelo anterior se hacen subwords de es-filtered.es y nah-filtered.nah
	python3 MT-Preparation/subwording/2-subword.py source.model target.model es-filtered.es nah-filtered.nah

- Se dividen los archivos es-filtered.es.subword y nah-filtered.nah.subword en corpus de entremamiento, desarrollo y evaluación:
	python3 MT-Preparation/train_dev_split/train_dev_test_split.py 2000 2000 nah-filtered.nah.subword es-filtered.es.subword

## Aquí se ejecuta practica8_pt2.py
 
- Se generan nuevos vocabularios para náhuatl e inglés:
	onmt_build_vocab -config /content/nmt/practica/config.yaml -n_sample -1 -num_threads 2

- Configuración del proceso de entrenamiento:
	onmt_train -config /content/nmt/practica/config.yaml

- Entrenamiento del modelo de traducción:
	onmt_translate -model models/model.enes_step_3000.pt -src nah-filtered.nah.subword.test -output es.practice.translated -gpu 0 -min_length 1

- Estas líneas quitan los subwords de los conjuntos es.practice.translated y es-filtered.es.subword.test: 
	python MT-Preparation/subwording/3-desubword.py target.model es.practice.translated
	python MT-Preparation/subwording/3-desubword.py target.model es-filtered.es.subword.test

- Cálculo de las métircas de evaluación BLEU y chrF2, mediante la línea:
	python evaluate.py --system_output /content/es.practice.translated.desubword --gold_reference /content/es-filtered.es.subword.test.desubword
