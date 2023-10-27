# Required package installations

- elotl
- nltk
- subword-nmt

# Specifications

1. Models and tokenized files were obtained by the following executions in command line:
	- cess.model:
	  subword-nmt learn-bpe -s 300 < cess_plain.txt > cess.model
	- spa-bible.txt:
	  subword-nmt apply-bpe -c cess.model < spa-bible.txt > spa_bible_tokenized.txt
	- axolotl_vanilla.model:
	  subword-nmt learn-bpe -s 300 < axolotl_plain_vanilla.txt > axolotl_vanilla.model
	- axolotl_vanilla_tokenized.txt:
	  subword-nmt apply-bpe -c axolotl_vanilla.model < axolotl_plain.txt > axolotl_vanilla_tokenized.txt

2. The creations of the prior files is pointed out in practica_5.py using uppercase comments. For example:
	"""HERE, WE TRAIN cess.model""

