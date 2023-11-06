import re

def preprocess_corpus(corpus: list[str]) -> list[list[str]]:
    cleaned = [ delete_scape_sequences(sentence) for sentence in corpus]
    cleaned = [ sentence for sentence in cleaned if not is_empty_sentence(sentence)]
    cleaned = [ preprocess_sentence(sentence) for sentence in cleaned]
    cleaned = [ delete_empty_spaces(sentence) for sentence in cleaned]

    return [ sentence.split() for sentence in cleaned ]

def preprocess_sentence(sentence: str) -> str:
    return "".join([word.lower() for word in sentence if re.match("^(?![0-9]+)[\w\s]+", word)])

def delete_scape_sequences(sentence: str) -> str:
    return sentence.replace("\n", "").replace("\t", "").replace("\r", "")

def is_empty_sentence(sentence: str) -> bool:
    return not sentence.strip()

def delete_empty_spaces(sentece: str) -> str:
    words = sentece.split()
    return " ".join(words)