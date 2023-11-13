import streamlit as st
import sentencepiece as spm
import ctranslate2
from nltk import sent_tokenize

CT2_MODEL_PATH = "models/model_ct2_v2/"
SP_SOURCE_MODEL_PATH = "models/source.model"
SP_TARGET_MODEL_PATH = "models/target.model"


def translate(source: str, translator, sp_source_model, sp_target_model) -> str:
    sentence = sent_tokenize(source)
    print(sentence)
    tokenized_sent = sp_source_model.encode(sentence, out_type=str)
    print(tokenized_sent)
    translations = translator.translate_batch(tokenized_sent)
    print(translations)
    detokenized_translation = sp_target_model.decode([translation[0]["tokens"] for translation in translations])
    print(detokenized_translation)
    return " ".join(detokenized_translation)

def main():
    translator = ctranslate2.Translator(CT2_MODEL_PATH, "cpu")
    sp_source = spm.SentencePieceProcessor(SP_SOURCE_MODEL_PATH)
    sp_target = spm.SentencePieceProcessor(SP_TARGET_MODEL_PATH)

    st.set_page_config(page_title="Neural Machine Translation Demo")

    st.title("Traductor en->es")

    with st.form("my_form"):
        user_input = st.text_area("Source Text", max_chars=200)
        translation = translate(user_input, translator, sp_source, sp_target)

        submitted = st.form_submit_button("Translate")
        if submitted:
            st.write("Translation")
            st.info(translation)

if __name__ == "__main__":
    main()