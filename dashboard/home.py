import streamlit as st


def write_home():
    st.markdown('This tool aims to perform relation extraction given a sentence, a text vectorization model,'
                ' and some hyperparameters.'
                '\n\nThe text vectorization models that are available are **Tok2Vec** and **PubmedBERT**.'
                '\n\nYou can choose whether to perform Named Entity Recognition to find the different entities in '
                'the text using the same chosen vectorization model, or to enter these entities manually. '
                'When entering entities, please make sure to type them correctly.'
                '\n\nYou can also try different threshold values.')
