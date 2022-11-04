import sys

sys.path.append('../RE')

import streamlit as st
import re
import spacy
from spacy import displacy

from settings import SENTENCES, ENTITIES
from syntax_model import get_syntax_predictions
from RE.RelationExtractor import RelationExtractor


def define_inputs():
    example_text = """Saposin c, in a cell type-specific manner, upregulates upa/upar and immediate early gene c-jun 
    expression, stimulates cell proliferation, migration and invasion and activates p42/44 and sapk/jnk mapk pathways 
    in prostate stromal and cancer cells """
    type_or_select = st.radio('Type a sentence or select an example', ['Type my sentence', 'Select an example'])
    if type_or_select == 'Type my sentence':
        input_sentence = st.text_area('Enter your sentence here', value=example_text)
    else:
        input_sentence = st.selectbox('Select a sentence', SENTENCES)

    input_entities = st.radio('Do you want to compute NER or manually specify your entities ?',
                              ['Compute NER', 'Type entities', 'Select entities'])
    if input_entities == 'Type entities':
        type_entities = True
        entities = st.text_input('Type your entities here')
        entities = [ent.strip() for ent in entities.split(',')]
        spans = get_spans(input_sentence, entities)
    elif input_entities == 'Select entities':
        type_entities = True
        entities = st.multiselect('Select your entities', ENTITIES)
        spans = get_spans(input_sentence, entities)
    else:
        type_entities = False
        spans = None
    st.markdown('**Your sentence :**')
    display_spans(input_sentence, spans)
    return input_sentence, type_entities, spans


def get_spans(sentence, entities):
    spans = []
    if len(entities) > 0:
        for entity in entities:
            matches = [x.span() for x in re.finditer(entity, sentence)]
            starts = list(list(zip(*matches))[0])
            ends = list(list(zip(*matches))[1])
            labels = ['ENT'] * len(starts)
            spans += list(zip(starts, ends, labels))
        return spans
    else:
        return None


def get_prediction(model, sentence, type_entities=False, spans=None, threshold=0.4):
    if model != 'syntax':
        rel = RelationExtractor(model=model)
        if type_entities:
            predicted_spans, predicted_relations = rel.get_predictions(text=sentence,
                                                                       threshold=threshold,
                                                                       disable_ner=True,
                                                                       ents=spans)
        else:
            predicted_spans, predicted_relations = rel.get_predictions(text=sentence,
                                                                       threshold=threshold,
                                                                       disable_ner=False)
    else:
        if not type_entities or not spans:
            st.write('**This model does not have a NER module. Please enter entities manually.**')
            predicted_spans = None
            predicted_relations = None
        else:
            predicted_spans = spans
            predicted_relations = get_syntax_predictions(sentence, spans)
    return predicted_spans, predicted_relations


def display_spans(sentence, spans):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    doc.ents = tuple()
    if spans:
        for span in spans:
            try:
                doc.ents += (doc.char_span(span[0], span[1], span[2]),)
            except TypeError:
                pass
    colors = {"ENT": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
    options = {"ents": ["ENT"], "colors": colors}
    ent_html = displacy.render(doc, style="ent", jupyter=False, options=options)
    st.markdown(ent_html, unsafe_allow_html=True)
    return
