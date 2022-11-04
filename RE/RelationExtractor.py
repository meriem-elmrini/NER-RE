import os
import re
import spacy
from spacy.tokenizer import Tokenizer

from utils.load_yaml import load_yaml_file
from RE.scripts.rel_pipe import make_relation_extractor, score_relations
from RE.scripts.rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors


class RelationExtractor:
    def __init__(self, model='trf'):
        if model == 'trf':
            model_name = 'transformer'
        elif model == 'tok2vec':
            model_name = 'tok2vec'
        else:
            model_name = model
            raise ValueError("Please choose between 'trf' and 'tok2vec' for the model arg")
        self.__model = spacy.load('../NER/training_' + model + '/model-best', exclude=['tagger', 'parser'])
        self.__re_project_config = load_yaml_file('../RE/project.yml')
        self.__re = spacy.load(os.path.join('../RE', self.__re_project_config.vars['trained_' + model]), 
                               exclude=['tagger', 'parser'])
        self.__model.add_pipe(model_name, name="re_" + model, source=self.__re)
        self.__model.add_pipe("relation_extractor", source=self.__re, last=True)

    def get_predictions(self, text: str, threshold=0.01, disable_ner=True, ents=[], log=True):
        if disable_ner:
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(text)
            doc.ents = [doc.char_span(*ent) for ent in ents]
            doc = self.__re(doc)

        else:
            doc = self.__model(text)

        spans = [(e.start, e.text, e.label_) for e in doc.ents]
        if log:
            print(f"spans: {spans}")

        predicted_relations = []
        for value, rel_dict in doc._.rel.items():
            for e in doc.ents:
                for b in doc.ents:
                    if e.start == value[0] and b.start == value[1]:
                        for val in rel_dict.keys():
                                if rel_dict[val] >= threshold:
                                    predicted_relations.append([(e.text, b.text, val, rel_dict[val])])
                                    if log:
                                        print(f"entities: {e.text, b.text} --> predicted relation: {val}")
        return spans, predicted_relations


if __name__ == "__main__":
    extractor = RelationExtractor()
    text = "The formation of a stable tetrameric DCoH-HNF-1 alpha complex, which required the dimerization domain of " \
           "HNF-1 alpha, did not change the DNA binding characteristics of HNF-1 alpha, but enhanced its " \
           "transcriptional activity. "
    extractor.get_predictions(text)
