import os
import spacy
from utils.load_yaml import load_yaml_file


class EntityExtractor:
    def __init__(self, model='trf'):
        self.__project_config = load_yaml_file('./NER/project.yml')
        self.__ner = spacy.load(os.path.join('./NER', self.__project_config.vars['trained_' + model]))

    def get_predictions(self, text: str, log=True):
        preds = []
        for doc in self.__ner.pipe(text, disable=["tagger", "parser"]):
            preds.append([{'text': ent.text,
                           'start': ent.start_char,
                           'token_start': ent.start, 
                           'token_end': ent.end, 
                           'end': ent.end_char, 
                           'type': 'span', 
                           'label': ent.label_} 
                          for ent in doc.ents])
            if log:
                print([(ent.text, ent.label_) for ent in doc.ents])
        return preds


if __name__ == "__main__":
    extractor = EntityExtractor()
    extractor.get_predictions(text=["CASP1 also activates proinflammatory interleukins, IL1B and IL18, via proteolysis",
                                    "(2) How does the NLRP3 caspase 1 IL 1b axis in the cartilaginous endplates of patients with Modic changes compare with control (trauma) specimens"])
