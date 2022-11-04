import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans

from settings import VERBS
from utils_syntax_re import extract_subj_verb_obj, extract_relations


def get_syntax_predictions(sentence, spans):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    tagged_verbs = get_verbs(doc)
    entities = format_spans(sentence, spans)

    relations = []

    for verb_text, verb_tag, verb_start, verb_end, is_negative in tagged_verbs:
        verb_start = int(verb_start)
        verb_end = int(verb_end)

        triplet = extract_subj_verb_obj(
            verb_text,
            verb_start,
            verb_end,
            verb_tag,
            is_negative,
            doc,
        )

        relation = extract_relations(triplet, entities)

        relations += relation
    return [[[rel['effector'], rel['effectee'], rel['link'].replace('!', '')] for rel in relations]]


def get_verbs(doc, verbs=VERBS):
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)
    pattern = [
        {"DEP": "neg", "OP": "*"},
        {
            "LEMMA": {"IN": verbs},
            "TAG": {"IN": ["VBD", "VBZ", "VBN", "VB"]},
        },
    ]

    matcher.add("bio_verbs_matcher", [pattern])
    matches = matcher(doc)
    return get_tagged_verbs(doc, matches)


def get_tagged_verbs(doc, matches):
    spans = filter_spans([doc[s:e] for _, s, e in matches])

    tagged_verbs = []

    for span in spans:

        if len(span) > 1:
            is_negative = True
        else:
            is_negative = False

        verb = span[-1]
        text = verb.text
        tag = verb.tag_
        start_char = verb.idx
        end_char = start_char + len(text)
        tagged_verbs.append((text, tag, start_char, end_char, is_negative))

    return tagged_verbs


def format_spans(sentence, spans):
    entities = []
    for span in spans:
        start, end, label = span[0], span[1], span[2]
        text = sentence[start: end]
        entities.append({'end': end,
                         'entity_text': text,
                         'entity_type': label,
                         'start': start})
    return entities
