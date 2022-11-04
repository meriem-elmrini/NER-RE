import sys
import json
import threading
import _thread as thread

import re
import regex
from itertools import chain
import spacy
from spacy import displacy
import networkx as nx
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords as nltk_en_stopwords


def load_json_from_gcs(gcs_path):
    gcs_file_system = gcsfs.GCSFileSystem(project=project_id)
    with gcs_file_system.open(gcs_path, "r") as f:
        data = json.load(f)
    return data


##############################################################################

labels_targets = (
    "DNA",
    "PROTEIN",
    "RNA",
)
labels_diseases = ("DISEASE",)
stemmer = SnowballStemmer("english")

VERB_TAGS = ["VBD", "VBP", "VBZ", "VB"]


##############################################################################


def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    print("{0} took too long".format(fn_name), file=sys.stderr)
    sys.stderr.flush()  # Python 3 stderr is likely buffered.
    thread.interrupt_main()  # raises KeyboardInterrupt


def exit_after(s):
    """
    use as decorator to exit process if
    function takes longer than s seconds
    """

    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result

        return inner

    return outer


##############################################################################


def clean_text(text):
    if text.startswith("["):
        text = text[1:]

    if text.endswith("]"):
        text = text[:-1]

    text = re.sub(
        "INTRODUCTION|METHODS|CORRIGENDUM|OBJECTIVE|BACKGROUND|RESULTS|PURPOSE|UNASSIGNED|MATERIALS|CONCLUSIONS|STUDY|FINDINGS|TREATMENT|REVIEW|UNLABELLED|SETTING",
        " ",
        text,
    )

    return text


@exit_after(3)
def extract_entity_from_original_text(entity, original_text, entity_type):
    number_errors = entity.count("_")

    if number_errors == 0:
        regular_expression = fr"\b{entity}\b"
        r = regex.compile(regular_expression, flags=regex.IGNORECASE)
        matches = list(r.finditer(original_text))

    else:
        n_matches = 0
        counter = 0

        while n_matches == 0:

            regular_expression = "e<=" + str(number_errors)
            regular_expression = "(" + entity + ")" + "{" + regular_expression + "}"

            r = regex.compile(regular_expression, flags=regex.IGNORECASE)

            matches = list(r.finditer(original_text))

            n_matches = len(matches)
            counter += 1

            number_errors += 1

            if counter == 5:
                break

    output = []

    for match in matches:

        d = {
            "entity_text": match.group(0),
            "start": match.span()[0],
            "end": match.span()[1],
            "entity_type": entity_type.upper(),
        }
        output.append(d)

    return output


def extract_entities(row):
    text = row["text"]
    match = row["match"]

    targets = match["target"]
    diseases = match["disease"]
    action_modes = match["action_mode"]

    targets_from_original = []
    diseases_from_original = []
    action_modes_from_original = []

    for target in targets:
        try:
            output = extract_entity_from_original_text(target, text, "TARGET")
        except:
            output = None

        if (output not in targets_from_original) and (output is not None):
            targets_from_original.append(output)

    for disease in diseases:
        try:
            output = extract_entity_from_original_text(disease, text, "DISEASE")
        except:
            output = None
        if (output not in diseases_from_original) and (output is not None):
            diseases_from_original.append(output)

    for action_mode in action_modes:
        try:
            output = extract_entity_from_original_text(action_mode, text, "ACTION_MODE")
        except:
            output = None
        if (output not in action_modes_from_original) and (output is not None):
            action_modes_from_original.append(output)

    entities = (
        targets_from_original + diseases_from_original + action_modes_from_original
    )

    entities = list(chain(*entities))

    return entities


def merge_entities(all_ents, labels_targets, labels_diseases, entity_type=None):

    if entity_type is not None:
        all_ents = [ent for ent in all_ents if ent["entity_type"] == entity_type]

    if len(all_ents) == 0:
        return []

    all_ents = [
        ent
        for ent in all_ents
        if (
            (ent["entity_type"] in (list(labels_targets) + ["TARGET"]))
            or (ent["entity_type"] in labels_diseases)
        )
    ]

    for ent in all_ents:
        if ent["entity_type"] in labels_targets:
            ent["entity_type"] = "TARGET"

    sorted_entities = sorted(all_ents, key=lambda ent: ent["start"], reverse=False)
    if len(sorted_entities) == 0:
        return []

    kept_entities = [sorted_entities[0]]

    start_init = sorted_entities[0]["start"]
    end_init = sorted_entities[0]["end"]

    for entity in sorted_entities[1:]:
        start = entity["start"]
        end = entity["end"]

        if start > end_init:
            kept_entities.append(entity)
            start_init = entity["start"]
            end_init = entity["end"]

        else:
            len_init = len(kept_entities[-1]["entity_text"])
            len_entity = len(entity["entity_text"])

            if len_entity > len_init:
                kept_entities[-1] = entity
                start_init = entity["start"]
                end_init = entity["end"]

    return kept_entities


def score_predictions(predictions, targets, entity_type):
    predictions = [
        prediction
        for prediction in predictions
        if prediction["entity_type"] == entity_type
    ]
    targets = [target for target in targets if target["entity_type"] == entity_type]

    intersections = []

    for prediction in predictions:
        for target in targets:
            if (prediction == target) and (prediction not in intersections):
                intersections.append(prediction)

    tp = len(intersections)

    fp = len([prediction for prediction in predictions if prediction not in targets])

    fn = len([target for target in targets if target not in predictions])

    try:
        precision = tp / (tp + fp)
    except:
        precision = None

    try:
        recall = tp / (tp + fn)
    except:
        recall = None

    try:
        f1 = (2 * precision * recall) / (precision + recall)
    except:
        f1 = None

    return {
        "p": precision,
        "r": recall,
        "f": f1,
    }


def display_entities(text, entities, entity_type=None):
    nlp = spacy.blank("en")
    doc = nlp.make_doc(text)
    ents = []
    for entity in entities:
        if (entity["entity_type"] == entity_type) or (entity_type is None):
            span_start = entity["start"]
            span_end = entity["end"]
            label = entity["entity_type"]
            ent = doc.char_span(span_start, span_end, label=label)
            if ent is None:
                continue
            ents.append(ent)

    doc.ents = ents

    displacy.render(doc, style="ent", jupyter=True)


def generate_sentence_graph(sentence_spacy: spacy.tokens.doc.Doc) -> nx.Graph:
    sentence_graph = nx.Graph()

    token_to_id = {token: f"{token.text}" for i, token in enumerate(sentence_spacy)}
    for token in sentence_spacy:
        sentence_graph.add_node(token_to_id[token])
        for child in token.children:
            sentence_graph.add_edge(token_to_id[token], token_to_id[child])

    return sentence_graph


def extract_verbs(doc, mappings):
    verbs = []
    for token in doc:
        if token.tag_ in VERB_TAGS:
            text = token.text
            stem = stemmer.stem(text)
            if stem in mappings:
                verbs.append(token)
    return verbs


#####
#  utils for detecting noun chunks


def get_noun_chunks(obj):
    """
    Detect base noun phrases from a dependency parse. Works on both Doc and Span.
    """
    labels = [
        "nsubj",
        "dobj",
        "nsubjpass",
        "pcomp",
        "pobj",
        "dative",
        "appos",
        "attr",
        "ROOT",
        "nmod",
    ]
    doc = obj.doc  # Ensure works on both Doc and Span.
    np_deps = [doc.vocab.strings.add(label) for label in labels]
    conj = doc.vocab.strings.add("conj")
    np_label = doc.vocab.strings.add("NP")
    seen = set()
    for i, word in enumerate(obj):
        if word.pos_ not in ("NOUN", "PROPN", "PRON"):
            continue
        # Prevent nested chunks from being produced
        if word.i in seen:
            continue
        if word.dep in np_deps:
            if any(w.i in seen for w in word.subtree):
                continue
            seen.update(j for j in range(word.left_edge.i, word.i + 1))
            yield word.left_edge.i, word.i + 1, np_label
        elif word.dep == conj:
            head = word.head
            while head.dep == conj and head.head.i < head.i:
                head = head.head
            # If the head is an NP, and we're coordinated to it, we're an NP
            if head.dep in np_deps:
                if any(w.i in seen for w in word.subtree):
                    continue
                seen.update(j for j in range(word.left_edge.i, word.i + 1))
                yield word.left_edge.i, word.i + 1, np_label


def get_noun_phrases(doc):
    noun_phrases = set()
    noun_chunks = get_noun_chunks(doc)
    for s, e, _ in noun_chunks:
        nc = doc[s:e]
        for np in [nc, doc[nc.root.left_edge.i : nc.root.right_edge.i + 1]]:
            noun_phrases.add(np)

    # noun_phrases = {np.text for np in noun_phrases}

    seen_texts = []

    distinct_noun_phrases = []
    for np in noun_phrases:
        text = np.text
        if text not in seen_texts:
            seen_texts.append(text)
            distinct_noun_phrases.append(np)

    return distinct_noun_phrases


#####


def check_if_tag_in_chunk(noun_chunk, tag):
    for token in noun_chunk:
        if token.dep_ == tag:
            return True
    return False


def is_phrase_invalid(noun_phrase):
    if (len(noun_phrase) == 1) and (noun_phrase[0].tag_ in ["PRP", "WDT"]):
        return True
    return False


def keep_longest_noun_phrases(noun_phrases):

    noun_phrases = sorted(noun_phrases, key=lambda n: n.start_char, reverse=False)

    longest = [noun_phrases[0]]
    init_start = longest[-1].start_char
    init_end = longest[-1].end_char

    for noun_phrase in noun_phrases[1:]:
        start = noun_phrase.start_char
        end = noun_phrase.end_char

        if start > init_end:
            longest.append(noun_phrase)
            init_start = start
            init_end = end

        else:
            if len(noun_phrase.text) > len(longest[-1].text):
                longest[-1] = noun_phrase
                init_start = start
                init_end = end

    return longest


def extract_subj_verb_obj(
    verb_text,
    verb_start,
    verb_end,
    verb_tag,
    is_negative,
    doc,
):
    if is_negative == "True":
        is_negative = True
    elif is_negative == "False":
        is_negative = False

    noun_phrases = get_noun_phrases(doc)

    if len(noun_phrases) == 0:
        return {
            "subject": None,
            "verb": verb_text,
            "object": None,
            "subject_start_char": None,
            "subject_end_char": None,
            "object_start_char": None,
            "object_end_char": None,
        }

    noun_phrases = keep_longest_noun_phrases(noun_phrases)

    subjects = []
    objects = []

    for _, noun_phrase in enumerate(noun_phrases):
        start_char = noun_phrase.start_char
        end_char = noun_phrase.end_char

        # active form

        if verb_tag in ["VBD", "VBZ", "VB"]:

            if (end_char < verb_start) and (
                check_if_tag_in_chunk(noun_phrase, "nsubj")
            ):
                subjects.append(noun_phrase)

            elif (start_char >= verb_end) and (
                check_if_tag_in_chunk(noun_phrase, "dobj")
            ):
                objects.append(noun_phrase)

        # passive form

        elif verb_tag in ["VBN"]:

            if (end_char < verb_start) and (
                check_if_tag_in_chunk(noun_phrase, "nsubjpass")
            ):
                objects.append(noun_phrase)

            elif (start_char >= verb_end) and (
                check_if_tag_in_chunk(noun_phrase, "nmod")
            ):
                subjects.append(noun_phrase)

    if verb_tag != "VBN":
        subjects = sorted(subjects, key=lambda n: n.start_char, reverse=True)
        objects = sorted(objects, key=lambda n: n.start_char, reverse=False)
    else:
        subjects = sorted(subjects, key=lambda n: n.start_char, reverse=False)
        objects = sorted(objects, key=lambda n: n.start_char, reverse=True)

    subjects = list(filter(lambda s: not is_phrase_invalid(s), subjects))
    objects = list(filter(lambda o: not is_phrase_invalid(o), objects))

    output = {
        "subject": subjects[0].text if len(subjects) > 0 else None,
        "subject_start_char": subjects[0].start_char if len(subjects) > 0 else None,
        "subject_end_char": subjects[0].end_char if len(subjects) > 0 else None,
        "object_start_char": objects[0].start_char if len(objects) > 0 else None,
        "object_end_char": objects[0].end_char if len(objects) > 0 else None,
        "verb": ("!" if is_negative == True else "") + verb_text,
        "object": objects[0].text if len(objects) > 0 else None,
    }

    return output


def extract_relations(triplet, entities):
    sub = triplet["subject"]
    obj = triplet["object"]
    verb = triplet["verb"]

    subject_start_char = triplet["subject_start_char"]
    subject_end_char = triplet["subject_end_char"]
    object_start_char = triplet["object_start_char"]
    object_end_char = triplet["object_end_char"]

    effectors = []
    effectees = []

    entities = sorted(entities, key=lambda e: e["start"])

    if sub is not None:
        for entity in entities:
            entity_text = entity["entity_text"]
            entity_type = entity["entity_type"]
            start = entity["start"]
            end = entity["end"]
            if (
                (entity_text in sub)
                and (end <= subject_end_char)
                and (subject_start_char <= start)
            ):
                effectors.append((entity_text, sub, entity_type, start, end))

    if obj is not None:
        for entity in entities:
            entity_text = entity["entity_text"]
            entity_type = entity["entity_type"]
            start = entity["start"]
            end = entity["end"]

            if (
                (entity_text in obj)
                and (end <= object_end_char)
                and (object_start_char <= start)
            ):
                effectees.append((entity_text, obj, entity_type, start, end))

    if (sub is not None) and (effectors == []):
        effectors = [(sub, sub, None, subject_start_char, subject_end_char)]

    if (obj is not None) and (effectees == []):
        effectees = [(obj, obj, None, object_start_char, object_end_char)]

    relations = []

    for (
        effector,
        sub_,
        effector_entity_type,
        effector_start_char,
        effector_end_char,
    ) in effectors:
        for (
            effectee,
            obj_,
            effectee_entity_type,
            effectee_start_char,
            effectee_end_char,
        ) in effectees:
            relation = {
                "effector": effector,
                "link": verb,
                "effectee": effectee,
                "subject": sub_,
                "object": obj_,
                "effector_type": effector_entity_type,
                "effectee_type": effectee_entity_type,
                "effector_start_char": effector_start_char,
                "effector_end_char": effector_end_char,
                "effectee_start_char": effectee_start_char,
                "effectee_end_char": effectee_end_char,
            }
            if relation not in relations:
                relations.append(relation)

    return relations


def display_token_deps(doc):
    for token in doc:
        print(token.text, "-->", token.dep_, token.tag_)


def sample_row(df):
    row = df.sample(1).to_dict(orient="records")[0]
    return row


#####

try:
    # If not present, download NLTK stopwords.
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

DEFAULT = set(nltk_en_stopwords.words("english"))


TERMS_TO_DROP = {
    "cell",
    "age",
    "ph",
    "set",
    "nm",
    "c_6",
    "impact",
    "r_1",
    "aim",
    "r_2",
    "c_2",
    "c_3",
    "hr",
    "tert",
    "gt",
    "dd",
    "red",
    "mir",
    "ca_2",
    "met",
    "face",
    "kit",
    "peg",
    "th",
    "cf",
    "ar",
    "cd_4",
    "tnf",
    "ec",
    "tyr",
    "l_1",
    "2_1",
    "ir",
    "gc",
    "ice",
    "pc",
    "coil",
    "c_5",
    "es",
    "rest",
    "t_1",
    "pa",
    "se",
    "nos",
    "t_2",
    "egfr",
    "cat",
    "pe",
    "held",
    "ac",
    "sds",
    "pigs",
    "cs",
    "cp",
    "flap",
    "max",
    "tank",
    "proc",
    "np",
    "camp",
    "mb",
    "si",
    "pt",
    "c_7",
    "hf",
    "trp",
    "mn",
    "cam",
    "akt",
    "tg",
    "hdl",
    "gh",
    "pb",
    "thrombin",
    "ss",
    "myc",
    "psa",
    "p_1",
    "tnf_alpha",
    "d_3",
    "bcl_2",
    "dm",
    "il_4",
    "crp",
    "hrs",
    "cad",
    "her_2",
    "fr",
    "yes",
    "egf",
    "arc",
    "nps",
    "cg",
    "dt",
    "app",
    "p_2",
    "asph",
    "cm",
    "child",
    "mm",
}


WORDS_TO_IGNORE = DEFAULT | TERMS_TO_DROP


def add_abbreviation_ent(matcher, doc, i, matches):
    match = matches[i]
    start = match[1]
    end = match[2]

    # extract the span inside the parenthesis
    span = doc[start:end]
    old_entity = span.ents[0].text
    label = span.ents[0].label_

    abrv_span = doc[start + 2 : end - 1]

    if len(abrv_span.ents) == 0:
        new_entity = doc.char_span(
            abrv_span.start_char,
            abrv_span.end_char,
            label,
        )

        entities = list(doc.ents)
        entities.append(new_entity)

        doc.ents = entities

        
def get_subtrees(node):
    if not node.children:
        return []

    result = list(node.children)
    for child in node.children:
        result.extend(get_subtrees(child))

    return result