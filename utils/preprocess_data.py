import spacy
from spacy.util import filter_spans


# Drop labels that we are not interested in
def drop_label(x, labels):
    return [entity_dict for entity_dict in x if entity_dict['label'] not in labels]


# Preprocess text input
def preprocess_text(string):
    return string.lower()


# Get tokens
def get_tokens(x, i):
    doc = x['doc'].copy()
    text = str(doc[i])
    start = doc[i:i + 1].start_char
    end = doc[i:i + 1].end_char
    if len(str(doc)) > end:
        next_char = str(doc)[end]
    else:
        next_char = ''
    if next_char == ' ':
        ws = True
    else:
        ws = False
    return {'text': text,
            'start': start,
            'end': end,
            'id': i,
            'ws': ws,
            'disabled': True}


# Get spans
def get_spans(x):
    spans = []
    doc = x['doc'].copy()
    for d in x.entities:
        start = d['start_offset']
        end = d['end_offset']
        substring = x['text'][start: end]
        lspaces = len(substring) - len(substring.lstrip())
        rspaces = len(substring) - len(substring.rstrip())
        start += lspaces
        end -= rspaces
        span = doc.char_span(start, end, d['label'], kb_id=d['id'])
        if span is not None:
            spans.append(span)
    filtered_spans = filter_spans(spans)
    return [{'id': span.kb_id,
             'text': str(span),
             'start': span.start_char,
             'token_start': span.start,
             'token_end': span.end,
             'end': span.end_char,
             'type': 'span',
             'label': span.label_}
            for span in filtered_spans]


# Get relations
def get_head_child_attributes(spans, head_entity, child_entity, label):
    head_dicts = [span_dict for span_dict in spans if (span_dict['id'] == head_entity)]
    child_dicts = [span_dict for span_dict in spans if (span_dict['id'] == child_entity)]

    assert len(head_dicts) <= 1
    assert len(child_dicts) <= 1

    if (len(head_dicts) > 0) & (len(child_dicts) > 0):
        head_start = head_dicts[0]['start']
        head_end = head_dicts[0]['end']
        head_token_start = head_dicts[0]['token_start']
        head_token_end = head_dicts[0]['token_end']
        head_label = head_dicts[0]['label']

        child_start = child_dicts[0]['start']
        child_end = child_dicts[0]['end']
        child_token_start = child_dicts[0]['token_start']
        child_token_end = child_dicts[0]['token_end']
        child_label = child_dicts[0]['label']

        head = head_token_end
        child = child_token_end
        label = label

        relations_dict = {'head': head,
                          'child': child,
                          'head_span': {'start': head_start,
                                        'end': head_end,
                                        'token_start': head_token_start,
                                        'token_end': head_token_end,
                                        'label': head_label},
                          'child_span': {'start': child_start,
                                         'end': child_end,
                                         'token_start': child_token_start,
                                         'token_end': child_token_end,
                                         'label': child_label},
                          'label': label}
    else:
        relations_dict = {}
    return relations_dict


def get_relations(x, directed=True, add_no_rel=False, labels_to_drop=['target']):
    relations_form = []
    relations = x['relations'].copy()
    spans = x['spans'].copy()
    remaining_combinations = [(spans[i]['id'], spans[j]['id'])
                              for i in range(len(spans))
                              for j in range(len(spans))
                              if spans[i]['id'] != spans[j]['id']]
    for i in range(len(relations)):
        head_entity = relations[i]['from_id']
        child_entity = relations[i]['to_id']
        rel_label = x['relations'][i]['type'].split('!')[-1]
        if rel_label not in labels_to_drop:
            relations_dict = get_head_child_attributes(spans, head_entity, child_entity, rel_label)
            if relations_dict != {}:
                relations_form.append(relations_dict)
                if not directed:
                    relations_form.append(get_head_child_attributes(spans, child_entity, head_entity,
                                                                    x['relations'][i]['type'].split('!')[-1]))
            try:
                remaining_combinations.remove((head_entity, child_entity))
                if not directed:
                    remaining_combinations.remove((child_entity, head_entity))
            except ValueError:
                pass
    if add_no_rel:
        for c in range(len(remaining_combinations)):
            head_entity = remaining_combinations[c][0]
            child_entity = remaining_combinations[c][1]
            relations_dict = get_head_child_attributes(spans, head_entity, child_entity, 'No-Rel')
            relations_form.append(relations_dict)
    return relations_form


# Delete rows with 0 relation
def delete_rows_no_rel(x):
    to_drop = False
    labels = [x['relations_form'][i]['label'] for i in range(len(x['relations_form']))]
    if (list(set(labels)) == ['No-Rel']) or (len(labels) == 0):
        to_drop = True
    return to_drop


# Preprocess using all the functions above
def preprocess(our_data, drop_labels=[], directed=True, add_no_rel=False, rel_labels_to_drop=['target']):
    nlp = spacy.load('en_core_web_sm')
    # nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)
    if len(drop_labels) > 0:
        our_data['entities'] = our_data['entities'].apply(lambda x: drop_label(x, drop_labels))
    our_data['text_preprocessed'] = our_data['text'].str.lower()
    our_data['doc'] = our_data['text_preprocessed'].apply(lambda x: nlp(x))
    our_data['tokens'] = our_data.apply(lambda x: [get_tokens(x, i) for i in range(len(x.doc))], axis=1)
    our_data['spans'] = our_data.apply(lambda x: get_spans(x), axis=1)
    our_data['relations_form'] = our_data.apply(lambda x: get_relations(x,
                                                                        directed=directed,
                                                                        add_no_rel=add_no_rel,
                                                                        labels_to_drop=rel_labels_to_drop),
                                                axis=1)
    our_data['to_drop'] = our_data.apply(lambda x: delete_rows_no_rel(x), axis=1)
    our_data = our_data[our_data.to_drop == False].copy()
    our_data['answer'] = 'accept'
    our_data['meta'] = None
    our_data['meta'] = our_data['meta'].apply(lambda x: {'source': 'unkown'})
    formatted_data = our_data[['text_preprocessed', 'spans', 'tokens', 'relations_form', 'answer', 'meta']]
    formatted_data.columns = ['text', 'spans', 'tokens', 'relations', 'answer', 'meta']
    return formatted_data
