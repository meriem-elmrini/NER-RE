import json
import pandas as pd


def read_jsonl(data_path):
    with open(data_path, 'r') as f:
        json_list = list(f)
    data = []
    for j in json_list:
        data.append(json.loads(j))
    return pd.DataFrame(data)


data = read_jsonl('../additional_data/test_data/doccano_doc_test.jsonl')
examples = data.sample(10).text.values

with open('../additional_data/test_data/example.txt', 'w') as f:
    f.write(str(examples))
