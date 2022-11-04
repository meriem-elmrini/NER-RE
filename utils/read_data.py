import gcsfs
import pandas as pd
import json


# Read data
def read_jsonl(data_path):
    with open(data_path, 'r') as f:
        json_list = list(f)
    data = []
    for j in json_list:
        data.append(json.loads(j))
    return pd.DataFrame(data)


def read_jsonl_from_gcs(data_path):
    gcs_file_system = gcsfs.GCSFileSystem()
    with gcs_file_system.open(data_path, 'r', encoding="utf-8") as f:
        json_list = list(f)
    data = []
    for j in json_list:
        data.append(json.loads(j))
    return pd.DataFrame(data)


def read_parquet(data_path):
    return pd.read_parquet(data_path)


def doccano_doc_to_df(df):
    data = df[['doccano_doc']].copy()
    data['entities'] = data['doccano_doc'].apply(lambda x: x['entities'])
    data['text'] = data['doccano_doc'].apply(lambda x: x['text'])
    data['relations'] = data['doccano_doc'].apply(lambda x: x['relations'])
    return data[['text', 'entities', 'relations']]
