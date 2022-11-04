import json


def write_jsonl(df, path):
    our_jsonl = df.to_dict(orient='records')
    with open(path, 'w') as outfile:
        for json_line in our_jsonl:
            json.dump(json_line, outfile)
            outfile.write('\n')
