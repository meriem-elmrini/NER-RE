# NER-RE

This repository aims to implement Named Entity Recognition and Relation Extraction pipelines usind spaCy v3.0.


There are two separate tasks: a Named Entity Recognition task and a Relation Extraction one.

- Two separate directories host the code for these two tasks :
    - NER: Named Entity Recognition Model
    - RE: Relation Extraction Model

For each task, we trained two different models that aim to vectorize the input sentences : a tok2vec model and a pre-trained BioMedNLP-PubMedBert one. 
You can modify the config files to try other models and architectures.


### Important Commands:

The following commands are common to both the models:

- To run the whole flow:
    * `spacy project run all_gpu` (Using GPU)
    * `spacy project run all` (Using CPU)

P.S. : If using CPU, model will not use transformers and will be trained using tok2vec instead.

- All runnable commands are mentioned in project.yaml files

- For training, three annotated files are needed:
    * annotations_train.jsonl
    * annotations_dev.jsonl
    * annotations_test.jsonl

### Note:

Ensure all modules and libraries mentioned in requirements.txt are installed before running the project.