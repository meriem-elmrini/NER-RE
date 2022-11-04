import sys
from typing import IO, Tuple, Callable, Dict, Any, Optional
from functools import partial
from pathlib import Path
from typing import Iterable, Callable
import spacy
from spacy import Language
from spacy.training import Example
from spacy.tokens import DocBin, Doc
import numpy as np

# make the factory work
from scripts.rel_pipe import make_relation_extractor

# make the config work
from scripts.rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors


@spacy.registry.readers("Gold_ents_Corpus.v1")
def create_docbin_reader(file: Path) -> Callable[["Language"], Iterable[Example]]:
    return partial(read_files, file)


def read_files(file: Path, nlp: "Language") -> Iterable[Example]:
    """Custom reader that keeps the tokenization of the gold data,
    and also adds the gold GGP annotations as we do not attempt to predict these."""
    doc_bin = DocBin().from_disk(file)
    docs = doc_bin.get_docs(nlp.vocab)
    for gold in docs:
        pred = Doc(
            nlp.vocab,
            words=[t.text for t in gold],
            spaces=[t.whitespace_ for t in gold],
        )
        pred.ents = gold.ents
        yield Example(pred, gold)
     

@spacy.registry.loggers("my_custom_logger.v1")
def custom_logger(log_path):
    def setup_logger(
        nlp: Language,
        stdout: IO=sys.stdout,
        stderr: IO=sys.stderr
    ) -> Tuple[Callable, Callable]:
        stdout.write(f"Logging to {log_path}\n")
        log_file = Path(log_path).open("w", encoding="utf8")
        log_file.write("step\t")
        log_file.write("f-score\t")
        log_file.write("precision\t")
        log_file.write("recall\t")
        print("epoch\t", "step\t", "f-score\t", "precision\t", "recall\n")
        for pipe in nlp.pipe_names:
            log_file.write(f"loss_{pipe}\t")
        log_file.write("\n")

        def log_step(info: Optional[Dict[str, Any]]):
            if info:
                log_file.write(f"{info['step']}\t")
                log_file.write(f"{info['score']}\t")
                log_file.write(f"{info['other_scores']['rel_micro_p']}\t")
                log_file.write(f"{info['other_scores']['rel_micro_r']}\t")
                print(f"{info['epoch']}\t", f"{info['step']}\t", f"{np.round(info['score'], 2)}\t", 
                      f"{np.round(info['other_scores']['rel_micro_p'], 2)}\t", 
                      f"{np.round(info['other_scores']['rel_micro_r'], 2)}\n")
                for pipe in nlp.pipe_names:
                    log_file.write(f"{info['losses'][pipe]}\t")
                log_file.write("\n")

        def finalize():
            log_file.close()

        return log_step, finalize

    return setup_logger
