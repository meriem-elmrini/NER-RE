import sys
from typing import IO, Tuple, Callable, Dict, Any, Optional
import spacy
from spacy import Language
from pathlib import Path
import numpy as np

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
        log_file.write("scores_by_label\t")
        print("epoch\t", "step\t", "f-score\t", "precision\t", "recall\n")
        for pipe in nlp.pipe_names:
            log_file.write(f"loss_{pipe}\t")
        log_file.write("\n")
        def log_step(info: Optional[Dict[str, Any]]):
            if info:
                log_file.write(f"{info['step']}\t")
                log_file.write(f"{info['score']}\t")
                log_file.write(f"{info['other_scores']['ents_p']}\t")
                log_file.write(f"{info['other_scores']['ents_r']}\t")
                log_file.write(f"{info['other_scores']['ents_per_type']}\t")
                print(f"{info['epoch']}\t", f"{info['step']}\t", f"{np.round(info['score'], 2)}\t", 
                      f"{np.round(info['other_scores']['ents_p'], 2)}\t", 
                      f"{np.round(info['other_scores']['ents_r'], 2)}\n")
                for pipe in nlp.pipe_names:
                    log_file.write(f"{info['losses'][pipe]}\t")
                log_file.write("\n")

        def finalize():
            log_file.close()

        return log_step, finalize

    return setup_logger
