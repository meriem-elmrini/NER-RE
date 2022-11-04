import os
import subprocess
import configparser
import numpy as np
from bayes_opt import BayesianOptimization
from scripts.evaluate import main as score


def black_box_function(batch_size, lr, epochs):
    trf_config = 'configs/rel_trf.cfg'
    train_file = 'data/train.spacy'
    dev_file = 'data/dev.spacy'
    test_file = 'data/test.spacy'
    batch_size = int(batch_size)
    epochs = int(epochs)
    # Change config file according to hyperparameters
    file = configparser.RawConfigParser()
    file.optionxform = str # Keep case
    file.read(trf_config)
    file.set('nlp', 'batch_size', str(batch_size))
    file.set('training.optimizer.learn_rate', 'initial_rate', str(lr))
    file.set('training', 'max_epochs', str(epochs))
    new_cfg_suffix = str(batch_size) + '_' + str(np.round(lr, 5)) + '_' + str(epochs)
    new_trf_config = 'configs/rel_trf_' + new_cfg_suffix + '.cfg'
    with open(new_trf_config, 'w') as f:
        file.write(f)
    # Train model
    training_cmd = "python -m spacy train " + new_trf_config + " --output training_" + new_cfg_suffix + \
                   " --paths.train " + train_file + \
                   " --paths.dev " + dev_file + " -c ./scripts/custom_functions.py --gpu-id 0 "
    subprocess.run(training_cmd, shell=True)
    trained_model = os.path.join('./training_' + new_cfg_suffix, 'model-last')
    # Evaluate
    results = score(trained_model, test_file, False)
    f_scores = {}
    for d in results:
        f_scores[d] = float(results[d]['rel_micro_f'])
    key_max = max(f_scores, key=f_scores.get)
    f = f_scores[key_max]
    # Remove config and trained model folder
    subprocess.run('rm ' + new_trf_config)
    subprocess.run('rm -rf ' + trained_model)
    return f


if __name__ == '__main__':
    pbounds = {"batch_size": [10, 1000],
               "lr": [2e-5, 5e-5],
               "epochs": [500, 2000]}
    optimizer = BayesianOptimization(f=black_box_function,
                                     pbounds=pbounds, verbose=2,
                                     random_state=4)
    optimizer.maximize(n_iter=10, acq="ei", xi=1e-1)
    