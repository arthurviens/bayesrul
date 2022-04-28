from pathlib import Path
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


class PredLogger:
    def __init__(self, path):
        self.path = Path(path, 'predictions')
        self.path.mkdir(exist_ok=True)
        self.file_path = Path(self.path, 'preds.npy')

    def save(self, test_preds):
        to_save = np.array([test_preds['preds'], test_preds['labels']])
        np.save(self.file_path, to_save)

    def load(self):
        outputs = np.load(self.file_path)
        return {'preds': outputs[0, :], 'labels': outputs[1, :]}


def plot_rul_pred(out, std=False):
    preds = out['preds']
    labels = out['labels']
    n = len(preds)
    assert n == len(labels), "Inconsistent sizes predictions {}, labels {}"\
        .format(n, len(labels))


    fig = plt.figure(figsize = (15, 5))
    ax = plt.gca()
    plt.plot(range(n), preds, color='blue', label='predicted')
    plt.plot(range(n), labels, color='red', label='actual')
    plt.title("RUL prediction")
    plt.xlabel("Timeline")
    plt.ylabel("RUL")
    plt.grid()
    plt.legend()

    return fig, ax