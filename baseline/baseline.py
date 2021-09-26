""" Script consisting the baselines for the TIG model.
"""
# pylint: disable=inconsistent-return-statements
from tqdm import trange
import numpy as np

# Sklearn classifier
from sklearn.base import BaseEstimator
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from torch.utils.data import DataLoader


def get_model(model: str, **params):
    """ Function that return a baseline model given a specific name.

        Args:
            model: str
                Model name. Options: ["svm", "logistic regression", "mlp"]
            **params: Dict
                Paramter to configure the baseline model (sklearn model).
        Returns:
            sklearn-Model: The sklearn baseline model.
    """
    params = {**{"random_state": 0}, **params}
    if model == "svm":
        return SGDClassifier(loss='hinge', **params)
    elif model == "logistic regression":
        return SGDClassifier(loss='log', **params)
    elif model == "mlp":
        return MLPClassifier(max_iter=50, **params)

def train_baseline(data: [DataLoader, DataLoader], num_epochs: int = 2,
                   baseline: str = "svm", **baseline_args):
    """ Function to train the selected baseline model on the given data.

        Args:
            data: [DataLoader, DataLoader]
                List of two data loaders, where the first contains the train data and
                the second contains the validation data.
            num_epochs: int
                Number of epochs that are used.
            baseline: str
                Name of the baseline model.  Options: ["svm", "logistic regression", "mlp"]
            **baseline_Args: Dict
                Paramter to configure the baseline model (sklearn model).
    """
    train_loader, val_loader = data
    classes = np.unique(np.array(train_loader.dataset)[:, 1]).astype('int')

    model = get_model(baseline, *baseline_args)

    if baseline == "decision tree":
        pass
    else:
        pbar = trange(num_epochs, desc=f'Epochs ({baseline})')
        for _ in pbar:
            accuracy = []
            val_accuracy = []

            for batch_x, batch_y in train_loader:
                if isinstance(model, BaseEstimator):
                    model.partial_fit(batch_x, batch_y, classes)
                    yhat = model.predict(batch_x)
                    accuracy.append(accuracy_score(batch_y, yhat))
                else:
                    # For pytorch models
                    pass

            for batch_x, batch_y in val_loader:
                if isinstance(model, BaseEstimator):
                    yhat = model.predict(batch_x)
                    val_accuracy.append(accuracy_score(batch_y, yhat))
                else:
                    # For pytorch models
                    pass

            pbar.set_description('Epochs (%s), accuracy: %.2f, val acc: %.2f'
                                 % (baseline, np.mean(accuracy), np.mean(val_accuracy)))
