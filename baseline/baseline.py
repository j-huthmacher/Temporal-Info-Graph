# """
# """
from tqdm import tqdm, trange
import numpy as np

# Sklearn classifier
from sklearn.base import BaseEstimator
# from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


def get_model(model, **params):
    if model == "svm":
        return SGDClassifier(loss='hinge', random_state=0, *params)
    elif model == "logistic regression":
        return SGDClassifier(loss='log', random_state=0, *params)

def train_baseline(data, num_epochs=2, baseline="svm", **baseline_args):
    """ Function to train the selected baseline on the given data.
    """
    train_loader, val_loader = data
    classes = np.unique(np.array(train_loader.dataset)[:,1]).astype('int')

    model = get_model(baseline, *baseline_args)

    if baseline == "decision tree":
        pass    
    else:
        pbar = trange(num_epochs, desc=f'Epochs ({baseline})')
        for epoch in pbar:
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

# # def train_sklearn()
