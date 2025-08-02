import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def build_meta_dataset(model, X, y):
    """
    Generate meta-training data: confidence + correctness per trial
    """
    confidences = model.decision_function(X)
    predictions = model.predict(X)
    correct = (predictions == y).astype(int)

    # You can add more features here if you want!
    meta_X = np.vstack([confidences]).T
    meta_y = correct

    return meta_X, meta_y


def train_meta_model(meta_X, meta_y):
    """
    Train a logistic regression to predict correctness
    """
    clf = LogisticRegression()
    clf.fit(meta_X, meta_y)
    return clf


def predict_correctness(meta_model, confidence):
    """
    Predict if a given prediction is likely to be correct
    """
    return meta_model.predict_proba([[confidence]])[0][1]  # probability of being correct

