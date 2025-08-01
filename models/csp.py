import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from mne.decoding import CSP

def run_csp_svm(X, y, test_size=0.2, n_components=2):
    """
    Apply CSP + SVM to EEG data.
    Returns accuracy and fitted pipeline.
    """
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Create CSP instance
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)

    # Create pipeline: CSP → SVM
    clf = Pipeline([
        ('csp', csp),
        ('svm', SVC(kernel='linear', C=1.0, probability=True))
    ])

    # Train
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return acc, clf
