import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from mne.decoding import CSP

def run_csp_svm(X, y, test_size=0.2, n_components=2):
    """
    Applies CSP for feature extraction and trains an SVM classifier.
    
    Parameters:
    - X: EEG data (n_trials, n_channels, n_times)
    - y: Binary labels (0 or 1)
    - test_size: Proportion of data to use as test set
    - n_components: Number of CSP components to keep
    
    Returns:
    - accuracy: Test set accuracy
    - model: Trained CSP + SVM pipeline
    """
    # Train/test split
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

    # Train pipeline
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return acc, clf

