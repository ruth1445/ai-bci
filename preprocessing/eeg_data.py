import mne
import numpy as np

def load_and_preprocess(subject=1, runs=[6, 10], channels=['C3', 'C4'], l_freq=7, h_freq=30, target_freq=128):
    """
    Loads EEGBCI motor imagery data for a given subject and runs,
    applies bandpass filtering and resampling,
    and selects specified channels (default C3 and C4).
    Returns preprocessed MNE Raw object.
    """
    # Download and load data files
    file_paths = mne.datasets.eegbci.load_data(subject, runs)

    # Load and concatenate all runs
    raws = [mne.io.read_raw_edf(fp, preload=True) for fp in file_paths]
    raw = mne.concatenate_raws(raws)

    # Standardize channel names (e.g., "C3." → "C3")
    raw.rename_channels(lambda x: x.strip('.'))

    # Pick desired channels (default: motor cortex)
    raw.pick_channels(channels)

    # Apply bandpass filter
    raw.filter(l_freq, h_freq, fir_design='firwin')

    # Resample to lighter frequency
    raw.resample(target_freq)

    return raw


def extract_epochs_and_labels(raw, tmin=0, tmax=4):
    """
    Extracts motor imagery epochs and binary labels from Raw EEG.
    Left hand → 0, Right hand → 1
    Returns X (data) and y (labels)
    """
    # Get events from annotations
    events, event_id = mne.events_from_annotations(raw)
    print("Event IDs:", event_id)

    # Motor imagery classes: T1 = left (2), T2 = right (3)
    motor_event_id = {'left': 2, 'right': 3}

    # Epoch the data
    epochs = mne.Epochs(
        raw, 
        events, 
        event_id=motor_event_id, 
        tmin=tmin, 
        tmax=tmax, 
        baseline=None, 
        preload=True
    )

    # Get data and labels
    X = epochs.get_data()             # shape: (n_trials, n_channels, n_times)
    y = epochs.events[:, -1]         # 2 or 3

    # Convert to binary: 2 → 0 (left), 3 → 1 (right)
    y = (y == 3).astype(int)

    print("EEG data shape (X):", X.shape)
    print("Binary labels (y):", y)

    return X, y


if __name__ == "__main__":
    # Run everything end-to-end
    raw = load_and_preprocess()
    X, y = extract_epochs_and_labels(raw)

