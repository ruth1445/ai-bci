import mne
import numpy as np

def load_and_preprocess(subject=1, runs=[6, 10]):
    raw = mne.concatenate_raws([
        mne.io.read_raw_edf(f"{data_path}/S{subject:03d}/S{subject:03d}R{run:02d}.edf", preload=True)
        for run in runs
    ])
    
    raw.pick_types(eeg=True)
    raw.filter(7., 30., fir_design='firwin')
    
    events, _ = mne.events_from_annotations(raw)
    event_id = dict(T1=2, T2=3)  # T1: left, T2: right

    epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=2.0, 
                        picks='eeg', baseline=None, preload=True)
    X = epochs.get_data()
    y = epochs.events[:, -1] - 2  # 0 = left, 1 = right
    return raw, X, y


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

