import mne

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

    # Resample to lighter freq
    raw.resample(target_freq)

    return raw

