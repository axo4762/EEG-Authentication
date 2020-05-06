#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP


def get_eeg_data(subject, runs, event_id):
    tmin, tmax = -1.0, 4.0

    # Get and prepare the raw data
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)
    raw.rename_channels(lambda x: x.strip('.'))
    raw.filter(7.0, 30.0, fir_design='firwin', skip_by_annotation='edge')

    events, _ = events_from_annotations(raw, event_id=event_id)

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                exclude='bads')

    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)

    labels = epochs.events[:, -1]

    epochs_data = epochs.get_data()

    return raw.info['sfreq'], epochs_data, labels


def train(subjects, runs):
    sfreq = 0
    data = []
    labels = []

    # Get all the data
    for subject in subjects:
        print(subject)
        event_id = dict(T0=subject, T1=subject, T2=subject)

        sfreq_subj, data_subj, labels_subj = get_eeg_data(subject, runs, event_id)
        sfreq += sfreq_subj
        data.append(data_subj)
        labels.append(labels_subj)

    sfreq /= len(subjects)

    data = np.concatenate(tuple(data))
    labels = np.concatenate(labels)

    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    clf = Pipeline([('CSP', csp), ('LDA', lda)])

    csp.fit_transform(data, labels)

    cv = ShuffleSplit(10, test_size=0.2, random_state=None)
    split = cv.split(data)

    w_length = int(sfreq * 0.5)
    w_step = int(sfreq * 0.1)
    w_start = np.arange(0, data.shape[2] - w_length, w_step)

    scores_windows = []

    for train_idx, test_idx in split:
        y_train, y_test = labels[train_idx], labels[test_idx]

        X_train = csp.fit_transform(data[train_idx], y_train)
        X_test = csp.transform(data[test_idx])

        lda.fit(X_train, y_train)

        score_this_window = []
        for n in w_start:
            X_test = csp.transform(data[test_idx][:, :, n:(n + w_length)])
            score_this_window.append(lda.score(X_test, y_test))
        scores_windows.append(score_this_window)

    w_times = (w_start + w_length / 2.0) / sfreq

    plt.figure()
    plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
    plt.axvline(0, linestyle='--', color='k')
    plt.axhline(0.5, linestyle='-', color='k', label='Chance')
    plt.xlabel('time (s)')
    plt.ylabel('classification accuracy')
    plt.legend(loc='lower right')
    plt.show()

    return csp, lda


def main():

    num_subjects = 5

    subjects = list(range(1, num_subjects + 1))
    csp, lda = train(subjects, [6, 10])


if __name__ == '__main__':
    main()
