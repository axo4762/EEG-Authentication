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


def main():
    # Retrieve data for subjects
    event_id_1 = dict(T0=1, T1=1, T2=1)
    event_id_2 = dict(T0=2, T1=2, T2=2)
    sfreq_train_1, data_train_1, labels_train_1 = get_eeg_data(1, [6,10], event_id_1)
    sfreq_1, data_1, labels_1 = get_eeg_data(1, 14, event_id_1)
    sfreq_train_2, data_train_2, labels_train_2 = get_eeg_data(2, [6,10], event_id_2)
    sfreq_2, data_2, labels_2 = get_eeg_data(2, 14, event_id_2)

    # Combine training data
    data_train = np.concatenate((data_train_1, data_train_2))
    labels_train = np.concatenate((labels_train_1, labels_train_2))

    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    clf = Pipeline([('CSP', csp), ('LDA', lda)])

    csp.fit_transform(data_train, labels_train)

    cv = ShuffleSplit(10, test_size=0.2, random_state=None)
    cv_split = cv.split(data_train)

    w_length = int(sfreq_train_1 * 0.5)
    w_step = int(sfreq_train_1 * 0.1)
    w_start = np.arange(0, data_train.shape[2] - w_length, w_step)

    scores_windows = []

    for train_idx, test_idx in cv_split:
        y_train, y_test = labels_train[train_idx], labels_train[test_idx]

        X_train = csp.fit_transform(data_train[train_idx], y_train)
        X_test = csp.transform(data_train[test_idx])

        lda.fit(X_train, y_train)

        score_this_window = []
        for n in w_start:
            X_test = csp.transform(data_train[test_idx][:, :, n:(n + w_length)])
            score_this_window.append(lda.score(X_test, y_test))
        scores_windows.append(score_this_window)

    w_times = (w_start + w_length / 2.0) / sfreq_train_1

#    plt.figure()
#    plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
#    plt.axvline(0, linestyle='--', color='k', label='Onset')
#    plt.axhline(0.5, linestyle='-', color='k', label='Chance')
#    plt.xlabel('time (s)')
#    plt.ylabel('classification accuracy')
#    plt.legend(loc='lower right')
#    plt.show()

    X = csp.transform(data_1)
    test = lda.predict(X)
    print(test)


if __name__ == '__main__':
    main()
