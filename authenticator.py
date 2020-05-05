#!/usr/bin/env python3

import logging
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

class Authenticator:
    def __init__(self):
        self.lda = LinearDiscriminantAnalysis()
        self.csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        self.log = logging.getLogger('eeg-authentication')


    def get_user_data(self, subject, runs, event_id):
        self.log.debug('Retrieving data for subject {}, runs {}'.format(subject, runs))

        tmin, tmax = -1.0, 4.0

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


    def train(self, subjects, runs):
        self.log.debug('Beginning training')

        sfreq = 0
        data = []
        labels = []

        for subject in subjects:
            event_id = dict(T0=subject, T1=subject, T2=subject)

            sfreq_subj, data_subj, labels_subj = self.get_user_data(subject, runs, event_id)

            sfreq += sfreq_subj
            data.append(data_subj)
            labels.append(labels_subj)

        sfreq /= len(subjects)

        data = np.concatenate(tuple(data))
        labels = np.concatenate(labels)

        clf = Pipeline([('CSP', self.csp), ('LDA', self.lda)])

        self.csp.fit_transform(data, labels)

        cv = ShuffleSplit(10, test_size=0.2, random_state=None)
        split = cv.split(data)

        w_length = int(sfreq * 0.5)
        w_step = int(sfreq * 0.1)
        w_start = np.arange(0, data.shape[2] - w_length, w_step)

        scores_windows = []

        for train, test in split:
            y_train, y_test = labels[train], labels[test]

            X_train = self.csp.fit_transform(data[train], y_train)
            X_test = self.csp.transform(data[test])

            self.lda.fit(X_train, y_train)

            score_this_window = []
            for n in w_start:
                X_test = self.csp.transform(data[test][:, :, n:(n + w_length)])
                score_this_window.append(self.lda.score(X_test, y_test))
            scores_windows.append(score_this_window)

            # Maybe not worth testing if we're not showing the graph
            # Could put the test results in a log message or something I guess
        self.log.debug('Finished training')


    def authenticate(self, subject, data):
        self.log.debug('Beginning authentication for user {}'.format(subject))

        X = self.csp.transform(data)
        labels = self.lda.predict(X)

        return labels
