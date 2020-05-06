#!/usr/bin/env python3

import numpy as np
from authenticator import Authenticator


def main():
    num_subjects = 5
    train_runs = [6, 10]
    auth_runs = [14]
    num_tests = 5

    auth = Authenticator()
    subjects = list(range(1, num_subjects + 1))
    auth.train(subjects, train_runs)

    output = ''

    for subject in subjects:
        _, data, _ = auth.get_user_data(subject, auth_runs, dict(T0=subject, T1=subject, T2=subject))
        labels = auth.authenticate(subject, data)
        unique, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique, counts))

        try:
            confidence = float(label_counts[subject]) / float(len(labels))
        except KeyError:
            confidence = 0.0

        output += '{}: {}\n'.format(subject, confidence)
        output += '{}\n\n'.format(str(label_counts))

    file_name = 'auth_test_{}.txt'.format(str(num_subjects))
    with open(file_name, 'w') as f:
        f.write(output)


if __name__ == '__main__':
    main()
