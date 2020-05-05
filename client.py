#!/usr/bin/env python3

import socket
import os
import sys
import logging

from mne import Epochs, pick_types, events_from_annotations
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf


def get_data(subject):
    event_id = dict(T0=subject, T1=subject, T2=subject)
    tmin, tmax = -1.0, 4.0

    raw_fnames = eegbci.load_data(subject, [14])
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

    return epochs.get_data


def main():
    if not os.getenv('SOCK_PATH'):
        sock_path = '/tmp/eeg.sock'
    else:
        sock_path = os.getenv('SOCK_PATH')

    user = input('Enter user ID for authentication: ')

    # data = get_data(int(user))

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    try:
        sock.connect(sock_path)
    except socket.error as err:
        print('Error connecting to socket: {}'.format(err))
        sys.exit(1)

    try:
        sock.sendall(bytes(user, 'utf-8'))
        msg = sock.recv(4096)
        print(msg.decode('utf-8'))
    finally:
        sock.close()



if __name__ == '__main__':
    main()
