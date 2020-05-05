#!/usr/bin/env python3

import socket
import sys
import os
import atexit
import logging
import pickle

from authenticator import Authenticator


def train_model(auth):
    '''
    Trains the model on the established EEG data.
    Normally this is where data from a DB or scans would be read in and
    processed but we don't have that so we're pulling from the eegbci dataset.
    
    Parameters:
    auth (Authenticator): The Authenticator object used for authenticating users
    '''

    subjects = list(range(1, 21))
    runs = [6, 10]

    auth.train(subjects, runs)


def authenticate(auth, user):
    '''
    Authenticates a user with the Authenticator object.
    '''


def cleanup(auth, save_path):
    '''
    Cleanup function to be run when the script exits
    '''

    logging.debug('Saving Authenticator to {}'.format(save_path))
    with open(save_path, 'wb') as save_file:
        pickle.dump(auth, save_file)


def main():
    if not os.getenv('SAVE_DIR'):
        save_dir = '/var/local/eeg/'
    else:
        save_dir = os.getenv('SAVE_DIR')

    save_path = os.path.join(save_dir, 'auth.bin')

    if not os.getenv('SOCK_PATH'):
        sock_path = '/tmp/eeg.sock'
    else:
        sock_path = os.getenv('SOCK_PATH')

    # Restore or create the Authenticator object
    if os.path.exists(save_path):
        if not os.path.exists(save_dir):
            logging.debug('Creating {}'.format(save_dir))
            try:
                os.mkdir(save_dir)
            except OSError:
                logging.error('Creation of directory failed')
                sys.exit(1)
        # Restore from file
        logging.info('Saved authenticator found at {}'.format(save_path))
        auth = pickle.load(open(save_path, 'rb'))
    else:
        # If no saved file is found, create a new object and train the model
        logging.info('Saved authenticator not found, creating from scratch')
        auth = Authenticator()
        train_model(auth)

    # Run the cleanup function to save the Authenticator object on exit
    atexit.register(cleanup, auth, save_path)

    # Create the unix socket
    try:
        os.unlink(sock_path)
    except OSError:
        if os.path.exists(sock_path):
            logging.error('Error removing socket')
            sys.exit(1)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    logging.debug('Starting socket at {}'.format(sock_path))
    sock.bind(sock_path)

    sock.listen(1)

    while True:
        conn, addr = sock.accept()

        try:
            logging.info('Accepted connection from {}'.format(addr))

            message = ''
            while True:
                data = conn.recv(1024)

                if data:
                    message += data.decode('utf-8')
                else:
                    break
        finally:
            print(message)
            conn.close()



if __name__ == '__main__':
    main()
