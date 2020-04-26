#!/usr/bin/env python3

import mne
#import matplotlib
import matplotlib.pyplot as plt

raw_data = mne.io.read_raw_edf("data/s001/S001R04.edf", preload=True)
mne.set_log_level("WARNING")
print(raw_data.info)

raw_data.rename_channels(lambda s: s.strip("."))
#print(mne.channels.get_builtin_montages())
montage = mne.channels.read_montage("standard_1020")
#montage.plot()
#input("Close?")
#print(len(raw_data.ch_names))
raw_data.set_montage(montage)
raw_data.set_eeg_reference("average")

events = mne.find_events(raw_data, initial_event=True, consecutive=True)
plt.plot(raw_data._data[-1])
plt.show()


raw_data.plot(n_channels=64, scalings={"eeg": 75e-6}, events=events, event_color={1: "green", 2: "blue", 3: "red"})
input("Close?")
