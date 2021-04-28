import mne
import os
import matplotlib
matplotlib.use("Tkagg")
import numpy as np
from mne import events_from_annotations
import matplotlib.pyplot as plt

SOURCE = "E:\\datasets\\DEAP\\data_original\\s01.bdf"
data = mne.io.read_raw_bdf(SOURCE, preload=True)

"""
1 (First occurence)	N/A	start of experiment (participant pressed key to start)
1 (Second occurence)	120000 ms(2 m)	start of baseline recording
1 (Further occurences)	N/A	start of a rating screen
2	1000 ms	Video synchronization screen (before first trial, before and after break, after last trial)
3	5000 ms	Fixation screen before beginning of trial
4	60000 ms Start of music video playback
5	3000 ms	Fixation screen after music video playback
7	N/A	End of experiment
"""

stim_chan = data._data[-1].copy()
occurence = 0
index = 0
while occurence <= 11:
    if stim_chan[index] == 1:
        occurence += 1
    index += 1

data.crop(tmin=index/512, tmax=len(data)/512-1)
events = mne.find_events(data, stim_channel="Status")
mapping = { 1 : "A", 2: "B", 3: "C", 4: "D", 5:"E", 7: "F"}
annotation =  mne.annotations_from_events(events=events, event_desc=mapping,
                                          sfreq=data.info['sfreq'],
                                          orig_time=data.info['meas_date'])
data.set_annotations(annotation)

