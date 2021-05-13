import mne
import os
import matplotlib
matplotlib.use("Tkagg") #Tranquillo, nun te sfondo il computer è per plottare la roba meglio!
import numpy as np
from mne import events_from_annotations
import matplotlib.pyplot as plt
import pandas as pd

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
#
# PATH = "E:\\datasets\\DEAP\\data_original"
#
# names = dict()
#
# for sub_index, subject in enumerate(os.scandir(PATH)):
#     data = mne.io.read_raw_bdf(subject.path, preload=True)
#     names[sub_index] = data.ch_names[-1]
#     del data

ch_to_pick = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz',
              'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2',
              'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']

xs = list()
sub_ys = list()
total_ys = list()

PATH = "E:\\datasets\\DEAP\\data_original"
# PATH = "C:\\Users\pietr\Desktop\Master\datasets\DEAP"

for sub_index, subject in enumerate(os.scandir(PATH)):
    if sub_index < 27:
        pass
    else:
        data = mne.io.read_raw_bdf(subject.path, preload=True)
        print()
        print("Processing.. " + str(subject))
        print()
        if sub_index >= 23:
            events = mne.find_events(data, stim_channel=data.ch_names[-1])
            mapping = {1638145: "A", 1638146:"B", 1638147:"C", 1638148:"D", 1638149:"E", 1638151:"F"}
        else:
            events = mne.find_events(data, stim_channel=data.ch_names[-1])
            mapping = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 7: "F"}

        annotation = mne.annotations_from_events(events=events, event_desc=mapping,
                                                 sfreq=data.info['sfreq'],
                                                 orig_time=data.info['meas_date'])
        data.set_annotations(annotation)

        if sub_index == 28:
            print("hereee")
        matrix_data = []
        datas = data.reorder_channels(ch_to_pick).get_data(picks=ch_to_pick)
        for label, onset in zip(annotation.description, annotation.onset):
            if label == "D":
                matrix_data.append(datas[:, int(onset * 512): int((onset + 60) * 512)])

        xs.append(np.stack(matrix_data))  # <-- sta qui

        video_list = pd.read_csv("C:\\Users\\franc_pyl533c\OneDrive\Repository\eMotivation\deap"
                                 "\\video_list.csv")
        # medio

        partecipant_rating = pd.read_csv(
            "C:\\Users\\franc_pyl533c\\OneDrive\\Repository\\eMotivation\\deap\\participant_ratings"
            ".csv")  # prendere i valori di A e V

        sub_rating = partecipant_rating[partecipant_rating.Participant_id == sub_index+1]

        AVG_rating = list()  # y
        sub_rati = list()  # y2

        for row in sub_rating.iterrows():
            total_rating = video_list[video_list.Experiment_id == row[1].Experiment_id]
            AVG_rating.append(np.array([total_rating.AVG_Valence.item(),
                                        total_rating.AVG_Arousal.item()]))
            sub_rati.append(np.array([row[1].Valence, row[1].Arousal]))

        sub_ys.append(np.stack(AVG_rating))
        total_ys.append(np.stack(sub_rati))

        del data
    
import pickle
with open("E:\\datasets\\DEAP\\generated\\xs.pkl", "wb") as file:
    pickle.dump(xs,file)

with open("E:\\datasets\\DEAP\\generated\\sub_ys.pkl", "wb") as file:
    pickle.dump(sub_ys,file)

with open("E:\\datasets\\DEAP\\generated\\total_ys.pkl", "wb") as file:
    pickle.dump(total_ys, file)




import mne
s_29_path = "E:\\datasets\\DEAP\\data_original\\s28.bdf"
s = mne.io.read_raw_bdf(s_29_path, preload=True)

events = mne.find_events(s, stim_channel=s.ch_names[-1])
mapping = {1638145: "A", 1638146:"B", 1638147:"C", 1638148:"D", 1638149:"E", 1638151:"F"}

annotation = mne.annotations_from_events(events=events, event_desc=mapping,
                                                 sfreq=s.info['sfreq'],
                                                 orig_time=s.info['meas_date'])
s.set_annotations(annotation)

import matplotlib
matplotlib.use("Tkagg")


print(s.ch_names)
import matplotlib
matplotlib.use("Qt5Agg")
s.plot()

