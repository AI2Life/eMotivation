# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:28:14 2021

@author: bianc
"""
import mne
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cwd= os.getcwd()

data_dir= os.path.join(cwd, "DEAP")

sub_list= os.listdir(data_dir)

csv_files=os.path.join(cwd, "altro")
p_r_path=os.path.join(csv_files, "participant_ratings.csv")
video_list_path=os.path.join(csv_files, "video_list.csv")


partecipant_rating=pd.read_csv(p_r_path)
video_list=pd.read_csv(video_list_path)

mapping = { 1 : "A", 2: "B", 3: "C", 4: "D", 5:"E", 7: "F"}
# matrix_data=[]
# X=[]
X=np.empty((40 , 30720 ))
# y=np.empty((40 ,  2 ))
AVG_rating = list()  # y
y=[]

list_channel=['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1',                 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2',                  'Fz', 'Cz', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR2', 'Erg1',                 'Erg2', 'Resp', 'Plet', 'Temp']
# data = mne.io.read_raw_bdf(SOURCE, preload=True)

for i, sub in enumerate(sub_list):
    matrix_data=[]
    

    print("\n sub_num : {}".format(i)) #f fuori da "" e i dentro graffe alternativa
    data_path=os.path.join(data_dir, sub) 
    data = mne.io.read_raw_bdf(data_path, preload=True, exclude=list_channel)
    
    stim_chan = data._data[-1].copy()
    occurence = 0
    index = 0
    # while occurence <= 11:
    #     if stim_chan[index] == 1:
    #         occurence += 1
    #     index += 1
    
    # data=data.crop(tmin=index/512, tmax=len(data)/512-1)
    
    events = mne.find_events(data, stim_channel="Status")

    annotation =  mne.annotations_from_events(events=events, event_desc=mapping,
                                              sfreq=data.info['sfreq'],
                                              orig_time=data.info['meas_date'])
    data.set_annotations(annotation)
    

    datas=data.get_data()
    
    
    for label, onset in zip(annotation.description, annotation.onset):
        if label== "D":
            
            matrix_data.append(datas[:,int(onset*512): int((onset + 60)*512)])
            
    sub_rating = partecipant_rating[partecipant_rating.Participant_id == i+1]
    for row in sub_rating.iterrows():
        
        total_rating = video_list[video_list.Experiment_id == row[1].Experiment_id]
        AVG_rating.append(np.array([total_rating.AVG_Valence.item(),
                                    total_rating.AVG_Arousal.item()]))
        
    # y.append(AVG_rating)
    
       
    
    

    
    # sub_rati = list()  # y2
        
    x=np.stack(matrix_data)
    # X=np.stack( x[:, 0, :], axis=0)
    X=np.concatenate((X, x[:, 0, :]), axis=0)
    if i==20:
        break
    
    
X=X[40:,:]

y=np.array(AVG_rating)






np.save(os.path.join(cwd, "x.npy"), X)
np.save(os.path.join(cwd, "y.npy"), y)




# plt.plot(x[0, 0, :])
# plt.plot(x[0, 1, :])


   
    
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    