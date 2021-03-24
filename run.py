# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 16:13:48 2021

@author: adria
"""


#%%

from tkinter import Tk
from tkinter.filedialog import askopenfile
import os
import pandas as pd
from psychopy import visual, event, gui, core
import numpy as np
from psychopy.core import MonotonicClock





class Stimulation:
    """
    A class to stimulate subjects
    """
    def __init__(self, 
         exp_name = 'oasis stimulation',
         stim_types= ['images','videos', 'sounds'], 
         curr_stim_type = 'images',
         dataset_name='oasis',
         stim_genders = None, #['male','female']
         group_size = 24,
         stim_groups = ['pleasure', 'neutral', 'unpleasure'],
         group_values = {
            'ple_val_min': 4.5,
            'ple_val_max': 7.0,
            'ple_aro_min': 4.5,
            'ple_aro_max': 7.0,
            'neu_val_min': 3.8,
            'neu_val_max': 4.2,
            'neu_aro_min': 1.0,
            'neu_aro_max': 2.5,
            'unp_val_min': 1.0,
            'unp_val_max': 3.0,
            'unp_aro_min': 4.5,
            'unp_aro_max': 7.0,
            },
         ):
        """
        exp_name : TYPE str, optional
            DESCRIPTION select experiment name
            The default is 'oasis stimulation'.
        stim_types : TYPE list , optional
            DESCRIPTION list of possible stimuli.
            The default is ['images', 'videos', 'sounds'].
        stim_type : TYPE str , optional
            DESCRIPTION select current stimulation type.
            The default is 'images'.
        dataset_name : TYPE str, optional
            DESCRIPTION select dataset folder
            The default is 'oasis'.
        stim_genders : TYPE list , optional
            DESCRIPTION list of stimuli gender selection.
            The default is ['images', 'videos', 'sounds'].
        stim_groups : TYPE list , optional
            DESCRIPTION list of stimuli groups type.
            The default is ['pleasure', 'neutral', 'unpleasure'].
            
        """
        self.exp_name = exp_name
        self.stim_types = stim_types
        self.curr_stim_type = curr_stim_type
        self.dataset_name = dataset_name
        self.stim_genders = stim_genders
        self.stim_groups = stim_groups
        self.group_size = group_size
        self.group_values = group_values
    
        
        
        self.cwd = os.getcwd()
        
        self.build_gen_dirs()
        self.load_dataset()
        self.build_exp_dirs()
        
        
        if self.stim_genders == None:
            
            print(self.values_df)
            
            # select pleasure images
            self.selected_ple = self.values_df.loc[(self.values_df['Valence_mean'] >= self.group_values['ple_val_min']) 
                                                 & (self.values_df['Valence_mean'] <= self.group_values['ple_val_max'])
                                                 & (self.values_df['Arousal_mean'] >= self.group_values['ple_aro_min'])
                                                 & (self.values_df['Arousal_mean'] <= self.group_values['ple_aro_max'])]
            
            self.selected_ple = self.selected_ple.sample(n=self.group_size)
            self.selected_ple['group'] = pd.Series('pleasure', index=self.selected_ple.index)
            
            self.selected_unp = self.values_df.loc[(self.values_df['Valence_mean'] >= self.group_values['unp_val_min']) 
                                                 & (self.values_df['Valence_mean'] <= self.group_values['unp_val_max'])
                                                 & (self.values_df['Arousal_mean'] >= self.group_values['unp_aro_min'])
                                                 & (self.values_df['Arousal_mean'] <= self.group_values['unp_aro_max'])]
            
            self.selected_unp = self.selected_unp.sample(n=self.group_size)
            self.selected_unp['group'] = pd.Series('unpleasure', index=self.selected_unp.index)
            
            self.selected_neu = self.values_df.loc[(self.values_df['Valence_mean'] >= self.group_values['neu_val_min']) 
                                                 & (self.values_df['Valence_mean'] <= self.group_values['neu_val_max'])
                                                 & (self.values_df['Arousal_mean'] >= self.group_values['neu_aro_min'])
                                                 & (self.values_df['Arousal_mean'] <= self.group_values['neu_aro_max'])]
            
            self.selected_neu = self.selected_neu.sample(n=self.group_size)
            self.selected_neu['group'] = pd.Series('neutral', index=self.selected_neu.index) 
            
        
            self.selected_images = pd.concat([ self.selected_ple,  self.selected_unp,  self.selected_neu])
            
            
            self.selected_images['path'] = pd.Series(
                        [os.path.join(self.images_dir, image_name) for image_name in list(self.selected_images['Theme'])],
                        index=self.selected_images.index
                        )
            
            self.selected_images.to_csv(
                os.path.join(
                    self.curr_exp_dir,
                    'selected_images_nogender.csv'
                    )
                )
            
        else:
            
            for gender in self.stim_genders:
            
                # select pleasure images
                self.selected_ple = self.values_df.loc[(self.values_df['Valence_mean_' + gender] >= self.group_values['ple_val_min']) 
                                                     & (self.values_df['Valence_mean_' + gender] <= self.group_values['ple_val_max'])
                                                     & (self.values_df['Arousal_mean_' + gender] >= self.group_values['ple_aro_min'])
                                                     & (self.values_df['Arousal_mean_' + gender] <= self.group_values['ple_aro_max'])]
                
                self.selected_ple = self.selected_ple.sample(n=self.group_size)
                self.selected_ple['group'] = pd.Series('pleasure', index=self.selected_ple.index)
                
                
          
                
                self.selected_ple['path'] = pd.Series('pleasure', index=self.selected_ple.index)
                
                self.selected_unp = self.values_df.loc[(self.values_df['Valence_mean_' + gender] >= self.group_values['unp_val_min']) 
                                                     & (self.values_df['Valence_mean_' + gender] <= self.group_values['unp_val_max'])
                                                     & (self.values_df['Arousal_mean_' + gender] >= self.group_values['unp_aro_min'])
                                                     & (self.values_df['Arousal_mean_' + gender] <= self.group_values['unp_aro_max'])]
                
                self.selected_unp = self.selected_unp.sample(n=self.group_size)
                self.selected_unp['group'] = pd.Series('unpleasure', index=self.selected_unp.index)
                
                self.selected_neu = self.values_df.loc[(self.values_df['Valence_mean_' + gender] >= self.group_values['neu_val_min']) 
                                                     & (self.values_df['Valence_mean_' + gender] <= self.group_values['neu_val_max'])
                                                     & (self.values_df['Arousal_mean_' + gender] >= self.group_values['neu_aro_min'])
                                                     & (self.values_df['Arousal_mean_' + gender] <= self.group_values['neu_aro_max'])]
                
                self.selected_neu = self.selected_neu.sample(n=self.group_size)
                self.selected_neu['group'] = pd.Series('neutral', index=self.selected_neu.index) 
                
            
                self.selected_images = pd.concat([ self.selected_ple,  self.selected_unp,  self.selected_neu])
                
                
                self.selected_images['path'] = pd.Series(
                        [os.path.join(self.images_dir, image_name) for image_name in list(stim.selected_images['Theme'])],
                        index=self.selected_images.index
                        )
                
                
                self.selected_images.to_csv(
                        os.path.join(
                                self.curr_exp_dir,
                                'selected_images_' + gender + '.csv'
                                )
                        )
                        
                
            

     
        
    def build_exp_dirs(self):
        # build current experiment directory
        self.curr_stim_path_dict = {}
        self.curr_exp_dir = self.create_dir(
            folder_root=self.cwd,
            folder_name=self.exp_name
            )

                        
        
        
                
                    
                    
                
                
            
            
        
        
        
        
        
        
        
        
        
    def build_gen_dirs(self):
        # build dirs (first run)
        self.df_dir = os.path.join(self.cwd, 'database')
        if not os.path.exists(self.df_dir):
            print ('first time run: create dirs')
            os.makedirs(self.df_dir)            
#            for stim in self.stim_types:
#                # to update in future version
#                _ = self.create_dir(
#                        folder_root=self.df_dir, 
#                        folder_name=''
                      
                        
            
#    
#    
#    
    
    
    
    
    
    def load_dataset(self):
        """
        load dataset for current stimulation
        """
           
        # load curr stim directory         
        self.curr_stim_dir = os.path.join(
                self.df_dir, 
                self.curr_stim_type
                )  
        if not os.path.exists(self.df_dir):
            print('error: type is stimulation type is not allowed')
        
        # load current dataset
        self.dataset_dir = os.path.join(
            self.curr_stim_dir, 
            self.dataset_name
            )
        if not os.path.exists(self.dataset_dir):
            print('error: the dataset {} does not exists'.format(self.dataset_name))
            
        if self.stim_genders == None:
            self.values_path = os.path.join(
                self.dataset_dir, 
                self.dataset_name +'.csv'
            )
        else:
            self.values_path = os.path.join(
                self.dataset_dir, 
                self.dataset_name + '_gender' +'.csv'
            )

        self.values_df = pd.read_csv(self.values_path)       
        if self.curr_stim_type == 'images':
            self.images_dir = os.path.join(self.dataset_dir, 'Images')
            self.images_list = os.listdir(self.images_dir)
        elif self.curr_stim_type == 'videos':
            self.videos_dir = os.path.join(self.dataset_dir, 'Videos')
            self.videos_list = os.listdir(self.videos_dir)
        
        
#        # experiment stuff

#        
    def showImage(self, imageN=0, image_time=6.0):
        """
        show single image
        """
#        stim_keys = [] 
        currImagePath = self.stim_path_list[imageN]
        currImage = visual.ImageStim(
            self.stim_window, 
            currImagePath
            )
        currImage.draw()
        self.stim_window.flip()
        self.stims_timestamp.append(self.exp_clock.getTime())
        core.wait(image_time)
        
#        while self.exp_clock.getTime() - image_start < image_time:
#            
#            
#        # wait user 
#        while len(stim_keys) == 0:
#            stim_keys = event.getKeys(keyList=['space'])
#            
            
    def show_fixcross(self,fix_time):
        """
        show single image
        """
        fix = visual.GratingStim(
            win=self.stim_window,
            mask="cross", 
            size=0.05,
            pos=[0,0],
            sf=0
            )
        fix.draw()
        self.stim_window.flip()
        self.fix_timestamp.append(self.exp_clock.getTime())
        core.wait(fix_time)
        
#        while len(stim_keys) == 0:
#            stim_keys = event.getKeys(keyList=['space'])
#        
#        
    def serial_images(self, max_images=1, starting_image=20):
        """
        show serial images
        """
        
        self.stim_window_size = [1920,1080]
        self.stim_window = visual.Window(
            size = self.stim_window_size,
            monitor="testMonitor", 
            color='black', 
            units='norm',                                       
            fullscr = False
            )
        curr_image_num = starting_image
        image_counter = 0
        while image_counter < max_images:
            self.showImage(imageN=curr_image_num)
            curr_image_num +=1
            image_counter +=1
        self.stim_window.close()
        
    
    def get_subj_info(self):    
        """
        obtain subject general info
        """
        self.subject_data_dlg = gui.Dlg(title=os.path.basename(self.curr_exp_dir))
        self.subject_data_dlg.addField('Nickname:')
        self.subject_data_dlg.addField('Age:',
                                       choices=range(18,99))
        self.subject_data_dlg.addField('Gender:',
                                       choices=["Male", 
                                                "Female"])
        self.subject_data_dlg.addField('Sexual prefernce:',
                                       choices=["Heterosexual",
                                           #    "Bisexual"
                                                "Homosexual"])
        self.subject_data_show = self.subject_data_dlg.show()
        
    def get_subj_stim(self):
        
        if self.stim_genders == ['male', 'female']:
        
            if (self.subject_data_show[2] == 'Male' \
            and self.subject_data_show[3] == 'Heterosexual') \
            or (self.subject_data_show[2] == 'Female' \
            and self.subject_data_show[3] == 'Homosexual'):  
                
                self.selected_images = pd.read_csv(os.path.join(
                    self.curr_exp_dir,
                    'selected_images_men.csv'
                    )
                )
                
          
               
            elif (self.subject_data_show[2] == 'Male' \
            and self.subject_data_show[3] == 'Homosexual') \
            or (self.subject_data_show[2] == 'Female' \
            and self.subject_data_show[3] == 'Heterosexual'):
    
                self.selected_images = pd.read_csv(os.path.join(
                                    self.curr_exp_dir,
                                    'selected_images_women.csv'
                                    )
                            )
        elif self.stim_genders == None:
            
            self.selected_images = pd.read_csv(os.path.join(
                    self.curr_exp_dir,
                    'selected_images_nogender.csv'
                    )
                )
            
    def build_subj_dir(self):
        
        self.results_dir = self.create_dir(
                folder_root=self.curr_exp_dir, 
                folder_name= 'results'
                )
               
        subj_nick = self.subject_data_show[0]
        
        self.curr_subj_dir = self.create_dir(
                folder_root=self.results_dir,
                folder_name=subj_nick
                )
        
                
            
    
    def run_experiment(
            self, 
            image_time = 6,
            fix_time = 6 
            ):
#        
        
        self.get_subj_info()
        self.build_subj_dir()
        
        if self.stim_genders == None:
            
            self.curr_stims_csv = pd.read_csv(os.path.join(
                self.curr_exp_dir,
                'selected_images_nogender.csv'
                ) 
            )
                
            self.curr_stims_csv = self.curr_stims_csv.sample(frac=1)
            
            self.stim_path_list = list(self.curr_stims_csv['path']) 
            
            
            
            
        #init stim
        if self.curr_stim_type == 'images':
        
            self.stim_window_size = [1920,1080]
            self.stim_window = visual.Window(
                size = self.stim_window_size,
                monitor="testMonitor", 
                color='black', 
                units='norm',                                       
                fullscr = False
                )   
#            
            image_counter = 0
            self.exp_clock = MonotonicClock()
            self.stims_timestamp = []
            self.fix_timestamp = []
            while image_counter < len(self.stim_path_list)*2:
                
                if image_counter % 2 == 0:
                    fix_time = np.random.uniform(fix_time-2,fix_time+2)
                    self.show_fixcross(fix_time)
                    
                else:
                    
                    self.showImage(imageN=image_counter//2)
                image_counter +=1
            self.stim_window.close()
        
        
        
        
            
            
        
        
        
        
        
#        self.get_subj_stim()
        
#        """
#        show serial images
#        """
#        

#            
#            
        
    def create_dir(self,
        folder_root='root', 
        folder_name='name'
        ):
        
        folder_path = os.path.join(folder_root, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print ('first time run: create dir {}'.format(folder_name))
        else:
            print ('run: load dir {}'.format(folder_name))
        return folder_path
        
    
        
    
        
    
        
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
if __name__ == '__main__':
    
    stim = Stimulation(
        exp_name = 'oasis stimulation',
        curr_stim_type = 'images', 
        dataset_name='oasis',
        stim_genders= None, #['men', 'women'],
        stim_groups = ['pleasure', 'neutral', 'unpleasure'],
        group_size = 24,
        group_values = {
            'ple_val_min': 4.5,
            'ple_val_max': 7.0,
            'ple_aro_min': 4.5,
            'ple_aro_max': 7.0,
            'neu_val_min': 3.8,
            'neu_val_max': 4.2,
            'neu_aro_min': 1.0,
            'neu_aro_max': 2.5,
            'unp_val_min': 1.0,
            'unp_val_max': 3.0,
            'unp_aro_min': 4.5,
            'unp_aro_max': 7.0,
            }
        )
    
    stim.run_experiment()
#    a = stim.values_df
  
    
#    stim.serial_images(
#        max_images=5,
#        starting_image=20,
#        )
    
#    
    
#        
#        
#        

#    
#
#    selected_male_neu = values_df[
#        values_df['Valence_mean_men'] >= neutral_image_range[0]
#            ]
#    
#    selected_male_neu = selected_male_neu[
#        values_df['Valence_mean_men'] < neutral_image_range[1]
#            ]
#    
#    selected_male_neu = selected_male_neu[
#        values_df['Arousal_mean_men'] >= neutral_image_range[2]
#            ]
#    
#    selected_male_neu = selected_male_neu[
#        values_df['Arousal_mean_men'] < neutral_image_range[3]
#            ]
#    
#    male_neutral_image_id = np.array([selected_male_neu['Theme']]).flatten()
#    
#    male_neutral_images_idx = np.random.choice(
#        len(male_neutral_image_id),
#        groups_size,
#        replace = False
#        )
    

    
    # 1. Creare metodo build exp dir (più compatto possibile)
    # 2. Selezionare immagini positive neutre e negative per male e female
    # 3. dall'esercizio 2 creare un metodo della classe
    







