from typing import Union
import os
import time
import pandas as pd
from psychopy import visual, gui, core
import numpy as np
from psychopy.core import MonotonicClock
import static


class Stimulation:
    """
    A class to stimulate subjects
    """
    def __init__(self,
                 experiment_name: Union[None, str] = None,
                 curr_stim_type: str = 'images',
                 dataset_name: str ='oasis'):

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

        self.experiment_name = experiment_name if experiment_name is not None \
            else ("experiment_%s_%s" %(time.localtime().tm_hour, time.localtime().tm_min))
        self.curr_stim_type = curr_stim_type
        self.dataset_name = dataset_name
        self.stim_genders = ["male", "female", "nonbin"]
        self.stim_types = ["images", "videos", "audios"]
        self.stim_groups = ['pleasure', 'neutral', 'unpleasure']

        #todo: test with MacOS
        self.meta_path: str = os.path.join(os.environ["USERPROFILE"], ".emot_data") \
            if os.name == "nt" else os.path.join(os.environ["HOME"], ".emot_data")
        if not os.path.exists(self.meta_path):
            static.create_meta_dir(self.meta_path)
        self.config: dict = static.get_config_file(self.meta_path)
        dataset_path = os.path.join(self.config['database_path'], self.dataset_name)
        if not os.path.isdir(dataset_path):
            oasis_paths = static.get_oasis(save_path=self.config['database_path'])
            static.add_config_file_record(self.meta_path, oasis_paths)


        self.build_exp_dirs()


    def build_exp_dirs(self):
        # build current experiment directory
        self.curr_stim_path_dict = {}
        self.curr_exp_dir = self.create_dir(
            folder_root=self.cwd,
            folder_name=self.experiment_name)


    def showImage(self, imageN=0, image_time=6.0):
        """
        show single image
        """
#       stim_keys = []
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
        experiment_name='oasis stimulation',
        curr_stim_type = 'images', 
        dataset_name='oasis',
        group_size = 24)


    # stim.run_experiment()
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
    







