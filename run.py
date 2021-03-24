# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 16:13:48 2021

@author: lore
"""

# %%

from tkinter import Tk
from tkinter.filedialog import askopenfile
import os
import pandas as pd
from psychopy import visual, event, gui
import numpy as np


class Stimulation:
    """
    A class to stimulate subjects
    """

    def __init__(self,
                 exp_name='oasis stimulation',
                 stim_types=['images', 'videos', 'sounds'],
                 curr_stim_type='images',
                 dataset_name='oasis',
                 stim_genders=None,  # ['male','female']
                 group_size=24,
                 stim_groups=['pleasure', 'neutral', 'unpleasure'],
                 group_values={
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

        #if self.stim_genders == None:
            # devo fargli legere, in questo caso, il database oasis.csv

            #self.selection()

            # print(self.values_df)

            # # select pleasure images
            # self.selected_ple = self.values_df.loc[(self.values_df['Valence_mean_man'] >= self.group_values['ple_val_min']) 
            #                                       & (self.values_df['Valence_mean_man'] <= self.group_values['ple_val_max'])
            #                                       & (self.values_df['Arousal_mean_man'] >= self.group_values['ple_aro_min'])
            #                                       & (self.values_df['Arousal_mean_man'] <= self.group_values['ple_aro_max'])]

            # self.selected_ple = self.selected_ple.sample(n=self.group_size)
            # self.selected_ple['group'] = pd.Series('pleasure', index=self.selected_ple.index)

            # self.selected_unp = self.values_df.loc[(self.values_df['Valence_mean'] >= self.group_values['unp_val_min']) 
            #                                       & (self.values_df['Valence_mean'] <= self.group_values['unp_val_max'])
            #                                       & (self.values_df['Arousal_mean'] >= self.group_values['unp_aro_min'])
            #                                       & (self.values_df['Arousal_mean'] <= self.group_values['unp_aro_max'])]

            # self.selected_unp = self.selected_unp.sample(n=self.group_size)
            # self.selected_unp['group'] = pd.Series('unpleasure', index=self.selected_unp.index)

            # self.selected_neu = self.values_df.loc[(self.values_df['Valence_mean'] >= self.group_values['neu_val_min']) 
            #                                       & (self.values_df['Valence_mean'] <= self.group_values['neu_val_max'])
            #                                       & (self.values_df['Arousal_mean'] >= self.group_values['neu_aro_min'])
            #                                       & (self.values_df['Arousal_mean'] <= self.group_values['neu_aro_max'])]

            # self.selected_neu = self.selected_neu.sample(n=self.group_size)
            # self.selected_neu['group'] = pd.Series('neutral', index=self.selected_neu.index) 

            # self.selected_images = pd.concat([ self.selected_ple,  self.selected_unp,  self.selected_neu])

            # self.selected_images['path'] = pd.Series(
            #             [os.path.join(self.images_dir, image_name) for image_name in list(stim.selected_images['Theme'])],
            #             index=self.selected_images.index
            #             )

            # self.selected_images.to_csv(os.path.join(self.curr_exp_dir, 'selected_images_nogender.csv') )

        # else:

        # for gender in self.stim_genders:

        #     # select pleasure images
        #     self.selected_ple = self.values_df.loc[(self.values_df['Valence_mean_' + gender] >= self.group_values['ple_val_min'])
        #                                           & (self.values_df['Valence_mean_' + gender] <= self.group_values['ple_val_max'])
        #                                           & (self.values_df['Arousal_mean_' + gender] >= self.group_values['ple_aro_min'])
        #                                           & (self.values_df['Arousal_mean_' + gender] <= self.group_values['ple_aro_max'])]

        #     self.selected_ple = self.selected_ple.sample(n=self.group_size)
        #     self.selected_ple['group'] = pd.Series('pleasure', index=self.selected_ple.index)

        #     self.selected_ple['path'] = pd.Series('pleasure', index=self.selected_ple.index)

        #     self.selected_unp = self.values_df.loc[(self.values_df['Valence_mean_' + gender] >= self.group_values['unp_val_min'])
        #                                           & (self.values_df['Valence_mean_' + gender] <= self.group_values['unp_val_max'])
        #                                           & (self.values_df['Arousal_mean_' + gender] >= self.group_values['unp_aro_min'])
        #                                           & (self.values_df['Arousal_mean_' + gender] <= self.group_values['unp_aro_max'])]

        #     self.selected_unp = self.selected_unp.sample(n=self.group_size)
        #     self.selected_unp['group'] = pd.Series('unpleasure', index=self.selected_unp.index)

        #     self.selected_neu = self.values_df.loc[(self.values_df['Valence_mean_' + gender] >= self.group_values['neu_val_min'])
        #                                           & (self.values_df['Valence_mean_' + gender] <= self.group_values['neu_val_max'])
        #                                           & (self.values_df['Arousal_mean_' + gender] >= self.group_values['neu_aro_min'])
        #                                           & (self.values_df['Arousal_mean_' + gender] <= self.group_values['neu_aro_max'])]

        #     self.selected_neu = self.selected_neu.sample(n=self.group_size)
        #     self.selected_neu['group'] = pd.Series('neutral', index=self.selected_neu.index)

        #     self.selected_images = pd.concat([ self.selected_ple,  self.selected_unp,  self.selected_neu])

        #     self.selected_images['path'] = pd.Series([os.path.join(self.images_dir, image_name) for image_name in list(stim.selected_images['Theme'])],
        #             index=self.selected_images.index)

        #     self.selected_images.to_csv(os.path.join(self.curr_exp_dir,'selected_images_' + gender + '.csv') )

        # self.selected_image

        # selected_unp = gen_values_df.loc[(gen_values_df['Valence_mean_' + self.gender] >= self.val_range[4])
        #                           & (gen_values_df['Valence_mean_' + self.gender] <= self.val_range[5])
        #                           & (gen_values_df['Arousal_mean_' + self.gender] >= self.aro_range[4])
        #                           & (gen_values_df['Arousal_mean_' + self.gender] <= self.aro_range[5])]

        # select pleasure images

    def selection(self):

        tipo = self.stim_groups  # sei neu, piacevole o spiacevole?
        gender = self.stim_genders  # sei maschio o femmina o none?
        
        self.lista = list()
        values_df = self.values_df

        for j in range(len(tipo)):
            if tipo[j] == self.stim_groups[1]:
                motivation = 'neutral'

            elif tipo[j] == self.stim_groups[0]:
                motivation = 'pleasure'

            elif tipo[j] == self.stim_groups[2]:
                motivation = 'unpleasure'

        for i in range(len(values_df)):
            
            if gender == None:
                if tipo[0] <= values_df["Valence_mean_"][i] < tipo[1] and tipo[2] <= values_df["Arousal_mean_"][i] < tipo[3]:
                    self.lista_no_gender.append({
                            'Theme': values_df['Theme'][i],
                            "Valence_mean_men": values_df["Valence_mean_men"][i],
                            "Valence_SD_men": values_df["Valence_SD_men"][i],
                            "Valence_N_men": values_df["Valence_N_men"][i],
                            "Valence_mean_woman": values_df["Valence_mean_women"][i],
                            "Valence_SD_women": values_df["Valence_SD_women"][i],
                            "Valence_N_women": values_df["Valence_N_women"][i],
                            "Arousal_mean_men": values_df["Arousal_mean_men"][i],
                            "Arousal_SD_men": values_df["Arousal_SD_men"][i],
                            "Arousal_N_men": values_df["Arousal_N_men"][i],
                            "Arousal_mean_women": values_df["Arousal_mean_women"][i],
                            "Arousal_SD_women": values_df["Arousal_SD_women"][i],
                            "Arousal_N_women": values_df["Arousal_N_women"][i],
                            "Motivation_state": motivation,
                            #"Path": "D:/gitkranen/eMotivation/database/images/oasis/" + values_df["Theme"][i] + ".jpg"
                    })

            elif tipo[0] <= values_df["Valence_mean_" + gender][i] < tipo[1] and tipo[2] <= values_df["Arousal_mean_" + gender][i] < tipo[3]:
                
                self.lista.append({
                    # 'Unnamed: 0': values_df["Unnamed: 0"][i],  id foto ma non serve  
                    'Theme': values_df["Theme"][i],  # nomi foto
                    # 'Category': values_df["Category"][i], # categoria foto ma non serve
                    # "Source": values_df["Source"][i], # dove prendiamo le foto ma non ci serve
                    "Valence_mean_men": values_df["Valence_mean_men"][i],
                    "Valence_SD_men": values_df["Valence_SD_men"][i],
                    "Valence_N_men": values_df["Valence_N_men"][i],
                    "Valence_mean_woman": values_df["Valence_mean_women"][i],
                    "Valence_SD_women": values_df["Valence_SD_women"][i],
                    "Valence_N_women": values_df["Valence_N_women"][i],
                    "Arousal_mean_men": values_df["Arousal_mean_men"][i],
                    "Arousal_SD_men": values_df["Arousal_SD_men"][i],
                    "Arousal_N_men": values_df["Arousal_N_men"][i],
                    "Arousal_mean_women": values_df["Arousal_mean_women"][i],
                    "Arousal_SD_women": values_df["Arousal_SD_women"][i],
                    "Arousal_N_women": values_df["Arousal_N_women"][i],
                    "Motivation_state": motivation,  # discrimina foto se sono ple, unple o neu
                     #"Path": "D:/gitkranen/eMotivation/database/images/oasis/" + values_df["Theme"][i] + ".jpg"  # dove sta la foto nel nostro computer
                })

        df_gender = pd.DataFrame(self.lista)
        df_gender = df_gender.sample(n=24)
        
        
        df_no_gender = pd.DataFrame(self.lista_no_gender)
        df_no_gender = df_no_gender.sample(n=24)
        
        return df_gender
        return df_no_gender

    # def n_path(self, dove, nome):
        
    #     self.dove = dove
    #     self.nome = nome
    #     n_path = os.path.join(dove, nome)
    #     if not os.path.exists(n_path):
    #         os.makedirs(n_path)

    def build_exp_dirs(self):
        
        # build current experiment directory
        
        self.curr_stim_path_dict = {}
        self.curr_exp_dir = self.create_dir(folder_root = self.cwd, folder_name = self.exp_name)

        # self.selected_stim = self.create_dir(folder_root=self.curr_exp_dir, folder_name = 'selected_stim_dir')
        
        # if self.stim_genders == None:
        #         self.gender_stim = self.create_dir(folder_root=self.selected_stim, folder_name= 'no_gender_diff')
    
        # else:
        #     for gender in self.stim_genders:
        #         self.gender_stim = self.create_dir(folder_root=self.selected_stim, folder_name=gender)
        #         for stim_group in self.stim_groups:
        #             self.group_stim = self.create_dir(folder_root=self.gender_stim, folder_name=stim_group)
        #             self.curr_stim_path_dict[gender + '_' + stim_group] =  self.group_stim
    

    def build_gen_dirs(self):
        
        # build dirs (first run)
        
        self.df_dir = os.path.join(self.cwd, 'database')
        
        if not os.path.exists(self.df_dir):
            print('first time run: create dirs')
            os.makedirs(self.df_dir)
            
        #            for stim in self.stim_types:

    #                # to update in future version
    #                _ = self.create_dir(folder_root=self.df_dir, folder_name='')
    

    def load_dataset(self):
        
        """
        load dataset for current stimulation
        """

        # load curr stim directory         
        self.curr_stim_dir = os.path.join(self.df_dir, self.curr_stim_type)
        if not os.path.exists(self.df_dir):
            raise Exception('error: type is stimulation type is not allowed')

        # load current dataset
        self.dataset_dir = os.path.join(self.curr_stim_dir, self.dataset_name)
        if not os.path.exists(self.dataset_dir):
            raise Exception('error: the dataset {} does not exists'.format(self.dataset_name))

        if self.stim_genders == None:
            self.values_path = os.path.join(self.dataset_dir, self.dataset_name + '.csv')

        else:
            self.values_path = os.path.join(self.dataset_dir, self.dataset_name + '_gender' + '.csv')

        self.values_df = pd.read_csv(self.values_path)

        if self.curr_stim_type == 'images':
            self.images_dir = os.path.join(self.dataset_dir, 'Images')
            self.images_list = os.listdir(self.images_dir)
        elif self.curr_stim_type == 'videos':
            self.videos_dir = os.path.join(self.dataset_dir, 'Videos')
            self.videos_list = os.listdir(self.videos_dir)

    #        # experiment stuff

    #
    #    def showImage(self, imageN=0):
    #        """
    #        show single image
    #        """
    #        stim_keys = []
    #        currImagePath = os.path.join(
    #                self.imagesDir,
    #                self.imagesList[imageN]
    #                )
    #
    #        # present start message
    #        currImage = visual.ImageStim(
    #            self.stim_window,
    #            currImagePath
    #            )
    #        currImage.draw()
    #        self.stim_window.flip()
    #        # wait user
    #        while len(stim_keys) == 0:
    #            stim_keys = event.getKeys(keyList=['space'])
    #
    #
    def serial_images(self, max_images=1, starting_image=20):
        
        """
        show serial images
        """

        self.stim_window_size = [1920, 1080]
        self.stim_window = visual.Window(size=self.stim_window_size, monitor="testMonitor",
                                         color='black', units='norm', fullscr=False)
        curr_image_num = starting_image
        image_counter = 0
        while image_counter < max_images:
            self.showImage(imageN=curr_image_num)
            curr_image_num += 1
            image_counter += 1
        self.stim_window.close()

    def get_subj_info(self):
        
        """
        obtain subject general info
        """

        self.subject_data_dlg = gui.Dlg(title=os.path.basename(self.curr_exp_dir))

        self.subject_data_dlg.addField('Nickname:')

        self.subject_data_dlg.addField('Age:', choices=range(18, 30))

        self.subject_data_dlg.addField('Gender:', choices = ["Male", "Female", 'None'])

        self.subject_data_dlg.addField('Sexual prefernce:', choices=["Heterosexual","Bisexual","Homosexual"])
        
        self.subject_data_show = self.subject_data_dlg.show()

    def get_subj_stim(self):

        if self.stim_genders == ['male', 'female']:

            if (self.subject_data_show[2] == 'Male' and self.subject_data_show[3] == 'Heterosexual') or (self.subject_data_show[2] == 'Female' \
            and self.subject_data_show[3] == 'Homosexual'):

                self.selected_images = pd.read_csv(os.path.join(self.curr_exp_dir, 'selected_images_men.csv'))



            elif (self.subject_data_show[2] == 'Male' and self.subject_data_show[3] == 'Homosexual') or (self.subject_data_show[2] == 'Female' \
            and self.subject_data_show[3] == 'Heterosexual'):

                self.selected_images = pd.read_csv(os.path.join(self.curr_exp_dir, 'selected_images_women.csv'))


        elif self.stim_genders == None:
            
            self.selected_images = pd.read_csv(os.path.join(self.curr_exp_dir, 'selected_images_nogender.csv'))
            
            #if not os.path.exists(self.selected_images):
                #os.makedirs(self.selected_images)


    def create_dir(self, folder_root='root', folder_name='name'):

        folder_path = os.path.join(folder_root, folder_name)
        
        if not os.path.exists(folder_path):
            
            os.makedirs(folder_path)
            
            print('first time run: create dir {}'.format(folder_name))
        else:
            
            
            print('run: load dir {}'.format(folder_name))
        return folder_path    

    

    def run_experiment(self):

        self.get_subj_info()
        self.get_subj_stim()
        self.selection()

    #        """
    #        show serial images
    #        """
    #
    #        if self.curr_stim_type == 'images':
    #
    #            self.stim_window_size = [1920,1080]
    #            self.stim_window = visual.Window(
    #                size = self.stim_window_size,
    #                monitor = "testMonitor",
    #                color = 'black',
    #                units = 'norm',
    #                fullscr = False
    #                )
    #
    #            image_counter = 0
    #            while image_counter < max_images:
    #                self.showImage(imageN = curr_image_num)
    #                curr_image_num += 1
    #                image_counter += 1
    #            self.stim_window.close()

    
    
    
    


if __name__ == '__main__':
    
    
    stim = Stimulation(
        exp_name = 'oasis_stimulation',
        curr_stim_type = 'images',
        dataset_name = 'oasis',
        stim_genders = None,  # ['men', 'women'],
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
            'unp_aro_max': 7.0, })

    dataframe = stim.values_df
    stim.run_experiment()
    #df_men= stim.selection(tipo = 'neutral', gender = 'men')
#


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
