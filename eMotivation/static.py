from typing import Dict
import requests
import os
from tqdm import tqdm
import zipfile
import pandas as pd
import numpy as np
import json

#os.environ["USERPROFILE"]
#os.environ["HOME"]


default_group_values = {'ple_val_min': 4.5,
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
                        'unp_aro_max': 7.0}

def create_meta_dir(path: str):
    os.mkdir(path)
    path_database = os.path.join(path, "database")
    os.mkdir(path_database)
    with open(os.path.join(path, "config.json"), "w") as file:
        json.dump({"database_path": path_database}, file)
    
def get_config_file(meta_path: str):
    with open(os.path.join(meta_path, "config.json"), "rb") as file:
        loaded_file = json.load(file)
    return loaded_file

def add_config_file_record(meta_path: str, to_add : Dict):
    new_config_file = {**to_add, **get_config_file(meta_path=meta_path)}
    with open(os.path.join(meta_path, "config.json"), "w") as file:
        json.dump(new_config_file, file)



def get_oasis(save_path = os.getcwd(), name: str = "oasis"):
    """

    :param save_path:
    :param name:
    :return:
    """
    # definisco un path dove salvo il downlaod
    complete_save_path = os.path.join(save_path, name + ".zip")
    # definsico un path dove estraggo il dataset
    data_path = os.path.join(save_path, name)
    os.mkdir(data_path)
    # download database in zip
    download_link = "http://benedekkurdi.com/oasis.php"
    response = requests.get(download_link, allow_redirects=True, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(complete_save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    #etraggo il file zip precedentemente scaricato in data_path
    with zipfile.ZipFile(complete_save_path, "r") as zip_ref:
        zip_ref.extractall(data_path)
    #elimino il file zip
    os.remove(complete_save_path)
    #aggiusto gli errori nei file csv
    filenames = ["OASIS.csv", "OASIS_bygender_CORRECTED_092617.csv"]
    for file in filenames:
        sheet = pd.read_csv(os.path.join(data_path, file))
        sheet.Theme[864] = sheet.Theme[864].rstrip()
        sheet.Theme[191] = sheet.Theme[191].rstrip()
        sheet.to_csv(os.path.join(data_path, file))
    dic_path = generate_oasis_selection(data_path, seed=42)
    dic_path["oasis_path"] = data_path
    return dic_path


def generate_oasis_selection(dataset_path: str, 
                             seed: int = 42,
                             group_values: Dict = default_group_values):
    
    rng = np.random.default_rng(seed=seed)
    result = list()
    filenames = ["OASIS.csv", "OASIS_bygender_CORRECTED_092617.csv"]
    genders = ["men", "women", "nonbin"]
    neutral_image_range = [group_values[value] for value in group_values if "neu" in value]
    pleasure_image_range = [group_values[value] for value in group_values if "ple" in value]
    unpleasure_image_range = [group_values[value] for value in group_values if "unp" in value]
    gender_df = pd.read_csv(os.path.join(dataset_path, filenames[1]))
    values_df_all = pd.read_csv(os.path.join(dataset_path, filenames[0]))
    img_names = gender_df["Theme"]
    gender_df["Valence_mean_nonbin"] = values_df_all["Valence_mean"]
    gender_df["Arousal_mean_nonbin"] = values_df_all["Arousal_mean"]

    for gender in genders:
        this_gender_stimulus = list()
        for type in [neutral_image_range,  pleasure_image_range, unpleasure_image_range]:
            df_type = list()
            for i in range(len(gender_df)):
                if type[0] <= gender_df["Valence_mean_" + gender][i] < type[1] and type[2] <= \
                        gender_df["Arousal_mean_" + gender][i] < type[3]:
                    df_type.append({
                        'Unnamed: 0': gender_df["Unnamed: 0"][i],
                        'Theme': gender_df["Theme"][i],
                        'Category': gender_df["Category"][i],
                        "Source": gender_df["Source"][i],
                        "Valence_mean_nonbin": gender_df["Valence_mean_nonbin"][i],
                        "Valence_mean_men": gender_df["Valence_mean_men"][i],
                        "Valence_SD_men": gender_df["Valence_SD_men"][i],
                        "Valence_N_men": gender_df["Valence_N_men"][i],
                        "Valence_mean_woman": gender_df["Valence_mean_women"][i],
                        "Valence_SD_women": gender_df["Valence_SD_women"][i],
                        "Valence_N_women": gender_df["Valence_N_women"][i],
                        "Arousal_mean_nonbin": gender_df["Arousal_mean_nonbin"][i],
                        "Arousal_mean_men": gender_df["Arousal_mean_men"][i],
                        "Arousal_SD_men": gender_df["Arousal_SD_men"][i],
                        "Arousal_N_men": gender_df["Arousal_N_men"][i],
                        "Arousal_mean_women": gender_df["Arousal_mean_women"][i],
                        "Arousal_SD_women": gender_df["Arousal_SD_women"][i],
                        "Arousal_N_women": gender_df["Arousal_N_women"][i],
                        "Motivation_state": type,
                        "Path": os.path.join(os.path.join("oasis", "Images") + img_names[i] + ".jpg")})
            type_sample = rng.choice(df_type, 24)
            this_gender_stimulus.append(type_sample)
        result.append(pd.DataFrame([item for sublist in this_gender_stimulus for item in sublist]))

    stimulation_data_path = os.path.join(dataset_path, "stimulation_data")
    os.mkdir(stimulation_data_path)
    path_map = dict()
    for gender, table in zip(genders, result):
        table.to_csv(os.path.join(stimulation_data_path, gender + ".csv"))
        path_map[gender] = os.path.join(stimulation_data_path, gender + ".csv")

    return path_map





