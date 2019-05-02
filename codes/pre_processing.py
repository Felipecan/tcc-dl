import os
import random
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from util import split_audio, wav2spectrogram


CLASSES_DF = {
    '1': [], 
    '2': []
}
TABLE_COLUMN = 'Pres, Desvio EAV-G (VGe)'

def get_patient_folder_name(patient_number):
    if(int(patient_number)< 10):                
        folder_name = 'pac00{}'.format(patient_number)                
    elif(int(patient_number) < 100):                
        folder_name = 'pac0{}'.format(patient_number)
    else:
        folder_name = 'pac{}'.format(patient_number)
    return folder_name


def pre_processing(csv_path, path_to_audios_folders):    
    '''
        Description:
            Faz o pré-processamento dos dados contidos no csv para os áudios e em seguida o espetrograma.

        Use:
            pre_processing("/path/to/file.csv", "path/to/the/audios/folders").

        Return:                
            csv_path:
                Caminho até o arquivo csv desejado.
            path_to_audios_folders:
                Caminho até as pastas que contem os áudios. Atenção que os áudios devem tá separados por pasta, nesse caso.
    ''' 
        
    # getting csv path correct
    if(not os.path.isabs(csv_path)):
        dirname, _ = os.path.split(os.path.abspath(__file__))  
        csv_path = os.path.join(dirname, csv_path)

    # getting the better path to patients folders.
    # this path is gonna be used to pre processing the patients for predicting and to splitting audios to training
    if(not os.path.isabs(path_to_audios_folders)):
        dirname, _ = os.path.split(os.path.abspath(__file__))  
        path_to_audios_folders = os.path.normpath(os.path.join(dirname, path_to_audios_folders))

    if(not os.path.isdir(path_to_audios_folders)):
        raise Exception('Path [{}] does not exist... Leaving program.'.format(path_to_audios_folders))
    
    path_to_preprocessed_files = os.path.normpath(os.path.join(path_to_audios_folders, '../pre_processing'))            


    try:        
        csv_file = pd.read_csv(csv_path, sep=',', encoding='utf-8', low_memory=False, dtype='str')         
    except IOError as e:        
        raise Exception('Could not read the file [{}] correctly. Closing program...'.format(csv_path))
                     
    
    # drop all elements with null and all columns that no matters (all columns except NÚMERO PACT and Pres, Desvio EAV-G (VGe)).
    csv_file.dropna(subset=[TABLE_COLUMN], inplace=True)
    csv_file.drop(csv_file.columns.difference(['NÚMERO PACT', TABLE_COLUMN]), axis=1, inplace=True)

    # AQUI DEVE ENTRAR O TRATAMENTO COM AS PASTAS SEM ÁUDIO
    
    # for this pre processing, we're using only deviation presence, deviation 1 and deviation 2. 
    # then, we fill the dictionary with keys to classes with these values
    for key, value in CLASSES_DF.items():
        CLASSES_DF[key] = csv_file[csv_file[TABLE_COLUMN].str.contains(key, case=False)]      

    # getting the smallest dataframe to balanced the classes and select the patients randomly
    len_smallest_df = min([len(value.index) for value in CLASSES_DF.values()])      
    for key, value in CLASSES_DF.items():
        temp_select_list = random.sample(range(0, len(CLASSES_DF[key].index)), len_smallest_df)
        CLASSES_DF[key] = CLASSES_DF[key].iloc[temp_select_list]
    
    
    # separating a set of patient to use on test of predict, that is, use this patient with the complete audio and observe your result.
    for key, value in CLASSES_DF.items():

        temp = random.sample(range(0, len(CLASSES_DF[key].index)), int(len_smallest_df*0.1)) # 10% of patients are for predict test    
        temp_df = CLASSES_DF[key].iloc[temp]
        CLASSES_DF[key] = pd.concat([CLASSES_DF[key], temp_df]) # remove patient to dataframe for training, validation e test
        CLASSES_DF[key].drop_duplicates(subset='NÚMERO PACT', keep=False, inplace=True)

        # -----> coping folders to other directory... it's gonna be used in predict after.                         
        path_predict_deviation = os.path.join(path_to_preprocessed_files, 'predict', '{}'.format(key))
        os.makedirs(path_predict_deviation, exist_ok=True)                
        
        for row in temp_df.itertuples():
            patient_path = os.path.join(path_to_audios_folders,  get_patient_folder_name(row[1]))            
            os.system('cp -r {} {}'.format(patient_path, path_predict_deviation))              


    # --- getting all splited audios from data csv ---        
    pool = multiprocessing.Pool(40)

    try:
        audios_by_class = {}             
        for key, value in CLASSES_DF.items():            
            
            results = []
            for row in value.itertuples():               
                audio_path = os.path.join(path_to_audios_folders,  get_patient_folder_name(row[1]))
                results.append(pool.apply_async(split_audio, args=(audio_path, 200)))

            splitted_audios = [results[i].get(timeout=None) for i in range(len(results))]
            audios_by_class.update({key: sum(splitted_audios, [])})    
    except:
        print('Removing {} folder.'.format(path_to_preprocessed_files))
        os.system('rm -r {}'.format(path_to_preprocessed_files))
        raise Exception('Some unexpected error occurred while processing the audio...')
    print('Áudios obtidos e cortados...')

    # --- saving all spectrograms from audios above ---   
    spectrogram_path = os.path.join(path_to_preprocessed_files, 'spectrograms')  

    try:
        len_smallest_audio = min([len(value) for value in audios_by_class.values()])    
        for key, value in audios_by_class.items():  
            
            os.makedirs(os.path.join(spectrogram_path, '{}'.format(key)), exist_ok=True)      
            for i in range(len_smallest_audio):            
                pool.apply_async(wav2spectrogram, args=(value[i], os.path.join(spectrogram_path, '{}'.format(key), '{}.png'.format(i))))
        
        pool.close()
        pool.join()
    except:
        print('Removing {} folder.'.format(path_to_preprocessed_files))
        os.system('rm -r {}'.format(path_to_preprocessed_files))
        raise Exception('Some unexpected error occurred while generating the spectrograms...')

    print('Espctrogramas das seções dos áudios salvas...')
    print('Espectrogramas salvos em: {}'.format(spectrogram_path))
    return spectrogram_path


if(__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Script pré-processar os áudios. Ele lê o csv, separa os áudios e os converte para espectrogramas.')
    parser.add_argument('--csv', action='store', dest='csv', default='../dados/db-spect.csv', required=True, help='Nome/caminho do .csv com os dados para categorizar.')
    parser.add_argument('--audios', action='store', dest='audio_folders', default='../dados/pac-audios', required=True, help='Nome/caminho da pasta contendo as pastas com os áudios.')
    arguments = parser.parse_args()

    pre_processing(arguments.csv, arguments.audio_folders)

        
