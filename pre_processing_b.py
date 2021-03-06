import os
import glob   
import random
import argparse
import pandas as pd
import multiprocessing
import distutils.dir_util as dir_util
from util import split_audio, wav_to_spectrogram

TIME_STEP_SPLIT = 200
PER_PREDICT = 0.3

#dictionary that contains the classes to be trained.
CLASSES = {
    '1': None,
    '2': None
}

#table at csv that will be analyzed on pre processing.
TABLE_COLUMN = 'Pres, Desvio EAV-G (VGe)'


global AUDIOS_DIR = './pacientes_audios'
global CSV_DIR = './pacientes.csv'
global SPECT_DIR = './'
global PRE_PROCESSED_DIR = './pre_processed_files'





# def pre_processing(csv_path, path_to_audios_folders):
#     '''
#         Description:
#             It preprocesses the data contained in the csv for the audios and then the spectrogram.

#         Use:
#             pre_processing('/path/to/file.csv', 'path/to/the/audios/folders')

#         Parameters:
#             csv_path:
#                 Path to the desired csv file.
#             path_to_audios_folders:
#                 Path to the folder containing the audio. 
#                 NOTE: The audios should be separated by folder, in this case.
#     '''

#     if(not os.path.isabs(csv_path)):
#         dirname, _ = os.path.split(os.path.abspath(__file__))
#         csv_path = os.path.join(dirname, csv_path)

#     if(not os.path.isabs(path_to_audios_folders)):
#         dirname, _ = os.path.split(os.path.abspath(__file__))
#         path_to_audios_folders = os.path.normpath(os.path.join(dirname, path_to_audios_folders))

#     if(not os.path.isdir(path_to_audios_folders)):
#         raise Exception('Path [{}] does not exist... Leaving program.'.format(path_to_audios_folders))

#     path_to_preprocessed_files = os.path.normpath(os.path.join(path_to_audios_folders, '../pre_processing'))


#     try:
#         csv_file = pd.read_csv(csv_path, sep=',', encoding='utf-8', low_memory=False, dtype='str')
#     except IOError:
#         raise IOError('Could not read the file [{}] correctly. Closing program...'.format(csv_path))


#     all_patients_valid = get_all_patients_with_valid_audio(path_to_audios_folders)
#     csv_file = csv_file.loc[csv_file['NÚMERO PACT'].isin(all_patients_valid)] 


#     csv_file.dropna(subset=[TABLE_COLUMN], inplace=True)
#     csv_file.drop(csv_file.columns.difference(['NÚMERO PACT', TABLE_COLUMN]), axis=1, inplace=True)


#     for key, value in CLASSES_DF.items():
#         CLASSES_DF[key] = csv_file[csv_file[TABLE_COLUMN].str.contains(key, case=False)]


#     len_smallest_df = min([len(value.index) for value in CLASSES_DF.values()])
#     for key, value in CLASSES_DF.items():
#         temp_select_list = random.sample(range(0, len(CLASSES_DF[key].index)), len_smallest_df)
#         CLASSES_DF[key] = CLASSES_DF[key].iloc[temp_select_list]


#     for key, value in CLASSES_DF.items():

#         temp = random.sample(range(0, len(CLASSES_DF[key].index)), int(len_smallest_df*0.4)) # 30% of patients are for predict test
#         temp_df = CLASSES_DF[key].iloc[temp]
#         CLASSES_DF[key] = pd.concat([CLASSES_DF[key], temp_df]) # remove patient to dataframe for training, validation and test
#         CLASSES_DF[key].drop_duplicates(subset='NÚMERO PACT', keep=False, inplace=True)

#         # -----> coping folders to other directory... it's gonna be used in predict after.
#         path_predict = os.path.join(path_to_preprocessed_files, 'predict', '{}'.format(key.replace(' ', '-')))
#         os.makedirs(path_predict, exist_ok=True)

#         for row in temp_df.itertuples():
#             patient_folder_name = get_patient_folder_name(row[1])            
#             patient_path_to_copy = os.path.join(path_to_audios_folders, patient_folder_name)
#             patient_path_to_save = os.path.join(path_predict, patient_folder_name)            
#             dir_util.copy_tree(patient_path_to_copy, patient_path_to_save)
#     print('csv file cleared...')



#     pool = multiprocessing.Pool(40)

#     try:
#         audios_by_class = {}
#         for key, value in CLASSES_DF.items():

#             results = []
#             for row in value.itertuples():
#                 audio_path = os.path.join(path_to_audios_folders,  get_patient_folder_name(row[1]))
#                 results.append(pool.apply_async(split_audio, args=(audio_path, TIME_STEP_SPLIT)))

#             splitted_audios = [results[i].get(timeout=None) for i in range(len(results))]
#             audios_by_class.update({key: sum(splitted_audios, [])})
#     except:
#         pool.terminate()
#         print('Removing {} folder.'.format(path_to_preprocessed_files))
#         dir_util.remove_tree(path_to_preprocessed_files)
#         raise Exception('Some unexpected error occurred while processing the audio...')
#     print('Audios obtained and cut...')



#     spectrogram_path = os.path.join(path_to_preprocessed_files, 'spectrograms')

#     try:
# 	    # talvez fazer um shuffle nos audios para embaralha-los, fazendo com que não pegue somente audios na sequencia que foram cortados
#         for key, value in audios_by_class.items():
#             random.shuffle(audios_by_class[key])
            
#         len_smallest_audio = min([len(value) for value in audios_by_class.values()])
#         for key, value in audios_by_class.items():

#             class_path = os.path.join(spectrogram_path, '{}'.format(key.replace(' ', '-')))
#             os.makedirs(class_path, exist_ok=True)
#             for i in range(len_smallest_audio):
#                 pool.apply_async(wav_to_spectrogram, args=(value[i], os.path.join(class_path, '{}.png'.format(i))))

#         pool.close()
#         pool.join()
#     except:
#         pool.terminate()
#         print('Removing {} folder.'.format(path_to_preprocessed_files))        
#         dir_util.remove_tree(path_to_preprocessed_files)
#         raise Exception('Some unexpected error occurred while generating the spectrograms...')

#     print('Spectrograms of the audio sections were saved...')
#     print('Spectrograms saved on: {}'.format(spectrogram_path))
#     print('Time step: {}'.format(TIME_STEP_SPLIT))
#     print('Number of saved images: {}'.format(len_smallest_audio))
#     print('Number of patients saved for prediction: {}'.format(int(len_smallest_df*0.4)))
#     return spectrogram_path


# if(__name__ == '__main__'):

#     parser = argparse.ArgumentParser(description='Script to preprocess the audios. It reads the csv, separates the audios and converts them to spectrograms.')
#     parser.add_argument('--csv', action='store', dest='csv', required=True, help='Name/path of .csv with data to categorize.')
#     parser.add_argument('--audios', action='store', dest='audios', required=True, help='Name/path of the folder containing the folder with audio.')
#     arguments = parser.parse_args()
    
#     pre_processing(arguments.csv, arguments.audios)



train():

    print('Starting load of files...')
    print('Audios from: {}', AUDIOS_DIR)

    all_patients_valid = get_all_patients_with_valid_audio(AUDIOS_DIR)

    print('Read CSV file from: {}', CSV_DIR)
    print('Starting CSV structure creation and clearing data...')

    try:
        csv_file = pd.read_csv(CSV_DIR, sep=',', encoding='utf-8', low_memory=False, dtype='str')
    except IOError:
        raise IOError('Could not read the file [{}] correctly. Closing program...'.format(CSV_DIR))

    csv_file = csv_file.loc[csv_file['NÚMERO PACT'].isin(all_patients_valid)]    
    csv_file.dropna(subset=[TABLE_COLUMN], inplace=True)
    csv_file.drop(csv_file.columns.difference(['NÚMERO PACT', TABLE_COLUMN]), axis=1, inplace=True)

    for key, value in CLASSES.items():
        CLASSES[key] = csv_file[csv_file[TABLE_COLUMN].str.contains(key, case=False)]

    len_smallest_class = min([len(value.index) for value in CLASSES.values()])
    for key, value in CLASSES_DF.items():
        temp_select_list = random.sample(range(0, len(CLASSES_DF[key].index)), len_smallest_class)
        CLASSES[key] = CLASSES[key].iloc[temp_select_list]

    for key, value in CLASSES.items():

        temp = random.sample(range(0, len(CLASSES[key].index)), int(len_smallest_class*PER_PREDICT)) # 30% of patients are for predict test
        temp_df = CLASSES[key].iloc[temp]
        CLASSES[key] = pd.concat([CLASSES[key], temp_df]) # remove patient to dataframe for training, validation and test
        CLASSES[key].drop_duplicates(subset='NÚMERO PACT', keep=False, inplace=True)

        # -----> coping folders to other directory... it's gonna be used in predict after.
        predict_dir = os.path.join(PRE_PROCESSED_DIR, 'predict', '{}'.format(key.replace(' ', '-')))
        os.makedirs(path_predict, exist_ok=True)

        for row in temp_df.itertuples():
            patient_folder_name = get_patient_folder_name(row[1])            
            path_to_copy = os.path.join(AUDIOS_DIR, patient_folder_name)
            path_to_paste = os.path.join(predict_dir, patient_folder_name)            
            dir_util.copy_tree(path_to_copy, path_to_paste)
    
    print('Files to be used in predict was saved on: {}...', predict_dir)
    print('Structure created and data cleared...')


    print('Starting spctogram generate...')
    pool = multiprocessing.Pool(40)

    print('Getting audios and splitting them...')
    try:

        audios_by_class = {}
        for key, value in CLASSES.items():

            results = []
            for row in value.itertuples():
                audio_path = os.path.join(AUDIOS_DIR,  get_patient_folder_name(row[1]))
                results.append(pool.apply_async(split_audio, args=(audio_path, TIME_STEP_SPLIT)))

            splitted_audios = [results[i].get(timeout=None) for i in range(len(results))]
            audios_by_class.update({key: sum(splitted_audios, [])})
    except:

        pool.terminate()
        print('Removing {} folder.'.format(PRE_PROCESSED_DIR))
        dir_util.remove_tree(PRE_PROCESSED_DIR)
        raise Exception('Some unexpected error occurred while processing the audio...')

    spectrogram_path = os.path.join(PRE_PROCESSED_DIR, 'spectrograms')
    
    try:
	    # talvez fazer um shuffle nos audios para embaralha-los, fazendo com que não pegue somente audios na sequencia que foram cortados
        for key, value in audios_by_class.items():
            random.shuffle(audios_by_class[key])
            
        len_smallest_audio = min([len(value) for value in audios_by_class.values()])
        for key, value in audios_by_class.items():

            class_path = os.path.join(spectrogram_path, '{}'.format(key.replace(' ', '-')))
            os.makedirs(class_path, exist_ok=True)
            for i in range(len_smallest_audio):
                pool.apply_async(wav_to_spectrogram, args=(value[i], os.path.join(class_path, '{}.png'.format(i))))

        pool.close()
        pool.join()
    except:
        pool.terminate()
        print('Removing {} folder.'.format(PRE_PROCESSED_DIR))        
        dir_util.remove_tree(PRE_PROCESSED_DIR)
        raise Exception('Some unexpected error occurred while generating the spectrograms...')

    print('Spectrograms saved on: {}'.format(spectrogram_path))
    print('Time step: {}'.format(TIME_STEP_SPLIT))
    print('Number of saved images: {}'.format(len_smallest_audio))
    print('Number of patients saved for prediction: {}'.format(int(len_smallest_class*PER_PREDICT)))