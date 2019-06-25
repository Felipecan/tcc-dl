import os
import glob   
import random
import argparse
import pandas as pd
import multiprocessing
import distutils.dir_util as dir_util
from util import split_audio, wav_to_spectrogram


''' 
dictionary that contains the classes to be trained.
example: if the classes are in relation to deviation, use: key: 1, 2, ..., 4...
always check at csv or database the key you will use.
'''
CLASSES_DF = {
    '1': None,
    '2': None
}

'''
table at csv that will be analyzed on pre processing.
always check at csv or database the column you will use.
'''
TABLE_COLUMN = 'Pres, Desvio EAV-G (VGe)'


def get_all_patients_with_valid_audio(path_to_audios_folders):   
    '''
        Description:
            Get all the patients with valid audios, ie, those patients that have in their folders 
            the files of audios qv001.wav or qv012.wav.

        Use:
            get_all_patients_with_valid_audio('path/to/the/audios/folders')

        Parameters:
            path_to_audios_folders:
                Path to the folder containing the audio. 
                NOTE: The audios should be separated by folder, in this case.

        Return:
            A list containing the number of patients selected.
    '''

    patient_qv001 = glob.glob(os.path.join(path_to_audios_folders, 'pac*/qv001.wav'))
    patient_qv012 = glob.glob(os.path.join(path_to_audios_folders, 'pac*/qv012.wav'))

    patient_qv001 = [i.replace('/qv001.wav', '') for i in patient_qv001]
    patient_qv012 = [i.replace('/qv012.wav', '') for i in patient_qv012]
    
    all_patients_valid = patient_qv001.copy()
    all_patients_valid.extend([patient for patient in patient_qv012 if not(patient in patient_qv001)])

    all_patients_valid = [int(''.join(filter(str.isdigit, p))) for p in all_patients_valid]
    all_patients_valid = [str(p) for p in all_patients_valid]

    return all_patients_valid

def get_patient_folder_name(patient_number):
    '''
        Description:
            Given the patient's number, he gets the name of the folder containing his files.

        Use:
            get_patient_folder_name('1')

        Parameters:
            patient_number:
                Patient number recorded in csv or database.           
        
        Return:
            A string containing the name of the patient folder. 
            Example: input: '1' => output: 'pac001'.
    '''

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
            It preprocesses the data contained in the csv for the audios and then the spectrogram.

        Use:
            pre_processing('/path/to/file.csv', 'path/to/the/audios/folders')

        Parameters:
            csv_path:
                Path to the desired csv file.
            path_to_audios_folders:
                Path to the folder containing the audio. 
                NOTE: The audios should be separated by folder, in this case.
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
    except IOError:
        raise IOError('Could not read the file [{}] correctly. Closing program...'.format(csv_path))


    # patients with valid audio are patient that has audio file qv001.wav or qv012.wav in their folders.
    all_patients_valid = get_all_patients_with_valid_audio(path_to_audios_folders)
    csv_file = csv_file.loc[csv_file['NÚMERO PACT'].isin(all_patients_valid)] # seletct only pacients with valid audio

    # drop all elements with null and all columns that no matters.
    csv_file.dropna(subset=[TABLE_COLUMN], inplace=True)
    csv_file.drop(csv_file.columns.difference(['NÚMERO PACT', TABLE_COLUMN]), axis=1, inplace=True)

    # for this pre processing, we're using only deviation presence, deviation 1 and deviation 2.
    # then, we fill the dictionary with keys to classes with these values
    for key, value in CLASSES_DF.items():
        CLASSES_DF[key] = csv_file[csv_file[TABLE_COLUMN].str.contains(key, case=False)]

    # getting the smallest dataframe to balanced the classes and select the patients randomly
    len_smallest_df = min([len(value.index) for value in CLASSES_DF.values()])
    for key, value in CLASSES_DF.items():
        temp_select_list = random.sample(range(0, len(CLASSES_DF[key].index)), len_smallest_df)
        CLASSES_DF[key] = CLASSES_DF[key].iloc[temp_select_list]


    # separating a set of patient to use on test of predict, that is, use this patient with the complete 
    # audio and observe your result.
    for key, value in CLASSES_DF.items():

        temp = random.sample(range(0, len(CLASSES_DF[key].index)), int(len_smallest_df*0.4)) # 30% of patients are for predict test
        temp_df = CLASSES_DF[key].iloc[temp]
        CLASSES_DF[key] = pd.concat([CLASSES_DF[key], temp_df]) # remove patient to dataframe for training, validation and test
        CLASSES_DF[key].drop_duplicates(subset='NÚMERO PACT', keep=False, inplace=True)

        # -----> coping folders to other directory... it's gonna be used in predict after.
        path_predict_deviation = os.path.join(path_to_preprocessed_files, 'predict', '{}'.format(key.replace(' ', '-')))
        os.makedirs(path_predict_deviation, exist_ok=True)

        for row in temp_df.itertuples():
            patient_folder_name = get_patient_folder_name(row[1])            
            patient_path_to_copy = os.path.join(path_to_audios_folders, patient_folder_name)
            patient_path_to_save = os.path.join(path_predict_deviation, patient_folder_name)            
            dir_util.copy_tree(patient_path_to_copy, patient_path_to_save)
    print('csv file cleared...')

    # --- getting all splited audios from data csv ---
    pool = multiprocessing.Pool(40)

    try:
        audios_by_class = {}
        for key, value in CLASSES_DF.items():

            results = []
            for row in value.itertuples():
                audio_path = os.path.join(path_to_audios_folders,  get_patient_folder_name(row[1]))
                results.append(pool.apply_async(split_audio, args=(audio_path, 100)))

            splitted_audios = [results[i].get(timeout=None) for i in range(len(results))]
            audios_by_class.update({key: sum(splitted_audios, [])})
    except:
        pool.terminate()
        print('Removing {} folder.'.format(path_to_preprocessed_files))
        dir_util.remove_tree(path_to_preprocessed_files)
        raise Exception('Some unexpected error occurred while processing the audio...')
    print('Audios obtained and cut...')

    # --- saving all spectrograms from audios above ---
    spectrogram_path = os.path.join(path_to_preprocessed_files, 'spectrograms')

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
        print('Removing {} folder.'.format(path_to_preprocessed_files))        
        dir_util.remove_tree(path_to_preprocessed_files)
        raise Exception('Some unexpected error occurred while generating the spectrograms...')

    print('Spectrograms of the audio sections were saved...')
    print('Spectrograms saved on: {}'.format(spectrogram_path))
    return spectrogram_path


if(__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Script to preprocess the audios. It reads the csv, separates the audios and converts them to spectrograms.')
    parser.add_argument('--csv', action='store', dest='csv', required=True, help='Name/path of .csv with data to categorize.')
    parser.add_argument('--audios', action='store', dest='audios', required=True, help='Name/path of the folder containing the folder with audio.')
    arguments = parser.parse_args()
    
    pre_processing(arguments.csv, arguments.audios)
