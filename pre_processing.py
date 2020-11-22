import os
import util
import random
import threading
import pandas as pd
import multiprocessing
import distutils.dir_util as dir_util

AUDIO_DIR = './tcc_files/patients_audios'
POST_PROCESSING_DIR = './tcc_files/post_processing_files'
CSV = './tcc_files/patients_db.csv'

GROUPS = {
    '1': None,
    '2': None
}

TABLE_COLUMN = 'Pres, Desvio EAV-G (VGe)'

PREDICT_PER = 0.3
TIME_STEP_SPLIT = 200

pool = multiprocessing.Pool(40)

def save_files_to_predict(predict_group):

    print('save_files_to_predict routine started...')

    for key, value in predict_group.items():
    
        predict_dir = os.path.join(POST_PROCESSING_DIR, 'predict', '{}'.format(key.replace(' ', '-')))
        os.makedirs(predict_dir, exist_ok=True)

        for row in value.itertuples():

            patient_folder_name = util.get_patient_folder_name(row[1])            
            copy_from = os.path.join(AUDIO_DIR, patient_folder_name)
            paste_to = os.path.join(predict_dir, patient_folder_name)            
            dir_util.copy_tree(copy_from, paste_to)    

def get_audios_and_process():

    print('Getting audios and splitting them...')

    try:

        audios_by_group = {}
        for key, value in GROUPS.items():

            results = []
            for row in value.itertuples():
                audio_path = os.path.join(AUDIO_DIR,  util.get_patient_folder_name(row[1]))
                results.append(pool.apply_async(util.split_audio, args=(audio_path, TIME_STEP_SPLIT)))

            splitted_audios = [results[i].get(timeout=None) for i in range(len(results))]
            audios_by_group.update({key: sum(splitted_audios, [])})
    except:

        pool.terminate()
        print('Removing {} folder.'.format(POST_PROCESSING_DIR))
        dir_util.remove_tree(POST_PROCESSING_DIR)
        raise Exception('Some unexpected error occurred while processing the audio...')

    return audios_by_group

def load_and_process_csv_file():

    try:
        print('Trying read CSV file from: {}'.format(CSV))
        csv_file = pd.read_csv(CSV, sep=',', encoding='utf-8', low_memory=False, dtype='str')
    except IOError:
        raise IOError('Could not read the file [{}] correctly. Closing program...'.format(CSV))

    print('Starting CSV structure creation and clearing data...\n')
    
    csv_file = csv_file.loc[csv_file['NÚMERO PACT'].isin(util.get_all_patients_with_valid_audio(AUDIO_DIR))]     
    csv_file.dropna(subset=[TABLE_COLUMN], inplace=True)
    csv_file.drop(csv_file.columns.difference(['NÚMERO PACT', TABLE_COLUMN]), axis=1, inplace=True)

    for key, _ in GROUPS.items():
        GROUPS[key] = csv_file[csv_file[TABLE_COLUMN].str.contains(key, case=False)]

    # normalizing, randomly by patients, the groups size...
    smallest_group_size = min([len(value.index) for value in GROUPS.values()])
    for key, value in GROUPS.items():
        temp_select_list = random.sample(range(0, len(value.index)), smallest_group_size)
        GROUPS[key] = value.iloc[temp_select_list]

    print('separating train/evaluate dataset from predict dataset...')
    predict_group = GROUPS.copy()

    for key, value in GROUPS.items():

        temp = random.sample(range(0, len(value.index)), int(smallest_group_size * PREDICT_PER)) # 30% of patients are for predict test
        predict_group[key] = value.iloc[temp]

        # remove predict patients from dataframe for training, validation and test
        common = value.merge(predict_group[key], on=['NÚMERO PACT'])        
        GROUPS[key] = GROUPS[key][~GROUPS[key]['NÚMERO PACT'].isin(common['NÚMERO PACT'])]

    print('Starting routine to copy predict files...')
    save_predict_thread = threading.Thread(target=save_files_to_predict, args=(predict_group,))
    save_predict_thread.start()


def generate_spectrograms():

    print('Starting spctogram generate...')

    audios_group = get_audios_and_process()

    spectrogram_path = os.path.join(POST_PROCESSING_DIR, 'spectrograms')
    
    try:
	    # talvez fazer um shuffle nos audios para embaralha-los, fazendo com que não pegue somente audios na sequencia que foram cortados
        for key, value in audios_group.items():
            random.shuffle(audios_group[key])
            
        smallest_audio_group_size = min([len(value) for value in audios_group.values()])

        for key, value in audios_group.items():

            class_path = os.path.join(spectrogram_path, '{}'.format(key.replace(' ', '-')))
            os.makedirs(class_path, exist_ok=True)

            for i in range(smallest_audio_group_size):
                pool.apply_async(util.wav_to_spectrogram, args=(value[i], os.path.join(class_path, '{}.png'.format(i))))

        pool.close()
        pool.join()
    except:
        pool.terminate()
        print('Removing {} folder.'.format(POST_PROCESSING_DIR))        
        dir_util.remove_tree(POST_PROCESSING_DIR)
        raise Exception('Some unexpected error occurred while generating the spectrograms...')
    

def start():

    print('\nStarting pre processing...\n')

    load_and_process_csv_file()
    generate_spectrograms()

    print('\nPre processing finished.\n')



