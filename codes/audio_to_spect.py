import matplotlib.pyplot as plt
from scipy import signal
from pydub import AudioSegment
import numpy as np
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, wait 

def split_audio(path_to_audio, step):
    '''
        Descrição:
            O funçao split_audio() separada um determinado áudio em várias partes.

        Utilização:
            split_audio(/path/to/audio.wav, 1000).

        Parâmetros:                
            path_to_audio:
                Caminho até o arquivo de áudio desejado.
            step:
                Paço a que o áudio será divido, dado em milisegundos. Por exemplo 1000 equivale a 1000 ms, que é 1 s.
    '''   
    audio = AudioSegment.from_wav(path_to_audio)

    audios_list = [audio[start:start+step] for start in range(0, len(audio), step)]
    
    frequencies, times, spectrogram = signal.spectrogram(np.array(audio.get_array_of_samples()), audio.frame_rate)
    # frequencies, times, spectrogram = signal.spectrogram(np.array(audios_list[12].get_array_of_samples()), audio.frame_rate)
    
    plt.pcolormesh(times, frequencies, np.log(spectrogram))    
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.show()


def wav2spectrogram(path_to_file):
    
    audio = AudioSegment.from_wav(path_to_file)

    frequencies, times, spectrogram = signal.spectrogram(np.array(audio.get_array_of_samples()), audio.frame_rate)
    # 20.*np.log10(np.abs(spectrogram)/10e-6) decibel
    # 10.*np.log10(spectrogram)
    plt.pcolormesh(times, frequencies, np.log(spectrogram))
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.box(False)    
    plt.savefig('./teste.png', bbox_inches='tight', pad_inches=0)

def pre_processing(path_to_csv, path_to_audios_folders):
    
    dirname, _ = os.path.split(os.path.abspath(__file__))   

    try:
        csv_file = pd.read_csv(os.path.join(dirname, path_to_csv), sep=',', encoding='utf-8', low_memory=False)
    except IOError as e:
        print('Não foi possivel ler o arquivo[] corretamente. Encerrando programa...')
        print(e)
        return 0  
        
    csv_file.dropna(subset=['Diagnóstico (descritivo)'], inplace=True)
    csv_file.drop(csv_file.columns.difference(['NÚMERO PACT','Diagnóstico (descritivo)']), axis=1, inplace=True)
    
    # healthy_df = csv_file[(csv_file['Diagnóstico (descritivo)'] == 'LARINGE NORMAL') | 
    #                    (csv_file['Diagnóstico (descritivo)'] == 'laringe normal')]      
    healthy_df = csv_file[csv_file['Diagnóstico (descritivo)'].str.contains('LARINGE NORMAL', case=False)]       
    polypo_df = csv_file[csv_file['Diagnóstico (descritivo)'].str.contains('POLIPO', case=False)]                    
    edema_df = csv_file[csv_file['Diagnóstico (descritivo)'].str.contains('EDEMA', case=False)]                   
    nodule_df = csv_file[csv_file['Diagnóstico (descritivo)'].str.contains('NODULO', case=False)]                   
    
    path_spect_cat = os.path.join(dirname, '../dados/spect')    
    path_to_audios_folders = os.path.join(dirname, path_to_audios_folders)    
    # pac_folders = os.listdir(dirname+'/../dados/pac-audios')

    for row in polypo_df.itertuples():
        # print(int(row[1]))
        if(row[1] < 100):
            pac_folder = os.listdir(path_to_audios_folders + '/pac0'+str(int(row[1])))            
    
    
pre_processing('../dados/db-spect.csv', '../dados/pac-audios')
# split_audio(1000)


'''
    1 - Ler o csv (done)
    2 - Tratá-lo para separar todas as categorias e sub-categorias (done)
    3 - A partir de um doença casada com o eavg (qualquer que seja), separar os áudios por pastas: doença -> eavg -> arquivos
    4 - Feito a divisão ou durante a divisão, separar o áudio e converter para o espectograma.
'''


'''
def pre_processing(path_to_csv=None, path_to_audios=None):

    try:
        csv_file = pd.read_csv('../dados/db-spect.csv', sep=',', encoding='utf-8', low_memory=False)
    except IOError as e:
        print('Não foi possivel ler o arquivo[] corretamente. Encerrando programa...')
        print(e)
        return 0  
    
    healthy = csv_file[(csv_file['Diagnóstico (descritivo)'] == 'LARINGE NORMAL') | 
                       (csv_file['Diagnóstico (descritivo)'] == 'laringe normal')]
    print('healthy: ' + str(len(healthy.index)))
    
    healthy_and_eavg_1 = healthy[healthy['Pres, Desvio EAV-G (VGe)'] == 1]
    print('healthy+eav1: ' + str(len(healthy_and_eavg_1.index)))

    healthy_and_eavg_2 = healthy[healthy['Pres, Desvio EAV-G (VGe)'] == 2]
    print('healthy+eav2: ' + str(len(healthy_and_eavg_2.index)))
   
    # EAV-Grau geral (VGe) classif
    for i in range(1, 5):
        healthy_and_eavg = healthy[healthy['EAV-Grau geral (VGe) classif'] == i]
        print('eav ' + str(i) + ': ' + str(len(healthy_and_eavg.index)))
'''