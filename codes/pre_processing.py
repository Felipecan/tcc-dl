import os
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from scipy import signal
from pydub import AudioSegment
import matplotlib.pyplot as plt


def split_audio(path_to_audio, step):
    '''
        Descrição:
            O funçao split_audio() separada um determinado áudio em várias partes.

        Utilização:
            split_audio("/path/to/audio", 1000).

        Parâmetros:                
            path_to_audio:
                Caminho até a pasta do áudio desejado.
            step:
                Paço a que o áudio será divido, dado em milisegundos. Por exemplo 1000 equivale a 1000 ms, que é 1 s.

        Retorno:
            Um lista de áudios que contem em cada posição, uma lista de áudios divididos. 
            Ex: audio[0] pode conter uma lista com 10 áudios da mesma fonte.
    '''   
    
    files = os.listdir(path_to_audio)
    if('qv001.wav' in files):
        path_to_audio = os.path.join(path_to_audio, 'qv001.wav')
    elif('qv012.wav' in files):
        path_to_audio = os.path.join(path_to_audio, 'qv012.wav')
    else:
        return []
    
    audio = AudioSegment.from_wav(path_to_audio)
    audios_list = [audio[start:start+step] for start in range(0, len(audio), step)]
    return audios_list
    
def wav2spectrogram(audio_file, path_to_save):
    '''
        Descrição:
            O funçao wav2spectrogram() gera um espectrograma a partir de um áudio e salva sua imagem.

        Utilização:
            wav2spectrogram(audio, "/path/to/save/image.png").

        Parâmetros:                
            audio_file:
                Arquivo de áudio do tipo AudioSegment.
            path_to_save:
                Caminho de onde deve ser salvo a imagem, juntamente com seu nome.
    ''' 
    
    frequencies, times, spectrogram = signal.spectrogram(np.array(audio_file.get_array_of_samples()), audio_file.frame_rate)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.box(False)  
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.pcolormesh(times, frequencies, np.log(spectrogram)) # 20.*np.log10(np.abs(spectrogram)/10e-6) decibel    
    plt.gcf().set_size_inches(2.24, 2.24)
    plt.gcf().set_frameon(False)
    plt.savefig(path_to_save)
    plt.clf()

def pre_processing(path_to_csv, path_to_audios_folders):
    '''
        Descrição:
            Faz o pré-processamento dos dados contidos no csv para os áudios e em seguida o espetrograma.

        Utilização:
            pre_processing("/path/to/file.csv", "path/to/the/audios/folders").

        Parâmetros:                
            path_to_csv:
                Caminho até o arquivo csv desejado.
            path_to_audios_folders:
                Caminho até as pastas que contem os áudios. Atenção que os áudios devem tá separados por pasta, nesse caso.
    ''' 

    dirname, _ = os.path.split(os.path.abspath(__file__))   

    try:
        csv_file = pd.read_csv(os.path.join(dirname, path_to_csv), sep=',', encoding='utf-8', low_memory=False)
    except IOError as e:
        print('Não foi possivel ler o arquivo[{}] corretamente. Encerrando programa...'.format(path_to_csv))
        raise SystemExit(e)        
        
    csv_file.dropna(subset=['Diagnóstico (descritivo)'], inplace=True)
    csv_file.drop(csv_file.columns.difference(['NÚMERO PACT','Diagnóstico (descritivo)']), axis=1, inplace=True)
       
    pathologies_df = {
        'laringe normal': [],
        'nodulo': [],
        'polipo': [],
        'edema': []
    }
    for k, v in pathologies_df.items():
        pathologies_df[k] = csv_file[csv_file['Diagnóstico (descritivo)'].str.contains(k, case=False)]                                  
    
    path_to_audios_folders = os.path.join(dirname, path_to_audios_folders)    
    path_spect_cat = os.path.join(dirname, '../dados/spect')    
    for k in pathologies_df.keys():
        os.makedirs(os.path.join(path_spect_cat, k), exist_ok=True)

    # getting all splited audios from data csv
    pathologies_audios_list = {}     
    min_len = min([len(value.index) for value in pathologies_df.values()])                    
    pool = multiprocessing.Pool(10)
    for key, value in pathologies_df.items():
    
        index = 0
        results = []
        for row in value.itertuples():
    
            if index == min_len:
                break            
            index += 1

            if(row[1] < 10):                
                audio_path = os.path.join(path_to_audios_folders, 'pac00' + str(int(row[1])))
            elif(row[1] < 100):                
                audio_path = os.path.join(path_to_audios_folders, 'pac0' + str(int(row[1])))
            else:
                audio_path = os.path.join(path_to_audios_folders, 'pac' + str(int(row[1])))                  
            
            results.append(pool.apply_async(split_audio, args=(audio_path, 1000)))

        splitted_audios = [results[i].get(timeout=None) for i in range(len(results))]
        pathologies_audios_list.update({key: sum(splitted_audios, [])})

    pool.close()
    pool.join()

    # saving all spectrograms from audios above
    min_len = min([len(value) for value in pathologies_audios_list.values()])
    pool = multiprocessing.Pool(30)
    for key, value in pathologies_audios_list.items():        
        for i in range(min_len):            
            pool.apply_async(wav2spectrogram, args=(value[i], os.path.join(path_spect_cat, key, str(i) + '.png')))
    pool.close()
    pool.join()
    
    print("Espectrogramas salvos em: " + path_spect_cat)
    print('end...')


parser = argparse.ArgumentParser(description='Script pré-processar os áudios. Ele lê o csv, separa os áudios e os converte para espectrogramas.')
parser.add_argument('--csv', action='store', dest='csv', default='../dados/db-spect.csv', required=True, help='Nome/caminho do .csv com os dados para categorizar.')
parser.add_argument('--audio_folders', action='store', dest='audio_folders', default='../dados/pac-audios', required=True, help='Nome/caminho do a pasta contendo as pastas com os áudios.')
arguments = parser.parse_args()

pre_processing(arguments.csv, arguments.audio_folders)