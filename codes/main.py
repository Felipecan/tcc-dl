import sys
import argparse
import pre_processing
from VGG19 import VGG19

parser = argparse.ArgumentParser(description='Script para executar tarefas gerais como: fazer o pré-processamento e em seguida treinar, treinar a rede diretamente.')
parser.add_argument('--spect_folders', action='store', dest='spect_folders', default='../dados/spect', required='training' in sys.argv, help='Nome/caminho da pasta contendo as pastas com os espectrogramas.')
parser.add_argument('--mode', choices=['training', 'all_proc'], action='store', dest='mode', default='../dados/spect', required=True, help='Modo em que o arquivo deve ser executado. training: somente o treinamente. all_proc: pre-processar e treinar.')
parser.add_argument('--csv', action='store', dest='csv', default='../dados/db-spect.csv', required='all_proc' in sys.argv, help='Nome/caminho do .csv com os dados para categorizar.')
parser.add_argument('--audio_folders', action='store', dest='audio_folders', default='../dados/pac-audios', required='all_proc' in sys.argv, help='Nome/caminho da pasta contendo as pastas com os áudios.')
arguments = parser.parse_args()

if(arguments.mode == 'training'):
    print("Entering on training mode...")
    vgg19 = VGG19()
    vgg19.config_db(arguments.spect_folders)    
    vgg19.train()
else:
    print('Entering on all mode...')    
    print('First, starting the pre processing of the datas...')    
    path_spect = pre_processing.pre_processing(arguments.csv, arguments.audio_folders)

    print('Starting training...')
    vgg19 = VGG19()
    vgg19.config_db(path_spect)
    print(len(vgg19.training_set['pics']))
    vgg19.train()