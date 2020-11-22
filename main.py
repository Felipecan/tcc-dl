# import sys
# import argparse
# from pre_processing import *
# from Model import Model
# from VGG19 import VGG19

# parser = argparse.ArgumentParser(description='Script to perform general tasks like: pre-processing and then train, train the network directly.')
# parser.add_argument('--mode', choices=['training', 'all_process'], action='store', dest='mode', required=True, help='How the file should be executed. training: only the training. all_process: pre-process and train.')
# parser.add_argument('--spectrograms', action='store', dest='spectrograms', required='training' in sys.argv, help='Nome/caminho da pasta contendo as pastas com os espectrogramas.')
# parser.add_argument('--csv', action='store', dest='csv', required='all_process' in sys.argv, help='Name/path of .csv with data to categorize.')
# parser.add_argument('--audios', action='store', dest='audios', required='all_process' in sys.argv, help='Name/path of the folder containing folders with the audio.')
# arguments = parser.parse_args()

# if(arguments.mode == 'training'):
#     print("Entering on training mode...")
#     vgg19 = VGG19()
#     vgg19.config_dataset(arguments.spectrograms)    
#     vgg19.train()
#     vgg19.test_predict()
# else:
#     print('Entering on all mode...')    
#     print('First, starting the pre processing of the datas...')    
#     spectrograms = pre_processing.pre_processing(arguments.csv, arguments.audios)

#     print('Starting training...')
#     vgg19 = VGG19()
#     vgg19.config_dataset(spectrograms)    
#     vgg19.train()
#     vgg19.test_predict()



# if(__name__ == '__main__'):

#     spectrograms = pre_processing()

#     vgg19.config_dataset(spectrograms)    
#     vgg19.train()



if(__name__ == '__main__'):
    import pre_processing
    pre_processing.start()