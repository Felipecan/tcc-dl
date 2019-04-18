import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, model_from_json, load_model

import pre_processing
import matplotlib.pyplot as plt

class VGG19:


    def __init__(self):
        
        self.model = Sequential([
            InputLayer(input_shape=[224,224,3]),
            ZeroPadding2D((1,1),input_shape=(3,224,224)),
            Conv2D(64, (3, 3), activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),

            ZeroPadding2D((1,1)),
            Conv2D(128, (3, 3), activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),

            ZeroPadding2D((1,1)),
            Conv2D(256, (3, 3), activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(256, (3, 3), activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(256, (3, 3), activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),

            ZeroPadding2D((1,1)),
            Conv2D(512, (3, 3), activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(512, (3, 3), activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(512, (3, 3), activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(512, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),

            ZeroPadding2D((1,1)),
            Conv2D(512, (3, 3), activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(512, (3, 3), activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(512, (3, 3), activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(512, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),

            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])

        # self.model.summary()

    def config_dataset(self, path_to_spect_folders):        
        '''
            Descrição:
                

            Utilização:
                

            Parâmetros:                
                

            Retorno:
                
        '''   

        self.test_set = {
            'pics': [],
            'labels': []
        }     
        self.training_set = {
            'pics': [],
            'labels': []
        }
                
        dirname, _ = os.path.split(os.path.abspath(__file__))   
        if(not os.path.isabs(path_to_spect_folders)):
            path_to_spect_folders = os.path.join(dirname, path_to_spect_folders)
        
        folders = os.listdir(path_to_spect_folders) # lista de pastas (classes)
        indexes = list(range(len(folders))) #criando uma lista com os índeces
        one_hot_encondings = to_categorical(np.array(indexes)) # one_hot_enconding para todos as pastas (classes)

        for folder in folders:
            
            print('folder: {}'.format(folder))

            files = os.listdir(os.path.join(path_to_spect_folders, folder))
            print('Number of files: {}'.format(len(files))) 
                
            for i in range(len(files)):
                
                im = cv2.imread(os.path.join(path_to_spect_folders, folder, files[i]))
                if(i < int(len(files)*0.9)):                    
                    self.training_set['pics'].append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                    self.training_set['labels'].append(one_hot_encondings[folders.index(folder)])
                else:                    
                    self.test_set['pics'].append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                    self.test_set['labels'].append(one_hot_encondings[folders.index(folder)])               
        
        print('training and validation set: {} images'.format(len(self.training_set['pics'])))
        print('test set: {} images'.format(len(self.test_set['pics'])))

    def train(self):
        '''
            Descrição:
                

            Utilização:
                

            Parâmetros:                
                

            Retorno:
                
        '''   
        

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)        
        # loss='sparse_categorical_crossentropy'
        callbacks_list = [keras.callbacks.ModelCheckpoint('./best_weights.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)]
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(x=np.array(self.training_set['pics']).reshape(-1, 224, 224, 3), y=np.array(self.training_set['labels']), batch_size=16, epochs=15, validation_split=0.111, callbacks=callbacks_list)        
        evaluate = self.model.evaluate(np.array(self.test_set['pics']).reshape(-1, 224, 224, 3), np.array(self.test_set['labels']))         
        
        print(evaluate)

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig('./acc.png')
        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig('./loss.png')

        plt.show()

    def predict(self, path_to_audio):
        '''
            Descrição:
                

            Utilização:
                

            Parâmetros:                
                

            Retorno:
                
        '''   

          
        if(not os.path.isabs(path_to_audio)):
            dirname, _ = os.path.split(os.path.abspath(__file__))
            path_to_audio = os.path.join(dirname, path_to_audio)

        audio_list = pre_processing.split_audio(path_to_audio, 200)
        print('Audios splitted in rate of {} ms by section.'.format(200))
        
        spectrograms_folder_temp = os.path.join(path_to_audio, '../temp')
        os.makedirs(spectrograms_folder_temp, exist_ok=True)
        for i in range(len(audio_list)):            
            pre_processing.wav2spectrogram(audio_list[i], os.path.join(spectrograms_folder_temp, '{}.png'.format(i)))
        print('Images saved in a temp folder. Images total: {}'.format(len(audio_list)))
        
        print('Predicting audio file from saved images...')
        pred = []
        for archive in os.listdir(spectrograms_folder_temp):
            im = cv2.imread(os.path.join(spectrograms_folder_temp, archive))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            pred.append(self.model.predict(np.array(im).reshape(-1, 224, 224, 3)).flatten())
        
        result_pred = [0]*len(pred[0])
        for i in range(len(pred)):
            for j in range(len(pred[i])):                
                result_pred[j] += pred[i][j]
        
        print('Non-standardized result', result_pred)
        result_pred = result_pred/sum(result_pred)
        print('Standardized result', result_pred)
        
        os.system('rm -r {}'.format(spectrograms_folder_temp))
        print('Deleted folder {}'.format(spectrograms_folder_temp))

    def test_predict(self, path_to_class):
        dirname, _ = os.path.split(os.path.abspath(__file__))  
        if(not os.path.isabs(path_to_class)):
            path_to_class = os.path.join(dirname, path_to_class)

        for c in os.listdir(path_to_class):
            print('predict to {}'.format(c))
            for patient in os.listdir(os.path.join(path_to_class, c)):
                self.predict(os.path.join(path_to_class, c, patient))
