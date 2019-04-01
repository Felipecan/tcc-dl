import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, model_from_json, load_model


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

    def config_db(self, path_to_spect_folders):        
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
        self.validation_set = {
            'pics': [],
            'labels': []
        } 
        
        dirname, _ = os.path.split(os.path.abspath(__file__))   
        if(not os.path.isabs(path_to_spect_folders)):
            path_to_spect_folders = os.path.join(dirname, path_to_spect_folders)
        
        for folder in os.listdir(path_to_spect_folders):
            
            files = os.listdir(os.path.join(path_to_spect_folders, folder))
            files = files[:3]
            
            if '1' in folder:
                one_hot_encoding = np.zeros((2,), dtype=np.int)
                one_hot_encoding[0] = 1
                #one_hot_encoding = [1, 0]                
            else:
                one_hot_encoding = np.zeros((2,), dtype=np.int)
                one_hot_encoding[1] = 1
                #one_hot_encoding = [0, 1]
                
            for i in range(len(files)):
                
                im = cv2.imread(os.path.join(path_to_spect_folders, folder, files[i]))
                if(i < int(len(files)*0.7)):                    
                    self.training_set['pics'].append(im)
                    self.training_set['labels'].append(one_hot_encoding)

                elif(i < int(len(files)*0.9)):                    
                    self.validation_set['pics'].append(im)
                    self.validation_set['labels'].append(one_hot_encoding)

                else:                    
                    self.test_set['pics'].append(im)
                    self.test_set['labels'].append(one_hot_encoding)               


    def train(self):
        '''
            Descrição:
                

            Utilização:
                

            Parâmetros:                
                

            Retorno:
                
        '''   
        
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy')
        self.model.fit(x=np.array(self.training_set['pics']).reshape(-1, 224, 224, 3), y=np.array(self.training_set['labels']), epochs=3, batch_size=1)        
        self.model.evaluate(np.array(self.validation_set['pics']).reshape(-1, 224, 224, 3), np.array(self.validation_set['labels']))         
