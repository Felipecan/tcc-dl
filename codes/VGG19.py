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
        pass
        # self.model = Sequential([
        #     InputLayer(input_shape=[224,224,3]),
        #     ZeroPadding2D((1,1),input_shape=(3,224,224)),
        #     Conv2D(64, 3, 3, activation='relu'),
        #     ZeroPadding2D((1,1)),
        #     Conv2D(64, 3, 3, activation='relu'),
        #     MaxPooling2D((2,2), strides=(2,2)),

        #     ZeroPadding2D((1,1)),
        #     Conv2D(128, 3, 3, activation='relu'),
        #     ZeroPadding2D((1,1)),
        #     Conv2D(128, 3, 3, activation='relu'),
        #     MaxPooling2D((2,2), strides=(2,2)),

        #     ZeroPadding2D((1,1)),
        #     Conv2D(256, 3, 3, activation='relu'),
        #     ZeroPadding2D((1,1)),
        #     Conv2D(256, 3, 3, activation='relu'),
        #     ZeroPadding2D((1,1)),
        #     Conv2D(256, 3, 3, activation='relu'),
        #     ZeroPadding2D((1,1)),
        #     Conv2D(256, 3, 3, activation='relu'),
        #     MaxPooling2D((2,2), strides=(2,2)),

        #     ZeroPadding2D((1,1)),
        #     Conv2D(512, 3, 3, activation='relu'),
        #     ZeroPadding2D((1,1)),
        #     Conv2D(512, 3, 3, activation='relu'),
        #     ZeroPadding2D((1,1)),
        #     Conv2D(512, 3, 3, activation='relu'),
        #     ZeroPadding2D((1,1)),
        #     Conv2D(512, 3, 3, activation='relu'),
        #     MaxPooling2D((2,2), strides=(2,2)),

        #     ZeroPadding2D((1,1)),
        #     Conv2D(512, 3, 3, activation='relu'),
        #     ZeroPadding2D((1,1)),
        #     Conv2D(512, 3, 3, activation='relu'),
        #     ZeroPadding2D((1,1)),
        #     Conv2D(512, 3, 3, activation='relu'),
        #     ZeroPadding2D((1,1)),
        #     Conv2D(512, 3, 3, activation='relu'),
        #     MaxPooling2D((2,2), strides=(2,2)),

        #     Flatten(),
        #     Dense(4096, activation='relu'),
        #     Dropout(0.5),
        #     Dense(4096, activation='relu'),
        #     Dropout(0.5),
        #     Dense(4, activation='softmax')
        # ])

        # model.summary()

    def config_db(self):        
        
        self.test_set = []       
        self.training_set = []
        self.validation_set = []
        
        class_folders = os.listdir('../dados/spect')
        for folder in class_folders:
            
            files = os.listdir('../dados/spect/'+folder)
            if '1' in folder:
                #one_hot_encoding = np.zeros((2,), dtype=np.int)
                one_hot_encoding = [1, 0]                
            else:
                #one_hot_encoding = np.zeros((2,), dtype=np.int)
                one_hot_encoding = [0, 1]
                
            for i in range(len(files)):
                
                if(i < int(len(files)*0.7)):
                    im = cv2.imread('../dados/spect/'+folder+'/'+files[i])
                    self.training_set.append([im, one_hot_encoding])

                elif(i < int(len(files)*0.9)):
                    im = cv2.imread('../dados/spect/'+folder+'/'+files[i])
                    self.validation_set.append([im, one_hot_encoding])

                else:
                    im = cv2.imread('../dados/spect/'+folder+'/'+files[i])
                    self.test_set.append([im, one_hot_encoding])                

        print('train', len(self.training_set))
        print('test', len(self.test_set))
        print('validation', len(self.validation_set))

    def train(self):

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy')
        self.model.fit(x=training_class1, y=training_class2, epochs=3, batch_size=1) 
        #model.summary()
