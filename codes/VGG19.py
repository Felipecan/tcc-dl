import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, model_from_json, load_model
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *


class VGG19:


    def __init__(self):
        
        self.model = Sequential([
            InputLayer(input_shape=[224,224,3]),
            ZeroPadding2D((1,1),input_shape=(3,224,224)),
            Conv2D(64, 3, 3, activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(64, 3, 3, activation='relu'),
            MaxPooling2D((2,2), strides=(2,2)),

            ZeroPadding2D((1,1)),
            Conv2D(128, 3, 3, activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(128, 3, 3, activation='relu'),
            MaxPooling2D((2,2), strides=(2,2)),

            ZeroPadding2D((1,1)),
            Conv2D(256, 3, 3, activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(256, 3, 3, activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(256, 3, 3, activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(256, 3, 3, activation='relu'),
            MaxPooling2D((2,2), strides=(2,2)),

            ZeroPadding2D((1,1)),
            Conv2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(512, 3, 3, activation='relu'),
            MaxPooling2D((2,2), strides=(2,2)),

            ZeroPadding2D((1,1)),
            Conv2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1,1)),
            Conv2D(512, 3, 3, activation='relu'),
            MaxPooling2D((2,2), strides=(2,2)),

            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4, activation='softmax')
        ])

        # model.summary()

    def config_db(self):
        # 1. Ir até as pasta com as classes
        # 2. Separar a proporção (70% treino, 20% validação e 10% teste)
        # 3. Carregar as imagens com opencv para cada classe
        pass

    def train(self):

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy')
        self.model.fit(x=training_class1, y=training_class2, epochs=3, batch_size=1) 
        #model.summary()
