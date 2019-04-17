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


class MobileNet:


    def __init__(self):
        self.model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False, input_shape=(224, 224, 3)),
            BatchNormalization(),

            DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),
            
            DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),
            
            DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),
            
            DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),
            
            DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),
            
            DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            BatchNormalization(),

            GlobalAveragePooling2D(),
            Dense(2, kernel_initializer='he_normal', activation='softmax')
        ])

        self.model.summary()

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
                
        dirname, _ = os.path.split(os.path.abspath(__file__))   
        if(not os.path.isabs(path_to_spect_folders)):
            path_to_spect_folders = os.path.join(dirname, path_to_spect_folders)
        
        for folder in os.listdir(path_to_spect_folders):
            
            print('folder: {}'.format(folder))

            files = os.listdir(os.path.join(path_to_spect_folders, folder))
            print('Number of files: {}'.format(len(files))) 

            
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
                if(i < int(len(files)*0.9)):                    
                    self.training_set['pics'].append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                    self.training_set['labels'].append(one_hot_encoding)
                else:                    
                    self.test_set['pics'].append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                    self.test_set['labels'].append(one_hot_encoding)               
        
        print('training and validation set: {} images'.format(len(self.training_set['pics'])))
        print('test set: {} images'.format(len(self.test_set['pics'])))

    def train(self):
        '''
            Descrição:
                

            Utilização:
                

            Parâmetros:                
                

            Retorno:
                
        '''   
        

        # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        # # loss='sparse_categorical_crossentropy'
        # callbacks_list = [keras.callbacks.ModelCheckpoint('./best_weights.hdf5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)]
        # self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        # history = self.model.fit(x=np.array(self.training_set['pics']).reshape(-1, 224, 224, 3), y=np.array(self.training_set['labels']), batch_size=16, epochs=15, validation_split=0.111, callbacks=callbacks_list)        
        # self.model.evaluate(np.array(self.test_set['pics']).reshape(-1, 224, 224, 3), np.array(self.test_set['labels']))         
        
        batch_size = 32
        learning_rate = 5e-4
        decay = 0.0
        momentum = 0.9
        num_classes = 200
        epochs = 10
        adam = Adam(lr=learning_rate, decay=decay)
        #sgd = optimizers.SGD(lr=0.1, decay=0.01)
        self.model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        
        checkpointer = keras.callbacks.ModelCheckpoint(filepath='./weights.hdf5', verbose=1, save_best_only=True,save_weights_only=True)
        history = self.model.fit(x=np.array(self.training_set['pics']).reshape(-1, 224, 224, 3), y=np.array(self.training_set['labels']), batch_size=batch_size, epochs=epochs, validation_split=0.111, callbacks=[checkpointer])        

        pred = self.model.evaluate(np.array(self.test_set['pics']).reshape(-1, 224, 224, 3), np.array(self.test_set['labels']))

        print(pred)
        import matplotlib.pyplot as plt
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='lower right')
        plt.show()
        plt.savefig('loss.png')
        plt.savefig('loss.pdf')
        plt.clf()

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='lower right')
        plt.show()
        plt.savefig('acc.png')
        plt.savefig('acc.pdf')
        plt.clf()

    # def predict(self, path_to_audios):
    #     '''
    #         Descrição:
                

    #         Utilização:
                

    #         Parâmetros:                
                

    #         Retorno:
                
    #     '''   

    #     # temos que fazer o predict
    #     # vou pegar um áudio completo
    #     dirname, _ = os.path.split(os.path.abspath(__file__))  

    #     if(not os.path.isabs(path_to_audios_folders)):
    #         path_to_audios_folders = os.path.join(dirname, path_to_audios_folders)




