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

from util import split_audio, wav_to_spectrogram
import matplotlib.pyplot as plt
import multiprocessing

class Model:
    '''
    '''    

    def __init__(self, name='VGG19'):

        if(name == 'VGG19'):
            self.model = self.build_vgg19()
        elif(name == 'mobilenet'):
            self.model = build_mobilenet()

        self.model.summary()

    def config_dataset(self, path_to_spect_folders):
        '''
            Description:


            Use:


            Parameters:


            Return:

        '''

        self.test_set = {
            'pics': [],
            'labels': []
        }
        self.training_set = {
            'pics': [],
            'labels': []
        }


        if(not os.path.isabs(path_to_spect_folders)):
            dirname, _ = os.path.split(os.path.abspath(__file__))
            path_to_spect_folders = os.path.join(dirname, path_to_spect_folders)

        self.folders = os.listdir(path_to_spect_folders) # lista de pastas (classes)
        indexes = list(range(len(self.folders))) #criando uma lista com os índeces
        one_hot_encondings = to_categorical(np.array(indexes)) # one_hot_enconding para todos as pastas (classes)

        for folder in self.folders:

            print('folder: {}'.format(folder))

            files = os.listdir(os.path.join(path_to_spect_folders, folder))
            # files = files[0:100]
            print('Number of files: {}'.format(len(files)))

            for i in range(len(files)):

                im = cv2.imread(os.path.join(path_to_spect_folders, folder, files[i]))
                if(i < int(len(files)*0.9)):
                    self.training_set['pics'].append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (224, 224)))
                    self.training_set['labels'].append(one_hot_encondings[self.folders.index(folder)])
                else:
                    self.test_set['pics'].append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (224, 224)))
                    self.test_set['labels'].append(one_hot_encondings[self.folders.index(folder)])

        print('training and validation set: {} images'.format(len(self.training_set['pics'])))
        print('test set: {} images'.format(len(self.test_set['pics'])))


    def train(self):
        '''
            Description:


            Use:


            Parameters:


            Return:

        '''

        # batch_size = 32
        # learning_rate = 5e-4
        # decay = 0.0
        # momentum = 0.9
        # num_classes = 200
        # epochs = 10
        # adam = Adam(lr=learning_rate, decay=decay)
        # #sgd = optimizers.SGD(lr=0.1, decay=0.01)
        # self.model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        
        # checkpointer = keras.callbacks.ModelCheckpoint(filepath='./weights.hdf5', verbose=1, save_best_only=True,save_weights_only=True)
        # history = self.model.fit(x=np.array(self.training_set['pics']).reshape(-1, 224, 224, 3), y=np.array(self.training_set['labels']), batch_size=batch_size, epochs=epochs, validation_split=0.111, callbacks=[checkpointer])        

        # pred = self.model.evaluate(np.array(self.test_set['pics']).reshape(-1, 224, 224, 3), np.array(self.test_set['labels']))

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)        
        callbacks_list = [keras.callbacks.ModelCheckpoint('./best_weights.h5', monitor='val_acc', verbose=1,
                                                          save_best_only=True, save_weights_only=True)]


        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        history = self.model.fit(x=np.array(self.training_set['pics']).reshape(-1, 224, 224, 3),
                                 y=np.array(self.training_set['labels']), batch_size=16, epochs=12,
                                 validation_split=0.111, callbacks=callbacks_list)

        evaluate = self.model.evaluate(np.array(self.test_set['pics']).reshape(-1, 224, 224, 3),
                                       np.array(self.test_set['labels']))

        print(evaluate)

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        
        # plt.style.use('ggplot')
        plt.plot(epochs, acc, label='Training acc')
        plt.plot(epochs, val_acc, label='Validation acc')
        plt.grid(axis='both', alpha=.3)
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig('./acc.png')
        plt.figure()

        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.grid(axis='both', alpha=.3)
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig('./loss.png')

        plt.show()

        
    def predict(self, path_to_audio):
        '''
            Description:


            Use:


            Parameters:


            Return:

        '''

        if(not os.path.isabs(path_to_audio)):
            dirname, _ = os.path.split(os.path.abspath(__file__))
            path_to_audio = os.path.join(dirname, path_to_audio)
        
        time_step = 100
        audio_list = split_audio(path_to_audio, time_step)
        # print('\tAudios splitted in {} ms.'.format(time_step))

        spectrograms_folder_temp = os.path.join(path_to_audio, '../temp')
        os.makedirs(spectrograms_folder_temp)
        for i in range(len(audio_list)):
            wav_to_spectrogram(audio_list[i], os.path.join(spectrograms_folder_temp, '{}.png'.format(i)))
        # print('\tImages saved...')

        print('\tPredicting audio file...')
        pred = []
        for archive in os.listdir(spectrograms_folder_temp):
            im = cv2.imread(os.path.join(spectrograms_folder_temp, archive))
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (224, 224))
            temp = self.model.predict(np.array(im).reshape(-1, 224, 224, 3)).flatten()
            pred.append(temp)


        if pred == []:
            return -1

        result_pred = [0]*len(pred[0])
        for i in range(len(pred)):
            for j in range(len(pred[i])):
                result_pred[j] += pred[i][j]

                
        print('\tStandardized result', result_pred)
        category = np.argmax(result_pred)
        print('\tDeviation type: {}'.format(self.folders[category]))

        os.system('rm -r {}'.format(spectrograms_folder_temp))        

        return self.folders[category]


    def test_predict(self, path_to_class):
        '''
            Description:


            Use:


            Parameters:


            Return:

        '''

        if(not os.path.isabs(path_to_class)):
            dirname, _ = os.path.split(os.path.abspath(__file__))
            path_to_class = os.path.join(dirname, path_to_class)

        right = 0
        wrong = 0
        err = 0
        for c in os.listdir(path_to_class):
            
            print('\x1b[6;30;42m' + '\nPredict to deviation {}\n'.format(c) + '\x1b[0m')
            for patient in os.listdir(os.path.join(path_to_class, c)):
                
                category = self.predict(os.path.join(path_to_class, c, patient))
                if(category == c):
                    right += 1
                elif(category == -1):
                    err += 1
                else:
                    wrong += 1
                print("right: {}; wrong: {}; error: {}".format(right, wrong, err))
                print('---------------------------------')
        print('{}/{} right'.format(right, right+wrong+err))
        print('{}/{} wrong'.format(wrong, right+wrong+err))
        print('{}/{} with some erros in the files'.format(err, right+wrong+err))


    def build_vgg19(self):
        '''
            Description:


            Use:
                build_vgg19()

            Return:

        '''
        return Sequential([        

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


    def build_mobilenet():
        '''
            Description:


            Use:
                build_mobilenet()        

            Return:

        '''
        
        # dim = (64,64,3)
        dim = (224,224,3)

        return Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False, input_shape=dim),
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

