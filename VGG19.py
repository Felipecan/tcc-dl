import os
import cv2
import random
import numpy as np
import tensorflow as tf

from util import split_audio, wav_to_spectrogram, stacked_bar
import matplotlib.pyplot as plt
import multiprocessing

# num_test = '0009/'

class VGG19:
    '''
    '''    

    def __init__(self):

        self.model = tf.keras.models.Sequential([        

            tf.keras.layers.InputLayer(input_shape=[224,224,3]),
            tf.keras.layers.ZeroPadding2D((1,1),input_shape=(3,224,224)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.ZeroPadding2D((1,1)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

            tf.keras.layers.ZeroPadding2D((1,1)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.ZeroPadding2D((1,1)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

            tf.keras.layers.ZeroPadding2D((1,1)),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.ZeroPadding2D((1,1)),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.ZeroPadding2D((1,1)),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.ZeroPadding2D((1,1)),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

            tf.keras.layers.ZeroPadding2D((1,1)),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
            tf.keras.layers.ZeroPadding2D((1,1)),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
            tf.keras.layers.ZeroPadding2D((1,1)),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
            tf.keras.layers.ZeroPadding2D((1,1)),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

            tf.keras.layers.ZeroPadding2D((1,1)),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
            tf.keras.layers.ZeroPadding2D((1,1)),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
            tf.keras.layers.ZeroPadding2D((1,1)),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
            tf.keras.layers.ZeroPadding2D((1,1)),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.4)
            # tf.keras.layers.Dense(4, activation='softmax')
        ])

        # self.model.summary()

    def config_dataset(self, path_to_spect_folders):
        '''
            Description:


            Use:


            Parameters:


            Return:

        '''

        self.test_set = []
        self.training_set = []

        if(not os.path.isabs(path_to_spect_folders)):
            dirname, _ = os.path.split(os.path.abspath(__file__))
            path_to_spect_folders = os.path.join(dirname, path_to_spect_folders)

        self.folders = os.listdir(path_to_spect_folders) # lista de pastas (classes)                    
        
        indexes = list(range(len(self.folders))) #criando uma lista com os índeces
        one_hot_encondings = tf.keras.utils.to_categorical(np.array(indexes)) # one_hot_enconding para todos as pastas (classes)

        self.model.add(tf.keras.layers.Dense(len(self.folders), activation='softmax'))
        self.model.summary()


        for folder in self.folders:

            print('folder: {}'.format(folder))

            files = os.listdir(os.path.join(path_to_spect_folders, folder))            
            print('Number of files: {}'.format(len(files)))

            for i in range(len(files)):

                im = cv2.imread(os.path.join(path_to_spect_folders, folder, files[i]))
                if(i < int(len(files)*0.9)):
                    self.training_set.append( (cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (224, 224)), 
                                              one_hot_encondings[self.folders.index(folder)]) )
                else:
                    self.test_set.append( (cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (224, 224)), 
                                          one_hot_encondings[self.folders.index(folder)]) )

        random.shuffle(self.training_set)
        random.shuffle(self.test_set)

        print('training and validation set: {} images'.format(len(self.training_set)))
        print('test set: {} images'.format(len(self.test_set)))


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
        
        # lr=0.00146, decay=1e-6, momentum=0.9, nesterov=True        
        sgd = tf.keras.optimizers.SGD(lr=0.00146, decay=1e-6, momentum=0.9, nesterov=True)        
        callbacks_list = [
            # tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=1),
            # tf.keras.callbacks.ModelCheckpoint('./best_weights.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)
            tf.keras.callbacks.ModelCheckpoint('./best_model.h5', monitor='val_acc', verbose=1, save_best_only=True)
        ]


        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        
        training_x = [item[0] for item in self.training_set]
        training_y = [item[1] for item in self.training_set]        
        history = self.model.fit(
            x=np.array(training_x).reshape(-1, 224, 224, 3),
            y=np.array(training_y), 
            batch_size=16, 
            epochs=12,
            validation_split=0.111, 
            callbacks=callbacks_list
        )
        
        
        test_x = [item[0] for item in self.test_set]
        test_y = [item[1] for item in self.test_set]        
        evaluate = self.model.evaluate(
            np.array(test_x).reshape(-1, 224, 224, 3),
            np.array(test_y)
        )
        
        # plot graph
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
                
        plt.plot(epochs, acc, label='Training acc')
        plt.plot(epochs, val_acc, label='Validation acc')
        plt.grid(axis='both', alpha=.3)
        plt.title('Training and validation accuracy')
        plt.legend()
        
        # os.makedirs("/content/gdrive/My Drive/datasets/tests/"+num_test)
        # plt.savefig("/content/gdrive/My Drive/datasets/tests/"+num_test+"acc.png")
        plt.savefig("acc.png")
        plt.figure()

        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.grid(axis='both', alpha=.3)
        plt.title('Training and validation loss')
        plt.legend()
        # plt.savefig("/content/gdrive/My Drive/datasets/tests/"+num_test+"loss.png")
        plt.savefig("loss.png")

        plt.show()

        
    def predict(self, path_to_audio, class_test):
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

        spectrograms_folder_temp = os.path.join(path_to_audio, '../temp')
        os.makedirs(spectrograms_folder_temp)
        for i in range(len(audio_list)):
            wav_to_spectrogram(audio_list[i], os.path.join(spectrograms_folder_temp, '{}.png'.format(i)))        

        print('\tPredicting audio file...')
        predictions = []                
        for archive in os.listdir(spectrograms_folder_temp):
            img = cv2.imread(os.path.join(spectrograms_folder_temp, archive))
            img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224))
            temp = self.model.predict(np.array(img).reshape(-1, 224, 224, 3)).flatten()
            predictions.append(temp)


        if predictions == []:
            print('Error! Probably without audio...')
            return -1
        
        ################################
        # for statistics
        for pred in predictions:            
            cat = self.folders[np.argmax(pred)]
            if cat == class_test:
                self.right_by_partion[self.folders.index(class_test)] += 1
            else:
                self.wrong_by_portion[self.folders.index(class_test)] += 1                     
        #################################
        
        predictions_result = [0]*len(predictions[0])
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):                
                predictions_result[j] += predictions[i][j]            

                
        # print('\tStandardized result', predictions_result)
        category = np.argmax(predictions_result)
        print('\tCategory: {}'.format(self.folders[category]))

        os.system('rm -r {}'.format(spectrograms_folder_temp))                        

        return self.folders[category]
    
    
    def predict_total_img(self, path_to_audio):
    
        if(not os.path.isabs(path_to_audio)):
            dirname, _ = os.path.split(os.path.abspath(__file__))
            path_to_audio = os.path.join(dirname, path_to_audio)
        
        from pydub import AudioSegment
        
        if(not os.path.isfile(path_to_audio)):
            files = os.listdir(path_to_audio)
            if('qv001.wav' in files):
                audio_name = 'qv001.wav'
            elif('qv012.wav' in files):
                audio_name = 'qv012.wav'
        
        audio = AudioSegment.from_wav(os.path.join(path_to_audio, audio_name))
        
        spectrograms_folder_temp = os.path.join(path_to_audio, '../temp')
        os.makedirs(spectrograms_folder_temp)
        
        wav_to_spectrogram(audio, os.path.join(spectrograms_folder_temp, 'img.png'))
                
        print('\tPredicting audio file...')
        img = cv2.imread(os.path.join(spectrograms_folder_temp, 'img.png'))
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224))
        prediction = self.model.predict(np.array(img).reshape(-1, 224, 224, 3)).flatten()
        
        print('\tCategory: {}'.format(self.folders[np.argmax(prediction)]))
        os.system('rm -r {}'.format(spectrograms_folder_temp))  

        return self.folders[np.argmax(prediction)]               
                

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
        
        ################################
        # for statistics in predict function
        self.right_by_partion = [0]*len(self.folders)
        self.wrong_by_portion = [0]*len(self.folders)
        ################################
        right = [0]*len(self.folders)
        wrong = [0]*len(self.folders)
        err = [0]*len(self.folders)
        for cat in os.listdir(path_to_class):
            
            print('\033[1;33m' + '\nPredict to {}\n'.format(cat) + '\033[0m')
            for patient in os.listdir(os.path.join(path_to_class, cat)):
                
                category = self.predict(os.path.join(path_to_class, cat, patient), cat)                
                
                if(category == cat):
                    right[self.folders.index(cat)] += 1
                elif(category == -1):
                    err[self.folders.index(cat)] += 1
                else:                    
                    wrong[self.folders.index(cat)] += 1

        print('{}/{} right'.format(sum(right), sum(right+wrong)))
        print('{}/{} wrong'.format(sum(wrong), sum(right+wrong)))
        print('Precision: {:.2f}'.format(sum(right)/sum(right+wrong)))

        
        # plot
        series_labels = ['Right', 'Error']
        category_labels = self.folders.copy() 
        data = [right, wrong]
        stacked_bar(
            data, 
            series_labels, 
            category_labels=category_labels, 
            show_values=True, 
            value_format="{:.0f}",
            y_label="Quantidade"
        )
        # plt.savefig("/content/gdrive/My Drive/datasets/tests/"+num_test+"bar.png")
        plt.savefig("bar.png")
        plt.show()
        
        data = [self.right_by_partion, self.wrong_by_portion]
        stacked_bar(
            data, 
            series_labels, 
            category_labels=category_labels, 
            show_values=True, 
            value_format="{:.0f}",
            y_label="Quantidade"
        )
        # plt.savefig("/content/gdrive/My Drive/datasets/tests/"+num_test+"bar2.png")
        plt.savefig("bar2.png")
        plt.show()
        
        
    def test_predict2(self, path_to_class):
        '''
            Description:


            Use:


            Parameters:


            Return:

        '''

        if(not os.path.isabs(path_to_class)):
            dirname, _ = os.path.split(os.path.abspath(__file__))
            path_to_class = os.path.join(dirname, path_to_class)

        right = [0]*len(self.folders)
        wrong = [0]*len(self.folders)
        err = [0]*len(self.folders)
        
        for cat in os.listdir(path_to_class):
            
            print('\033[1;33m' + '\nPredict to {}\n'.format(cat) + '\033[0m')
            for patient in os.listdir(os.path.join(path_to_class, cat)):
                                
                category = self.predict_total_img(os.path.join(path_to_class, cat, patient))
                
                if(category == cat):
                    right[self.folders.index(cat)] += 1
                elif(category == -1):
                    err[self.folders.index(cat)] += 1
                else:                    
                    wrong[self.folders.index(cat)] += 1

        print('{}/{} right'.format(sum(right), sum(right+wrong)))
        print('{}/{} wrong'.format(sum(wrong), sum(right+wrong)))
        print('Precision: {:.2f}'.format(sum(right)/sum(right+wrong)))
        
        # plot
        series_labels = ['Right', 'Error']
        category_labels = self.folders.copy() 
        data = [right, wrong]
        stacked_bar(
            data, 
            series_labels, 
            category_labels=category_labels, 
            show_values=True, 
            value_format="{:.0f}",
            y_label="Quantidade"
        )
        # plt.savefig("/content/gdrive/My Drive/datasets/tests/"+num_test+"bar3.png")
        plt.savefig("bar3.png")
        plt.show()
        

    # def build_mobilenet(self):
    #     '''
    #         Description:


    #         Use:
    #             build_mobilenet()        

    #         Return:

    #     '''
        
    #     # dim = (64,64,3)
    #     dim = (224,224,3)

    #     return tf.keras.models.Sequential([
            
    #         tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False, input_shape=dim),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),
    #         tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),
    #         tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),
    #         tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),
    #         tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),
    #         tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
    #         tf.keras.layers.BatchNormalization(),

    #         tf.keras.layers.GlobalAveragePooling2D(),
    #         tf.keras.layers.Dense(2, kernel_initializer='he_normal', activation='softmax')
    #     ])        

