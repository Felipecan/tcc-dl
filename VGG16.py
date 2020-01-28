from Model import Model
import tensorflow as tf

class VGG16(Model):

    def __init__(self):
        
        Model.__init__(self)

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
            tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

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
            tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5)
            # tf.keras.layers.Dense(4, activation='softmax')
        ])

        # self.model.summary()