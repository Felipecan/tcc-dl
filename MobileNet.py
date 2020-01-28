from Model import Model
import tensorflow as tf

class MobileNet(Model):

    def __init__(self):
        
        Model.__init__(self)
        
        # dim = (64,64,3)
        dim = (224,224,3)

        self.model = tf.keras.models.Sequential([
            
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False, input_shape=dim),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu', use_bias=False),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2, kernel_initializer='he_normal', activation='softmax')
        ])       