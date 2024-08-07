import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


image_shape = (128, 128, 3)

# LeNet-5
lenet5 = Sequential([
            Conv2D(6, 5, activation='relu', input_shape=image_shape),
            MaxPooling2D(strides=(2,2)),
            Conv2D(16, 5, activation='relu'),
            MaxPooling2D(strides=(2,2)),
            Flatten(),
            Dense(120, activation='relu'),
            Dropout(0.5),
            Dense(84, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
], name='lenet5')

# AlexNet
alex = Sequential([
            Conv2D(96, 11, strides=(4,4), activation='relu', input_shape=image_shape),
            MaxPooling2D(pool_size=(3,3), strides=(2,2)),
            Conv2D(256, 5, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(3,3), strides=(2,2)),
            Conv2D(384, 3, padding='same', activation='relu'),
            Conv2D(384, 3, padding='same', activation='relu'),
            Conv2D(256, 3, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(3,3), strides=(2,2)),
            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
], name='alex')

# VGG-16
vgg16 = Sequential([
            Conv2D(64, 3, padding='same', activation='relu', input_shape=image_shape),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            Conv2D(128, 3, padding='same', activation='relu'),
            Conv2D(128, 3, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            Conv2D(256, 3, padding='same', activation='relu'),
            Conv2D(256, 3, padding='same', activation='relu'),
            Conv2D(256, 3, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            Conv2D(512, 3, padding='same', activation='relu'),
            Conv2D(512, 3, padding='same', activation='relu'),
            Conv2D(512, 3, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            Conv2D(512, 3, padding='same', activation='relu'),
            Conv2D(512, 3, padding='same', activation='relu'),
            Conv2D(512, 3, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
], name='vgg16')

# Inception
inception = Sequential([
            Conv2D(64, 7, strides=(2,2), padding='same', activation='relu', input_shape=image_shape),
            MaxPooling2D(pool_size=(3,3), strides=(2,2)),
            Conv2D(64, 1, activation='relu'),
            Conv2D(192, 3, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(3,3), strides=(2,2)),
            Conv2D(64, 1, activation='relu'),
            Conv2D(128, 3, padding='same', activation='relu'),
            Conv2D(256, 3, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(3,3), strides=(2,2)),
            Conv2D(128, 1, activation='relu'),
            Conv2D(192, 3, padding='same', activation='relu'),
            Conv2D(256, 3, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(3,3), strides=(2,2)),
            Conv2D(256, 1, activation='relu'),
            Conv2D(384, 3, padding='same', activation='relu'),
            Conv2D(256, 3, padding='same', activation='relu'),
            Conv2D(256, 3, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(3,3), strides=(2,2)),
            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
], name='inception')
