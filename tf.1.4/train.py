from tensorflow.keras import models, layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, initializers, regularizers, metrics
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, \
    ZeroPadding2D, Add
from tensorflow.keras.applications.resnet50 import preprocess_input
from model import conv1_layer, conv2_layer, conv3_layer, conv4_layer, conv5_layer
import os
import numpy as np
import math

#########################################

try:
    import tensorflow as tf
except AttributeError:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

#########################################

# number of classes
K = 6

input_tensor = Input(shape=(224, 224, 3), dtype='float32', name='input')


def create_model(input_tensor, K):
    x = conv1_layer(input_tensor)
    x = conv2_layer(x)
    x = conv3_layer(x)
    x = conv4_layer(x)
    x = conv5_layer(x)

    x = GlobalAveragePooling2D()(x)
    output_tensor = Dense(K, activation='softmax')(x)

    model = Model(input_tensor, output_tensor)

    return model


resnet50 = create_model(input_tensor, K)
resnet50.summary()

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
resnet50.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

image_size = 224


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_dir = os.path.join('dataset/Defectimage_202000821/train')
val_dir = os.path.join('dataset/Defectimage_202000821/val')

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=16, target_size=(224, 224), color_mode='rgb')
val_generator = val_datagen.flow_from_directory(val_dir, batch_size=16, target_size=(224, 224), color_mode='rgb')

fit_history = resnet50.fit_generator(
        train_generator,
        epochs=10,
        validation_data=val_generator,
)

resnet50.save("./working/defect_model_keras.h5")
