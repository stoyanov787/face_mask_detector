"""This module creates face mask recognition model"""

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import os

INIT_LR = 1e-4  # initial learning rate
EPOCHS = 20
BATCH_SIZE = 32

DIRECTORY = r'dataset'  # path to dataset directory
CATEGORIES = ('with_mask', 'without_mask')  # names of subdirectories in dataset

data = []
labels = []


def img_sort():
    """Gets the pictures from the dataset and sort it

    :param path: The path to subdirectories of the dataset
    :type path: str
    :param img_path: The path to images of the subdirectories
    :type img_path: str
    :param imgage: The loaded image, converted to array
    :type imgage: numpy.ndarray
    """

    for category in CATEGORIES:
        path = os.path.join(DIRECTORY, category)  # path to subdirectories
        for img in os.listdir(path):
            img_path = os.path.join(path, img)  # path to image
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            data.append(image)
            labels.append(category)


def labels_encoding():
    """Encode the labels and make np.array the lables and data"""

    global labels
    global data

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)  # make the names of categories(str) in labels to binary representation
    labels = to_categorical(labels)  # make binary representation into categories
    data = np.array(data, dtype='float32')
    labels = np.array(labels)


def construct_base_head_model():
    """Construct the base model and the head model

    :param base_model: Create base model
    :type base_model: keras.engine.functional.Functional
    :param head_model: Create head model
    :type head_model: keras.engine.keras_tensor.KerasTensor
    :returns: the head_model and the base_model
    """

    # construct the base module
    base_model = MobileNetV2(weights='imagenet', include_top=False,
                             input_tensor=Input(shape=(224, 224, 3)))

    # construct the head module
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name='flatten')(head_model)
    head_model = Dense(128, activation='relu')(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(2, activation='softmax')(head_model)

    return head_model, base_model


def create_model():
    """Create, train and save face mask recognition model

    :param aug: Construct the training image generator for data augmentation
    :type aug: keras.preprocessing.image.ImageDataGenerator
    :param model: The model that will be trained
    :type model: keras.engine.functional.Functional
    :param opt: The optimizer for the model compilation
    :type opt: keras.optimizer_v2.adam.Adam
    """

    train_x, test_x, train_y, test_y = train_test_split(data, labels,
                                                        test_size=0.20, stratify=labels, random_state=42)

    # construct the training image generator for data augmentation
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest')

    # construct the model that will be trained
    head_model, base_model = construct_base_head_model()
    model = Model(inputs=base_model.input, outputs=head_model)

    # freeze all layers from the base model to not be included in the first training
    for layer in base_model.layers:
        layer.trainable = False

    # optimizer for the model compilation
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

    # compile the model
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy'])

    # train the head
    model.fit(
        aug.flow(train_x, train_y, batch_size=BATCH_SIZE),
        steps_per_epoch=len(train_x) // BATCH_SIZE,
        validation_data=(test_x, test_y),
        validation_steps=len(test_x) // BATCH_SIZE,
        epochs=EPOCHS)

    model.save(r'mask_recognition.h5', save_format='h5')
    print('The model is saved!')


def run():
    """Run the program and make the model"""

    img_sort()
    labels_encoding()
    create_model()
