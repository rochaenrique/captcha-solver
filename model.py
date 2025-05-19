import os
import string
from glob import glob
import numpy as np
import pandas as pd
from profiler import Profile

import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.preprocessing import image
from keras import layers, models, callbacks, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from collections import Counter

import cv2
import sys

if len(sys.argv) < 2:
    print('Error: Expected train, test, and submit filenames')
    print(f'Usage: {sys.argv[0]} <submit.csv>')
    exit(1)

submit_file = sys.argv[1]

SEED = 1337
NUM_DIGITS = 6
CHARS = string.digits + string.ascii_lowercase + string.ascii_uppercase
char_to_idx = { c: i for i, c in enumerate(CHARS)}
idx_to_char = { i: c for c, i in char_to_idx.items()}

def encode_label(label):
    return [char_to_idx[c] for c in str(label)]

def decode_label(encoded):
    return ''.join([idx_to_char[i] for i in encoded])

def try_gpu(): 
    gpus = tf.config.list_physical_devices('GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(len(gpus), 'Physical GPUs, ', len(logical_gpus), 'Logical GPUs')
        except RuntimeError as e:
            print(e)
    else:
        print("No GPUs detected")

def load_clean_imgs(img_dir, cache_path, id_to_label=None):
    if os.path.exists(cache_path):
        print(f'Loading cached from \'{cache_path}\'')
        data = np.load(cache_path, allow_pickle=True)
        return data['X'], data['y'], data['ids'].tolist()
    else:
        print(f'Cache not found, loading...')
        X, y, ids = load_imgs(img_dir, id_to_label)
        X = clean(X)
        np.savez(cache_path, X=X, y=y, ids=np.array(ids))
    return X, y, ids 

def load_imgs(img_dir, id_to_label=None):
    skipped = 0
    X, y, ids = [], [], []

    files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
    print(f'Found {len(files)} .png images in {img_dir}')
    
    for filename in files:
        img_id = int(filename[:-4])
        
        if id_to_label is not None and id_to_label.get(img_id) is None:
            print(f'[WARN] Skipping unknown image: {img_id}')
            skipped += 1
            continue

        # Load image as grayscale and convert to array
        img_path = os.path.join(img_dir, filename)
        img = image.load_img(img_path, color_mode='grayscale')
        img_array = image.img_to_array(img).astype(np.uint8)

        X.append(img_array)
        ids.append(img_id)

        if id_to_label is not None:
            y.append(encode_label(id_to_label[img_id]))

    print(f'Skipped {skipped} images')
    return np.array(X), np.array(y), ids

def clean(X):
    clean = []
    for img in X:
        img = img.squeeze()
        img = cv2.fastNlMeansDenoising(img, None, h=15, templateWindowSize=7, searchWindowSize=21)
        img = cv2.GaussianBlur(img, (3,3), 0)
        img = cv2.adaptiveThreshold(img, 255, 
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 
                                    11, 2
                                    )
        clean.append(img)
    return np.expand_dims(np.array(clean), -1).astype('float32') / 255.

def split(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=SEED)

def prep_labels(y):
    y = np.stack(np.array(y))
    return [to_categorical(y[:, i], num_classes=len(CHARS)) for i in range(NUM_DIGITS)]

def build(input_shape=(80, 200, 1), num_classes=len(CHARS)):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)

    outputs = [Dense(num_classes, activation='softmax', name=f'char_{i}')(x) for i in range(NUM_DIGITS)]
    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == '__main__':
    with Profile(f'Trying gpus'):
        try_gpu()
    
    train_dir = 'data/train/train'
    train_labels_path = 'data/train.csv'
    train_cache = 'build/train_cache.npy'

    with Profile(f'Loading train labels \'{train_labels_path}\''):
        label_df = pd.read_csv(train_labels_path)
        id_to_label = dict(zip(label_df['Id'], label_df['Label']))

    with Profile(f'Loading train data'):
        X_train, y_train, ids = load_clean_imgs(train_dir, train_cache, id_to_label)

    test_dir = 'data/test/test'
    test_cache = 'build/test_cache.npy'
    with Profile('Loading test data'):
        X_test, _, test_ids = load_clean_imgs(test_dir, test_cache)

    with Profile('Split train test'):
        X_tr, X_val, y_tr, y_val = split(X_train, y_train)

    with Profile('Build model'):
        input_shape = X_tr.shape[1:]
        model = build(input_shape, num_classes)

    best_model = 'build/best.h5'
    with Profile('Setting callbacks'):
        point = ModelCheckpoint(
            best_model, save_best_only=True, monitor='val_accuracy', mode='max'
        )
        earlystop = EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True
        )

    with Profile('Building model'):
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=64,
            callbacks=[point, earlystop]
        )

    with Profile('Loading weights'):
        model.load_weights(best_model)

    with Profile('Predict'):
        predictions = model.predict(X_test)
        results = []
        for pred in zip(*predictions):
            results.append(''.join([idx_to_char[np.argmax(dl)] for dl in pred]))

    with Profile('Create submission'):
        subm = pd.DataFrame({'Id': test_ids, 'Label': results})
        subm.to_csv(submit_file, index=False)

    print("-------- DONE! --------")
