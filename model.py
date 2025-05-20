import os
import gc
import cv2
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

SEED = 6969 
NUM_DIGITS = 6
BATCH_SIZE = 64
SHUFFLE = 1_000
CHARS = string.digits + string.ascii_lowercase + string.ascii_uppercase
char_to_idx = { c: i for i, c in enumerate(CHARS)}
idx_to_char = { i: c for c, i in char_to_idx.items()}

tf.config.optimizer.set_jit(True)
np.random.seed(SEED)

def encode_label(label):
    s = str(label)
    if len(s) < NUM_DIGITS: 
        s = s.rjust(NUM_DIGITS, '0')
    elif len(s) > NUM_DIGITS:
        s = s[:NUM_DIGITS]
    return [char_to_idx[c] for c in s]

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
    if not os.path.exists(cache_path):
        print('Cache not found, loading...')
        X, y, ids = load_imgs(img_dir, id_to_label)
        print('Cleaning data')
        X = clean(X)
        y = np.array(y, dtype=np.int32) if id_to_label is not None else None

        np.savez(cache_path.strip('.npz'),
                X=X, 
                y=y,
                ids=np.array(ids))

        del X, y, ids
        gc.collect()

    print(f'Loading \'{cache_path}\' in mmap mode')
    data = np.load(cache_path, mmap_mode='r', allow_pickle=True)
    return data['X'], data['y'], data['ids']

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
    return X, y, ids

def clean(X):
    clean = []
    for img in X:
        img = img.squeeze()
        img = cv2.fastNlMeansDenoising(img, None, h=15, templateWindowSize=7, searchWindowSize=21)
#        img = cv2.GaussianBlur(img, (3,3), 0)
#        img = cv2.adaptiveThreshold(img, 255, 
#                                    cv2.ADAPTIVE_THRESH_MEAN_C,
#                                    cv2.THRESH_BINARY_INV, 
#                                    11, 2
#                                    )
        clean.append(img)
    return np.expand_dims(np.array(clean), -1).astype('float32') / 255.

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
        loss=['sparse_categorical_crossentropy'] * NUM_DIGITS,
        metrics=['accuracy'] * NUM_DIGITS
    )
    return model

def make_dataset(X, y, indices, shuffle=True):
    print('Making dataset')

    idx = np.array(indices, dtype=np.int32)
    N = idx.shape[0]

    def generator(idx): 
        for i in idx:
            yield (X[i], *y[i])

    output_sig = (
            tf.TensorSpec(shape=X.shape[1:], dtype=tf.float32),
            *[tf.TensorSpec(shape=(), dtype=tf.int32) for _ in range(NUM_DIGITS)],
            )

    ds = tf.data.Dataset.from_generator(
            lambda: generator(idx), 
            output_signature=output_sig
            )
    if shuffle:
        ds = ds.shuffle(len(idx), seed=SEED)

    ds = ds.batch(BATCH_SIZE)

    ds = ds.map(
            lambda x, *y: (x, y),
            num_parallel_calls=tf.data.AUTOTUNE
            )
    return ds.prefetch(tf.data.AUTOTUNE)

if __name__ == '__main__':
    with Profile(f'Trying gpus'):
        try_gpu()
    
    train_dir = 'data/train/train'
    train_labels_path = 'data/train.csv'
    train_cache = 'build/train_cache.npz'

    test_dir = 'data/test/test'
    test_cache = 'build/test_cache.npz'

    with Profile(f'Loading train labels \'{train_labels_path}\''):
        label_df = pd.read_csv(train_labels_path)
        id_to_label = dict(zip(label_df['Id'], label_df['Label']))

    with Profile(f'Loading train data'):
        X_train, y_train, ids = load_clean_imgs(train_dir, train_cache, id_to_label)

    with Profile('Loading test data'):
        X_test, _, test_ids = load_clean_imgs(test_dir, test_cache)

    with Profile('Created data batches'):
        N = X_train.shape[0]
        perm = np.random.permutation(N)
        split_at = int(0.8 * N)
        train_idx, val_idx = perm[:split_at], perm[split_at:]
        train_ds = make_dataset(X_train, y_train, train_idx)
        val_ds = make_dataset(X_train, y_train, val_idx, False)

    with Profile('Building model'):
        model = build(X_train.shape[1:])

    best_model = 'build/best.h5'
    with Profile('Setting callbacks'):
        point = ModelCheckpoint(
            best_model, save_best_only=True, monitor='val_accuracy', mode='max'
        )
        earlystop = EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True
        )

    with Profile('Fitting'):
        print('Let the games begin!')
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=20,
            callbacks=[point, earlystop]
        )

    with Profile('Loading weights'):
        model.load_weights(best_model)

    with Profile('Predict'):
        test_ds = tf.data.Dataset.from_tensor_slices(X_test).batch(BATCH_SIZE)
        predictions = model.predict(test_ds)
        results = []
        for pred in zip(*predictions):
            results.append(''.join([idx_to_char[np.argmax(dl)] for dl in pred]))

    with Profile('Create submission'):
        subm = pd.DataFrame({'Id': test_ids, 'Label': results})
        subm.to_csv(submit_file, index=False)

    print("-------- DONE! --------")



