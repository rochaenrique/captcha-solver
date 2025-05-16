import os
from glob import glob
import numpy as np
import pandas as pd
from profiler import Profile

import tensorflow as tf
import keras
from keras import utils
from keras.preprocessing import image
from keras import layers, models, callbacks, optimizers

from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

import sys
if len(sys.argv) < 2:
    print('Error: Expected train, test, and submit filenames')
    print(f'Usage: {sys.argv[0]} <submit.csv>')
    exit(1)

submit_file = sys.argv[1]

SEED=1337

def try_gpu(): 
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Detected {len(gpus)} GPU(s). Using GPU acceleration.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected. Running on CPU.")


def load_imgs(img_dir):
    paths = glob(os.path.join(img_dir, "*.png"))
    X, y, ids = [], [], []
    for p in paths:
        label = os.path.basename(p).split('.')[0]
        img = image.load_img(p, color_mode='grayscale')
        data = image.img_to_array(img).astype(np.uint8)
        X.append(data)
        y.append(int(label))
        ids.append(label)
    return np.array(X), np.array(y), ids

def svm_clean(X):
    X_clean = []
    for img in X:
        flat = img.reshape(-1, 1)
        svm = OneClassSVM(gamma='auto', nu=0.01)
        svm.fit(flat)

        pred = svm.predict(flat)
        mask = pred.reshape(img.shape) == -1

        med = np.median(img)
        clean = img.copy()
        clean[mask] = med
        X_clean.append(clean)
        
    return np.array(X_clean)

def prep(X, y, num_classes):
    X = X.astype('float32') / 255.0
    X = np.expand_dims(X, -1)
    y = utils.to_categorical(y, num_classes)
    return X, y

def build(input_shape, num_classes):
    model = models.Sequential([
    ])
    model.compile(
        optimizer=optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == '__main__':
    # with Profile(f'Trying gpus'):
    #     try_gpu()
    
    train_dir = 'data/train'
    with Profile(f'Loading images from {train_dir}'):
        X_train, y_train, ids = load_imgs(train_dir)

    with Profile(f'Cleaning {len(X_train)} with svm'):
        X_train = svm_clean(X_train)

    with Profile(f'Split train test'):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
        )

    with Profile(f'Preprocess'):
        num_classes = len(np.unique(y_train))
        X_val, y_val = prep(X_val, y_val, num_classes)

    with Profile('Build model'):
        input_shape = X_tr.shape[1:]
        model = build_model(input_shape, num_classes)

    best_model = 'build/best.h5'
    with Profile('Setting callbacks'):
        point = callbacks.ModelCheckpoint(
            best_model, save_best_only=True, monitor='val_accuracy', mode='max'
        )
        earlystop = callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True
        )

    with Profile('Building model'):
        model.fit(
            X_tr, y_tr,
            validations_data=(X_val, y_val),
            epochs=50,
            batch_size=64,
            callbacks=[point, earlystop]
        )

    with Profile('Loadin weights'):
        model.load_weights(best_model)


    test_dir = 'data/train'
    with Profile('Submission'):
        X_test, _, test_ids = load_imgs(test_dir)
        X_test = svm_clean(X_test)
        X_test = X_test.astype('float32') / 255.0
        X_test = np.expand_dims(X_test, -1)

    with Profile('Predict'):
        preds = model.predict(X_test)
        results = np.argmax(preds, axis=1)

    with Profile('Create submission'):
        subm = pd.DataFrame({'id': test_ids, 'results': results})
        subm.to_csv(submit_file, index=False)

    print("-------- DONE! --------")
