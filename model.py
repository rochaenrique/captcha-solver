import keras
from keras import utils

import sys
if len(sys.argv) < 2:
    print('Error: Expected train, test, and submit filenames')
    print(f'Usage: {sys.argv[0]} <submit.csv>')
    exit(1)

submit_file = sys.argv[1]

SEED=1337

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "data/train",
    validation_split=0.2,
    subset="both",
    seed=SEED,
    color_mode="rgba",
    image_size=(200, 80),
    batch_size=32,
    verbose=True
)

print(train_ds)
print(val_ds)
