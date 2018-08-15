# Keras Preprocessing

[![Build Status](https://travis-ci.org/keras-team/keras-preprocessing.svg?branch=master)](https://travis-ci.org/keras-team/keras-preprocessing)

Keras Preprocessing is the data preprocessing
and data augmentation module of the Keras deep learning library.
It provides utilities for working with image data, text data,
and sequence data.

Read the documentation at: https://keras.io/

Keras Preprocessing may be imported directly
from an up-to-date installation of Keras:

```
from keras import preprocessing
```

Keras Preprocessing is compatible with Python 2.7-3.6
and is distributed under the MIT license.

# Keras Preprocessing (Multi-Threading)

the repo can be placed in local folder and imported as following:
```
sys.path.insert(0,"path-to-code/keras-preprocessing")
import keras_preprocessing.image as T
```


usage:
```
import multiprocessing.dummy
    
n_process = 16
pool = multiprocessing.dummy.Pool(processes=n_process)
train_datagen = T.ImageDataGenerator(
    ...,
    pool=pool
    )
```
