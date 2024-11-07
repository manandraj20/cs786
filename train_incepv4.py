import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def read_dataset(filename='alldata.data'):
    import gzip, pickle
    with (gzip.open if filename.endswith('.gz') else open)(filename, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def preprocessing_img(x):
    return np.asarray(x).astype(np.float32) / 255.0

def build_inceptionV4(input_shape):
    # Load InceptionV4 from TensorFlow Hub as base model
    base_model = hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v4/feature_vector/5", 
                                trainable=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return base_model, model

def main(argv=['ds1', 'None', '0']):
    ds, resume_model, im_noise = argv[0], argv[1] if argv[1] != 'None' else None, int(argv[2]) == 1
    nb_epoch2 = 70
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess data
    tx, ty = read_dataset(ds + '/train.data')
    vx, vy = read_dataset(ds + '/valid.data')
    tx, vx = preprocessing_img(tx), preprocessing_img(vx)
    tx, vx = np.repeat(tx[:, np.newaxis, :, :], 3, axis=1), np.repeat(vx[:, np.newaxis, :, :], 3, axis=1)
    ty, vy = to_categorical(ty, 2), to_categorical(vy, 2)
    
    # Build model
    base_model, model = build_inceptionV4(input_shape=(200, 200, 3))

    datagen = ImageDataGenerator(
        preprocessing_function=add_random_noise if im_noise else None,
        rotation_range=5, width_shift_range=0.02, height_shift_range=0.02,
        horizontal_flip=True, vertical_flip=True
    )
    
    if resume_model:
        model.load_weights(resume_model)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint2 = ModelCheckpoint("models/" + ds + "_incep4-{epoch:02d}-{val_accuracy:.3f}.hdf5",
                                  monitor='val_accuracy', save_best_only=True, save_weights_only=True)
    early_stopper = EarlyStopping(monitor='val_accuracy', patience=40, min_delta=0.0001)
    
    model.fit(datagen.flow(tx, ty, batch_size=40, shuffle=True),
              steps_per_epoch=len(ty) // 40, epochs=nb_epoch2,
              validation_data=(vx, vy), callbacks=[early_stopper, checkpoint2])

    model.save_weights(ds + "_incep4-%d.hdf5" % nb_epoch2)
    print("Saved last weights to disk")

if __name__ == "__main__":
    main(sys.argv[1:])