import tensorflow as tf
import matplotlib.pyplot as plt
from time import *
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import callbacks

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ??????????????????????imgheight*imgwidth????????batch
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    # ?????
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=123,
        #color_mode="grayscale",
        image_size=(img_height, img_width),
        batch_size=batch_size)
    # ?????
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        #color_mode="grayscale",
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    # ?????????????????
    return train_ds, val_ds, class_names


# ??CNN??
def model_load(IMG_SHAPE=(224, 224,3), class_num=10):
    # ????
    model = tf.keras.models.Sequential([
        # ????????????0-255??????????0?1??
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),
        # ????????????32???????????3*3??????relu
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # ?????????kernel???2*2
        tf.keras.layers.MaxPooling2D(2, 2),
        # Add another convolution
        # ???????64??????????3*3??????relu
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # ??????????2*2?????????
        tf.keras.layers.MaxPooling2D(2, 2),
        # ???????????
        tf.keras.layers.Flatten(),
        # The same 128 dense layers, and 10 output layers as in the pre-convolution example:
        tf.keras.layers.Dense(128, activation='relu'),
        # ??softmax????????????????????????softmax?????
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    # ??????
    model.summary()
    # ??????????????sgd????????????????
    opt = keras.optimizers.SGD(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # ????
    return model


# ?????????
def show_loss_acc(history):
    # ?history??????????????????????
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # ???????????
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('Spatial/results_png.png', dpi=100)


def train(epochs):
    # ???????????
    begin_time = time()
    # todo ?????? ???????????
    train_ds, val_ds, class_names = data_load("./Spatial/slice_png_train/train",
                                              "./Spatial/slice_png_train/val", 224, 224, 32)
    print(train_ds)
    print(val_ds)
    print(class_names)

    db_train = train_ds.shuffle(10000)
    db_train = db_train.batch(32)

    db_val = val_ds.shuffle(10000)
    db_val = db_val.batch(32)
    # ????
    model = model_load(class_num=len(class_names))

    # early stopping
    earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=1)

    mc = ModelCheckpoint(filepath='./Spatial/models/cnn_slice_model.h5',
                         monitor='val_accuracy',
                         mode='max',
                         verbose=1,
                         save_best_only=True)

    # ???????epoch?????
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1, callbacks=[mc, earlystop])
    # ???????epoch?????
   # history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    # todo ????? ?????????????
    model.save("./Spatial/models/cnn_picture_video_text.h5")
    # ??????
    end_time = time()
    run_time = end_time - begin_time
    print('??????????', run_time, "s")  # ?????????? 1.4201874732
    # ?????????
    show_loss_acc(history)


if __name__ == '__main__':
    train(epochs=1500)
