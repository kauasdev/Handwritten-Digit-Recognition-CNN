import datetime
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import plot_model
import numpy as np
from preprocess import reshape_data, normalize_data


IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_CHANNELS = 1


def load_data() -> tuple:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    return x_train, y_train, x_test, y_test


def preprocess(data: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x_train, x_test = data

    x_train, x_test = reshape_data((x_train, x_test), IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    x_train, x_test = normalize_data((x_train, x_test))

    return x_train, x_test


def create_model() -> Sequential:
    global IMG_WIDTH
    global IMG_HEIGHT
    global IMG_CHANNELS

    # Create model
    sequential_model = Sequential()

    # Add layers
    sequential_model.add(Convolution2D(
        input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS),
        kernel_size=5,
        filters=8,
        strides=1,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling()
    ))

    sequential_model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))

    sequential_model.add(Convolution2D(
        kernel_size=5,
        filters=16,
        strides=1,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling()
    ))

    sequential_model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ))

    sequential_model.add(Flatten())
    sequential_model.add(Dropout(.3))

    sequential_model.add(Dense(
        units=10,
        activation=tf.keras.activations.softmax,
        kernel_initializer=tf.keras.initializers.VarianceScaling()
    ))

    # Compile the model
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    sequential_model.compile(
        optimizer=adam_optimizer,
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    return sequential_model


def create_tensorboard_callback() -> tf.keras.callbacks.TensorBoard:
    log_dir = f'.logs/fit/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )


def create_early_stop() -> tf.keras.callbacks.EarlyStopping:
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=20
    )


def training_model(model_to_train: Sequential):
    x_train, y_train, x_test, y_test = load_data()
    x_train, x_test = preprocess((x_train, x_test))

    tensorboard_callback = create_tensorboard_callback()
    early_stop = create_early_stop()

    model_to_train.fit(
        x=x_train,
        y=y_train,
        epochs=100,
        validation_data=(x_test, y_test),
        callbacks=[tensorboard_callback, early_stop]
    )

    evaluate_model(model_to_train, train_data=(x_train, y_train), test_data=(x_test, y_test))


def save_model(model_to_save: Sequential, model_name: str):
    model_to_save.save(model_name, save_format='h5')


def evaluate_model(
        model_to_evaluate: Sequential,
        train_data: tuple[np.ndarray, np.ndarray],
        test_data: tuple[np.ndarray, np.ndarray]
) -> None:
    train_loss, train_accuracy = model_to_evaluate.evaluate(*train_data)
    validation_loss, validation_accuracy = model_to_evaluate.evaluate(*test_data)

    print(f"""
    - Train
    Training Loss: {train_loss}
    Training Accuracy: {train_accuracy}
    
    - Validation
    Validation Loss: {validation_loss}
    Validation Accuracy: {validation_accuracy}
    """)


if __name__ == '__main__':
    model = create_model()
    training_model(model)

    if not os.path.exists('model.h5'):
        pass
    else:
        os.remove('model.h5')

    save_model(model, 'model.h5')
