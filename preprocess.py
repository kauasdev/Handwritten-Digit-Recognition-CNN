import numpy as np


def reshape_data(
        data: tuple[np.ndarray, np.ndarray],
        img_width=28,
        img_height=28,
        img_channels=1
):

    x_train_reshaped = data[0].reshape((
        data[0].shape[0],
        img_width,
        img_height,
        img_channels
    ))

    x_test_reshaped = data[1].reshape((
        data[1].shape[0],
        img_width,
        img_height,
        img_channels
    ))

    return x_train_reshaped, x_test_reshaped


def normalize_data(data: tuple[np.ndarray, np.ndarray]):
    x_train_normalized = data[0] / 255
    x_test_normalized = data[1] / 255

    return x_train_normalized, x_test_normalized
