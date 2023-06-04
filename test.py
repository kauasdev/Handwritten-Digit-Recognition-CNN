import numpy as np
from PIL import Image

def resize_array(array, new_shape):
    # Reduzir a escala do array para (28, 28, 4)
    scaled_array = array[::8, ::8, :]

    # Converter para escala de cinza
    grayscale_array = np.mean(scaled_array, axis=2)

    # Redimensionar para (28, 28, 1)
    resized_array = grayscale_array.reshape((*new_shape[:-1], 1))

    return resized_array

# Exemplo de uso
array = np.random.random((224, 224, 4))  # Substitua pelo seu array de entrada
resized_array = resize_array(array, (28, 28, 1))
print(resized_array.shape)
