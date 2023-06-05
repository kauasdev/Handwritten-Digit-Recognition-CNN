import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import base64
import numpy as np
import io
import uvicorn


class Request(BaseModel):
    img_base64: str


PORT = 2940
HOST = '0.0.0.0'
app = FastAPI()

origins = [
    "http://127.0.0.1:8080",
    "http://127.0.0.1:2940",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model('model.h5')


def base64_to_bytes(base64_str: str) -> bytes:
    base64_img = base64_str.split(',')[1]
    decoded_bytes = base64.b64decode(base64_img)

    return decoded_bytes


def bytes_to_image(bytes_data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(bytes_data)).resize((28, 28))

    return img


def sharpen_image(img: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Sharpness(img)
    image = enhancer.enhance(5.0)

    return image


def reduce_noise(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.MedianFilter)


def improve_image(image: Image.Image) -> Image.Image:
    image = sharpen_image(image)
    image = reduce_noise(image)

    return image


def image_to_numpy(image: Image) -> np.ndarray:
    return np.array(image)


def resize_image(image: np.ndarray) -> np.ndarray:
    # Reduce from (224, 224, 4) to (28, 28, 4)
    scaled_img = image[::8, ::8, :1]

    grayscale_img = np.mean(scaled_img, axis=2)

    resized_img = grayscale_img.reshape((28, 28, 1))

    return resized_img


def convert_to_grayscale(image_array: np.ndarray) -> np.ndarray:
    return np.mean(image_array, axis=2, keepdims=True).reshape((1, 28, 28, 1))


def normalize_img(image_array: np.ndarray) -> np.ndarray:
    return image_array / 255


def process_image(image_array: np.ndarray) -> np.ndarray:
    reshaped_img = convert_to_grayscale(image_array)
    normalized_img = normalize_img(reshaped_img)

    return normalized_img


@app.get('/')
def index():
    return {"message": "Welcome to the API! To make a prediction just make a request to '/predict' with the image..."}


@app.post('/predict')
def predict(req_body: Request):
    # Get base64 image
    base64_img_data = req_body.img_base64

    # Convert base64 to bytes
    bytes_from_base64 = base64_to_bytes(base64_img_data)

    # Convert bytes to image
    image_from_bytes = bytes_to_image(bytes_from_base64)

    # Convert image to numpy array
    image_array = image_to_numpy(image_from_bytes)

    # Process image
    processed_img = process_image(image_array)

    predictions = model.predict(processed_img)
    prediction = np.argmax(predictions)

    return {'data': str(prediction)}


if __name__ == '__main__':
    uvicorn.run('app:app', port=PORT, host=HOST, reload=True)
