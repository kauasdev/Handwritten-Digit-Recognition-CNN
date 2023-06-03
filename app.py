import tensorflow as tf
from fastapi import FastAPI, UploadFile
from PIL import Image
import numpy as np
import uvicorn


PORT = 2940
HOST = '0.0.0.0'
app = FastAPI()
model = tf.keras.models.load_model('model.h5')


def process_img(img: UploadFile = UploadFile(...)) -> np.ndarray:
    # Open image
    img = Image.open(img.file)

    # Resize image to 28x28 pixels
    img_resized = img.resize((28, 28))

    # Convert to gray scale
    img_gray = img_resized.convert("L")

    # Convert to numpy array
    img_array = np.array(img_gray).reshape((1, 28, 28, 1))

    return img_array / 255


@app.get('/')
def index():
    return {"message": "Welcome to the API! To make a prediction just make a request to '/predict' with the image..."}


@app.post('/predict')
def predict(img: UploadFile = UploadFile(...)):
    processed_img = process_img(img)
    predictions = model.predict(processed_img)

    return {"result": str(np.argmax(predictions))}


if __name__ == '__main__':
    uvicorn.run('app:app', port=PORT, host=HOST, reload=True)
