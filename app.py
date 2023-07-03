from flask import Flask, render_template, request, jsonify
from PIL import Image
from keras.models import load_model
import numpy as np
import json




app = Flask(__name__)

# Cargar el modelo
loaded_model = load_model('ModeloFinal.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener la imagen cargada desde la solicitud
    image = request.files['input-image']

    pil_image = Image.open(image)

    resized_image = pil_image.resize((32, 32))

    # Convertir la imagen a un arreglo numpy
    image_array = np.array(resized_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)


    
    # Realizar la predicción utilizando el modelo cargado
    
    prediction = loaded_model.predict(image_array).tolist()
    prediction_rounded = [round(value, 2) for value in prediction[0]]

    response = {
        'prediction': {
            'akiec': prediction_rounded[0],
            'bcc': prediction_rounded[1],
            'bkl': prediction_rounded[2],
            'df': prediction_rounded[3],
            'mel': prediction_rounded[4],
            'nv': prediction_rounded[5],
            'vasc': prediction_rounded[6]
        }
    }


    return jsonify(response)

if __name__ == '__main__':
    app.run()