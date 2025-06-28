
from flask import Flask, render_template, request, jsonify, url_for, redirect
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
import os
import tensorflow as tf
app = Flask(__name__)
model = tf.keras.models.load_model('internship12.keras')
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/predict', methods=['GET', 'POST'])
def output():
    if request.method == 'POST':
        f = request.files['pc_image']
        img_path = "static/uploads/" + f.filename
        f.save(img_path)

        img = load_img(img_path, target_size=(224, 224))
        image_array = np.array(img)
        image_array = np.expand_dims(image_array, axis=0)

        pred = np.argmax(model.predict(image_array), axis=1)
        index = ['Biodegradable Images (1)', 'Recyclable Images (1)', 'Trash Images (2)']
        prediction = index[int(pred)]

        return render_template("portfolio-details.html", predict=prediction)
if __name__=="__main__":
    
    # app.run(host='0.0.0.0', port=8000,debug=True)    # running the app
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
