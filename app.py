from werkzeug.utils import secure_filename
import numpy as np
from glob import glob
from flask import Flask, flash, request, redirect, render_template, url_for, send_from_directory
import urllib.request
import os
from flask import Flask
from keras.preprocessing import image

import keras
app = Flask(__name__)

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

IMG_FOLDER = '/temp/'

app.config['IMG_FOLDER'] = IMG_FOLDER


def _get_img_path(image):
    return os.path.join(app.config['IMG_FOLDER'], image)


def load_model():
    # load the pre-trained Keras model
    global model
    model = keras.models.load_model('./keras_models')


def extract_Xception(tensor):
    from keras.applications.xception import Xception, preprocess_input
    return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def img_to_tensor(img):
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def Xception_predict_breed(img_path):
    dog_names = [item[20:-1]
                 for item in sorted(glob("../data/dogImages/train/*/"))]

    # extract bottleneck features
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    print(bottleneck_feature.shape)
    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            load_model()
            file.save(os.path.join(app.config['IMG_FOLDER'], filename))
            # preds = Xception_predict_breed(_get_img_path(filename))
            # print(preds)
            # return send_from_directory(app.config['IMG_FOLDER'], filename)
            return redirect(url_for('display_image', filename=filename))
            flash('File successfully uploaded')
            # return redirect('/')

        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)


@ app.route('/displayimages/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return send_from_directory(app.config['IMG_FOLDER'], filename)


if __name__ == "__main__":
    app.run()
