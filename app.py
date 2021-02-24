from werkzeug.utils import secure_filename
import numpy as np
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

IMG_FOLDER = '/tmp/'

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

    dog_names = ['Affenpinscher',
                 'Afghan hound',
                 'Airedale terrier',
                 'Akita',
                 'Alaskan malamute',
                 'American eskimo dog',
                 'American foxhound',
                 'American staffordshire terrier',
                 'American water spaniel',
                 'Anatolian shepherd dog',
                 'Australian cattle dog',
                 'Australian shepherd',
                 'Australian terrier',
                 'Basenji',
                 'Basset hound',
                 'Beagle',
                 'Bearded collie',
                 'Beauceron',
                 'Bedlington terrier',
                 'Belgian malinois',
                 'Belgian sheepdog',
                 'Belgian tervuren',
                 'Bernese mountain dog',
                 'Bichon frise',
                 'Black and tan coonhound',
                 'Black russian terrier',
                 'Bloodhound',
                 'Bluetick coonhound',
                 'Border collie',
                 'Border terrier',
                 'Borzoi',
                 'Boston terrier',
                 'Bouvier des flandres',
                 'Boxer',
                 'Boykin spaniel',
                 'Briard',
                 'Brittany',
                 'Brussels griffon',
                 'Bull terrier',
                 'Bulldog',
                 'Bullmastiff',
                 'Cairn terrier',
                 'Canaan dog',
                 'Cane corso',
                 'Cardigan welsh corgi',
                 'Cavalier king charles spaniel',
                 'Chesapeake bay retriever',
                 'Chihuahua',
                 'Chinese crested',
                 'Chinese shar-pei',
                 'Chow chow',
                 'Clumber spaniel',
                 'Cocker spaniel',
                 'Collie',
                 'Curly-coated retriever',
                 'Dachshund',
                 'Dalmatian',
                 'Dandie dinmont terrier',
                 'Doberman pinscher',
                 'Dogue de bordeaux',
                 'English cocker spaniel',
                 'English setter',
                 'English springer spaniel',
                 'English toy spaniel',
                 'Entlebucher mountain dog',
                 'Field spaniel',
                 'Finnish spitz',
                 'Flat-coated retriever',
                 'French bulldog',
                 'German pinscher',
                 'German shepherd dog',
                 'German shorthaired pointer',
                 'German wirehaired pointer',
                 'Giant schnauzer',
                 'Glen of imaal terrier',
                 'Golden retriever',
                 'Gordon setter',
                 'Great dane',
                 'Great pyrenees',
                 'Greater swiss mountain dog',
                 'Greyhound',
                 'Havanese',
                 'Ibizan hound',
                 'Icelandic sheepdog',
                 'Irish red and white setter',
                 'Irish setter',
                 'Irish terrier',
                 'Irish water spaniel',
                 'Irish wolfhound',
                 'Italian greyhound',
                 'Japanese chin',
                 'Keeshond',
                 'Kerry blue terrier',
                 'Komondor',
                 'Kuvasz',
                 'Labrador retriever',
                 'Lakeland terrier',
                 'Leonberger',
                 'Lhasa apso',
                 'Lowchen',
                 'Maltese',
                 'Manchester terrier',
                 'Mastiff',
                 'Miniature schnauzer',
                 'Neapolitan mastiff',
                 'Newfoundland',
                 'Norfolk terrier',
                 'Norwegian buhund',
                 'Norwegian elkhound',
                 'Norwegian lundehund',
                 'Norwich terrier',
                 'Nova scotia duck tolling retriever',
                 'Old english sheepdog',
                 'Otterhound',
                 'Papillon',
                 'Parson russell terrier',
                 'Pekingese',
                 'Pembroke welsh corgi',
                 'Petit basset griffon vendeen',
                 'Pharaoh hound',
                 'Plott',
                 'Pointer',
                 'Pomeranian',
                 'Poodle',
                 'Portuguese water dog',
                 'Saint bernard',
                 'Silky terrier',
                 'Smooth fox terrier',
                 'Tibetan mastiff',
                 'Welsh springer spaniel',
                 'Wirehaired pointing griffon',
                 'Xoloitzcuintli',
                 'Yorkshire terrier']

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
    if not os.path.exists('/tmp/'):
        directory = "tmp"
        path = os.path.join(directory)
        os.mkdir(path)

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
            file.save(_get_img_path(filename))
            preds = Xception_predict_breed(_get_img_path(filename))
            print(preds)
            # return send_from_directory(app.config['IMG_FOLDER'], filename)
            # return redirect(url_for('display_image', filename=filename))
            flash('Thist dog is a {}'.format(preds))
            # return redirect('/')
            return render_template('upload.html', filename=filename)

        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)


@ app.route('/displayimage/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return send_from_directory(app.config['IMG_FOLDER'], filename)


if __name__ == "__main__":
    # app.run()
    app.run(host='0.0.0.0', port=3000, debug=True)
