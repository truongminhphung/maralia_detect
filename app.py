from flask import Flask, render_template, url_for, request
import cv2
import tensorflow as tf
from werkzeug.utils import redirect

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        img_url = request.form['img1']
        img_url = "images/" + img_url
        # detect
        result = run_detect(img_url)
        
        return render_template("index.html", result = result)
    
    return render_template('index.html')

def run_detect(img_url):
    class_name = ["Parasitized", "Uninfected"]
    img_url = "static/" +img_url 
    print(img_url)
    img = cv2.imread(img_url)
    # preprocess image suitable for efficently running model
    img_preprocess = preprocess_img(img)
    #load model to predict
    model = load_model()

    # prediction
    predictions = model.predict(img_preprocess)
    score = tf.nn.sigmoid(predictions)
    if score > 0.5:
        return class_name[1]
    else:
        return class_name[0]
    
    

def preprocess_img(img):
    img_resize = tf.image.resize(img, tf.constant([224,224]))
    img_preprocess = tf.expand_dims(img_resize, 0)
    print(img_preprocess.shape)
    return img_preprocess

def load_model():
    model = tf.keras.models.load_model('best_weights.hdf5')
    # model.summary()
    return model

if __name__ == '__main__':
    app.run(debug=True)