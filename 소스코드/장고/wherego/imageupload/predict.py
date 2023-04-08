import joblib 
from keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input 
import numpy as np

def pred_picture(img):
    kmeans_model = joblib.load('imageupload/static/model/kmeans.pkl')
    model = InceptionV3()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

    reshaped_img = img.reshape(1,299,299,3) 

    imgx = preprocess_input(reshaped_img)

    feat = model.predict(imgx, use_multiprocessing=True)

    feat = np.array(list(feat))
    feat = feat.reshape(-1, 2048)

    return kmeans_model.predict(feat) # nature