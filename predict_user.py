import pickle
import numpy as np
from matplotlib.pyplot import imread
import skimage.transform
from dnn_app_utils import load_data, L_model_forward

filename = 'parameters/data.pkl'
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
num_px = train_x_orig.shape[1]

a_file = open(filename, "rb")
parameters = pickle.load(a_file)
a_file.close()

def organize_image(uploaded_file, num_px = num_px):
    '''organizing the input data'''

    image = np.array(imread(uploaded_file))
    image = image/255.
    my_image = skimage.transform.resize(image,(num_px,num_px)).reshape((1, num_px*num_px*3)).T

    return my_image

def predict(X, parameters = parameters, classes = classes):
    '''This function is used to predict the results of a  L-layer neural network.'''
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
        
    return p, classes