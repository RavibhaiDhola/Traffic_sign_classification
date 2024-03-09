import pickle
import numpy as np
import pandas as pd
from preprocess import preprocess
from keras.utils.np_utils import to_categorical



def data_load():
    
    '''
    Data loader function for reading and loading the images
    
    returns: inputs, targets
    
    '''

    #load the data
    with open('german-traffic-signs/train.p', 'rb') as f:
        train_data = pickle.load(f)
    with open('german-traffic-signs/valid.p', 'rb') as f:
        val_data = pickle.load(f)
    with open('german-traffic-signs/test.p', 'rb') as f:
        test_data = pickle.load(f)

     # Split out features and labels
    X_train, y_train = train_data['features'], train_data['labels']
    X_val, y_val = val_data['features'], val_data['labels']
    X_test, y_test = test_data['features'], test_data['labels']

    y_labels =  y_test

    #already 4 dimensional
    print(X_train.shape)
    print(X_test.shape)
    print(X_val.shape)

    assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
    assert(X_train.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."
    assert(X_val.shape[0] == y_val.shape[0]), "The number of images is not equal to the number of labels."
    assert(X_val.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."
    assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels."
    assert(X_test.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."
    data = pd.read_csv('german-traffic-signs/signnames.csv')


    X__train = np.array(list(map(preprocess, X_train)))
    X__test = np.array(list(map(preprocess, X_test)))
    X__val = np.array(list(map(preprocess, X_val)))


    # Merge inputs and targets
    inputs = np.concatenate((X__train, X__val, X__test), axis=0)
    targets = np.concatenate((y_train,y_val,y_test), axis=0)

    inputs = inputs.reshape(inputs.shape[0], 32, 32, 1)
    targets = to_categorical(targets, 43)

    return inputs, targets
