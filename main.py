import numpy as np
from keras.utils.np_utils import to_categorical
from data_loader  import data_load
from model import CNN_model
from utils import plot_train
from sklearn.model_selection import KFold
from keras.preprocessing.image import ImageDataGenerator


 
def main(inputs, targets):
    '''
    inputs: Images of traffic signs
    targets: Labels
    '''

    # Data augmentation 
    datagen = ImageDataGenerator(width_shift_range=0.1,
                                height_shift_range=0.1,
                                zoom_range=0.2,
                                shear_range=0.1,
                                rotation_range=10.)
    
    
    # Define the K-fold Cross Validator
    num_folds = 10
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(inputs, targets):
        
        model = CNN_model()
        
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        # Fit data to model
        history = model.fit(datagen.flow(inputs[train], targets[train]),
                    batch_size=500,
                    epochs=10,
                    verbose=1)

        # Generate generalization metrics
        scores = model.evaluate(datagen.flow(inputs[test], targets[test]), verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1


    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
        print('------------------------------------------------------------------------')



if __name__ == "__main__":

    # Data loader 
    inputs, targets = data_load()
    main(inputs, targets)
