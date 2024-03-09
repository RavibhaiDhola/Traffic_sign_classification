
import matplotlib.pyplot as plt
from random import randint, choice


def plot_train(data, X_train, y_train):
    '''
    data: total images
    X_train: images (2D np.array())
    y_train: labels of images
    '''
    num_of_samples=[]
    cols = 5
    num_classes = 43

    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,50))
    fig.tight_layout()

    for i in range(cols):
        for j, row in data.iterrows():
            x_selected = X_train[y_train == j]
            axs[j][i].imshow(x_selected[randint(0,(len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
            axs[j][i].axis("off")
            if i == 2:
                axs[j][i].set_title(str(j) + " - " + row["SignName"])
                num_of_samples.append(len(x_selected))
    print(num_of_samples)
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, num_classes), num_of_samples)
    plt.title("Distribution of the train dataset")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()
