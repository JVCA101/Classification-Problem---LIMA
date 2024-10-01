import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import ResNet50, DenseNet121
import numpy as np
import matplotlib.pyplot as plt

from models import *

from sklearn import metrics

import os



def main(name_model):
    batch_size = 16

    image_shape = (224, 224, 3)


    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.getcwd() + '/Train',
        image_size=(224,224),
        batch_size=batch_size,
        label_mode = 'categorical'
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.getcwd() + '/Test',
        image_size=(224,224),
        batch_size=batch_size,
        label_mode = 'categorical'
    )

    num_classes = len(train_ds.class_names)

    class_names = train_ds.class_names
    print(class_names)



    # model
    model = models(name_model, image_shape)

    model.compile(optimizer=SGD(learning_rate=1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

    model.summary()

    history = model.fit(train_ds, epochs=80)

    train_acc = history.history["accuracy"]
    train_loss = history.history["loss"]

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)


    # print results
    print('\nTest accuracy:', test_acc)

    # write loss and accuracy to file
    f = open("output/acc_loss.txt", "a")
    f.write(model.name + " " + str(test_loss) + " ")
    y_pred = model.predict(test_ds)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = []
    for _, labels in test_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_true = np.array(y_true)

    def class_accuracy(y_true, y_pred, class_label):
        indices = y_true == class_label
        return metrics.accuracy_score(y_true[indices], y_pred[indices])

    accuracies = {}
    for class_label in range(num_classes):
        acc = class_accuracy(y_true, y_pred, class_label)
        accuracies[class_label] = acc

    for class_label in accuracies:
        f.write(str(accuracies[class_label]) + " ")

    f.write(str(test_acc) + "\n")
    f.close()

    # plot
    epochs = range(1, len(train_acc) + 1)

    plt.plot(epochs, train_acc, 'b', label='Training acc')
    plt.title('Training accuracy')
    plt.legend()
    plt.savefig("output/" + model.name + "_acc.png")
    plt.figure()

    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.title('Training loss')
    plt.legend()
    plt.savefig("output/" + model.name + "_loss.png")

    # plt.show()


    # confusion matrix
    predictions = np.array([])
    labels =  np.array([])
    for x, y in test_ds:
        predictions = np.concatenate([predictions, np.argmax(model.predict(x, verbose=0), axis = -1)])
        labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

    confusion_matrix = metrics.confusion_matrix(labels, predictions)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=class_names).plot(cmap=plt.cm.Reds)

    cm_display.figure_.savefig("output/" + model.name + "_confusion_matrix.png")

main("lenet5")
main("alex")
main("vgg16")
main("inception")
# main("resnet50")
# main("densenet")