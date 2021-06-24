import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math

import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout, MaxPooling2D, Flatten, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import load_model

from Parameters import Parameters


class Network:
    dirs = []
    listNames = []
    parameters = Parameters()

    @classmethod
    def start(cls):
        cls.dirs = os.listdir(os.path.abspath(os.getcwd()) + "/" + cls.parameters.folder)
        X, Y = cls.get_data()

        print(len(Y), len(X))
        model = cls.creatModel(X)
        print(K.eval(model.optimizer.lr))
        history = model.fit([X[:, 0], X[:, 1]], Y, batch_size=64, epochs=cls.parameters.num_epochs, verbose=2,
                            validation_split=.25)
        print(K.eval(model.optimizer.lr))
        cls.graph(history)

        return model

    @classmethod
    def test(cls, img, model):
        print("Тест...")
        cls.dirs = os.listdir(os.path.abspath(os.getcwd()) + "/" + cls.parameters.folder)
        matrImage = cls.readDataTest()

        masImg = np.zeros([1, img.shape[0], img.shape[1], 1])
        masImg[0, :, :, 0] = img
        masImg = masImg / 255
        list = []

        # распознаём изображение
        i = 0
        while i < cls.parameters.numMan * cls.parameters.sample:
            #img = matrImage[i] * 255
            #cv2.imwrite("image/" + str(i) + ".jpg", img[0,:,:,:])
            res = model.predict([masImg, matrImage[i]])
            list.append(res)
            #print(res, str(i))
            i += 1

        minZn = min(list)
        if minZn < 0.4:
            p = list.index(minZn)
            #print(minZn)
            p += 1
            print(cls.dirs[math.ceil(p / cls.parameters.sample) - 1])
        else:
            print("Изображение не идентифицировано.")

    @classmethod
    def creatModel(cls, mas):
        img_a = Input(shape=mas.shape[2:])
        img_b = Input(shape=mas.shape[2:])
        base_network = cls.build_base_network(mas.shape[2:])
        # Получаем вектора признаков
        feat_vecs_a = base_network(img_a)
        feat_vecs_b = base_network(img_b)
        distance = Lambda(cls.euclidean_distance)([feat_vecs_a, feat_vecs_b])
        rms = RMSprop(learning_rate=cls.parameters.learning_rate)
        model = Model([img_a, img_b], distance)

        model.compile(loss=cls.contrastive_loss, optimizer=rms, metrics=[cls.accuracy])
        #opt = tensorflow.keras.optimizers.Adam(lr=0.001)
        #model.compile(optimizer=opt, loss='contrastive_crossentropy', metrics=['accuracy'])

        return model


    @classmethod
    def get_data(cls):
        total_sample_size = cls.parameters.total_sample_size
        width = cls.parameters.width
        height = cls.parameters.height
        sample = cls.parameters.sample
        numMan = cls.parameters.numMan
        nameFolder = cls.parameters.folder
        img = cv2.imread(nameFolder + "\\" + cls.dirs[0] + "\\" + str(1) + '.' + cls.parameters.typeel)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dim1 = image.shape[0]
        dim2 = image.shape[1]

        count = 0

        x_geuine_pair = np.zeros([total_sample_size, 2, dim1, dim2, 1])
        y_genuine = np.zeros([total_sample_size, 1])

        for i in range(numMan):
            print(cls.dirs[i])
            #for j in range(int((numMan * (numMan - 1)) / 2)):
            for j in range(int(total_sample_size / numMan)):
                ind1 = 0
                ind2 = 0
                # read images from same directory (genuine pair)
                while ind1 == ind2:
                    ind1 = np.random.randint(sample)
                    ind2 = np.random.randint(sample)

                # read the two images
                img1 = cv2.imread(nameFolder + "\\" + cls.dirs[i] + "\\" + str(ind1 + 1) + "." + cls.parameters.typeel)
                img2 = cv2.imread(nameFolder + "\\" + cls.dirs[i] + "\\" + str(ind2 + 1) + "." + cls.parameters.typeel)

                img1 = cv2.resize(img1, (width, height), interpolation=cv2.INTER_AREA)
                img2 = cv2.resize(img2, (width, height), interpolation=cv2.INTER_AREA)
                # to gray
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

                # store the images to the initialized numpy array
                x_geuine_pair[count, 0, :, :, 0] = img1
                x_geuine_pair[count, 1, :, :, 0] = img2

                # as we are drawing images from the same directory we assign label as 1. (genuine pair)
                y_genuine[count] = 1
                count += 1

        count = 0
        x_imposite_pair = np.zeros([total_sample_size, 2, dim1, dim2, 1])
        y_imposite = np.zeros([total_sample_size, 1])

        #for i in range(int((numMan * (numMan - 1)) / 2)):
        for i in range(int(total_sample_size / sample)):
            for j in range(sample):
                ind1 = 0
                ind2 = 0
                # read images from different directory (imposite pair)
                while ind1 == ind2:
                    ind1 = np.random.randint(numMan)
                    ind2 = np.random.randint(numMan)

                img1 = cv2.imread(nameFolder + "\\" + cls.dirs[ind1] + "\\" + str(j + 1) + "." + cls.parameters.typeel)
                img2 = cv2.imread(nameFolder + "\\" + cls.dirs[ind2] + "\\" + str(j + 1) + "." + cls.parameters.typeel)

                img1 = cv2.resize(img1, (width, height), interpolation=cv2.INTER_AREA)
                img2 = cv2.resize(img2, (width, height), interpolation=cv2.INTER_AREA)
                # to gray
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

                x_imposite_pair[count, 0, :, :, 0] = img1
                x_imposite_pair[count, 1, :, :, 0] = img2
                # as we are drawing images from the different directory we assign label as 0. (imposite pair)
                y_imposite[count] = 0
                count += 1

        # now, concatenate, genuine pairs and imposite pair to get the whole data
        X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0) / 255
        Y = np.concatenate([y_genuine, y_imposite], axis=0)

        return X, Y

    @classmethod
    def build_base_network(cls, input_shape):
        """
        filters - кол-во выходных фильтров
        kernel_size - ширина и высота ядра двумерной свертки
        pool_size - размер окна пулинга
        strides - шаг сдвига окна
        activation - функция активации
        """
        seq = Sequential()

        nb_filter = [6, 12]
        kernel_size = 3

        seq.add(Conv2D(filters=nb_filter[0], kernel_size=(kernel_size, kernel_size), padding='valid',
                       input_shape=input_shape, activation='relu'))
        seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        seq.add(Dropout(0.25))

        seq.add(Conv2D(filters=nb_filter[1], kernel_size=(kernel_size, kernel_size), padding='valid', activation='relu'))
        seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        seq.add(Dropout(0.25))

        seq.add(Flatten())
        seq.add(Dense(100, activation='softmax'))

        return seq

    @classmethod
    def continueTraining(cls, model):
        cls.dirs = os.listdir(os.path.abspath(os.getcwd()) + "/" + cls.parameters.folder)
        newX, newY = cls.get_data()
        history = model.fit([newX[:, 0], newX[:, 1]], newY, validation_split=.25, batch_size=64,
                            epochs=cls.parameters.num_epochs, verbose=2)
        cls.graph(history)

        return model

    @classmethod
    def euclidean_distance(cls, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    @classmethod
    def contrastive_loss(cls, y_true, y_pred):
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    @classmethod
    def compute_accuracy(cls, predictions, labels):
        pred = labels.ravel() < 0.5
        return np.mean(pred == predictions)

    @classmethod
    def accuracy(cls, y_true, y_pred):
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

    @classmethod
    def graph(cls, history):
        plt.subplot(2, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.title('Train accuracy')
        plt.ylabel('Точность')
        plt.xlabel('Эпохи')
        plt.legend(['Обучающая'], loc='upper left')

        plt.subplot(2, 2, 2)
        plt.plot(history.history['loss'])
        plt.title('Train loss')
        plt.ylabel('Потеря')
        plt.xlabel('Эпохи')
        plt.legend(['Обучающая'], loc='upper left')

        plt.subplot(2, 2, 3)
        plt.plot(history.history['val_accuracy'])
        plt.title('Val accuracy')
        plt.ylabel('Точность')
        plt.xlabel('Эпохи')
        plt.legend(['Обучающая'], loc='upper left')

        plt.subplot(2, 2, 4)
        plt.plot(history.history['val_loss'])
        plt.title('Val loss')
        plt.ylabel('Потеря')
        plt.xlabel('Эпохи')
        plt.legend(['Обучающая'], loc='upper left')
        plt.show()

    @classmethod
    def modelSave(cls, model):
        #model.save_weights('Models/model_w.h5')
        model.save('Models/model_w.h5')

    @classmethod
    def modelLoad(cls, fl):
        mas = np.zeros([cls.parameters.total_sample_size * 2, 2, cls.parameters.height, cls.parameters.width, 1])
        new_model = cls.creatModel(mas)
        new_model.load_weights(fl)
        return new_model

    @classmethod
    def readDataTest(cls):
        count = 0
        sample = cls.parameters.sample
        numMan = cls.parameters.numMan
        width = cls.parameters.width
        height = cls.parameters.height
        matrImage = np.zeros([sample * numMan, 1, height, width, 1])
        for i in range(0, numMan):
            for j in range(0, sample):
                img = cv2.imread(cls.parameters.folder + "\\" + cls.dirs[i] + "\\" + str(j + 1) +'.' + cls.parameters.typeel)
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                matrImage[count, 0, :, :, 0] = img
                count += 1
        matrImage = matrImage / 255
        return matrImage
