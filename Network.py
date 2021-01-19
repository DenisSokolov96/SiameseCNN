import datetime

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout, MaxPooling2D, Flatten, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop


class Network:
    #nameFolder = "transports"
    nameFolder = "att_faces1"
    dirs = os.listdir(os.path.abspath(os.getcwd()) + "/" + nameFolder)
    listNames = []
    #typeel = 'jpg'
    typeel = 'pgm'

    @classmethod
    def start(cls, params):
        X, Y = cls.get_data(params)
        print(len(Y), len(X))
        model = cls.creatModel(X)
        history = model.fit([X[:, 0], X[:, 1]], Y, batch_size=48, epochs=params['num_epochs'], verbose=2)
        cls.graph(history)

        return model

    @classmethod
    def test(cls, img, model, params):
        print("Тест...")
        matrImage = cls.readDataTest(params)

        masImg = np.zeros([1, img.shape[0], img.shape[1], 1])
        masImg[0, :, :, 0] = img
        masImg = masImg / 255
        list = []

        # распознаём изображение
        i = 0
        while i < params['numMan'] * params['sample']:
            #img = matrImage[i] * 255
            #cv2.imwrite("image/" + str(i) + ".jpg", img[0,:,:,:])
            res = model.predict([masImg, matrImage[i]])
            list.append(res)
            #print(res, str(i))
            i += 1

        minZn = min(list)
        if minZn < 0.035:
            p = list.index(minZn)
            #print(minZn)
            p += 1
            print(cls.dirs[math.ceil(p / params['sample']) - 1])
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
        # distance = Lambda(cls.euclidean_distance, output_shape=cls.eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])
        distance = Lambda(cls.euclidean_distance)([feat_vecs_a, feat_vecs_b])
        # rms = RMSprop()
        model = Model([img_a, img_b], distance)

        # model.compile(loss=cls.contrastive_loss, optimizer=rms, metrics=[cls.accuracy])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model


    @classmethod
    def get_data(cls, params):
        total_sample_size = params['total_sample_size']
        width = params['width']
        height = params['height']
        sample = params['sample']
        numMan = params['numMan']
        img = cv2.imread(cls.nameFolder + "\\" + cls.dirs[0] + "\\" + str(1) + '.' + cls.typeel)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dim1 = image.shape[0]
        dim2 = image.shape[1]

        count = 0

        x_geuine_pair = np.zeros([total_sample_size, 2, dim1, dim2, 1])
        y_genuine = np.zeros([total_sample_size, 1])

        for i in range(numMan):
            print(cls.dirs[i])
            for j in range(int(total_sample_size / numMan)):
                ind1 = 0
                ind2 = 0
                # read images from same directory (genuine pair)
                while ind1 == ind2:
                    ind1 = np.random.randint(sample)
                    ind2 = np.random.randint(sample)

                # read the two images
                img1 = cv2.imread(cls.nameFolder + "\\" + cls.dirs[i] + "\\" + str(ind1 + 1) + "." + cls.typeel)
                img2 = cv2.imread(cls.nameFolder + "\\" + cls.dirs[i] + "\\" + str(ind2 + 1) + "." + cls.typeel)

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

        for i in range(int(total_sample_size / sample)):
            for j in range(sample):
                ind1 = 0
                ind2 = 0
                # read images from different directory (imposite pair)
                while ind1 == ind2:
                    ind1 = np.random.randint(numMan)
                    ind2 = np.random.randint(numMan)

                img1 = cv2.imread(cls.nameFolder + "\\" + cls.dirs[ind1] + "\\" + str(j + 1) + "." + cls.typeel)
                img2 = cv2.imread(cls.nameFolder + "\\" + cls.dirs[ind2] + "\\" + str(j + 1) + "." + cls.typeel)

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
        seq = Sequential()

        nb_filter = [6, 12]
        kernel_size = 3

        seq.add(Conv2D(nb_filter[0], (kernel_size, kernel_size), input_shape=input_shape,
                       activation='relu'))
        seq.add(MaxPooling2D(pool_size=(2, 2)))
        seq.add(Dropout(0.25))

        seq.add(Conv2D(nb_filter[1], (kernel_size, kernel_size), activation='relu'))
        seq.add(MaxPooling2D(pool_size=(2, 2)))
        seq.add(Dropout(0.25))

        seq.add(Flatten())
        seq.add(Dense(100, activation='softmax'))
        #################################################################################
        # seq.add(Conv2D(32, kernel_size=(3, 3),
        #                  activation='relu',
        #                  input_shape=input_shape))
        # #Операция максимальной подвыборки
        # seq.add(MaxPooling2D(pool_size=(2, 2)))
        # # слой выравнивания
        # seq.add(Dropout(0.25))
        # #одно измерение
        # seq.add(Flatten())
        # # полносвязный слой
        # seq.add(Dense(128, activation='relu'))
        # # слой выравнивания
        # seq.add(Dropout(0.5))
        # # полносвязный слой
        # seq.add(Dense(10, activation='softmax'))
        ###################################################################################
        # seq.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=input_shape))
        # seq.add(MaxPooling2D())
        # seq.add(Conv2D(128, kernel_size=3, activation='relu'))
        # seq.add(Flatten())
        # seq.add(Dense(10, activation='softmax'))
        ###################################################################################
        return seq

    @classmethod
    def euclidean_distance(cls, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    @classmethod
    def eucl_dist_output_shape(cls, shapes):
        shape1, shape2 = shapes
        return shape1[0], 1

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
        plt.subplot(2, 1, 1)
        plt.plot(history.history['accuracy'])
        plt.title('Модель Точность')
        plt.ylabel('Точность')
        plt.xlabel('Эпохи')
        plt.legend(['Обучающая'], loc='upper left')

        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'])
        plt.title('Модель потерь')
        plt.ylabel('Потеря')
        plt.xlabel('Эпохи')
        plt.legend(['Обучающая'], loc='upper left')
        plt.show()

    @classmethod
    def modelSave(cls, model):
        model.save_weights('Models/model_w.h5')

    @classmethod
    def modelLoad(cls, params, fl):
        mas = np.zeros([params['total_sample_size'] * 2, 2, params['height'], params['width'], 1])
        new_model = cls.creatModel(mas)
        new_model.load_weights(fl)
        return new_model

    @classmethod
    def readDataTest(cls, params):
        count = 0
        sample = params['sample']
        numMan = params['numMan']
        width = params['width']
        height = params['height']
        matrImage = np.zeros([sample * numMan, 1, height, width, 1])
        for i in range(0, numMan):
            for j in range(0, sample):
                img = cv2.imread(cls.nameFolder + "\\" + cls.dirs[i] + "\\" + str(j + 1) +'.' + cls.typeel)
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                matrImage[count, 0, :, :, 0] = img
                count += 1
        matrImage = matrImage / 255
        return matrImage
