import os
import GUI
import cv2

from Network import Network
from Parameters import Parameters
from Stream import Stream
from Person import Person


def trainNetwork(model):
    print('Выполняется обучение...')
    try:
        model = Network.start()
    except Exception as inst:
        print(inst)
    return model


def readact():
    print('Редактирование...')
    GUI.editWin()


def recognition(sg, model):
    print('Распознавание...')
    ftypes = [('Изображения', '*.jpg'), ('Изображения', '*.pgm'), ('Все файлы', '*')]
    dlg = sg.filedialog.Open(filetypes=ftypes)
    fl = dlg.show()
    if fl != '':
        img = cv2.imread(fl)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (Parameters().width, Parameters().height), interpolation=cv2.INTER_AREA)
        if model is not None:
            try:
                Network.test(img, model)
            except Exception as inst:
                print(inst)
        else:
            print("Сеть не обучена.")


def loadModel(model, sg):
    print('Загрузка модели...')
    ftypes = [('Веса', '*.h5'), ('Все файлы', '*')]
    dlg = sg.filedialog.Open(filetypes=ftypes)
    fl = dlg.show()
    if fl != '':
        model = Network.modelLoad(fl)
        print('Модель загружена.')
    return model


def continueTraining(model):
    if model is not None:
        try:
            model = Network.continueTraining(model)
            print('Выполнено.')
        except Exception as inst:
            print(inst)
            print('Модель не загруженна.')
    return model


def saveModel(model):
    if model is not None:
        Network.modelSave(model)
        print('Модель сохранена.')
    else:
        print("Ошибка сохранения.")


def startCam(model):
    print('Запуск камеры.')
    if model is not None:
        Stream(Parameters().width, Parameters().height, model)
    else:
        print("Сеть не обучена.")


def getInfo():
    parameters = Parameters()
    folder = "s1"
    person = Person()
    path = os.path.abspath(os.getcwd()) + "/" + parameters.folder + "/" + folder + "/" + "Info.txt"
    try:
        handle = open(path, "r")
        for line in handle:
            masSplit = line.split(":")
            if masSplit[0] == "FIO":
                person.firstName = masSplit[1]
            elif masSplit[0] == "DateBirth":
                person.dateBirth = masSplit[1]
            elif masSplit[0] == "Position":
                person.position = masSplit[1]
        print(person.firstName)
        print(person.dateBirth)
        print(person.position)
        handle.close()

    except Exception as inst:
        print(inst)
        print('Ошибка получения данных.')
