import GUI
import cv2

from Network import Network
from Parameters import Parameters
from Stream import Stream


def trainNetwork(event, model):
    if event == "Обучить":
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
        img = cv2.resize(img, (Parameters.width, Parameters.height), interpolation=cv2.INTER_AREA)
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


def saveModel(model):
    if model is not None:
        Network.modelSave(model)
        print('Модель сохранена.')
    else:
        print("Ошибка сохранения.")


def startCam(model):
    print('Запуск камеры.')
    if model is not None:
        Stream.start(Parameters.width, Parameters.height, model)
    else:
        print("Сеть не обучена.")