import GUI
import cv2

from Network import Network
from Stream import Stream


def trainNetwork(event, model, params):
    if event == "Обучить":
        print('Выполняется обучение...')
        try:
            model = Network.start(params)
        except Exception as inst:
            print(inst)
    return model


def readact(params):
    print('Редактирование...')
    params = GUI.editWin(params)
    return params


def recognition(params, sg, model):
    print('Распознавание...')
    ftypes = [('Изображения', '*.jpg'), ('Изображения', '*.pgm'), ('Все файлы', '*')]
    dlg = sg.filedialog.Open(filetypes=ftypes)
    fl = dlg.show()
    if fl != '':
        img = cv2.imread(fl)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (params['width'], params['height']), interpolation=cv2.INTER_AREA)
        if model is not None:
            try:
                Network.test(img, model, params)
            except Exception as inst:
                print(inst)
        else:
            print("Сеть не обучена.")


def loadModel(model, sg, params):
    print('Загрузка модели...')
    ftypes = [('Веса', '*.h5'), ('Все файлы', '*')]
    dlg = sg.filedialog.Open(filetypes=ftypes)
    fl = dlg.show()
    if fl != '':
        model = Network.modelLoad(params, fl)
        print('Модель загружена.')
    return model


def saveModel(model):
    if model is not None:
        Network.modelSave(model)
        print('Модель сохранена.')
    else:
        print("Ошибка сохранения.")


def startCam(params, model):
    print('Запуск камеры.')
    if model is not None:
        Stream.start(params['width'], params['height'], model)
    else:
        print("Сеть не обучена.")
