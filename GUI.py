import os

import PySimpleGUI as sg

from Binding import *


def mainWind():
    model = None
    sg.theme('Light Green')
    menu_def = [['&Выбрать задачу', ['&Обучить', '&Распознать', '&Включить камеру', '&Сохранить модель',
                                     '&Продолжить обучение модели']],
                ['&Параметры обучения', ['&Редактирование', '&Загрузить модель']]]
    layout = [
        [sg.Menu(menu_def, tearoff=False)],
        [sg.Text("Инфо:")],
        [sg.Output(size=(88, 20), key='out')]
    ]
    window = sg.Window('SCNN', layout)
    while True:
        event, values = window.read()

        if event == "Обучить":
            # getInfo()
            model = trainNetwork(model)

        if event == "Редактирование":
            readact()

        if event == "Распознать":
            recognition(sg, model)

        if event == "Загрузить модель":
            model = loadModel(model, sg)

        if event == 'Сохранить модель':
            saveModel(model)

        if event == 'Продолжить обучение модели':
            model = continueTraining(model)

        # добавить распознавание внутрь камеры
        if event == "Включить камеру":
            startCam(model)

        if event in (sg.WIN_CLOSED, 'Quit'):
            break
    window.close()


def editWin():
    parameters = Parameters()
    listFolder = filter(os.path.isdir, os.listdir(os.path.abspath(os.getcwd())))
    layout1 = [[sg.Text("Ширина"), sg.Input(key='-Width-', default_text=parameters.width, size=(7, 1), justification='center'),
                sg.Text("Высота"), sg.Input(key='-Height-', default_text=parameters.height, size=(7, 1), justification='center')],
               [sg.Text("Максимальное количество примеров"),
                sg.Input(key='-MaxSam-', default_text=parameters.total_sample_size, size=(7, 1), justification='center')],
               [sg.Text("Кол-во классов"), sg.Input(key='-NumCl-', default_text=parameters.numMan, size=(7, 1), justification='center')],
               [sg.Text("Изображений в классе"), sg.Input(key='-Sample-', default_text=parameters.sample, size=(7, 1), justification='center')],
               [sg.Text("Эпохи"), sg.Input(key='-Epoch-', default_text=parameters.num_epochs, size=(7, 1), justification='center')],
               [sg.Text("Скорость обучения"), sg.Input(key='-Lr-', default_text=parameters.learning_rate, size=(7, 1), justification='center')],
               [sg.Text("Формат изображений"), sg.Combo(["jpg", "pgm"], enable_events=True, key='-Combo-', default_value=parameters.typeel, size=(5, 1))],
               [sg.Text("Dataset"), sg.Combo([folder for folder in listFolder], enable_events=True, key='-Folder-', size=(15, 1), default_value=parameters.folder)],
               [sg.Button("Применить")]]
    sg.theme('Reds')
    window = sg.Window("Параметры", layout1)
    while True:
        event, values = window.read()
        if event == "Применить":
            flag = False
            try:
                parameters.num_epochs = int(values['-Epoch-'])
                parameters.total_sample_size = int(values['-MaxSam-'])
                parameters.width = int(values['-Width-'])
                parameters.height = int(values['-Height-'])
                parameters.numMan = int(values['-NumCl-'])
                parameters.sample = int(values['-Sample-'])
                parameters.learning_rate = float(values['-Lr-'])
                parameters.typeel = values['-Combo-']
                parameters.folder = values['-Folder-']
                flag = True
            except ValueError:
                print("Ошибка ввода")
            if flag:
                break
        if event in (sg.WIN_CLOSED, 'Quit'):
            break
    window.close()