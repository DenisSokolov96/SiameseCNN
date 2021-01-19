import PySimpleGUI as sg

from Stream import *
from Binding import *


def mainWind(params):
    model = None
    sg.theme('Light Green')
    menu_def = [['&Выбрать задачу', ['&Обучить', '&Распознать', '&Включить камеру', '&Сохранить модель']],
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
            model = trainNetwork(event, model, params)

        if event == "Редактирование":
            params = readact(params)

        if event == "Распознать":
            recognition(params, sg, model)

        if event == "Загрузить модель":
            model = loadModel(model, sg, params)

        if event == 'Сохранить модель':
            saveModel(model)

        # добавить распознавание внутрь камеры
        if event == "Включить камеру":
            startCam(params, model)

        if event in (sg.WIN_CLOSED, 'Quit'):
            break
    window.close()


def editWin(params):
    layout1 = [[sg.Text("Ширина"), sg.Input(key='-Width-', default_text=params['width'], size=(7, 1), justification='center'),
                sg.Text("Высота"), sg.Input(key='-Height-', default_text=params['height'], size=(7, 1), justification='center')],
               [sg.Text("Максимальное количество примеров"),
                sg.Input(key='-MaxSam-', default_text=params['total_sample_size'], size=(7, 1), justification='center')],
               [sg.Text("Эпохи"), sg.Input(key='-Epoch-', default_text=params['num_epochs'], size=(7, 1), justification='center')],
               [sg.Button("Применить")]]
    sg.theme('Reds')
    window = sg.Window("Параметры", layout1)
    while True:
        event, values = window.read()
        if event == "Применить":
            flag = False
            try:
                params['num_epochs'] = int(values['-Epoch-'])
                params['total_sample_size'] = int(values['-MaxSam-'])
                params['width'] = int(values['-Width-'])
                params['height'] = int(values['-Height-'])
                flag = True
            except ValueError:
                print("Ошибка ввода")
            if flag:
                break
        if event in (sg.WIN_CLOSED, 'Quit'):
            break
    window.close()
    return params