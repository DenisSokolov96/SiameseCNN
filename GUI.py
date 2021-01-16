import PySimpleGUI as sg

from Stream import *


def mainWind(num_epochs, total_sample_size, width, height, sample, numMan):
    model = None
    sg.theme('Light Green')
    menu_def = [['&Выбрать задачу', ['&Обучить', '&Распознать', '&Включить камеру', '&Сохранить модель']],
                ['&Параметры обучения', ['&Редактирование', '&Загрузить модель']]]
    layout = [
        [sg.Menu(menu_def, tearoff=False)],
        [sg.Text("Инфо:")],
        [sg.Output(size=(88, 20), key='out')]
    ]
    window = sg.Window('SNN', layout)
    while True:
        event, values = window.read()

        if event == "Обучить":
            print('Выполняется обучение...')
            try:
                model = Network.start(num_epochs, total_sample_size, width, height, sample, numMan)
            except Exception as inst:
                print(inst)

        if event == "Редактирование":
            print('Редактирование...')
            num_epochs, total_sample_size, width, height = editWin(num_epochs, total_sample_size, width, height)

        if event == "Распознать":
            print('Распознавание...')
            ftypes = [('Изображения', '*.jpg'), ('Изображения', '*.pgm'), ('Все файлы', '*')]
            dlg = sg.filedialog.Open(filetypes=ftypes)
            fl = dlg.show()
            if fl != '':
                img = cv2.imread(fl)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (width, height),  interpolation=cv2.INTER_AREA)
                if model is not None:
                    try:
                        Network.test(img, model, sample, numMan,  width, height)
                    except Exception as inst:
                        print(inst)
                else:
                    print("Сеть не обучена.")

        if event == "Загрузить модель":
            print('Загрузка модели...')
            model = Network.modelLoad(total_sample_size, height, width)
            print('Модель загружена.')

        if event == 'Сохранить модель':
            if model is not None:
                Network.modelSave(model)
                print('Модель сохранена.')
            else:
                print("Ошибка сохранения.")

        # добавить распознавание внутрь камеры
        if event == "Включить камеру":
            print('Запуск камеры.')
            if model is not None:
                Stream.start(width, height, model)
            else:
                print("Сеть не обучена.")
        if event in (sg.WIN_CLOSED, 'Quit'):
            break
    window.close()


def editWin(num_epochs, total_sample_size, width, height):
    layout1 = [[sg.Text("Ширина"), sg.Input(key='-Width-', default_text=width, size=(7, 1), justification='center'),
                sg.Text("Высота"), sg.Input(key='-Height-', default_text=height, size=(7, 1), justification='center')],
               [sg.Text("Максимальное количество примеров"),
                sg.Input(key='-MaxSam-', default_text=total_sample_size, size=(7, 1), justification='center')],
               [sg.Text("Эпохи"), sg.Input(key='-Epoch-', default_text=num_epochs, size=(7, 1), justification='center')],
               [sg.Button("Применить")]]
    sg.theme('Reds')
    window = sg.Window("Параметры", layout1)
    while True:
        event, values = window.read()
        if event == "Применить":
            flag = False
            try:
                num_epochs = int(values['-Epoch-'])
                total_sample_size = int(values['-MaxSam-'])
                width = int(values['-Width-'])
                height = int(values['-Height-'])
                flag = True
            except ValueError:
                print("Ошибка ввода")
            if flag:
                break
        if event in (sg.WIN_CLOSED, 'Quit'):
            break
    window.close()
    return num_epochs, total_sample_size, width, height