from GUI import *

#для лиц 10 10 200 240 5 40
#для двух классов 10 10 300 240 8 2
def main():
    params = {'num_epochs': 10,
              'total_sample_size': 10,
              'width': 200,
              'height': 240,
              'sample': 5,
              'numMan': 40}
    mainWind(params)


if __name__ == '__main__':
    main()
