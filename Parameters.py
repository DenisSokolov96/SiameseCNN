# для лиц 40 классов:  10 10 200 240 5 40
# для двух классов (auto - plane): 10 10 300 240 8 2
# для 4 классов : 30 8 200 240 4 4
"""
    Pattern singleton for storing parameters.
"""


class Parameters:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls._num_epochs = 20
            cls._width = 200
            cls._height = 240
            cls._sample = 8
            cls._numMan = 5
            cls._total_sample_size = 10  # int((numMan * (numMan - 1)) / 2)
            cls._learning_rate = 0.0001
            cls._typeel = "pgm"
            cls._folder = "att_faces1"
            cls.instance = super(Parameters, cls).__new__(cls)
        return cls.instance

    @property
    def num_epochs(cls):
        return cls._num_epochs

    @num_epochs.setter
    def num_epochs(cls, _num_epochs):
        cls._num_epochs = _num_epochs

    @property
    def width(cls):
        return cls._width

    @width.setter
    def width(cls, _width):
        cls._width = _width

    @property
    def height(cls):
        return cls._height

    @height.setter
    def height(cls, _height):
        cls._height = _height

    @property
    def sample(cls):
        return cls._sample

    @sample.setter
    def sample(cls, _sample):
        cls._sample = _sample

    @property
    def numMan(cls):
        return cls._numMan

    @numMan.setter
    def numMan(cls, _numMan):
        cls._numMan = _numMan

    @property
    def total_sample_size(cls):
        return cls._total_sample_size

    @total_sample_size.setter
    def total_sample_size(cls, _total_sample_size):
        cls._total_sample_size = _total_sample_size

    @property
    def typeel(cls):
        return cls._typeel

    @typeel.setter
    def typeel(cls, _typeel):
        cls._typeel = _typeel

    @property
    def folder(cls):
        return cls._folder

    @folder.setter
    def folder(cls, _folder):
        cls._folder = _folder

    @property
    def learning_rate(cls):
        return cls._learning_rate

    @learning_rate.setter
    def learning_rate(cls, _learning_rate):
        cls._learning_rate = _learning_rate
