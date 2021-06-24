class Person:
    def __new__(cls):#, firsteName, secondName, date, position):
        cls._firstName = ""
        cls._secondName = ""
        cls._dateBirth = ""
        cls._position = ""
        # cls._firstName = firsteName
        # cls._secondName = secondName
        # cls._dateBirth = date
        # cls._position = position

    @property
    def firstName(cls):
        return cls._firstName

    @firstName.setter
    def firstName(cls, _firstName):
        cls._firstName = _firstName

    @property
    def secondName(cls):
        return cls._secondName

    @secondName.setter
    def secondName(cls, _secondName):
        cls._secondName = _secondName

    @property
    def dateBirth(cls):
        return cls._dateBirth

    @dateBirth.setter
    def dateBirth(cls, _dateBirth):
        cls._dateBirth = _dateBirth

    @property
    def position(cls):
        return cls._position

    @position.setter
    def position(cls, _position):
        cls._position = _position
