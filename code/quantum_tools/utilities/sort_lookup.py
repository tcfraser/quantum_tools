class SortLookup():

    def __init__(self, data):
        for d in data:
            assert(isinstance(d, str))
        self.list = data
        self.dict = dict((a,b) for b,a in enumerate(data))
        self.__size = len(data)

    def __getitem__(self, i):
        if isinstance(i, str):
            return self.dict[i]
        else:
            return self.list[i]

    def __len__(self):
        return self.__size
