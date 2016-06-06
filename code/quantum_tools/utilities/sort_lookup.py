import collections

class SortLookup():

    def __init__(self, data):
        for d in data:
            assert(isinstance(d, str))
        self.list = data
        self.dict = dict((a,b) for b,a in enumerate(data))
        self.__size = len(data)

    def __getitem__(self, i):
        if not isinstance(i, str) and isinstance(i, collections.Iterable):
            return [self[ii] for ii in i]
        else:
            if isinstance(i, str):
                return self.dict[i]
            elif isinstance(i, int):
                return self.list[i]
            else:
                raise Exception("SortLookup object only supports ints and strings bi-lookup.")

    def __len__(self):
        return self.__size
