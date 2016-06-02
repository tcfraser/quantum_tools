"""
    This was a very bad idea. Sets can only be taken of hashable types anyway.
"""
class MAM():
    """ Multi Association Map """

    def __init__(self, *args):
        for arg in args:
            assert(isinstance(arg, (list, tuple)))
            assert(len(arg) == len(set(arg)))
        for i in range(len(args) - 1):
            assert(len(args[i]) == len(args[i+1]))
        self.c = args

    def __getitem__(self, slice):
        return [data[slice] for data in self.c]

    def __call__(self, elem, at, to):
        return self.c[to][self.c[at].index(elem)]

    def multi(self, elems, at, to):
        return [self.c[to][i] for i, elem in enumerate(self.c[at]) if elem in elems]

def perform_tests():
    mam = MAM(["Cat", "Dog"], [0, 1], [{}, []])
    print(mam[0:2])
    print(mam("Cat", 0, 2))
    print(mam.multi(["Cat", "Dog"], 0, 2))

if __name__ == '__main__':
    perform_tests()