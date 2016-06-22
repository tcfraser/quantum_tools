class SortedFrozenSet(frozenset):

    @classmethod
    def __sort__(cls, elem):
        return elem

    def __new__(cls, *args, **kwargs):
        args, kwargs = cls.__sanitize_init_args__(*args, **kwargs)
        return super(SortedFrozenSet,cls).__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        elems = args[0]
        should_sort = not kwargs.get('sorted', False)
        elems = list(super().__iter__())
        # print(self.__class__)
        if should_sort:
            elems.sort(key=self.__class__.__sort__)
        self.__sorted_elems = elems
        self.__hash_map = None
        # print(args, kwargs)
        self.__post_init__(*args, **kwargs)

    def __post_init__(self, *args, **kwargs):
        pass

    def __getitem__(self, slice):
        return self.__sorted_elems[slice]

    def __get_hash_map(self):
        if self.__hash_map is None:
            self.__hash_map = dict((hash(self.__sorted_elems[i]), i) for i in range(len(self.__sorted_elems)))
        return self.__hash_map

    def index(self, elem):
        try:
            idx = self.__get_hash_map()[hash(elem)]
        except KeyError as e:
            raise KeyError("Element {0} not in set.".format(str(elem)))
        return idx

    @classmethod
    def __sanitize_init_args__(cls, *args, **kwargs):
        return args, kwargs

    # def _get_sort():
    #     if not hasattr(self, '__sorted_elems'):
    #         self.__sorted_elems = list(self).sort(key=self.__class__.__sort__)
    #     return self.__sorted_elems

    def __iter__(self):
        for i in self.__sorted_elems:
            yield i

class SubClassSFS(SortedFrozenSet):

    @classmethod
    def __sort__(cls, elem):
        return elem[0]

    def __post_init__(self):
        self.is_sub_class = True

class MinimalFrozenSet(frozenset):
    def __new__(cls, data):
        return super(MinimalFrozenSet,cls).__new__(cls, data)

_wrap_methods = [
    '__ror__',
    'symmetric_difference',
    '__rsub__',
    '__and__',
    '__rand__',
    'intersection',
    'difference',
    'union',
    '__or__',
    'copy',
    '__rxor__',
    '__xor__',
    '__sub__',
]

def WrapMethods(cls, target_super=frozenset, names=_wrap_methods):
    def wrap_method_closure(name):
        super_function = getattr(target_super, name)
        def inner(self, *args):
            return cls(super_function(self, *args))
        inner.fn_name = name
        setattr(cls, name, inner)
    for name in names:
        wrap_method_closure(name)

WrapMethods(SortedFrozenSet)
WrapMethods(SubClassSFS)

def tests():
    # a = SortedFrozenSet(['a', 'b', 'c', 'd'])
    # b = SortedFrozenSet(['f', 'e', 'c', 'c'])
    c = SubClassSFS([('f',), ('e',), ('c',), ('c',)])
    d = SubClassSFS([('f','j'), ('h',), ('c',), ('c',)])
    # e = SubClassSFS(['c', 'c'])
    # d = SubClassSFS([('f','j'), ('h',), ('c',), ('c',)])
    # f = MinimalFrozenSet(['a', 'a', 'b'])
    # print(f)
    # print(a)
    print(c)
    # print(b)
    print(d)
    # print(a.union(b))
    print(c.union(d))
    print(c.intersection(d))
    # print(frozenset.__new__(frozenset, ['c', 'c']))
    # print(d.is_sub_class)
    # print(e)
    # print(d)
    # print(list(d))
    # print(hash(d))
    # print(d.union(c))
    # for i in a.union(b):
    #     print(i)
    # print(a ^ b)

if __name__ == '__main__':
    tests()