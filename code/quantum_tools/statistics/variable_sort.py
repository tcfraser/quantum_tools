import re

def int_covert(text):
    return int(text) if text.isdigit() else text

def alphanum_key(key):
    return [int_covert(c) for c in re.split('([0-9]+)', key)]

def sort(seq):
    """ Sort the given iterable in the way that humans expect."""
    return sorted(seq, key=alphanum_key)

def argsort(seq):
    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    #by unutbu
    an_seq = list(alphanum_key(s) for s in seq)
    return sorted(range(len(seq)), key=an_seq.__getitem__)

def perform_tests():
    print(sort(['B1', 'A1']))
    print(argsort(['B1', 'A1']))
    print(argsort(['A01', 'A1']))

if __name__ == '__main__':
    perform_tests()