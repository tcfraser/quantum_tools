# import pyximport; pyximport.install()
from .utilities import integer_map

def test():
    a = integer_map.get_digits(22, [2,2,2])
    print(a)
    # a = np.array(list(itertools.product([0,1,2,3,4], [0,1,2], [0,1,2,3])))
    # im = IntMap((5,3,4))
    # print(im.get_digits(59))
    # print(im.get_integer((4,2,3)))
    # print(im.get_integer(np.array([[4,2,3],[4,2,3]])))


    # a = np.arange(11)
    # mask = np.array([0,3,4,7,8,9])
    # c_mask = comp_mask(mask, len(a))
    # print(a[c_mask])

if __name__ == '__main__':
    test()