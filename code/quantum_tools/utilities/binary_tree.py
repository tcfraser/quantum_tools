class Node:
    def __init__(self):
        self.l = None
        self.r = None
        self.v = None

class PositivityTree(Node):
    def __init__(self):
        super().__init__()

    def add_val(self, path, val):
        path_iter = iter(path)
        self.__iter_add_val(path_iter, val)

    def __iter_add_val(self, path_iter, val):
        path_head = next(path_iter, None)
        if path_head is not None:
            if path_head >= 0:
                # Going Right
                if self.r is None:
                    self.r = PositivityTree()
                self.r.__iter_add_val(path_iter, val)
            else:
                # Going Left
                if self.l is None:
                    self.l = PositivityTree()
                self.l.__iter_add_val(path_iter, val)
        else:
            self.v = val

    def find_iter(self, path):
        path_iter = iter(path)
        return self.__iter_find(path_iter)

    def __iter_find(self, path_iter):
        path_head = next(path_iter, None)
        if path_head is None:
            return self.v
        else:
            if path_head >= 0:
                # Going Right
                if self.r is None:
                    return None
                return self.r.__iter_find(path_iter)
            else:
                # Going Left
                if self.l is None:
                    return None
                return self.l.__iter_find(path_iter)

    def find(self, path):
        subtree = self
        for p in path:
            if subtree is None:
                return None
            if p >= 0:
                subtree = subtree.r
            else:
                subtree = subtree.l
        return subtree.v

    # def printTree(self):
    #     if(self.root != None):
    #         self._printTree(self.root)

    # def getRoot(self):
    #     return self.root

    # def add(self, val):
    #     if(self.root == None):
    #         self.root = Node(val)
    #     else:
    #         self._add(val, self.root)

    # def _add(self, val, node):
    #     if(val < node.v):
    #         if(node.l != None):
    #             self._add(val, node.l)
    #         else:
    #             node.l = Node(val)
    #     else:
    #         if(node.r != None):
    #             self._add(val, node.r)
    #         else:
    #             node.r = Node(val)

    # def find(self, val):
    #     if(self.root != None):
    #         return self._find(val, self.root)
    #     else:
    #         return None

    # def _find(self, val, node):
    #     if(val == node.v):
    #         return node
    #     elif(val < node.v and node.l != None):
    #         self._find(val, node.l)
    #     elif(val > node.v and node.r != None):
    #         self._find(val, node.r)

    # def deleteTree(self):
    #     # garbage collector will do this for us.
    #     self.root = None

    # def _printTree(self, node):
    #     if(node != None):
    #         self._printTree(node.l)
    #         print str(node.v) + ' '
    #         self._printTree(node.r)

# #     3
# # 0     4
# #   2      8
# tree = Tree()
# tree.add(3)
# tree.add(4)
# tree.add(0)
# tree.add(8)
# tree.add(2)
# tree.printTree()
# print (tree.find(3)).v
# print tree.find(10)
# tree.deleteTree()
# tree.printTree()

def tests():
    pt = PositivityTree()
    pt.add_val((1,-1,1,-1), (3, (1,2,3,4)))
    # print(pt.find((1,)))
    # print(pt.r.l.r.l.val)
    print(pt.find((1,-2,3,-2)))

if __name__ == '__main__':
    tests()