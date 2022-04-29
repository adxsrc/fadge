from collections import namedtuple
from jax.tree_util import tree_flatten, tree_unflatten, register_pytree_node_class

class NodeLeaf:
    def __init__(self, *args):
        if len(args) == 0:
            self.data = None
        elif len(args) == 1:
            self.data = args[0]
        else:
            self.data = args

        self.__class__ = NodeLeaf2

    def isleaf(self):
        return not isinstance(self.data, tuple)

    def __repr__(self):
        if self.isleaf():
            return '>'+self.data.__repr__()
        else:
            return '[]'

    def tree_flatten(self):
        children = self.data
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(aux_data, children):
        return Space(*children)

class NodeLeaf2(NodeLeaf):
    def cool(self):
        print('cool')

# Global registration
register_pytree_node_class(NodeLeaf2)

#from collections import UserList as Space
def testing(s):
    return s.x == 1

x = NodeLeaf(1)
y = NodeLeaf(x,x)
z = NodeLeaf(x,y)

f, t = tree_flatten(z, is_leaf=lambda x: x.isleaf())
print(f)
print(t)

print(z.__class__)
#z.__class__ = NodeLeaf2

z.cool()
