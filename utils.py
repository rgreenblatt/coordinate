import torch
import torch.nn as nn
import collections

class DictMathWrapper():
    def __init__(self, d):
        self.d = d
    def apply(self, f, o=None):
        if o is not None:
            is_dict = isinstance(o, collections.Mapping) or isinstance(o, DictMathWrapper)

            if is_dict:
                assert len(o) == len(self.d)

        new = {}
        for name in self.d:
            if o is not None:
                new[name] = f(self.d[name], o[name] if is_dict else o)
            else:
                new[name] = f(self.d[name])

        return DictMathWrapper(new)

    def zeros(self):
        def in_place_op_to_zeros(x):
            try:
                return x.zeros()
            except AttributeError:
                return torch.zeros_like(x)
                
        return self.apply(in_place_op_to_zeros)

    def __getitem__(self, idx):
        return self.d[idx]

    def __len__(self):
        return len(self.d)

    def __iter__(self):
        return iter(self.d)
        
    def __add__(self, o):
        return self.apply(lambda x, y : x + y, o)

    def __radd__(self, o):
        return self.apply(lambda x, y : y + x, o)

    def __sub__(self, o):
        return self.apply(lambda x, y : x - y, o)

    def __rsub__(self, o):
        return self.apply(lambda x, y : y - x, o)
        
    def __mul__(self, o):
        return self.apply(lambda x, y : x * y, o)

    def __rmul__(self, o):
        return self.apply(lambda x, y : y * x, o)
        
    def __truediv__(self, o):
        return self.apply(lambda x, y : x / y, o)

    def __rtruediv__(self, o):
        return self.apply(lambda x, y : y / x, o)
        
    def __and__(self, o):
        return self.apply(lambda x, y : x & y, o)

    def __rand__(self, o):
        return self.apply(lambda x, y : y & x, o)
        
    def __or__(self, o):
        return self.apply(lambda x, y : x | y, o)

    def __ror__(self, o):
        return self.apply(lambda x, y : y | x, o)

    def __lt__(self, o):
        return self.apply(lambda x, y : x < y, o)

    def __rlt__(self, o):
        return self.apply(lambda x, y : y < x, o)

    def __le__(self, o):
        return self.apply(lambda x, y : x <= y, o)

    def __rle__(self, o):
        return self.apply(lambda x, y : y <= x, o)

    def __eq__(self, o):
        return self.apply(lambda x, y : x == y, o)

    def __req__(self, o):
        return self.apply(lambda x, y : y == x, o)

    def __ne__(self, o):
        return self.apply(lambda x, y : x != y, o)

    def __rne__(self, o):
        return self.apply(lambda x, y : y != x, o)

    def __gt__(self, o):
        return self.apply(lambda x, y : x > y, o)

    def __rgt__(self, o):
        return self.apply(lambda x, y : y > x, o)

    def __ge__(self, o):
        return self.apply(lambda x, y : x >= y, o)

    def __rge__(self, o):
        return self.apply(lambda x, y : y >= x, o)

    def in_place_op(self, f):
        for name in self.d:
            f(self.d[name])

    def any(self):
        full = []
        for name in self.d:
            full.append(self.d[name].any())
        return torch.tensor(full).any()

    def all(self):
        full = []
        for name in self.d:
            full.append(self.d[name].all())
        return torch.tensor(full).all()

def coordinate_function(f):
    def f_function(inp_l):
        out = []

        for inp in inp_l:
            def do_f(x):
                try:
                    return x.apply(do_f)
                except AttributeError:
                    return f(x)
            o = do_f(inp)
            out.append(o)

        return out
    return f_function

coordinate_sigmoid = coordinate_function(torch.sigmoid)

#x+a(e^x-1) for x<0 and x-a(e^{-x}-1) for x>=0 is used to have a non linear symmetric function
def non_linear_symmetric(a):
    def f(x):
        return torch.where(x < 0, x + a * (torch.exp(x) - 1), x - a * (torch.exp(-x) - 1))
    return f
