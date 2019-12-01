class gradient():
    '''
        used to make the gradient averaging easier
    '''
    def __init__(self, size):
        self.size = size
        self.info = [
        [
            [
            0,
            [0 for j in range(size[l-1])]
            ]for k in range(size[l])
        ] for l in range(len(size))
        ]
        
    def __str__(self):
        return str(self.info)
        
    def __len__(self):
        return len(self.info)
        
    def __getitem__(self, n):
        if not isinstance(n, int):
            raise TypeError("Can only index by an integer")
        return self.info[n]
        
    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            new_gradient = gradient(self.size)
            for nl,ol in zip(new_gradient, self):
                for nk,ok in zip(nl,ol):
                    nk[0] = ok[0] / other
                    for j in range(len(nk[1])):
                        nk[1][j] = ok[1][j] / other
            return new_gradient
        elif isinstance(other, gradient):
            assert self.size == other.size, "Using operator `/` requires gradients of the same dimensions"
            new_gradient = gradient(self.size)
            for l in range(len(other)):
                for k in range(len(other[l])):
                    new_gradient[l][k][0] = self[l][k][0] / other[l][k][0]
                    for j in range(len(other[l][k][1])):
                        new_gradient[l][k][1][j] = self[l][k][1][j] / other[l][k][1][j]
            return new_gradient
        else:
            raise TypeError("unsupported operand type(s) for /: 'gradient' and '"+other.__class__.__name__+"'")
        return None
        
    def __itruediv__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            for l in self:
                for k in l:
                    k[0] /= other
                    for j in range(len(k[1])):
                        k[1][j] /= other
        elif isinstance(other, gradient):
            assert self.size == other.size, "Using operator `/` requires gradients of the same dimensions"
            for l in range(len(other)):
                for k in range(len(other[l])):
                    self[l][k][0] /= other[l][k][0]
                    for j in range(len(other[l][k][1])):
                        self[l][k][1][j] /= other[l][k][1][j]
        else:
            raise TypeError("unsupported operand type(s) for /: 'gradient' and '"+other.__class__.__name__+"'")
        return self
        
    def __mul__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            new_gradient = gradient(self.size)
            for nl,ol in zip(new_gradient, self):
                for nk,ok in zip(nl,ol):
                    nk[0] = ok[0] * other
                    for j in range(len(nk[1])):
                        nk[1][j] = ok[1][j] * other
            return new_gradient
        elif isinstance(other, gradient):
            assert self.size == other.size, "Using operator `*` requires gradients of the same dimensions"
            new_gradient = gradient(self.size)
            for l in range(len(other)):
                for k in range(len(other[l])):
                    new_gradient[l][k][0] = self[l][k][0] * other[l][k][0]
                    for j in range(len(other[l][k][1])):
                        new_gradient[l][k][1][j] = self[l][k][1][j] * other[l][k][1][j]
            return new_gradient
        else:
            raise TypeError("unsupported operand type(s) for *: 'gradient' and '"+other.__class__.__name__+"'")
        return None
        
    def __imul__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            for l in self:
                for k in l:
                    k[0] *= other
                    for j in range(len(k[1])):
                        k[1][j] *= other
        elif isinstance(other, gradient):
            assert self.size == other.size, "Using operator `*` requires gradients of the same dimensions"
            for l in range(len(other)):
                for k in range(len(other[l])):
                    self[l][k][0] *= other[l][k][0]
                    for j in range(len(other[l][k][1])):
                        self[l][k][1][j] *= other[l][k][1][j]
        else:
            raise TypeError("unsupported operand type(s) for *: 'gradient' and '"+other.__class__.__name__+"'")
        return self
        
    def __add__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            new_gradient = gradient(self.size)
            for nl,ol in zip(new_gradient, self):
                for nk,ok in zip(nl,ol):
                    nk[0] = ok[0] + other
                    for j in range(len(nk[1])):
                        nk[1][j] = ok[1][j] + other
            return new_gradient
        elif isinstance(other, gradient):
            assert self.size == other.size, "Using operator `+` requires gradients of the same dimensions"
            new_gradient = gradient(self.size)
            for l in range(len(other)):
                for k in range(len(other[l])):
                    new_gradient[l][k][0] = self[l][k][0] + other[l][k][0]
                    for j in range(len(other[l][k][1])):
                        new_gradient[l][k][1][j] = self[l][k][1][j] + other[l][k][1][j]
            return new_gradient
        else:
            raise TypeError("unsupported operand type(s) for +: 'gradient' and '"+other.__class__.__name__+"'")
        return None
        
    def __iadd__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            for l in self:
                for k in l:
                    k[0] += other
                    for j in range(len(k[1])):
                        k[1][j] += other
        elif isinstance(other, gradient):
            assert self.size == other.size, "Using operator `+` requires gradients of the same dimensions"
            for l in range(len(other)):
                for k in range(len(other[l])):
                    self[l][k][0] += other[l][k][0]
                    for j in range(len(other[l][k][1])):
                        self[l][k][1][j] += other[l][k][1][j]
        else:
            raise TypeError("unsupported operand type(s) for +: 'gradient' and '"+other.__class__.__name__+"'")
        return self
        
    def __sub__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            new_gradient = gradient(self.size)
            for nl,ol in zip(new_gradient, self):
                for nk,ok in zip(nl,ol):
                    nk[0] = ok[0] - other
                    for j in range(len(nk[1])):
                        nk[1][j] = ok[1][j] - other
            return new_gradient
        elif isinstance(other, gradient):
            assert self.size == other.size, "Using operator `-` requires gradients of the same dimensions"
            new_gradient = gradient(self.size)
            for l in range(len(other)):
                for k in range(len(other[l])):
                    new_gradient[l][k][0] = self[l][k][0] - other[l][k][0]
                    for j in range(len(other[l][k][1])):
                        new_gradient[l][k][1][j] = self[l][k][1][j] - other[l][k][1][j]
            return new_gradient
        else:
            raise TypeError("unsupported operand type(s) for -: 'gradient' and '"+other.__class__.__name__+"'")
        return None
        
    def __isub__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            for l in self:
                for k in l:
                    k[0] -= other
                    for j in range(len(k[1])):
                        k[1][j] -= other
        elif isinstance(other, gradient):
            assert self.size == other.size, "Using operator `-` requires gradients of the same dimensions"
            for l in range(len(other)):
                for k in range(len(other[l])):
                    self[l][k][0] -= other[l][k][0]
                    for j in range(len(other[l][k][1])):
                        self[l][k][1][j] -= other[l][k][1][j]
        else:
            raise TypeError("unsupported operand type(s) for -: 'gradient' and '"+other.__class__.__name__+"'")
        return self