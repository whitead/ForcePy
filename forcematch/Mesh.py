class UniformMesh:
    """Uniform mesh. 
    """
    def __init__(self, l, r, dx):
        self.l = l
        self.r = r
        self.dx = dx
        self.length = int(ceil((self.r - self.l) / self.dx))

    def max(self):
        return self.r

    def min(self):
        return self.l

    def mesh_index(self, x):
        return max(0, min(self.length - 1, int(floor( (x - self.l) / self.dx) )))
        
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return i * self.dx + self.l    

