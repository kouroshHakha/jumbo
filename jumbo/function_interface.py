

class EvalFunction:

    def __init__(self, name: str, function, bounds):
        self.name = name
        self.function = function
        self.bounds = bounds
        self.numDims = bounds.shape[0]


    def __call__(self, x, minimize=True, **kwargs):
        ret = self.function(x, minimize=minimize, **kwargs)
        return ret
