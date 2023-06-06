"""from ml.base.base import BaseOptimizer
from ml.optimizers import SGD
import numpy as np
from collections import defaultdict

loss_fn =  lambda x: x**4 - 10*x*3 + 35*x**2 - 50*x + 24
gradient_fn = lambda x, y: 4*x**3 - 30*x*2 + 70*x - 50
sgd = SGD(gradient_fn=gradient_fn, learning_rate=1e-3)
X = np.linspace(-5, 5, 1000)
y = np.zeros_like(X)
sgd.optimize(X, y)"""

if __name__ == "__main__":
    from mllib.optimizers.sgd import SGD
    import numpy as np
    
    loss_fn =  lambda x: x**4 - 10*x*3 + 35*x**2 - 50*x + 24
    gradient_fn = lambda x, y: 4*x**3 - 30*x*2 + 70*x - 50
    sgd = SGD(gradient_fn=gradient_fn, learning_rate=1e-3)
    X = np.linspace(-5, 5, 1000)
    y = np.zeros_like(X)
    sgd.optimize(X, y)