from ..base import BaseOptimizer
import numpy as np
from collections import defaultdict

"https://github.com/timvvvht/Neural-Networks-and-Optimizers-from-scratch/blob/main/optimizers.py"
"https://towardsdatascience.com/linear-regression-and-gradient-descent-using-only-numpy-53104a834f75"
"https://ml-lectures.org/docs/supervised_learning_wo_NNs/Linear-regression.htmlg"


class SGD(BaseOptimizer):
     def __init__(self, gradient_fn, learning_rate, tolerance=1e-5, max_iter=1000, batch_size=1):
        super().__init__(gradient_fn, learning_rate, tolerance, max_iter)
        self.batch_size = batch_size
    
     def update_parameters(self, parameters, gradient_fn):
        for key in parameters:
            parameters[key] -= self.learning_rate * gradient_fn[key]
        return parameters
    
     def optimize(self, X, y):
        parameters = defaultdict(float)
        n_samples = X.shape[0]
        iteration = 0
        error = np.inf
        
        while iteration < self.max_iter and error > self.tol:
            random_indices = np.random.choice(n_samples, size=self.batch_size, replace=False)
            X_batch, y_batch = X[random_indices], y[random_indices]
            
            gradients = self.gradient_fn(X_batch, y_batch, parameters)
            parameters = self.update_parameters(parameters, gradients)
            
            # Calculate error
            y_pred = self.predict(X)
            error = np.mean((y_pred - y) ** 2)
            
            # Save history
            self.history['error'].append(error)
            
            iteration += 1
        
        return parameters