from base import BaseOptimizer
import numpy as np
from collections import defaultdict

"https://github.com/timvvvht/Neural-Networks-and-Optimizers-from-scratch/blob/main/optimizers.py"
"https://towardsdatascience.com/linear-regression-and-gradient-descent-using-only-numpy-53104a834f75"
"https://ml-lectures.org/docs/supervised_learning_wo_NNs/Linear-regression.htmlg"


class SGD(BaseOptimizer):
     
    def __init__(self,  gradient_fn, parameters, learning_rate, loss_fn, batch_size=1, **kwargs):
        super().__init__(gradient_fn, parameters, learning_rate, **kwargs)
        self.batch_size = batch_size
        self.loss_fn = loss_fn

    def batch_generator(self, X, y):
        n_samples = X.shape[0]
        batch_size = self.batch_size
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield X[batch_indices], y[batch_indices]
    
    def update_parameters(self, parameters, gradients):
        parameters -= self.learning_rate * gradients
        return parameters
    
    def optimize(self, X, y):
        n_samples = X.shape[0]
        iteration = 0
        error = np.inf
        parameters = self.parameters
        prev_loss = None
        
        while iteration < self.max_iter and error > self.tol:
            random_indices = np.random.choice(n_samples, size=self.batch_size, replace=False)
            X_batch, y_batch = X[random_indices], y[random_indices]
            
            gradients = self.gradient_fn(X_batch, y_batch, parameters)
            if prev_loss is None:
                prev_loss = self.loss_fn(X, y, parameters)
            parameters = self.update_parameters(parameters, gradients)
            loss = self.loss_fn(X, y, parameters)
            error = abs(np.mean(loss - prev_loss))
            prev_loss = loss
            """
            # Calculate error
            y_pred = self.loss(X)
            error = np.mean((y_pred - y) ** 2)
            
            # Save history
            self.history['error'].append(error)
            """

            iteration += 1
        
        return parameters

if __name__ == "__main__":
    loss_fn =  lambda z, y, x: x**2 - 6*x + 9
    gradient_fn = lambda z, y, x: 2*x - 6
    sgd = SGD(gradient_fn=gradient_fn, parameters=np.array([20.]), loss_fn=loss_fn, learning_rate=1e-2, max_iter=1000, tolerance=1e-12)
    X = y = np.array([0])
    x_opt = sgd.optimize(X, y)
    assert abs(x_opt-3) <= 1e2, "Wrong answer"
    print(f"x_opt: {x_opt} f_opy: {loss_fn(None, None, x_opt)} grad_fn: {gradient_fn(None, None, x_opt)}")
  